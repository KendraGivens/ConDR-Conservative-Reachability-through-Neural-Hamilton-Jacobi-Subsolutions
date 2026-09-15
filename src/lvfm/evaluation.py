import math

import numpy as np
import torch

from lvfm.helpers import compute_metrics, squeeze_last


def _value_and_residual(model, residual, xt, tau_phys, deepreach=True):
    xt = xt.requires_grad_(True)
    V = squeeze_last(model(xt))
    r = residual.compute_deepreach_residual(model, xt)
    return V, None, None, None, r


def uniform_volume_frac(model, residual, x_net, tau_phys, device, deepreach=False, chunk_size=8192):
    """Monte-Carlo estimate of vol({V(., tau) <= 0}) over a FIXED UNIFORM sample.

    `x_net` is a fixed, seeded uniform draw in network coordinates; `tau_phys` is
    the scalar time to evaluate at. Kept separate from the training batch on
    purpose: the adaptive sampler's batch is drawn from a model-dependent measure
    (boundary- and residual-weighted), so a BRT fraction computed on it is not a
    volume and is not comparable across models or across steps (audit P1-2).
    """
    was_training = model.training
    model.eval()
    n = x_net.shape[0]
    # Network time, not physical time -- the model consumes tau/T when
    # scale_time_to_01 is set.
    tau_net = float(tau_phys) / residual.T if residual.scale_time_to_01 else float(tau_phys)
    fracs = []
    with torch.no_grad():
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            xn = x_net[start:end].to(device).float()
            tau_col = torch.full((end - start, 1), tau_net, device=device)
            xt = torch.cat([xn, tau_col], dim=-1)
            tau_v = torch.full((end - start,), float(tau_phys), device=device)
            V = model(xt)
            fracs.append((squeeze_last(V).reshape(-1) <= 0.0).float().cpu())
    if was_training:
        model.train()
    return float(torch.cat(fracs).mean())


def evaluate_unsupervised_batch(model, residual, batch, device, deepreach=False, band_width=0.1,
                                volume_probe=None, volume_tau=None):
    was_training = model.training
    model.eval()

    xt = batch["xt_interior"].to(device).float().requires_grad_(True)
    tau_phys = batch["tau_interior_phys"].to(device).float()

    with torch.enable_grad():
        V, raw, latent, dlatent, r = _value_and_residual(
            model=model,
            residual=residual,
            xt=xt,
            tau_phys=tau_phys,
            deepreach=deepreach,
        )
        grad_xt = torch.autograd.grad(V.sum(), xt, retain_graph=True, create_graph=False)[0]
        grad_x_phys = residual._unscale_spatial_gradient(grad_xt[:, : residual.coordinate_dim])

    r_abs = r.detach().abs().reshape(-1)
    V_det = V.detach().reshape(-1)
    band = V_det.abs() < float(band_width)
    grad_norm = grad_x_phys.detach().norm(dim=-1)

    out = {
        "residual_mean": float(r_abs.mean().cpu()),
        "residual_median": float(r_abs.median().cpu()),
        "residual_p90": float(torch.quantile(r_abs, 0.90).cpu()),
        "residual_p99": float(torch.quantile(r_abs, 0.99).cpu()),
        # Measured on the training batch -- kept under an honest name because the
        # batch measure is model-dependent under the adaptive sampler (audit P1-2).
        "batch_brt_frac": float((V_det <= 0.0).float().mean().cpu()),
        "grad_norm_mean": float(grad_norm.mean().cpu()),
        "grad_norm_p90": float(torch.quantile(grad_norm, 0.90).cpu()),
    }

    if band.any():
        out["boundary_residual_mean"] = float(r_abs[band].mean().cpu())
        out["boundary_grad_norm_mean"] = float(grad_norm[band].mean().cpu())
    else:
        out["boundary_residual_mean"] = math.nan
        out["boundary_grad_norm_mean"] = math.nan

    if latent is not None:
        out["latent_norm_mean"] = float(latent.detach().norm(dim=-1).mean().cpu())
    if dlatent is not None:
        out["latent_velocity_norm_mean"] = float(dlatent.detach().norm(dim=-1).mean().cpu())
    if raw is not None:
        out["raw_abs_mean"] = float(raw.detach().abs().mean().cpu())
        out["raw_rms"] = float(torch.sqrt(raw.detach().pow(2).mean()).cpu())

    # The comparable volume estimate: fixed uniform sample, current tau.
    if volume_probe is not None:
        tau_for_volume = batch["tau_interior_phys"].max().item() if volume_tau is None else float(volume_tau)
        out["pred_brt_volume_frac"] = uniform_volume_frac(
            model, residual, volume_probe, tau_for_volume, device, deepreach=deepreach,
        )
        out["pred_brt_volume_tau"] = tau_for_volume
    else:
        # No probe supplied: emit NaN rather than the batch number, so a missing
        # probe can never masquerade as a volume.
        out["pred_brt_volume_frac"] = math.nan

    if was_training:
        model.train()
    return out


def evaluate_gt_arrays(V_pred, V_gt, boundary_width=0.05):
    V_pred = np.asarray(V_pred)
    V_gt = np.asarray(V_gt)
    mse = float(np.mean((V_pred - V_gt) ** 2))
    _, iou = compute_metrics(V_pred, V_gt)

    pred_unsafe = V_pred <= 0.0
    gt_unsafe = V_gt <= 0.0
    sign_accuracy = float((pred_unsafe == gt_unsafe).mean())
    false_safe = float((~pred_unsafe & gt_unsafe).sum() / max(gt_unsafe.sum(), 1))
    false_unsafe = float((pred_unsafe & ~gt_unsafe).sum() / max((~gt_unsafe).sum(), 1))

    band = np.abs(V_gt) < boundary_width
    boundary_sign_accuracy = float((pred_unsafe[band] == gt_unsafe[band]).mean()) if band.any() else math.nan

    return {
        "value_mse": mse,
        "brt_iou": float(iou),
        "sign_accuracy": sign_accuracy,
        "false_safe_rate": false_safe,
        "false_unsafe_rate": false_unsafe,
        "boundary_sign_accuracy": boundary_sign_accuracy,
    }


def approximate_hausdorff_from_levelsets(V_pred, V_gt, x_axis, y_axis):
    pred_pts = _zero_band_points(np.asarray(V_pred), x_axis, y_axis)
    gt_pts = _zero_band_points(np.asarray(V_gt), x_axis, y_axis)
    if pred_pts.size == 0 or gt_pts.size == 0:
        return math.nan
    d_pg = _nearest_distances(pred_pts, gt_pts).max()
    d_gp = _nearest_distances(gt_pts, pred_pts).max()
    return float(max(d_pg, d_gp))


def _zero_band_points(values, x_axis, y_axis):
    sign = values <= 0.0
    edge = np.zeros_like(sign, dtype=bool)
    edge[:-1, :] |= sign[:-1, :] != sign[1:, :]
    edge[:, :-1] |= sign[:, :-1] != sign[:, 1:]
    yy, xx = np.where(edge)
    if xx.size == 0:
        return np.empty((0, 2))
    return np.stack([np.asarray(x_axis)[xx], np.asarray(y_axis)[yy]], axis=-1)


def _nearest_distances(a, b, chunk_size=2048):
    outs = []
    for start in range(0, a.shape[0], chunk_size):
        chunk = a[start : start + chunk_size]
        d2 = ((chunk[:, None, :] - b[None, :, :]) ** 2).sum(axis=-1)
        outs.append(np.sqrt(d2.min(axis=1)))
    return np.concatenate(outs, axis=0)
