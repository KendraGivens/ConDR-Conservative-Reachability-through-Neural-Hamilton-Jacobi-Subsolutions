import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
from lvfm.helpers import compute_metrics
from scipy.interpolate import RegularGridInterpolator
import math
from lvfm.grids import linear_axis, angle_axis, periodic_nearest_index

@torch.no_grad()

@torch.no_grad()
def plot_linear_oscillator_2d(
    model,
    device,
    tau,
    gt_values=None,              # shape: (nx, nx)
    x1_bounds=(-1.0, 1.0),
    x2_bounds=(-1.0, 1.0),
    nx=201,
    chunk_size=4096,
    scale_to_minus1_1=False,
    T=1.0,
    scale_time_to_01=True,
    title=None,
    show_heatmap=True,
    save_path=None,
    dpi=300,
    show=False,
):
    model_was_training = model.training
    model.eval()

    x1 = np.linspace(x1_bounds[0], x1_bounds[1], nx)
    x2 = np.linspace(x2_bounds[0], x2_bounds[1], nx)
    X1, X2 = np.meshgrid(x1, x2, indexing="xy")

    pts_phys = np.stack([X1.reshape(-1), X2.reshape(-1)], axis=-1).astype(np.float32)

    pts_net = pts_phys.copy()
    if scale_to_minus1_1:
        lo = np.array([x1_bounds[0], x2_bounds[0]], dtype=np.float32)
        hi = np.array([x1_bounds[1], x2_bounds[1]], dtype=np.float32)
        pts_net = 2.0 * (pts_net - lo) / (hi - lo) - 1.0

    tau_net = tau / T if scale_time_to_01 else tau
    tau_col = np.full((pts_net.shape[0], 1), tau_net, dtype=np.float32)

    # Model expects xt = [x1, x2, tau]
    xt_np = np.concatenate([pts_net, tau_col], axis=-1)

    vals = []
    for start in range(0, xt_np.shape[0], chunk_size):
        end = min(start + chunk_size, xt_np.shape[0])
        xt_chunk = torch.tensor(xt_np[start:end], dtype=torch.float32, device=device)

        V_chunk = model(xt_chunk).squeeze(-1)
        vals.append(V_chunk.detach().cpu().numpy())

    V_pred = np.concatenate(vals, axis=0).reshape(nx, nx)

    gt_values = None if gt_values is None else np.asarray(gt_values)
    if gt_values is not None and gt_values.shape != (nx, nx):
        raise ValueError(f"gt_values must have shape {(nx, nx)}, got {gt_values.shape}")

    abs_error = None
    if gt_values is not None:
        abs_error = np.abs(V_pred - gt_values)

    ncols = 3 if gt_values is not None else 1
    fig, axes = plt.subplots(1, ncols, figsize=(18, 5) if ncols == 3 else (7, 6))
    if ncols == 1:
        axes = [axes]

    if gt_values is not None:
        vmin = min(V_pred.min(), gt_values.min())
        vmax = max(V_pred.max(), gt_values.max())
    else:
        vmin = V_pred.min()
        vmax = V_pred.max()

    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-6

    levels = np.linspace(vmin, vmax, 40)

    # Left panel: prediction
    ax = axes[0]
    if show_heatmap:
        cf_pred = ax.contourf(X1, X2, V_pred, levels=levels, vmin=vmin, vmax=vmax)
        fig.colorbar(cf_pred, ax=ax, label="V(x, tau)")

    ax.contour(X1, X2, V_pred, levels=[0.0], colors="red", linewidths=2.5, linestyles="-")

    if gt_values is not None:
        ax.contour(X1, X2, gt_values, levels=[0.0], colors="black", linewidths=2.5, linestyles="--")

    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_xlim(x1_bounds)
    ax.set_ylim(x2_bounds)
    ax.set_aspect("equal")
    ax.set_title(f"Predicted at tau={tau:.2f}")

    if gt_values is not None:
        # Middle panel: ground truth
        ax = axes[1]
        if show_heatmap:
            cf_gt = ax.contourf(X1, X2, gt_values, levels=levels, vmin=vmin, vmax=vmax)
            fig.colorbar(cf_gt, ax=ax, label="V(x, tau)")

        ax.contour(X1, X2, gt_values, levels=[0.0], colors="black", linewidths=2.5, linestyles="--")

        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_xlim(x1_bounds)
        ax.set_ylim(x2_bounds)
        ax.set_aspect("equal")
        ax.set_title(f"Ground Truth at tau={tau:.2f}")

        # Right panel: absolute error
        ax = axes[2]
        err_levels = np.linspace(0.0, max(abs_error.max(), 1e-8), 40)
        cf_err = ax.contourf(X1, X2, abs_error, levels=err_levels)
        fig.colorbar(cf_err, ax=ax, label=r"$|V_{\mathrm{pred}} - V_{\mathrm{gt}}|$")

        ax.contour(X1, X2, V_pred, levels=[0.0], colors="red", linewidths=2.0, linestyles="-")
        ax.contour(X1, X2, gt_values, levels=[0.0], colors="black", linewidths=2.0, linestyles="--")

        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_xlim(x1_bounds)
        ax.set_ylim(x2_bounds)
        ax.set_aspect("equal")
        ax.set_title("Absolute Error")

    handles = [Line2D([0], [0], color="red", lw=2.5, linestyle="-", label="learned V=0")]
    if gt_values is not None:
        handles.append(Line2D([0], [0], color="black", lw=2.5, linestyle="--", label="ground-truth V=0"))

    if title is None:
        title = f"Predicted vs Ground Truth at tau={tau:.2f}"
    fig.suptitle(title, y=0.99)

    if gt_values is not None:
        mae, iou = compute_metrics(V_pred, gt_values)
        metric_text = f"MAE: {mae:.4e}    BRT Overlap: {iou:.4f}"
        fig.text(
            0.5,
            0.93,
            metric_text,
            ha="center",
            va="center",
        )

    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=len(handles),
        bbox_to_anchor=(0.5, 0.915),
    )

    fig.tight_layout(rect=[0, 0, 1, 0.84])

    if save_path is None:
        save_path = Path(f"linear_oscillator_deepreach_tau_{tau:.2f}.png")
    else:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    if model_was_training:
        model.train()


@torch.no_grad()

@torch.no_grad()


@torch.no_grad()

def _eval_air3d_slice(model, device, tau, psi_val, x_bounds, y_bounds, psi_bounds, nx,
                      chunk_size, scale_to_minus1_1, T, scale_time_to_01):
    """Return V_pred (nx, nx) for a given (tau, psi) slice. No grad."""
    x = np.linspace(x_bounds[0], x_bounds[1], nx)
    y = np.linspace(y_bounds[0], y_bounds[1], nx)
    X, Y = np.meshgrid(x, y, indexing="xy")
    psi = np.full((X.size, 1), psi_val, dtype=np.float32)
    pts_phys = np.concatenate([X.reshape(-1, 1).astype(np.float32),
                                Y.reshape(-1, 1).astype(np.float32), psi], axis=-1)
    pts_net = pts_phys.copy()
    if scale_to_minus1_1:
        pts_net[:, 0] = 2.0 * (pts_phys[:, 0] - x_bounds[0]) / (x_bounds[1] - x_bounds[0]) - 1.0
        pts_net[:, 1] = 2.0 * (pts_phys[:, 1] - y_bounds[0]) / (y_bounds[1] - y_bounds[0]) - 1.0
        pts_net[:, 2] = pts_phys[:, 2] / (1.2 * math.pi)
    tau_net = tau / T if scale_time_to_01 else tau
    tau_col = np.full((pts_net.shape[0], 1), tau_net, dtype=np.float32)
    xt_np = np.concatenate([pts_net, tau_col], axis=-1)
    vals = []
    for start in range(0, xt_np.shape[0], chunk_size):
        end = min(start + chunk_size, xt_np.shape[0])
        xt_chunk = torch.tensor(xt_np[start:end], dtype=torch.float32, device=device)
        tau_chunk = torch.full((end - start,), tau, dtype=torch.float32, device=device)
        V_chunk = model(xt_chunk)
        vals.append(V_chunk.detach().cpu().numpy())
    return np.concatenate(vals, axis=0).reshape(nx, nx)


@torch.no_grad()
def plot_air3d_psi_grid(
    model,
    device,
    tau,
    gt_values_3d,
    psi_values=None,
    x_bounds=(-1.0, 1.0),
    y_bounds=(-1.0, 1.0),
    psi_bounds=(-math.pi, math.pi),
    nx=101,
    chunk_size=4096,
    scale_to_minus1_1=True,
    T=1.0,
    scale_time_to_01=True,
    save_path=None,
    dpi=150,
    show=False,
):
    """
    Grid of V=0 contour overlays at multiple psi slices for a single tau.
    Red = predicted, black dashed = GT.  Per-panel IoU in the title.
    """
    model_was_training = model.training
    model.eval()

    gt_arr = np.asarray(gt_values_3d)  # (nx, ny, npsi)
    npsi_gt = gt_arr.shape[2]
    psi_axis = np.linspace(psi_bounds[0], psi_bounds[1], npsi_gt)

    if psi_values is None:
        psi_values = np.linspace(psi_bounds[0], psi_bounds[1], 8, endpoint=False)

    x_grid = np.linspace(x_bounds[0], x_bounds[1], nx)
    y_grid = np.linspace(y_bounds[0], y_bounds[1], nx)
    X, Y = np.meshgrid(x_grid, y_grid, indexing="xy")

    ncols = 4
    nrows = math.ceil(len(psi_values) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4))
    axes = np.array(axes).reshape(nrows, ncols)

    iou_list = []
    for idx, psi_val in enumerate(psi_values):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        psi_idx = int(np.argmin(np.abs(psi_axis - psi_val)))
        gt_slice = gt_arr[:, :, psi_idx].T  # (ny, nx)

        V_pred = _eval_air3d_slice(
            model, device, tau, float(psi_val), x_bounds, y_bounds, psi_bounds,
            nx, chunk_size, scale_to_minus1_1, T, scale_time_to_01,
        )

        _, iou = compute_metrics(V_pred, gt_slice)
        iou_list.append(iou)

        ax.contour(X, Y, V_pred, levels=[0.0], colors="red", linewidths=2.0, linestyles="-")
        ax.contour(X, Y, gt_slice, levels=[0.0], colors="black", linewidths=2.0, linestyles="--")
        ax.set_xlim(x_bounds)
        ax.set_ylim(y_bounds)
        ax.set_aspect("equal")
        ax.set_title(f"ψ={psi_val:.2f}  IoU={iou:.3f}", fontsize=9)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    for idx in range(len(psi_values), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    mean_iou = float(np.mean(iou_list))
    handles = [
        Line2D([0], [0], color="red", lw=2.0, linestyle="-", label="predicted V=0"),
        Line2D([0], [0], color="black", lw=2.0, linestyle="--", label="GT V=0"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.00))
    fig.suptitle(
        f"Air3D BRT boundary — τ={tau:.2f}   mean IoU={mean_iou:.3f}",
        y=1.02, fontsize=12,
    )
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    if model_was_training:
        model.train()

    return iou_list


def compute_air3d_control_accuracy(
    model,
    device,
    tau,
    gt_values_3d,
    psi_values=None,
    x_bounds=(-1.0, 1.0),
    y_bounds=(-1.0, 1.0),
    psi_bounds=(-math.pi, math.pi),
    nx=51,
    scale_to_minus1_1=True,
    T=1.0,
    scale_time_to_01=True,
    chunk_size=2048,
    angle_alpha_factor=1.2,
):
    """
    Fraction of grid points where the predicted optimal pursuer control sign matches GT.

    u* = sign(p_x * y - p_y * x - p_psi)  where p = spatial gradient of V.

    GT: finite-difference gradient of the GT value grid.
    Model: autograd spatial gradient of model V.
    """
    model_was_training = model.training
    model.eval()

    gt_arr = np.asarray(gt_values_3d)
    npsi_gt = gt_arr.shape[2]
    # psi is PERIODIC on the hj grid: nodes at lo + k*(hi-lo)/n, none at hi. The
    # old endpoint=True axis mis-sized dpsi and offset the slice index by up to a
    # cell (audit P1-1).
    psi_axis = angle_axis(psi_bounds, npsi_gt)
    dpsi = psi_axis[1] - psi_axis[0]

    if psi_values is None:
        psi_values = angle_axis(psi_bounds, 8)

    x_arr = linear_axis(x_bounds, nx)
    y_arr = linear_axis(y_bounds, nx)
    X, Y = np.meshgrid(x_arr, y_arr, indexing="xy")  # (ny, nx)

    dx = x_arr[1] - x_arr[0]
    dy = y_arr[1] - y_arr[0]

    results = {}
    for psi_val in psi_values:
        psi_idx = int(periodic_nearest_index(psi_axis, float(psi_val)))
        gt_slice = gt_arr[:, :, psi_idx].T  # (ny, nx)

        # GT gradients via finite differences. psi WRAPS, so the central
        # difference is taken modulo npsi_gt; the old code fell back to a
        # one-sided difference at the seam -- which is exactly where the default
        # psi_values start (psi = -pi), degrading 1/8 of the reported metric
        # (audit P1-1).
        gt_px = np.gradient(gt_slice, dx, axis=1)
        gt_py = np.gradient(gt_slice, dy, axis=0)
        nxt = (psi_idx + 1) % npsi_gt
        prv = (psi_idx - 1) % npsi_gt
        gt_ppsi = (gt_arr[:, :, nxt] - gt_arr[:, :, prv]).T / (2 * dpsi)

        gt_ctrl = np.sign(gt_px * Y - gt_py * X - gt_ppsi)

        # Model spatial gradients via autograd
        psi_col = np.full((X.size, 1), float(psi_val), dtype=np.float32)
        pts_phys = np.concatenate([X.reshape(-1, 1).astype(np.float32),
                                    Y.reshape(-1, 1).astype(np.float32), psi_col], axis=-1)
        pts_net = pts_phys.copy()
        if scale_to_minus1_1:
            pts_net[:, 0] = 2.0 * (pts_phys[:, 0] - x_bounds[0]) / (x_bounds[1] - x_bounds[0]) - 1.0
            pts_net[:, 1] = 2.0 * (pts_phys[:, 1] - y_bounds[0]) / (y_bounds[1] - y_bounds[0]) - 1.0
            pts_net[:, 2] = pts_phys[:, 2] / (float(angle_alpha_factor) * math.pi)

        tau_net = tau / T if scale_time_to_01 else tau
        tau_col_arr = np.full((pts_net.shape[0], 1), tau_net, dtype=np.float32)
        xt_np = np.concatenate([pts_net, tau_col_arr], axis=-1)

        model_grads = []
        for start in range(0, xt_np.shape[0], chunk_size):
            end = min(start + chunk_size, xt_np.shape[0])
            xt_chunk = torch.tensor(xt_np[start:end], dtype=torch.float32, device=device).requires_grad_(True)
            tau_chunk = torch.full((end - start,), tau, dtype=torch.float32, device=device)
            with torch.enable_grad():
                V_chunk = model(xt_chunk)
                grads = torch.autograd.grad(V_chunk.sum(), xt_chunk)[0]
            model_grads.append(grads[:, :3].detach().cpu().numpy())

        model_grad_np = np.concatenate(model_grads, axis=0)  # (N, 3) net coords
        if scale_to_minus1_1:
            model_grad_np[:, 0] *= 2.0 / (x_bounds[1] - x_bounds[0])
            model_grad_np[:, 1] *= 2.0 / (y_bounds[1] - y_bounds[0])
            model_grad_np[:, 2] /= 1.2 * math.pi

        model_px = model_grad_np[:, 0].reshape(nx, nx)
        model_py = model_grad_np[:, 1].reshape(nx, nx)
        model_ppsi = model_grad_np[:, 2].reshape(nx, nx)
        model_ctrl = np.sign(model_px * Y - model_py * X - model_ppsi)

        valid = (gt_ctrl != 0) & (model_ctrl != 0)
        acc = float((gt_ctrl[valid] == model_ctrl[valid]).mean()) if valid.any() else float("nan")
        results[float(psi_val)] = acc

    results["mean"] = float(np.nanmean([v for k, v in results.items() if k != "mean"]))
    if model_was_training:
        model.train()
    return results
