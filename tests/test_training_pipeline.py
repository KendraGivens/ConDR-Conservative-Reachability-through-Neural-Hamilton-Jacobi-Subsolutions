"""Pipeline smoke tests for the DeepReach 2x2 (replaces the ROM-era suite).

The previous file exercised INR_PNODE / MixtureINR_PNODE / the conditioned
decoders, all removed on 2026-08-27; it is kept at
archive/scripts_offthesis/test_unsupervised_features_ROM.py.
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import torch

from lvfm.datasets import Air3DDataset, LinearOscillator2DDataset
from lvfm.evaluation import evaluate_unsupervised_batch
from lvfm.managers import LossManager
from lvfm.models import DeepReachExact, DeepReachModel
from lvfm.residuals import Air3DResidual, LinearOscillator2DResidual
from lvfm.samplers import MixedCollocationSampler, SamplerConfig
from lvfm.training import causal_binned_loss


def make_air3d(n=64, correction="negative_softplus", loss_mode="complementarity"):
    dataset = Air3DDataset(num_batches=4, num_interior=n, num_terminal=8, T=1.0,
                           tau_max=1.0, x_bounds=(-1, 1), y_bounds=(-1, 1),
                           efficient=True, num_unique_taus=4)
    residual = Air3DResidual(T=1.0, x_bounds=(-1, 1), y_bounds=(-1, 1))
    lm = LossManager(residual=residual, hjvi_loss_mode=loss_mode,
                     use_boundary_weighting=True)
    backbone = DeepReachModel(in_dim=4, hidden_dim=32, num_layers=2, omega0=30.0)
    model = DeepReachExact(backbone=backbone, residual=residual, coordinate_dim=3,
                           correction_mode=correction)
    return dataset, residual, lm, model


def test_training_step_runs_and_updates():
    dataset, residual, lm, model = make_air3d()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    before = [p.detach().clone() for p in model.parameters()]
    for _, batch in zip(range(3), iter(dataset)):
        opt.zero_grad(set_to_none=True)
        xt = batch["xt_interior"].requires_grad_(True)
        out = lm.compute_losses(model=model, xt_interior=xt, deepreach=True)
        out["loss_train"].backward()
        opt.step()
    assert torch.isfinite(out["loss_train"])
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))


def test_conservative_correction_enforces_V_leq_g():
    """The structural half of the method: V = g - softplus(.) <= g."""
    _, residual, _, model = make_air3d(correction="negative_softplus")
    x_net = (torch.rand(2048, 3) * 2 - 1) * 0.95
    xt = torch.cat([x_net, torch.rand(2048, 1)], -1)
    V = model(xt).reshape(-1)
    g = residual.target_function(residual._unscale_x(x_net))
    assert (V - g).max().item() <= 1e-6, f"max V-g = {(V - g).max().item()}"


def test_free_correction_does_not_enforce_it():
    """Contrast: the vanilla model has no structural guarantee."""
    _, residual, _, model = make_air3d(correction="free")
    torch.manual_seed(0)
    x_net = (torch.rand(2048, 3) * 2 - 1) * 0.95
    xt = torch.cat([x_net, torch.rand(2048, 1)], -1)
    V = model(xt).reshape(-1)
    g = residual.target_function(residual._unscale_x(x_net))
    assert (V - g).max().item() > 0.0


def test_value_equals_target_at_tau_zero():
    _, residual, _, model = make_air3d()
    x_net = (torch.rand(256, 3) * 2 - 1) * 0.9
    xt = torch.cat([x_net, torch.zeros(256, 1)], -1)
    V = model(xt).reshape(-1)
    g = residual.target_function(residual._unscale_x(x_net))
    assert torch.allclose(V, g, atol=1e-5)


def test_sampler_returns_correct_batch_shapes():
    dataset, residual, _, model = make_air3d(n=32)
    sampler = MixedCollocationSampler(
        dataset, residual,
        SamplerConfig(type="mixed", uniform_frac=0.4, pred_boundary_frac=0.25,
                      target_boundary_frac=0.15, residual_frac=0.2,
                      candidate_multiplier=4.0, chunk_size=16),
        device=torch.device("cpu"), deepreach=True)
    batch = sampler.sample_batch(model, dataset[0])
    assert batch["xt_interior"].shape == (32, 4)
    assert batch["x_interior_phys"].shape == (32, 3)
    assert batch["tau_interior_phys"].shape == (32,)
    assert sum(batch["sampler_counts"].values()) == 32
    assert not batch["xt_interior"].requires_grad


def test_evaluation_needs_no_ground_truth():
    dataset, residual, _, model = make_air3d(n=32)
    m = evaluate_unsupervised_batch(model=model, residual=residual, batch=dataset[0],
                                    device=torch.device("cpu"), deepreach=True)
    assert "residual_mean" in m and "batch_brt_frac" in m


def test_causal_loss_handles_empty_bins():
    loss, w, b = causal_binned_loss(torch.ones(8), torch.zeros(8), T=1.0, num_bins=5,
                                    causal_epsilon=5.0, min_weight=0.05)
    assert torch.isfinite(loss) and w.shape == (5,) and b.shape == (5,)


def test_oscillator_pipeline():
    dataset = LinearOscillator2DDataset(num_batches=2, num_interior=64, num_terminal=8,
                                        T=1.0, tau_max=1.0, efficient=True, num_unique_taus=4)
    residual = LinearOscillator2DResidual(T=1.0)
    lm = LossManager(residual=residual, hjvi_loss_mode="complementarity")
    backbone = DeepReachModel(in_dim=3, hidden_dim=32, num_layers=2)
    model = DeepReachExact(backbone=backbone, residual=residual, coordinate_dim=2,
                           correction_mode="negative_softplus")
    xt = dataset[0]["xt_interior"].requires_grad_(True)
    out = lm.compute_losses(model=model, xt_interior=xt, deepreach=True)
    out["loss_train"].backward()
    assert torch.isfinite(out["loss_train"])


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"{name}: ok")
            except AssertionError as e:
                fails += 1; print(f"{name}: FAIL -- {e}")
    raise SystemExit(1 if fails else 0)
