"""Regression tests for the bugs found in the 2026-08-27 audit.

Each test corresponds to a finding and would have caught it. Run with
`pytest tests/` or directly (`python tests/test_audit_regressions.py`).
"""
import math
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import torch

from lvfm.grids import (angle_axis, linear_axis, nearest_index,
                        periodic_interpolator, periodic_nearest_index)
from lvfm.managers import LossManager
from lvfm.models import DeepReachExact, DeepReachModel
from lvfm.residuals import (Air3DResidual, Air6DJointResidual, Dubins3DResidual,
                            LinearOscillator2DResidual, MultiVehicle9DResidual,
                            Quadrotor13DResidual, SimpleQuadrotor13DResidual)

ALL_RESIDUALS = [
    (LinearOscillator2DResidual, dict(), 2),
    (Air3DResidual, dict(), 3),
    (Air6DJointResidual, dict(), 6),
    (MultiVehicle9DResidual, dict(), 9),
    (Quadrotor13DResidual, dict(), 13),
    (SimpleQuadrotor13DResidual, dict(), 13),
    (Dubins3DResidual, dict(), 3),
]


# ---- P0-1: every residual must expose _scale_x, and it must invert _unscale_x ----
def test_every_residual_has_scale_x_roundtrip():
    """load_mpc_data / eval scripts call residual._scale_x. Air3D and
    LinearOsc2D did not define it, so the MPC models crashed at startup."""
    for cls, kwargs, dim in ALL_RESIDUALS:
        r = cls(**kwargs)
        assert hasattr(r, "_scale_x"), f"{cls.__name__} has no _scale_x"
        x_net = (torch.rand(64, dim) * 2 - 1) * 0.9
        back = r._scale_x(r._unscale_x(x_net))
        assert torch.allclose(back, x_net, atol=1e-5), \
            f"{cls.__name__}: _scale_x is not the inverse of _unscale_x"


# ---- P0-4: violation diagnostics must be measured in EVERY loss mode ----
def _tiny_deepreach():
    res = Air3DResidual(x_bounds=(-1, 1), y_bounds=(-1, 1))
    bb = DeepReachModel(in_dim=4, hidden_dim=16, num_layers=2, omega0=30.0)
    return res, DeepReachExact(backbone=bb, residual=res, coordinate_dim=3,
                               correction_mode="free")


def test_violation_metrics_reported_in_every_loss_mode():
    """With hjvi_loss_mode: standard the components were never computed, so
    pde_positive_frac logged a hard 0.0 -- reading as 'no violations' and
    inverting the conservative-vs-vanilla comparison."""
    res, model = _tiny_deepreach()
    torch.manual_seed(0)
    xt = torch.rand(256, 4) * 2 - 1
    for mode in ("standard", "complementarity", "subsolution", "smoothmax"):
        lm = LossManager(residual=res, hjvi_loss_mode=mode)
        out = lm.compute_losses(model=model, xt_interior=xt.clone().requires_grad_(True),
                                deepreach=True)
        assert out["pde_positive_frac"].item() > 0.0, \
            f"{mode}: pde_positive_frac is exactly 0 -- not measured"
        assert out["loss_hjvi_ineq"].item() > 0.0, f"{mode}: loss_hjvi_ineq not measured"


def test_standard_loss_is_numerically_unchanged():
    """Computing the components unconditionally must not change any loss."""
    res, model = _tiny_deepreach()
    torch.manual_seed(0)
    xt = torch.rand(256, 4) * 2 - 1
    lm = LossManager(residual=res, hjvi_loss_mode="standard")
    got = lm.compute_losses(model=model, xt_interior=xt.clone().requires_grad_(True),
                            deepreach=True)["loss_pinn"]
    want = res.compute_deepreach_residual(
        model=model, xt=xt.clone().requires_grad_(True)).pow(2).mean()
    assert torch.allclose(got, want, atol=1e-6), f"{got.item()} != {want.item()}"


# ---- P1-1: evaluation axes must match hj_reachability's grid construction ----
def test_eval_axes_match_hj_grid():
    """hj builds periodic dims with endpoint=False and non-periodic with
    endpoint=True. Nine eval sites used endpoint=True for psi against a
    periodic GT grid, offsetting by up to a full cell."""
    try:
        import hj_reachability as hj
    except ImportError:
        print("  (skipped: hj_reachability not installed)")
        return
    n = 17
    grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
        hj.sets.Box(np.array([-1.0, -1.0, -math.pi]), np.array([1.0, 1.0, math.pi])),
        shape=(n, n, n), periodic_dims=2)
    assert np.allclose(np.asarray(grid.coordinate_vectors[0]), linear_axis((-1.0, 1.0), n))
    assert np.allclose(np.asarray(grid.coordinate_vectors[2]),
                       angle_axis((-math.pi, math.pi), n)), \
        "angle_axis does not match hj's periodic coordinate vector"


# ---- P1-6: lookups must use the NEAREST node, not searchsorted's upper one ----
def test_nearest_index_is_nearest():
    ax = linear_axis((-1.0, 1.0), 5)          # [-1, -0.5, 0, 0.5, 1]
    assert list(nearest_index(ax, [-0.6, -0.4, 0.24, 0.26])) == [1, 1, 2, 3]
    # np.searchsorted would give the upper neighbour and be wrong here
    assert list(np.searchsorted(ax, [-0.4])) != list(nearest_index(ax, [-0.4]))


def test_periodic_nearest_index_wraps():
    ax = angle_axis((-math.pi, math.pi), 8)
    # just below +pi is nearest to the -pi node, not the last node
    assert int(periodic_nearest_index(ax, math.pi - 1e-6)) == 0


def test_periodic_interpolator_crosses_the_seam():
    ax = [linear_axis((-1, 1), 4), linear_axis((-1, 1), 4), angle_axis((-math.pi, math.pi), 8)]
    v = np.random.default_rng(0).normal(size=(4, 4, 8))
    f = periodic_interpolator(ax, v, periodic_dim=2)
    near_seam = f(np.array([[0.0, 0.0, math.pi - 1e-3]]))
    assert np.isfinite(near_seam).all(), "interpolator cannot cross the psi seam"


# ---- P1-2: the BRT volume must come from a fixed uniform sample ----
def test_volume_probe_is_independent_of_the_batch():
    """pred_brt_volume_frac used to be computed on the adaptive sampler's
    batch, whose measure is model-dependent -- not a volume."""
    from lvfm.evaluation import evaluate_unsupervised_batch, uniform_volume_frac
    res, model = _tiny_deepreach()
    torch.manual_seed(0)
    probe = torch.rand(512, 3) * 2 - 1
    batch = {"xt_interior": torch.rand(128, 4) * 2 - 1,
             "tau_interior_phys": torch.rand(128)}
    out = evaluate_unsupervised_batch(model=model, residual=res, batch=batch,
                                      device=torch.device("cpu"), deepreach=True,
                                      volume_probe=probe, volume_tau=1.0)
    assert "batch_brt_frac" in out and not math.isnan(out["pred_brt_volume_frac"])
    direct = uniform_volume_frac(model, res, probe, 1.0, torch.device("cpu"), deepreach=True)
    assert abs(out["pred_brt_volume_frac"] - direct) < 1e-9
    # and omitting the probe must NOT silently fall back to the batch number
    out2 = evaluate_unsupervised_batch(model=model, residual=res, batch=batch,
                                       device=torch.device("cpu"), deepreach=True)
    assert math.isnan(out2["pred_brt_volume_frac"])


# ---- P0-2: MPC label domains must match the training configs ----
def test_mpc_label_domain_matches_configs():
    sys.path.insert(0, "scripts")
    from gen_mpc_labels import PARAMS
    assert PARAMS["air3d"]["xy"] == 1.0, "air3d labels sampled outside x_bounds=[-1,1]"
    assert PARAMS["linear_oscillator_2d"]["x_hi"] == [1.0, 1.0], \
        "oscillator labels sampled outside x1_bounds=[-1,1]"


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"{name}: ok")
            except AssertionError as e:
                fails += 1
                print(f"{name}: FAIL -- {e}")
    raise SystemExit(1 if fails else 0)
