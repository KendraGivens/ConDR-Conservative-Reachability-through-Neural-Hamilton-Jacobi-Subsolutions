"""
Closed-loop safety-filter evaluation on Air3D pursuit-evasion.

Simulates the relative-state Air3D game. The evader flies a nominal policy
(straight, u=0); the pursuer plays an adversarial policy. A least-restrictive
safety filter overrides the evader with the value-gradient optimal control
whenever the filter model says the state is (near) unsafe:

    if V_hat(T, x) <= margin:  u = u_bound * sign(p_x * y - p_y * x - p_psi)

Initial states are sampled from the truly-safe set (V_true > eps, from the
classical solver — evaluation-only GT usage). Reported per filter model:
  - collision rate over truly-safe starts (safety; a perfect filter -> 0)
  - intervention fraction (conservatism cost)
Baselines: --no-filter rollouts, plus any mix of LatentReach / conservative-DR
runs (our repo) and official DeepReach runs (external/deepreach).
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import yaml
from lvfm.config_utils import cfg_get, to_namespace
from lvfm.hj_solvers import solve_air3d_relative
from lvfm.residuals import Air3DResidual
from lvfm.grids import linear_axis, angle_axis, nearest_index, periodic_nearest_index

VP = VE = 0.75
U_BOUND = D_BOUND = 3.0
RADIUS = 0.25
T_HORIZON = 1.0
ANGLE_SCALE = 1.2 * math.pi


def load_repo_model(run_name, device):
    from train_unsupervised import create_model

    run_dir = REPO_ROOT / "scripts" / "runs" / "air_3d" / run_name
    cfg = to_namespace(yaml.safe_load(open(run_dir / "config.yaml")))
    cfg.device = str(device)
    # Read the run's OWN physics. These used to come from the module constants
    # while x_bounds came from the config, so a run trained with a different beta
    # would be rebuilt with the wrong target function g -- and V = g + tau*c*rho
    # is built ON g, so V itself would be wrong, silently (audit P1-7).
    # The simulated game must then use the same constants, so assert they match.
    for name, cfg_key, const in (("vp", "vp", VP), ("ve", "ve", VE),
                                 ("u_bound", "u_bound", U_BOUND),
                                 ("d_bound", "d_bound", D_BOUND),
                                 ("beta", "beta", RADIUS), ("T", "T", T_HORIZON)):
        got = float(cfg_get(cfg, cfg_key, const))
        if abs(got - float(const)) > 1e-9:
            raise ValueError(
                f"{run_name}: config {name}={got} differs from the rollout's "
                f"simulated dynamics ({const}). The closed-loop game in this "
                f"script is hard-coded to those constants, so the comparison "
                f"would be invalid. Update the constants or exclude this run.")
    angle_scale = float(cfg_get(cfg, "angle_alpha_factor", 1.2)) * math.pi
    residual = Air3DResidual(
        vp=cfg_get(cfg, "vp", VP), ve=cfg_get(cfg, "ve", VE),
        control_bound=cfg_get(cfg, "u_bound", U_BOUND),
        disturbance_bound=cfg_get(cfg, "d_bound", D_BOUND),
        radius=cfg_get(cfg, "beta", RADIUS), T=cfg_get(cfg, "T", T_HORIZON),
        x_bounds=cfg.x_bounds, y_bounds=cfg.y_bounds, psi_bounds=cfg.psi_bounds,
        scale_to_minus1_1=cfg.scale_to_minus1_1, scale_time_to_01=cfg.scale_time_to_01,
        angle_alpha_factor=cfg_get(cfg, "angle_alpha_factor", 1.2),
    )
    model, _, deepreach = create_model(cfg, residual, device)
    ckpt = torch.load(run_dir / "ckpts" / "final.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    def value_and_grad(states_phys):
        """states_phys: (N,3) numpy. Returns V (N,), grad_phys (N,3)."""
        pts = np.asarray(states_phys, dtype=np.float32)
        net = pts.copy()
        net[:, 0] = 2.0 * (pts[:, 0] - cfg.x_bounds[0]) / (cfg.x_bounds[1] - cfg.x_bounds[0]) - 1.0
        net[:, 1] = 2.0 * (pts[:, 1] - cfg.y_bounds[0]) / (cfg.y_bounds[1] - cfg.y_bounds[0]) - 1.0
        net[:, 2] = pts[:, 2] / angle_scale
        tau_net = 1.0 if cfg.scale_time_to_01 else T_HORIZON
        xt = torch.tensor(
            np.concatenate([net, np.full((net.shape[0], 1), tau_net, dtype=np.float32)], axis=-1),
            device=device,
        ).requires_grad_(True)
        V = model(xt).reshape(-1)
        grad_net = torch.autograd.grad(V.sum(), xt)[0][:, :3]
        grad = grad_net.detach().cpu().numpy()
        grad[:, 0] *= 2.0 / (cfg.x_bounds[1] - cfg.x_bounds[0])
        grad[:, 1] *= 2.0 / (cfg.y_bounds[1] - cfg.y_bounds[0])
        grad[:, 2] /= angle_scale
        return V.detach().cpu().numpy(), grad

    return value_and_grad


def load_madr_model(run_name, device):
    """Load a run trained in external/madr (the a3r_/a3g_/a3hi_/a3xh_ models).

    The existing loaders only cover LatentReach runs (scripts/runs/air_3d) and the
    official DeepReach fork, so the 2x2 models could not be safety-filtered at all.
    values_on(..., want_grad=True) already returns V and dV/ds in physical units,
    which is exactly the interface the filter needs.
    """
    sys.path.insert(0, str(REPO_ROOT / "external" / "madr"))
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from eval_madr_models import load_madr_run, values_on

    model, dyn, _ = load_madr_run(REPO_ROOT / "external" / "madr" / "runs" / run_name, device)

    def value_and_grad(states_phys):
        pts = np.asarray(states_phys, dtype=np.float32)
        V, G = values_on(model, dyn, pts, T_HORIZON, device, want_grad=True)
        return np.asarray(V), np.asarray(G)

    return value_and_grad


def load_official_model(run_name, device):
    sys.path.insert(0, str(REPO_ROOT / "external" / "deepreach"))
    from eval_deepreach_official import load_official_run

    model, dyn, _, _, _ = load_official_run(
        REPO_ROOT / "external" / "deepreach" / "runs" / run_name, device
    )

    def value_and_grad(states_phys):
        pts = np.asarray(states_phys, dtype=np.float32)
        coords = torch.tensor(
            np.concatenate([np.full((pts.shape[0], 1), T_HORIZON, dtype=np.float32), pts], axis=-1),
            device=device,
        )
        model_input = dyn.coord_to_input(coords).requires_grad_(True)
        results = model({"coords": model_input})
        V = dyn.io_to_value(results["model_in"], results["model_out"].squeeze(dim=-1))
        dv = dyn.io_to_dv(results["model_in"], results["model_out"].squeeze(dim=-1))
        return V.detach().cpu().numpy(), dv[:, 1:].detach().cpu().numpy()

    return value_and_grad


def dynamics(state, u, d):
    x, y, psi = state[:, 0], state[:, 1], state[:, 2]
    dx = -VE + VP * np.cos(psi) + u * y
    dy = VP * np.sin(psi) - u * x
    dpsi = d - u
    return np.stack([dx, dy, dpsi], axis=-1)


def pursuer_policy(state):
    """Heuristic adversary: pure pursuit — steer relative heading toward the
    evader (drive psi toward the bearing that closes distance fastest)."""
    x, y, psi = state[:, 0], state[:, 1], state[:, 2]
    bearing = np.arctan2(-y, -x)  # direction from pursuer-relative offset toward evader
    err = (bearing - psi + np.pi) % (2 * np.pi) - np.pi
    return np.clip(5.0 * err, -D_BOUND, D_BOUND)


def rollout(value_and_grad, states0, dt, margin, use_filter=True, record=False):
    n = states0.shape[0]
    state = states0.copy()
    collided = np.zeros(n, dtype=bool)
    interventions = np.zeros(n)
    steps = int(round(T_HORIZON / dt))
    traj = [state.copy()] if record else None

    for _ in range(steps):
        active = ~collided
        u = np.zeros(n)
        if use_filter and value_and_grad is not None:
            V, g = value_and_grad(state)
            unsafe = (V <= margin) & active
            coeff = g[:, 0] * state[:, 1] - g[:, 1] * state[:, 0] - g[:, 2]
            u[unsafe] = U_BOUND * np.sign(coeff[unsafe])
            interventions[unsafe] += 1
        d = pursuer_policy(state)
        # RK4 on the relative dynamics with zero-order-hold inputs
        k1 = dynamics(state, u, d)
        k2 = dynamics(state + 0.5 * dt * k1, u, d)
        k3 = dynamics(state + 0.5 * dt * k2, u, d)
        k4 = dynamics(state + dt * k3, u, d)
        state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        state[:, 2] = (state[:, 2] + np.pi) % (2 * np.pi) - np.pi
        dist = np.sqrt(state[:, 0] ** 2 + state[:, 1] ** 2)
        collided |= (dist <= RADIUS) & active
        if record:
            traj.append(state.copy())

    return {
        "collision_rate": float(collided.mean()),
        "intervention_frac": float(interventions.sum() / (n * steps)),
        "traj": np.stack(traj, axis=0) if record else None,
    }


def sample_truly_safe_starts(n, eps, seed, nx=101):
    gt = solve_air3d_relative(
        tau_steps=[T_HORIZON], xr_discretization=nx, yr_discretization=nx,
        theta_discretization=nx, xr_bounds=(-1.0, 1.0), yr_bounds=(-1.0, 1.0),
        theta_bounds=(-math.pi, math.pi), vp=VP, ve=VE,
        u_bound=U_BOUND, d_bound=D_BOUND, radius=RADIUS,
    )
    V_gt = np.asarray(gt[-1])
    # x/y are non-periodic (endpoint included); psi is periodic (no node at +pi).
    # Lookups use NEAREST node -- np.searchsorted returns the upper insertion
    # point, biasing every start by up to a cell (audit P1-1 / P1-6).
    xs = linear_axis((-1.0, 1.0), nx)
    psis = angle_axis((-math.pi, math.pi), nx)
    rng = np.random.default_rng(seed)
    starts = []
    while len(starts) < n:
        pts = rng.uniform([-1, -1, -math.pi], [1, 1, math.pi], size=(4 * n, 3))
        ix = nearest_index(xs, pts[:, 0])
        iy = nearest_index(xs, pts[:, 1])
        ip = periodic_nearest_index(psis, pts[:, 2])
        vals = V_gt[ix, iy, ip]
        good = pts[vals > eps]
        starts.extend(good.tolist())
    return np.asarray(starts[:n], dtype=np.float64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-runs", nargs="*", default=[], help="air_3d run names (LatentReach or cons-DR)")
    parser.add_argument("--official-runs", nargs="*", default=[], help="official DeepReach run names")
    parser.add_argument("--madr-runs", nargs="*", default=[], help="external/madr run names (the 2x2 models)")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--margin", type=float, default=0.0)
    parser.add_argument("--safe-eps", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", default=None)
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else f"cuda:{args.device}")
    starts = sample_truly_safe_starts(args.episodes, args.safe_eps, args.seed)
    print(f"Sampled {len(starts)} truly-safe initial states (V_true > {args.safe_eps})", flush=True)

    results = []
    base = rollout(None, starts, args.dt, args.margin, use_filter=False)
    results.append({"model": "no_filter", "collision_rate": base["collision_rate"], "intervention_frac": 0.0})
    print(f"[no_filter] collision {base['collision_rate']:.3f}", flush=True)

    for run in args.repo_runs:
        vg = load_repo_model(run, device)
        r = rollout(vg, starts, args.dt, args.margin)
        results.append({"model": run, "collision_rate": r["collision_rate"], "intervention_frac": r["intervention_frac"]})
        print(f"[{run}] collision {r['collision_rate']:.3f} intervention {r['intervention_frac']:.3f}", flush=True)

    for run in args.madr_runs:
        vg = load_madr_model(run, device)
        r = rollout(vg, starts, args.dt, args.margin)
        results.append({"model": run, "collision_rate": r["collision_rate"], "intervention_frac": r["intervention_frac"]})
        print(f"[{run}] collision {r['collision_rate']:.3f} intervention {r['intervention_frac']:.3f}", flush=True)

    for run in args.official_runs:
        vg = load_official_model(run, device)
        r = rollout(vg, starts, args.dt, args.margin)
        results.append({"model": f"official:{run}", "collision_rate": r["collision_rate"], "intervention_frac": r["intervention_frac"]})
        print(f"[official:{run}] collision {r['collision_rate']:.3f} intervention {r['intervention_frac']:.3f}", flush=True)

    out = {"episodes": args.episodes, "dt": args.dt, "margin": args.margin,
           "safe_eps": args.safe_eps, "seed": args.seed, "results": results}
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out_json, "w"), indent=2)

    if args.out_md:
        lines = ["# Air3D Closed-Loop Safety-Filter Results", "",
                 f"{args.episodes} episodes from truly-safe starts, dt={args.dt}, margin={args.margin}, "
                 "straight-line nominal evader, pure-pursuit adversary.", "",
                 "|filter model|collision rate|intervention frac|", "|---|---|---|"]
        for r in results:
            lines.append(f"|{r['model']}|{r['collision_rate']:.4f}|{r['intervention_frac']:.4f}|")
        open(args.out_md, "w").write("\n".join(lines) + "\n")
    print(f"Wrote {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
