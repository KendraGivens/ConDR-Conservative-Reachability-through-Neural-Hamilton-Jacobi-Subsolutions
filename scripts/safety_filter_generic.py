"""System-agnostic closed-loop safety filter for any MADR dynamics.

scripts/safety_filter_rollout.py hardcodes Air3D: its `dynamics()` writes the
relative-frame ODE by hand, its adversary is a hand-tuned pursuer policy, and its
start states come from the exact GT solver. Only the LAST of those is a real
obstacle above 3D -- MADR's Dynamics interface already supplies the other two for
every system:

    dsdt(state, control, disturbance)   -> the ODE
    optimal_control(state, dvds)        -> the filter's override
    optimal_disturbance(state, dvds)    -> the worst-case adversary
    boundary_fn(state)                  -> the collision test

The GT dependency exists because calling a collision a FILTER FAILURE requires
knowing the start was recoverable. Without GT we sample starts that are safe BY
MARGIN (boundary_fn >= --safe-margin), where recoverability is obvious without
solving the PDE. That biases toward easy states, so the resulting collision rate is
a LOWER bound on risk and the intervention fraction a LOWER bound on
restrictiveness -- conservative, not wrong. Report it as such; it is not a
substitute for the GT-based filter on Air3D.

What this adds over scripts/eval_recovery.py: recovery reports delta_level and
recovered_volume (a static volume proxy) but never how often the filter actually
fires. Intervention fraction is that number, and the thesis wants it -- "needs
delta ~ 0 AND intervenes less often" is stronger than either half.

Usage:
    python3 scripts/safety_filter_generic.py --prefix q13 --seeds 42 \
        --out-json scripts/runs/eval/filter_q13.json
"""
import argparse, io, contextlib, json, math, sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "external" / "madr"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

MODELS = ("vanilla", "ours", "madr", "madr_ours")
PURE_PURSUIT = False  # set from --adversary pursuit


def load(run_name, device):
    from eval_madr_models import load_madr_run
    with contextlib.redirect_stdout(io.StringIO()):
        model, dyn, opt = load_madr_run(ROOT / "external" / "madr" / "runs" / run_name, device)
    return model, dyn, opt


def value_and_grad(model, dyn, states, tau, device):
    from eval_madr_models import values_on
    V, G = values_on(model, dyn, states.astype(np.float32), tau, device, want_grad=True)
    return np.asarray(V), np.asarray(G)


def _clamp_physical(dyn, t):
    """Apply dyn.clamp_state_input to PHYSICAL states. The vendored convention
    (dataio.py call sites) is that clamp_state_input takes NORMALIZED model
    inputs -- F1tenth's version prepends a time column and un-normalizes before
    testing the track; feeding it physical states drops every sample (empty
    concatenate in values_on, seen 09-04). Quaternion systems are unaffected
    (normalize_q commutes with the unit-range scaling)."""
    if not hasattr(dyn, "clamp_state_input"):
        return t
    try:
        ones = torch.ones(t.shape[0], 1, dtype=t.dtype, device=t.device)
        inp = dyn.coord_to_input(torch.cat((ones, t), dim=-1))[..., 1:]
        kept = dyn.clamp_state_input(inp)
        if kept.shape[0] == 0:
            return kept
        return dyn.input_to_coord(torch.cat((ones[: kept.shape[0]], kept), dim=-1))[..., 1:]
    except Exception:
        return t


def sample_starts(dyn, n, safe_margin, rng, device, max_tries=200):
    """Uniform over the state box, keeping only states safe BY MARGIN."""
    rng_lo = np.array([r[0] for r in dyn.state_test_range()], dtype=np.float64)
    rng_hi = np.array([r[1] for r in dyn.state_test_range()], dtype=np.float64)
    out = []
    for _ in range(max_tries):
        p = rng.uniform(rng_lo, rng_hi, size=(4 * n, len(rng_lo)))
        t = torch.tensor(p, dtype=torch.float32, device=device)
        # quaternion systems must stay on S^3, else the "state" is not physical
        t = _clamp_physical(dyn, t)
        b = dyn.boundary_fn(t).detach().cpu().numpy()
        keep = t.detach().cpu().numpy()[b >= safe_margin]
        if len(keep):
            out.append(keep)
            if sum(len(k) for k in out) >= n:
                break
    if not out:
        return np.zeros((0, len(rng_lo)), dtype=np.float32)
    return np.concatenate(out, 0)[:n].astype(np.float32)


def sample_starts_gt6d(dyn, n, safe_eps, rng, device, max_tries=200, nx=161, half=2.0):
    """Air6D only: starts VERIFIED safe against the exact solution, via the SE(2)
    lift (V_6D(s) = V_3D(relative state)). Mirrors the Air3D verified-start
    protocol (safety_filter_rollout.py: V* > safe_eps) so the two systems can be
    compared under the same start rule. Uses the cached 161^3 solve on +/-2 if
    present (docs/figures/gt3_nx161_half2.npz), else solves it."""
    import math
    from lvfm.grids import angle_axis, linear_axis, periodic_interpolator
    from lvfm.hj_solvers import solve_air3d_relative
    from eval_madr_air6d import relative_state
    cache = ROOT / "docs" / "figures" / f"gt3_nx{nx}_half{half:g}.npz"
    if cache.exists():
        gt3 = np.load(cache)["gt"]
    else:
        gt3 = np.asarray(solve_air3d_relative(
            tau_steps=[1.0], xr_discretization=nx, yr_discretization=nx, theta_discretization=nx,
            xr_bounds=(-half, half), yr_bounds=(-half, half), theta_bounds=(-math.pi, math.pi),
            vp=dyn.pursuer_velocity, ve=dyn.evader_velocity, u_bound=dyn.omega_max,
            d_bound=dyn.omega_max, radius=dyn.goalR))[-1]
    interp = periodic_interpolator([linear_axis((-half, half), nx), linear_axis((-half, half), nx),
                                    angle_axis((-math.pi, math.pi), nx)], gt3, periodic_dim=2, fill_value=np.nan)
    lo = np.array([r[0] for r in dyn.state_test_range()], dtype=np.float64)
    hi = np.array([r[1] for r in dyn.state_test_range()], dtype=np.float64)
    out = []
    for _ in range(max_tries):
        p = rng.uniform(lo, hi, size=(4 * n, len(lo))).astype(np.float32)
        v = interp(relative_state(p))
        keep = p[np.isfinite(v) & (v > safe_eps)]
        if len(keep):
            out.append(keep)
            if sum(len(k) for k in out) >= n: break
    if not out:
        return np.zeros((0, len(lo)), dtype=np.float32)
    return np.concatenate(out, 0)[:n].astype(np.float32)


def sample_starts_gt3d(dyn, n, safe_eps, rng, device, max_tries=200, nx=161, half=1.0):
    """Air3D only: starts verified safe against the exact 3D solve (V* > safe_eps),
    the safety_filter_rollout.py start rule, but rolled out under THIS script's
    protocol (shared learned-gradient adversary, Euler) -- the control that
    separates adversary strength from gradient quality when comparing to Air6D."""
    import math
    from lvfm.grids import angle_axis, linear_axis, periodic_interpolator
    from lvfm.hj_solvers import solve_air3d_relative
    cache = ROOT / "docs" / "figures" / f"gt3_nx{nx}_half{half:g}.npz"
    if cache.exists():
        gt3 = np.load(cache)["gt"]
    else:
        gt3 = np.asarray(solve_air3d_relative(
            tau_steps=[1.0], xr_discretization=nx, yr_discretization=nx, theta_discretization=nx,
            xr_bounds=(-half, half), yr_bounds=(-half, half), theta_bounds=(-math.pi, math.pi),
            vp=dyn.pursuer_velocity, ve=dyn.evader_velocity, u_bound=dyn.omega_max,
            d_bound=dyn.omega_max, radius=dyn.goalR))[-1]
    interp = periodic_interpolator([linear_axis((-half, half), nx), linear_axis((-half, half), nx),
                                    angle_axis((-math.pi, math.pi), nx)], gt3, periodic_dim=2, fill_value=np.nan)
    lo = np.array([r[0] for r in dyn.state_test_range()], dtype=np.float64)
    hi = np.array([r[1] for r in dyn.state_test_range()], dtype=np.float64)
    out = []
    for _ in range(max_tries):
        p = rng.uniform(lo, hi, size=(4 * n, len(lo))).astype(np.float32)
        v = interp(p)
        keep = p[np.isfinite(v) & (v > safe_eps)]
        if len(keep):
            out.append(keep)
            if sum(len(k) for k in out) >= n: break
    if not out:
        return np.zeros((0, len(lo)), dtype=np.float32)
    return np.concatenate(out, 0)[:n].astype(np.float32)


def sample_starts_by_value(ref_model, ref_dyn, n, v_min, tau, rng, device, max_tries=200):
    """Starts the SHARED reference model scores as safe (V >= v_min at the full
    horizon). Unlike a boundary_fn margin this accounts for reachability over tau,
    and because the reference is the same for every model it introduces no per-model
    bias -- the four models are still compared on identical initial states."""
    lo = np.array([r[0] for r in ref_dyn.state_test_range()], dtype=np.float64)
    hi = np.array([r[1] for r in ref_dyn.state_test_range()], dtype=np.float64)
    out = []
    for _ in range(max_tries):
        p = rng.uniform(lo, hi, size=(4 * n, len(lo)))
        t = torch.tensor(p, dtype=torch.float32, device=device)
        t = _clamp_physical(ref_dyn, t)
        if t.shape[0] == 0:
            continue
        pn = t.detach().cpu().numpy()
        V, _ = value_and_grad(ref_model, ref_dyn, pn, tau, device)
        # also require the start itself be physically safe (l(x) >= 0): a
        # reference model can over-estimate V inside the failure set, and a
        # start already in collision is not a filter test.
        lx = ref_dyn.boundary_fn(t).detach().cpu().numpy()
        keep = pn[(np.asarray(V) >= v_min) & (lx >= 0)]
        if len(keep):
            out.append(keep)
            if sum(len(k) for k in out) >= n: break
    if not out:
        return np.zeros((0, len(lo)), dtype=np.float32)
    return np.concatenate(out, 0)[:n].astype(np.float32)


def rollout(model, dyn, starts, tau, dt, steps, margin, device, use_filter=True,
            adv_model=None, adv_dyn=None, true_dyn=None):
    """Euler rollout under a FIXED adversary; the filter overrides the nominal
    (trim) control whenever the learned value says the state is unsafe.

    The adversary MUST NOT depend on the model under test. It originally used the
    evaluated model's own gradient, so each of the four models played a different
    game -- and an model with a more accurate gradient generated a STRONGER
    adversary against itself, which inverted the comparison (on Air3D `ours`
    looked worse than vanilla here while beating it under the GT-based filter).
    `adv_model` is one designated model shared by every model of a set."""
    # MODEL MISMATCH: `dyn` is what the controller BELIEVES (value decode +
    # control choice); `true_dyn` is the world the state actually evolves in
    # (propagation, adversary authority, collision geometry). Identical unless
    # --mismatch is set -- the sim stand-in for sim-to-real error.
    td = true_dyn if true_dyn is not None else dyn
    s = torch.tensor(starts, dtype=torch.float32, device=device)
    n = s.shape[0]
    collided = np.zeros(n, dtype=bool)
    interventions = 0
    total = 0
    # audit 09-02 S4-3: stop counting interventions on already-collided
    # trajectories, else the intervention fraction is biased model-dependently.
    for k in range(steps):
        sn = s.detach().cpu().numpy()
        t_left = max(tau * (1.0 - k / steps), 1e-3)
        V, G = value_and_grad(model, dyn, sn, t_left, device)
        gt = torch.tensor(G, dtype=torch.float32, device=device)
        active = torch.tensor(~collided, device=device)
        unsafe = torch.tensor(V <= margin, device=device) & active
        # control_init is the system's TRIM, not zero: the 20D drone needs
        # [0,0,0.909] thrust just to hold altitude, so a zero nominal made every
        # rollout collide (no-filter collision 1.0000) and the metric carried no
        # signal at all.
        _ci = getattr(dyn, "control_init", None)
        if _ci is not None and _ci.numel() == max(dyn.control_dim, 1):
            u_nom = _ci.to(device).float().unsqueeze(0).repeat(n, 1)
        else:
            u_nom = torch.zeros(n, max(dyn.control_dim, 1), device=device)
        try:
            u_opt = dyn.optimal_control(s, gt)
            if u_opt.ndim == 1:
                u_opt = u_opt[..., None]
        except Exception:
            u_opt = u_nom
        u = torch.where(unsafe[:, None], u_opt.float(), u_nom) if use_filter else u_nom
        interventions += int(unsafe.sum().item()) if use_filter else 0
        total += int(active.sum().item())
        try:
            # adversary gradient comes from the SHARED reference model, never
            # from the model being evaluated
            if adv_model is not None:
                # decode the adversary with ITS OWN dynamics: passing the evaluated
                # model's `dyn` applied that model's value map (exact vs exact_cons) to
                # the adversary's outputs, so the "shared" adversary still differed
                # between models -- visible as a no-filter baseline that split exactly
                # along exact/exact_cons when it must be identical for all four.
                Va, Ga = value_and_grad(adv_model, adv_dyn if adv_dyn is not None else dyn,
                                        sn, t_left, device)
                g_adv = torch.tensor(Ga, dtype=torch.float32, device=device)
            else:
                g_adv = gt
            if PURE_PURSUIT and hasattr(td, "pursuer_velocity") and s.shape[1] == 6:
                # Air6D pure pursuit in absolute coordinates: the same rule as the
                # Air3D verified-start protocol (gain 5 on bearing error, saturated).
                bearing = torch.atan2(s[:, 4] - s[:, 1], s[:, 3] - s[:, 0])
                err = torch.remainder(bearing - s[:, 2] + math.pi, 2 * math.pi) - math.pi
                d = torch.clamp(5.0 * err, -td.omega_max, td.omega_max)[:, None]
            else:
                d = td.optimal_disturbance(s, g_adv)
                d = d[..., None] if (hasattr(d, "ndim") and d.ndim == 1) else d
                d = torch.as_tensor(d, dtype=torch.float32, device=device)
        except Exception:
            d = torch.zeros(n, max(td.disturbance_dim, 1), device=device)
        s = s + dt * td.dsdt(s, u, d)
        if hasattr(td, "equivalent_wrapped_state"):
            try:
                s = td.equivalent_wrapped_state(s)
            except Exception:
                pass
        collided |= (td.boundary_fn(s).detach().cpu().numpy() <= 0.0)
    return float(collided.mean()), (interventions / max(total, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--episodes", type=int, default=400)
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--tau", type=float, default=None,
                    help="rollout horizon; default reads tMax from the run's own config. "
                         "A fixed 1.0 silently queried the 20D drone (tMax=2.0) at half "
                         "its trained horizon and rolled out for half the time.")
    ap.add_argument("--margin", type=float, default=0.0)
    ap.add_argument("--safe-margin", type=float, default=0.15,
                    help="start states must have boundary_fn >= this (GT-free proxy "
                         "for 'truly safe'); larger = easier states = more conservative")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--start-mode", choices=["margin", "value", "gt6d", "gt3d"], default="margin",
                    help="how to pick recoverable starts. 'margin' keeps boundary_fn >= "
                         "--safe-margin, which works where the obstacle distance is "
                         "informative. On the 20D drone boundary_fn maxes out at 0.90, so "
                         "NO margin implies recoverability over its 2 s horizon and the "
                         "no-filter collision rate sits at 0.88 -- the metric cannot "
                         "discriminate. 'value' instead keeps states the SHARED reference "
                         "model scores V >= --start-value, which is reachability-aware and "
                         "identical for every model_name, so the comparison stays fair.")
    ap.add_argument("--start-value", type=float, default=0.05)
    ap.add_argument("--gt-safe-eps", type=float, default=0.02,
                    help="gt6d mode: keep starts with exact V* > this (Air3D protocol uses 0.02)")
    ap.add_argument("--mismatch", type=float, default=0.0,
                    help="model-mismatch fraction M for the TRUE dynamics only: "
                         "pursuer/adversary speed x(1+M) and evader speed x(1-M) "
                         "where the class has those attributes, else shared "
                         "velocity x(1+M) (world faster than modeled). The value "
                         "functions and control choices still use the nominal "
                         "model -- the sim proxy for sim-to-real error.")
    ap.add_argument("--adversary", choices=["learned", "pursuit"], default="learned",
                    help="learned: worst-case disturbance from the shared reference model's gradient; "
                         "pursuit: hand-written pure pursuit (Air6D only), the Air3D verified-start rule")
    ap.add_argument("--adversary-model", default="vanilla",
                    help="model whose gradient drives the adversary for EVERY model of "
                         "this set; fixed so all four play the same game")
    ap.add_argument("--out-json", default=None)
    a = ap.parse_args()
    global PURE_PURSUIT; PURE_PURSUIT = (a.adversary == "pursuit")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Horizon must come from the run, not a constant: the systems differ (tMax 1.0
    # for Air3D/Air6D/9D, 2.0 for the 20D drone).
    if a.tau is None:
        import pickle as _pk
        for _sd in a.seeds:
            _c = ROOT / "external/madr/runs" / f"{a.prefix}_vanilla_s{_sd}" / "orig_opt.pickle"
            if _c.exists():
                a.tau = float(_pk.load(open(_c, "rb")).tMax); break
        if a.tau is None:
            a.tau = 1.0
        print(f"  horizon tau = {a.tau} (from the run config)", flush=True)
    steps = int(round(a.tau / a.dt))
    results = []
    # One shared adversary for the whole set, loaded once.
    adv_model = adv_dyn = None
    for sd0 in a.seeds:
        ref = f"{a.prefix}_{a.adversary_model}_s{sd0}"
        if (ROOT / "external/madr/runs" / ref / "training/checkpoints/model_final.pth").exists():
            try:
                adv_model, adv_dyn, _ = load(ref, device)
                print(f"  adversary (shared by all model_names): {ref}", flush=True)
                break
            except Exception as e:
                print(f"  adversary load failed for {ref}: {e}", flush=True)
    if adv_model is None:
        print("  WARNING: no shared adversary available; falling back to each model_name's own "
              "gradient, which makes cross-model_name numbers NOT comparable", flush=True)
    print(f"  {'model':<11}{'seed':>5}{'collision':>11}{'interv':>9}{'no-filter':>11}", flush=True)
    for model_name in MODELS:
        for sd in a.seeds:
            name = f"{a.prefix}_{model_name}_s{sd}"
            if not (ROOT / "external/madr/runs" / name / "training/checkpoints/model_final.pth").exists():
                print(f"  {model_name:<11}{sd:>5}   (not trained)", flush=True); continue
            try:
                model, dyn, _ = load(name, device)
                true_dyn = None
                if a.mismatch > 0.0:
                    import copy as _copy
                    true_dyn = _copy.deepcopy(dyn)
                    if hasattr(true_dyn, "pursuer_velocity"):
                        true_dyn.pursuer_velocity *= (1.0 + a.mismatch)
                        true_dyn.evader_velocity *= (1.0 - a.mismatch)
                    elif hasattr(true_dyn, "velocity"):
                        true_dyn.velocity *= (1.0 + a.mismatch)
                    else:
                        raise SystemExit(f"no mismatch recipe for {type(dyn).__name__}")
                rng = np.random.default_rng(a.seed)
                if a.start_mode == "value" and adv_model is not None:
                    starts = sample_starts_by_value(adv_model, adv_dyn, a.episodes,
                                                    a.start_value, a.tau, rng, device)
                elif a.start_mode == "gt6d":
                    starts = sample_starts_gt6d(dyn, a.episodes, a.gt_safe_eps, rng, device)
                elif a.start_mode == "gt3d":
                    starts = sample_starts_gt3d(dyn, a.episodes, a.gt_safe_eps, rng, device)
                else:
                    starts = sample_starts(dyn, a.episodes, a.safe_margin, rng, device)
                if len(starts) == 0:
                    print(f"  {model_name:<11}{sd:>5}   (no safe starts at margin {a.safe_margin})", flush=True)
                    continue
                col, iv = rollout(model, dyn, starts, a.tau, a.dt, steps, a.margin,
                                  device, True, adv_model=adv_model, adv_dyn=adv_dyn,
                                  true_dyn=true_dyn)
                base, _ = rollout(model, dyn, starts, a.tau, a.dt, steps, a.margin,
                                  device, False, adv_model=adv_model, adv_dyn=adv_dyn,
                                  true_dyn=true_dyn)
                print(f"  {model_name:<11}{sd:>5}{col:>11.4f}{iv:>9.4f}{base:>11.4f}", flush=True)
                # A no-filter rate at 0 or 1 means every start is trivially safe
                # or doomed, so no filter can differ from any other. Report it
                # rather than emitting numbers that look like a comparison.
                degenerate = base >= 0.999 or base <= 0.001
                if degenerate:
                    print(f"      ^ DEGENERATE: no-filter collision {base:.4f}; "
                          f"this metric cannot separate model_names", flush=True)
                results.append(dict(model=name, model_name=model_name, seed=sd, n_starts=int(len(starts)),
                                    collision_rate=col, intervention_frac=iv,
                                    collision_rate_no_filter=base,
                                    degenerate=bool(degenerate)))
            except Exception as e:
                print(f"  {model_name:<11}{sd:>5}   FAILED: {type(e).__name__}: {str(e)[:60]}", flush=True)
    if a.out_json:
        Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
        json.dump(dict(prefix=a.prefix, safe_margin=a.safe_margin,
                       episodes=a.episodes, dt=a.dt, results=results),
                  open(a.out_json, "w"), indent=2)
        print(f"  wrote {a.out_json}", flush=True)


if __name__ == "__main__":
    main()
