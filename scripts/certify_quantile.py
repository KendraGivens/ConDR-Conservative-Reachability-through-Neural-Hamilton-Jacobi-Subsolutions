"""Conformal certification of the MADR-stack models, with the horizon factor.

Two certificates per run, both stated at confidence >= 1-beta over the sampling
distribution (uniform on the state box x tau ~ U(0, T]):

1. RESIDUAL certificate (unsupervised -- works in any dimension).
   v(x, tau) = relu( max( dV/dtau - H, V - g ) )   (VI subsolution violation)
   delta_sub = distribution-free (1-eps)-quantile upper bound of v
   (conformal_quantile_bound, the corrected binomial tail). A delta-approximate
   subsolution integrates to a value gap of at most delta * tau along
   characteristics (Gronwall / comparison; Theorem B), so the certified tube at
   the horizon is {V(., T) <= delta_sub * T} -- the horizon factor the
   2026-08-31 audit found missing (S1-5). We report its volume by Monte Carlo
   on fresh samples. Honest scope: the guarantee is distributional in the
   sampled measure; an eps-fraction violation set is not controlled along
   characteristics, so this is a probabilistic certificate, not a proof.

2. GT certificate (validation, where an exact solve exists: Air3D).
   Over grid states in the true BRT at tau = T, take the (1-eps)-quantile of
   (V - V*)+ = delta_gt; the tube {V <= delta_gt} misses at most an
   eps-fraction of BRT grid cells. STATISTICAL FRAMING (2026-09-02 audit): the
   evaluation grid is a fixed population, not an iid sample, so this is an
   EMPIRICAL coverage quantile (a census of the grid) -- no sampling
   confidence statement attaches, and none is needed. The conformal
   binomial-tail machinery applies only to certificate (1), whose draws are
   iid. delta_gt remains the stable replacement for the max-based vol@FN=0.
"""
import argparse, json, math, os, sys
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"
import numpy as np, torch

REPO = Path(__file__).resolve().parents[1]
MADR = REPO / "external" / "madr"
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(MADR))
from certify_conservative import conformal_quantile_bound
from eval_madr_models import MODEL_ORDER, load_madr_run


def residual_violations(model, dyn, n, tmax, device, rng, chunk=8192):
    """v = relu(max(dvdt - H, V - g)) at (x, tau) ~ U(box) x U(0, T]."""
    lo = np.array([r[0] for r in dyn.state_test_range()], np.float64)
    hi = np.array([r[1] for r in dyn.state_test_range()], np.float64)
    out = []
    done = 0
    while done < n:
        m = min(chunk, n - done)
        x = rng.uniform(lo, hi, size=(m, len(lo))).astype(np.float32)
        tau = rng.uniform(1e-6, tmax, size=(m, 1)).astype(np.float32)
        coords = torch.tensor(np.concatenate([tau, x], -1), device=device)
        # Project onto the state manifold before evaluating H and g. The quadrotor
        # carries a unit quaternion: box-sampling its four components and passing
        # the raw state to hamiltonian()/boundary_fn() evaluates the residual off
        # S^3, where the model was never trained (09-14: delta_sub 1.8 -> 0.16 on
        # q13_ours_s42 once normalised). No-op for systems without normalize_q.
        if hasattr(dyn, "normalize_q"):
            coords = torch.cat([coords[:, :1], dyn.normalize_q(coords[:, 1:])], -1)
        mi = dyn.coord_to_input(coords).requires_grad_(True)
        r = model({"coords": mi})
        V = dyn.io_to_value(r["model_in"], r["model_out"].squeeze(-1))
        dv = dyn.io_to_dv(r["model_in"], r["model_out"].squeeze(-1))
        state = coords[..., 1:]
        ham = dyn.hamiltonian(state.unsqueeze(0), dv[..., 1:].unsqueeze(0)).squeeze(0)
        g = dyn.boundary_fn(state)
        viol = torch.relu(torch.maximum(dv[..., 0] - ham, V - g))
        out.append(viol.detach().cpu().numpy())
        done += m
    return np.concatenate(out)


def mc_volume(model, dyn, level, tmax, n, device, rng, chunk=16384):
    """P_x[ V(x, T) <= level ] over the uniform box, by Monte Carlo."""
    lo = np.array([r[0] for r in dyn.state_test_range()], np.float64)
    hi = np.array([r[1] for r in dyn.state_test_range()], np.float64)
    hits = tot = 0
    while tot < n:
        m = min(chunk, n - tot)
        x = rng.uniform(lo, hi, size=(m, len(lo))).astype(np.float32)
        coords = torch.tensor(np.concatenate([np.full((m, 1), tmax, np.float32), x], -1),
                              device=device)
        if hasattr(dyn, "normalize_q"):
            coords = torch.cat([coords[:, :1], dyn.normalize_q(coords[:, 1:])], -1)
        with torch.no_grad():
            r = model({"coords": dyn.coord_to_input(coords)})
            V = dyn.io_to_value(r["model_in"], r["model_out"].squeeze(-1))
        hits += int((V <= level).sum()); tot += m
    return hits / tot


def air3d_gt(nx, dyn):
    from lvfm.grids import angle_axis, linear_axis
    from lvfm.hj_solvers import solve_air3d_relative
    half = float(dyn.state_max)
    xs = linear_axis((-half, half), nx); ps = angle_axis((-math.pi, math.pi), nx)
    X, Y, P = np.meshgrid(xs, xs, ps, indexing="ij")
    pts = np.stack([X.ravel(), Y.ravel(), P.ravel()], -1).astype(np.float32)
    gt = np.asarray(solve_air3d_relative(
        tau_steps=[1.0], xr_discretization=nx, yr_discretization=nx,
        theta_discretization=nx, xr_bounds=(-half, half), yr_bounds=(-half, half),
        theta_bounds=(-math.pi, math.pi), vp=dyn.pursuer_velocity, ve=dyn.evader_velocity,
        u_bound=dyn.omega_max, d_bound=dyn.omega_max, radius=dyn.goalR))[-1].reshape(-1)
    return pts, gt


def grid_values(model, dyn, pts, tau, device, chunk=16384):
    out = []
    for s in range(0, len(pts), chunk):
        c = np.concatenate([np.full((min(chunk, len(pts) - s), 1), tau, np.float32),
                            pts[s:s + chunk]], -1)
        with torch.no_grad():
            r = model({"coords": dyn.coord_to_input(torch.tensor(c, device=device))})
            out.append(dyn.io_to_value(r["model_in"], r["model_out"].squeeze(-1)).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default=None, help="score <prefix>_<model>_s<seed> runs")
    ap.add_argument("--models", nargs="*", default=MODEL_ORDER)
    ap.add_argument("--seeds", type=int, nargs="*", default=[42, 43, 44])
    ap.add_argument("--runs", nargs="*", default=None, help="explicit run names instead of --prefix")
    ap.add_argument("--n", type=int, default=200_000, help="certification samples")
    ap.add_argument("--n-vol", type=int, default=500_000, help="Monte-Carlo volume samples")
    ap.add_argument("--eps", type=float, default=0.01, help="allowed violation fraction")
    ap.add_argument("--beta", type=float, default=1e-3, help="1 - confidence")
    ap.add_argument("--gt", choices=["none", "air3d"], default="none")
    ap.add_argument("--nx", type=int, default=81)
    ap.add_argument("--seed", type=int, default=0, help="sampling seed (shared across runs)")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", default=None)
    a = ap.parse_args()
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    names = a.runs or [f"{a.prefix}_{model_name}_s{sd}" for model_name in a.models for sd in a.seeds]
    rows, gt_cache = [], None
    for name in names:
        rd = MADR / "runs" / name
        if not (rd / "training/checkpoints/model_final.pth").exists():
            print(f"  {name}: SKIP (missing)", flush=True); continue
        model, dyn, opt = load_madr_run(rd, dev)
        tmax = float(getattr(opt, "tMax", 1.0))
        rng = np.random.default_rng(a.seed)          # SAME samples for every run
        v = residual_violations(model, dyn, a.n, tmax, dev, rng)
        if not np.isfinite(v).all():
            print(f"  {name}: SKIP (non-finite violations)", flush=True); continue
        delta_sub, _ = conformal_quantile_bound(v, a.eps, a.beta)
        level = delta_sub * tmax                     # Gronwall horizon factor
        vol = mc_volume(model, dyn, level, tmax, a.n_vol, dev, rng)
        vol0 = mc_volume(model, dyn, 0.0, tmax, a.n_vol, dev, np.random.default_rng(a.seed + 1))
        row = {"run": name, "tmax": tmax, "n": a.n, "eps": a.eps, "beta": a.beta,
               "viol_frac": float((v > 1e-9).mean()),
               "delta_sub": float(delta_sub), "certified_level": float(level),
               "vol_raw": float(vol0), "vol_certified": float(vol)}
        if a.gt == "air3d":
            if gt_cache is None:
                gt_cache = air3d_gt(a.nx, dyn)
            pts, gt = gt_cache
            V = grid_values(model, dyn, pts, tmax, dev)
            inb = gt <= 0
            over = np.maximum(V[inb] - gt[inb], 0.0)
            # Grid cells are a census, not iid draws: plain empirical quantile,
            # matching the docstring (independent audit 09-02, S1-5).
            delta_gt = float(np.quantile(over, 1.0 - a.eps)) if len(over) else 0.0
            tube = V <= delta_gt
            row.update({"true_brt_volume": float(inb.mean()),
                        "delta_gt": float(delta_gt),
                        "vol_gt_certified": float(tube.mean()),
                        "fn_after_gt_shift": float(((~tube) & inb).sum() / inb.sum()),
                        "fn_raw": float(((V > 0) & inb).sum() / inb.sum())})
        rows.append(row)
        extra = (f"  dGT={row['delta_gt']:.4f} volGT={row['vol_gt_certified']:.4f}"
                 if "delta_gt" in row else "")
        print(f"  {name:22s} violFrac={row['viol_frac']:.4f} dSub={delta_sub:.4f} "
              f"level={level:.4f} vol {row['vol_raw']:.4f}->{vol:.4f}{extra}", flush=True)

    Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"config": vars(a), "rows": rows}, open(a.out_json, "w"), indent=2)
    if a.out_md and rows:
        hdr = ["run", "viol_frac", "delta_sub", "certified_level", "vol_raw", "vol_certified"]
        if any("delta_gt" in r for r in rows):
            hdr += ["delta_gt", "vol_gt_certified", "fn_raw", "fn_after_gt_shift"]
        L = ["# Conformal certification (horizon-corrected)", "",
             f"eps={a.eps}, beta={a.beta}, n={a.n}. `certified_level = delta_sub * T` "
             "(Gronwall factor); `vol_certified` = volume of the certified tube "
             "{V <= delta_sub*T}. GT columns certify (1-eps)-coverage of the true BRT.", "",
             "|" + "|".join(hdr) + "|", "|" + "|".join("---" for _ in hdr) + "|"]
        for r in rows:
            L.append("|" + "|".join(
                (f"{r[k]:.4f}" if isinstance(r.get(k), float) else str(r.get(k, "")))
                for k in hdr) + "|")
        Path(a.out_md).write_text("\n".join(L) + "\n")
        print("Wrote", a.out_md)
    print("Wrote", a.out_json)


if __name__ == "__main__":
    main()
