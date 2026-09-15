"""Evaluate the 6D joint Air3D 2x2 trained in the MADR codebase.

There is no tractable exact HJ solve on a 6D grid, but this system does not need
one. The 6D joint game is SE(2)-invariant: rotating and translating BOTH vehicles
leaves the value unchanged, so

    V_6D([x_p, y_p, th_p, x_e, y_e, th_e])  ==  V_3D(x_r, y_r, psi_r)

with (x_r, y_r) the pursuer's position in the evader's frame and psi_r = th_p -
th_e. The exact 3D Air3D solution therefore gives EXACT ground truth at any 6D
state, evaluated by Monte-Carlo over the 6D domain rather than on a grid.

Relative positions reach +/-2 per axis when both vehicles live in +/-1, so the 3D
reference is solved on +/-2 and states landing outside it are excluded (reported).
"""
import argparse, json, math, os, statistics, sys
from pathlib import Path
os.environ["JAX_PLATFORMS"] = "cpu"
import numpy as np, torch

REPO = Path(__file__).resolve().parents[1]
MADR = REPO / "external" / "madr"
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(MADR))
from lvfm.grids import angle_axis, linear_axis, periodic_interpolator
from lvfm.hj_solvers import solve_air3d_relative
from eval_madr_models import MODEL_LABEL, MODEL_ORDER, load_madr_run, values_on


def relative_state(s):
    """6D joint -> Air3D relative coords (pursuer in the evader's frame)."""
    dx = s[:, 0] - s[:, 3]; dy = s[:, 1] - s[:, 4]
    c, sn = np.cos(s[:, 5]), np.sin(s[:, 5])
    xr = c * dx + sn * dy
    yr = -sn * dx + c * dy
    pr = np.arctan2(np.sin(s[:, 2] - s[:, 5]), np.cos(s[:, 2] - s[:, 5]))
    return np.stack([xr, yr, pr], -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="a6r")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--n", type=int, default=400000, help="Monte-Carlo samples over the 6D domain")
    ap.add_argument("--ref-nx", type=int, default=161, help="3D reference grid resolution")
    ap.add_argument("--half", type=float, default=2.0, help="3D reference half-width")
    ap.add_argument("--targets", type=float, nargs="+", default=[0.05, 0.03, 0.01, 0.0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", default=None)
    a = ap.parse_args()
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    import dynamics.dynamics as D
    ref = D.Air6D(set_mode="avoid"); half6 = float(ref.state_max)

    print(f"solving the 3D reference ({a.ref_nx}^3 on +/-{a.half})...", flush=True)
    gt3 = np.asarray(solve_air3d_relative(
        tau_steps=[1.0], xr_discretization=a.ref_nx, yr_discretization=a.ref_nx,
        theta_discretization=a.ref_nx, xr_bounds=(-a.half, a.half),
        yr_bounds=(-a.half, a.half), theta_bounds=(-math.pi, math.pi),
        vp=ref.pursuer_velocity, ve=ref.evader_velocity,
        u_bound=ref.omega_max, d_bound=ref.omega_max, radius=ref.goalR))[-1]
    interp = periodic_interpolator(
        [linear_axis((-a.half, a.half), a.ref_nx), linear_axis((-a.half, a.half), a.ref_nx),
         angle_axis((-math.pi, math.pi), a.ref_nx)], gt3, periodic_dim=2, fill_value=np.nan)

    rng = np.random.default_rng(a.seed)
    S = np.stack([
        rng.uniform(-half6, half6, a.n), rng.uniform(-half6, half6, a.n),
        rng.uniform(-math.pi, math.pi, a.n),
        rng.uniform(-half6, half6, a.n), rng.uniform(-half6, half6, a.n),
        rng.uniform(-math.pi, math.pi, a.n)], -1).astype(np.float32)
    rel = relative_state(S)
    gt = interp(rel)
    ok = np.isfinite(gt)
    S, gt = S[ok], gt[ok]
    gu = gt <= 0
    print(f"  {ok.sum():,}/{a.n:,} samples inside the 3D reference "
          f"({1-ok.mean():.2%} excluded); true-unsafe fraction {gu.mean():.4f}", flush=True)

    g_target = np.linalg.norm(S[:, 0:2] - S[:, 3:5], axis=-1) - ref.goalR

    def vol_at_fn(v, t, lo=-1.0, hi=1.5):
        for _ in range(60):
            m = (lo + hi) / 2
            if float(((v > m) & gu).sum() / gu.sum()) <= t: hi = m
            else: lo = m
        tube = v <= hi
        return float(tube.mean()), float((tube & ~gu).sum() / max((~gu).sum(), 1)), float(hi)

    rows = []
    for model_name in MODEL_ORDER:
        for sd in a.seeds:
            rd = MADR / "runs" / f"{a.prefix}_{model_name}_s{sd}"
            if not (rd / "training/checkpoints/model_final.pth").exists():
                print(f"  {rd.name}: SKIP", flush=True); continue
            m, dyn, opt = load_madr_run(rd, dev)
            V, _ = values_on(m, dyn, S, 1.0, dev)
            pu = V <= 0.0
            inter = (pu & gu).sum(); uni = (pu | gu).sum()
            fr = [vol_at_fn(V, t) for t in a.targets]
            rows.append({
                "run": rd.name, "model": model_name, "seed": sd,
                "params": int(sum(p.numel() for p in m.parameters())),
                "deepReach_model": getattr(opt, "deepReach_model", "?"),
                "our_loss": bool(getattr(opt, "our_loss", False)),
                "use_MPC": not bool(getattr(opt, "not_use_MPC", False)),
                "iou_final": float(inter / max(uni, 1)),
                "mae_final": float(np.abs(V - gt).mean()),
                "fn_final": float((~pu & gu).sum() / max(gu.sum(), 1)),
                "fp_final": float((pu & ~gu).sum() / max((~gu).sum(), 1)),
                "volume_final": float(pu.mean()),
                "frac_V_le_Vstar": float((V - gt <= 1e-6).mean()),
                "max_v_minus_g": float((V - g_target).max()),
                "frontier_vol": [f[0] for f in fr],
                "frontier_fp": [f[1] for f in fr],
                "frontier_delta": [f[2] for f in fr],
            })
            r = rows[-1]
            print(f"  {rd.name:20s} IoU={r['iou_final']:.4f} FN={r['fn_final']:.4f} "
                  f"vol={r['volume_final']:.4f} maxV-g={r['max_v_minus_g']:+.4f} "
                  f"[{r['deepReach_model']}, our_loss={r['our_loss']}, MPC={r['use_MPC']}]", flush=True)

    agg = {}
    for model_name in MODEL_ORDER:
        rs = [r for r in rows if r["model"] == model_name]
        if not rs: continue
        sc = [k for k, v in rs[0].items() if isinstance(v, float)]
        agg[model_name] = {k: (statistics.fmean([r[k] for r in rs]),
                        statistics.stdev([r[k] for r in rs]) if len(rs) > 1 else 0.0,
                        len(rs)) for k in sc}
        agg[model_name]["frontier_vol"] = np.mean([r["frontier_vol"] for r in rs], axis=0).tolist()

    Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"n_used": int(len(S)), "true_unsafe_frac": float(gu.mean()),
               "targets": a.targets, "runs": rows,
               "aggregate": {k: {m: (list(v) if isinstance(v, (list, tuple)) else v)
                                 for m, v in d.items()} for k, d in agg.items()}},
              open(a.out_json, "w"), indent=2)
    if a.out_md:
        L = ["# Air6D (6D joint Air3D) 2x2 in the MADR codebase", "",
             "No 6D grid solve is needed: the joint game is SE(2)-invariant, so the exact",
             "3D Air3D solution gives exact ground truth at any 6D state. Evaluated by",
             f"Monte-Carlo over {len(S):,} samples of the 6D domain.",
             f"True-unsafe fraction {gu.mean():.4f}.", "",
             "|model_name|seeds|IoU|FN|FP|volume|frac(V<=V*)|max(V-g)|",
             "|---|---|---|---|---|---|---|---|"]
        for model_name in MODEL_ORDER:
            if model_name not in agg: continue
            g = agg[model_name]; f = lambda k, p=4: (f"{g[k][0]:.{p}f} ± {g[k][1]:.{p}f}" if g[k][1] else f"{g[k][0]:.{p}f}")
            L.append(f"|{MODEL_LABEL[model_name]}|{g['iou_final'][2]}|{f('iou_final')}|{f('fn_final')}|"
                     f"{f('fp_final')}|{f('volume_final')}|{f('frac_V_le_Vstar')}|{f('max_v_minus_g')}|")
        L += ["", "## Safety-tightness frontier (tube volume at matched FN)", "",
              "|model_name|" + "|".join(f"FN <= {t:g}" for t in a.targets) + "|",
              "|---|" + "|".join("---" for _ in a.targets) + "|"]
        for model_name in MODEL_ORDER:
            if model_name not in agg: continue
            L.append(f"|{MODEL_LABEL[model_name]}|" + "|".join(f"{v:.4f}" for v in agg[model_name]["frontier_vol"]) + "|")
        Path(a.out_md).write_text("\n".join(L) + "\n")
        print("Wrote", a.out_md)
    print("Wrote", a.out_json)


if __name__ == "__main__":
    main()
