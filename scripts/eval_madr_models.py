"""Evaluate the Air3D 2x2 trained in the MADR codebase (external/madr).

All four models are scored on ONE shared grid at ONE tau against ONE exact HJ
solve, using the canonical DeepReach Air3D parameters the MADR dynamics class has
been aligned to. Metrics match scripts/eval_arms.py so the two studies are
directly comparable.

  model         not_use_MPC  deepReach_model  our_loss
  vanilla        True        exact          False
  ours           True        exact_cons     True
  madr           False       exact          False
  madr_ours      False       exact_cons     True
"""
import argparse, json, math, os, pickle, statistics, sys
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"
import numpy as np, torch

REPO = Path(__file__).resolve().parents[1]
MADR = REPO / "external" / "madr"
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(MADR))

from lvfm.evaluation import approximate_hausdorff_from_levelsets
from lvfm.grids import angle_axis, linear_axis
from lvfm.helpers import compute_metrics
from lvfm.hj_solvers import solve_air3d_relative

MODEL_ORDER = ["vanilla", "ours", "madr", "madr_ours"]
MODEL_LABEL = {"vanilla": "vanilla DeepReach", "ours": "ours (cons + compl.)",
             "madr": "MADR", "madr_ours": "MADR + ours"}

# What each model's saved config MUST say: (deepReach_model, our_loss, use_MPC).
# Checked per run so a mislabeled or wrongly-launched run can never be scored
# under the wrong column.
MODEL_TREATMENT = {"vanilla": ("exact", False, False), "ours": ("exact_cons", True, False),
                 "madr": ("exact", False, True), "madr_ours": ("exact_cons", True, True)}

# Everything OUTSIDE the treatment must be identical across the four models of a
# set, or the "2x2" is not a controlled comparison (audit S4-3: the stack-A
# guard checked physics only, not training). gpinn_weight is here deliberately:
# gPINN is generic PINN infrastructure, so an model getting it while another does
# not is a confound, not a treatment. our_loss_* weights are NOT here: they are
# part of the `our_loss` treatment and inert on the baseline models.
SHARED_OPT_KEYS = ["dynamics_class", "tMax", "minWith", "num_epochs", "numpoints",
                   "num_src_samples", "num_nl", "num_hl", "lr", "clip_grad",
                   "counter_end", "pretrain", "pretrain_iters", "model", "model_mode",
                   "gpinn_weight",
                   # independent audit 09-02 S4-2: MPC-supervision scale must not
                   # drift between madr and madr_ours (inert on non-MPC models)
                   "MPC_batch_size", "num_MPC_batches", "num_MPC_data_samples",
                   "num_iterative_refinement", "MPC_dt", "MPC_loss_type",
                   "residual_norm"]
# Old checkpoints predate some argparse flags; an absent attribute means the
# then-current default, not a different setting. Fill these before comparing.
OPT_DEFAULTS = {"gpinn_weight": 0.0, "pretrain": False}


def baseline_residual_norm(opt):
    """Which VI-residual norm a BASELINE (non-our_loss) run was trained with.

    The launcher originally substituted a squared (L2) residual as the silent
    default; both DeepReach and MADR publish L1, and the default was later
    flipped. The flag set in the saved config identifies the code generation:
      - `baseline_l2` attr present  -> post-flip code: L1 unless baseline_l2=True
      - `baseline_l1` == True       -> pre-flip code, L1 opted in explicitly
      - otherwise                   -> pre-flip code, silent L2 default
    """
    if hasattr(opt, "baseline_l2"):
        return "l2" if getattr(opt, "baseline_l2") else "l1"
    return "l1" if getattr(opt, "baseline_l1", False) else "l2"


def load_madr_run(run_dir, device):
    import dynamics.dynamics as D
    import utils.modules as modules
    run_dir = Path(run_dir)
    opt = pickle.load(open(run_dir / "orig_opt.pickle", "rb"))
    # Was hardcoded to Air3D, so loading any other system silently built the wrong
    # network shape and failed with a state_dict size mismatch. Rebuild whatever
    # class the run was actually trained with, passing only the ctor args the saved
    # config carries (LessLinearND has no set_mode, Quadrotor needs collisionR, ...).
    import inspect
    cls = getattr(D, getattr(opt, "dynamics_class", "Air3D"))
    kw = {}
    for pname, par in inspect.signature(cls).parameters.items():
        if hasattr(opt, pname):
            kw[pname] = getattr(opt, pname)
        elif par.default is inspect._empty:
            raise ValueError(f"{cls.__name__} needs '{pname}' but the run config has no such field")
    dyn = cls(**kw)
    # CRITICAL: the class hard-codes deepReach_model in __init__ and
    # run_experiment.py overrides it from the CLI at train time. Restore it, or
    # an exact_cons model is decoded with the plain `exact` value map and every
    # number is silently wrong.
    dyn.set_model(getattr(opt, "deepReach_model", "exact"))
    # Same trap as deepReach_model: --cons_shift overrides the constructor's
    # scale-aware default at TRAIN time (run_experiment.py:751), so a checkpoint
    # trained with a custom shift decodes wrongly unless it is restored here too.
    if getattr(opt, "cons_shift", None) is not None:
        dyn.cons_shift = float(opt.cons_shift)
    model = modules.SingleBVPNet(
        in_features=dyn.input_dim, out_features=1, type="sine", mode="mlp",
        final_layer_factor=1.0, hidden_features=opt.num_nl,
        num_hidden_layers=opt.num_hl, periodic_transform_fn=dyn.periodic_transform_fn)
    ck = torch.load(run_dir / "training" / "checkpoints" / "model_final.pth",
                    map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    return model, dyn, opt


def values_on(model, dyn, states, tau, device, chunk=16384, want_grad=False):
    """states: (N,3) physical. Returns V (N,) and optionally dV/ds (N,3)."""
    V_out, G_out = [], []
    for s in range(0, states.shape[0], chunk):
        e = min(s + chunk, states.shape[0])
        c = np.concatenate([np.full((e - s, 1), float(tau), np.float32),
                            states[s:e].astype(np.float32)], -1)
        coords = torch.tensor(c, device=device)
        mi = dyn.coord_to_input(coords)
        if want_grad:
            mi = mi.requires_grad_(True)
            r = model({"coords": mi})
            V = dyn.io_to_value(r["model_in"], r["model_out"].squeeze(-1))
            dv = dyn.io_to_dv(r["model_in"], r["model_out"].squeeze(-1))
            G_out.append(dv[..., 1:].detach().cpu().numpy())
            V_out.append(V.detach().cpu().numpy())
        else:
            with torch.no_grad():
                r = model({"coords": mi})
                V_out.append(dyn.io_to_value(r["model_in"],
                                             r["model_out"].squeeze(-1)).cpu().numpy())
    V = np.concatenate(V_out)
    return (V, np.concatenate(G_out)) if want_grad else (V, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default=str(MADR / "runs"))
    ap.add_argument("--prefix", default="a3r", help="run-name prefix, e.g. a3 or a3hi")
    ap.add_argument("--models", nargs="*", default=MODEL_ORDER)
    ap.add_argument("--seeds", type=int, nargs="*", default=[42, 43, 44])
    ap.add_argument("--nx", type=int, default=81)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", default=None)
    ap.add_argument("--allow-partial", action="store_true",
                    help="report even if some (model, seed) runs are missing or "
                         "diverged. Without it a partial set aborts, because a "
                         "well-formatted table over a half-trained 2x2 reads as a "
                         "finished result (audit S4-2).")
    a = ap.parse_args()
    dev = torch.device("cpu" if a.cpu or not torch.cuda.is_available() else "cuda:0")

    import dynamics.dynamics as D
    ref = D.Air3D(set_mode="avoid")
    half = float(ref.state_max)
    xs = linear_axis((-half, half), a.nx)
    ys = linear_axis((-half, half), a.nx)
    ps = angle_axis((-math.pi, math.pi), a.nx)          # periodic: no node at +pi
    X, Y, P = np.meshgrid(xs, ys, ps, indexing="ij")
    pts = np.stack([X.ravel(), Y.ravel(), P.ravel()], -1).astype(np.float32)

    print(f"grid {a.nx}^3 on +/-{half}; system v={ref.evader_velocity} "
          f"omega={ref.omega_max} R={ref.goalR}", flush=True)
    gt = np.asarray(solve_air3d_relative(
        tau_steps=[a.tau], xr_discretization=a.nx, yr_discretization=a.nx,
        theta_discretization=a.nx, xr_bounds=(-half, half), yr_bounds=(-half, half),
        theta_bounds=(-math.pi, math.pi), vp=ref.pursuer_velocity,
        ve=ref.evader_velocity, u_bound=ref.omega_max, d_bound=ref.omega_max,
        radius=ref.goalR))[-1].reshape(-1)
    gu = gt <= 0.0
    g_target = np.linalg.norm(pts[:, :2], axis=-1) - ref.goalR
    print(f"exact BRT volume {gu.mean():.4f} (shared by every model_name)", flush=True)

    rows, cfgs, problems = [], [], []
    for model_name in a.models:
        for sd in a.seeds:
            rd = Path(a.runs_root) / f"{a.prefix}_{model_name}_s{sd}"
            ck = rd / "training" / "checkpoints" / "model_final.pth"
            if not ck.exists():
                print(f"  {rd.name}: SKIP (no model_final.pth)", flush=True)
                problems.append(f"{rd.name}: missing"); continue
            model, dyn, opt = load_madr_run(rd, dev)
            # A run that diverged after its last good checkpoint still writes
            # model_final.pth and LOOKS complete; non-finite values are the
            # cheapest reliable tell at eval time (audit S4-8).
            V, _ = values_on(model, dyn, pts, a.tau, dev)
            if not np.isfinite(V).all():
                print(f"  {rd.name}: SKIP (non-finite V -- diverged run)", flush=True)
                problems.append(f"{rd.name}: non-finite V (diverged)"); continue
            # The saved config must match the model the filename claims.
            want = MODEL_TREATMENT.get(model_name)
            got = (getattr(opt, "deepReach_model", "?"), bool(getattr(opt, "our_loss", False)),
                   not bool(getattr(opt, "not_use_MPC", False)))
            if want is not None and got != want:
                print(f"  {rd.name}: SKIP (treatment mismatch: config says "
                      f"model={got[0]}, our_loss={got[1]}, MPC={got[2]}; "
                      f"model_name '{model_name}' requires {want})", flush=True)
                problems.append(f"{rd.name}: treatment mismatch {got} != {want}"); continue
            cfgs.append((rd.name, opt))
            mae, iou = compute_metrics(V, gt)
            pu = V <= 0.0
            d = V - gt
            band = np.abs(gt) < 0.05
            hd = approximate_hausdorff_from_levelsets(
                V.reshape(a.nx, a.nx, a.nx)[:, :, a.nx // 2],
                gt.reshape(a.nx, a.nx, a.nx)[:, :, a.nx // 2], xs, ys)
            rows.append({
                "run": rd.name, "model": model_name, "seed": sd,
                "params": int(sum(p.numel() for p in model.parameters())),
                "deepReach_model": getattr(opt, "deepReach_model", "?"),
                "our_loss": bool(getattr(opt, "our_loss", False)),
                "use_MPC": not bool(getattr(opt, "not_use_MPC", False)),
                "iou_final": float(iou), "mae_final": float(mae),
                "fp_final": float((pu & ~gu).sum() / max((~gu).sum(), 1)),
                "fn_final": float((~pu & gu).sum() / max(gu.sum(), 1)),
                "volume_final": float(pu.mean()),
                "frac_V_le_Vstar": float((d <= 1e-6).mean()),
                "max_V_minus_Vstar": float(d.max()),
                "max_v_minus_g": float((V - g_target).max()),
                "hausdorff_midpsi": float(hd),
                "boundary_sign_acc": float((pu[band] == gu[band]).mean()) if band.any() else float("nan"),
                "delta_fn0": float(max(V[gu].max(), 0.0)) if gu.any() else 0.0,
            })
            r = rows[-1]
            print(f"  {rd.name:20s} IoU={r['iou_final']:.4f} FN={r['fn_final']:.4f} "
                  f"vol={r['volume_final']:.4f} maxV-g={r['max_v_minus_g']:+.4f} "
                  f"[{r['deepReach_model']}, our_loss={r['our_loss']}, MPC={r['use_MPC']}]",
                  flush=True)

    # ---- fairness guard: every non-treatment setting identical across the set.
    if cfgs:
        val = lambda opt, k: getattr(opt, k, OPT_DEFAULTS.get(k))
        ref_name, ref_opt = cfgs[0]
        for name, opt in cfgs[1:]:
            for k in SHARED_OPT_KEYS:
                v0, v1 = val(ref_opt, k), val(opt, k)
                if v0 != v1:
                    problems.append(f"shared-setting mismatch on '{k}': "
                                    f"{ref_name}={v0!r} vs {name}={v1!r}")
        # Baseline models must carry the PUBLISHED L1 residual; a leftover
        # pre-flip checkpoint was trained against a silent L2 substitute that
        # neither DeepReach nor MADR publish.
        for name, opt in cfgs:
            if not bool(getattr(opt, "our_loss", False)) and baseline_residual_norm(opt) != "l1":
                problems.append(f"{name}: stale baseline trained with the substituted "
                                "L2 residual, not the published L1 -- retrain it")

    # ---- completeness guard: all requested (model, seed) cells present.
    have = {(r["model"], r["seed"]) for r in rows}
    absent = [f"{a.prefix}_{model_name}_s{sd}" for model_name in a.models for sd in a.seeds
              if (model_name, sd) not in have]
    if problems or absent:
        print("\n!! SET NOT CLEAN:", flush=True)
        for p in problems: print(f"   - {p}", flush=True)
        if absent: print(f"   - cells absent from the table: {', '.join(absent)}", flush=True)
        if not a.allow_partial:
            raise SystemExit(
                f"refusing to report: {len(problems)} problem(s), "
                f"{len(absent)}/{len(a.models)*len(a.seeds)} cells missing. "
                "Fix the set or pass --allow-partial (and say so wherever the "
                "numbers are quoted).")

    agg = {}
    for model_name in a.models:
        rs = [r for r in rows if r["model"] == model_name]
        if not rs: continue
        keys = [k for k, v in rs[0].items() if isinstance(v, float)]
        def _fin(k):
            # a degenerate model (tube = whole box) leaves some derived metrics NaN;
            # aggregate over finite values so one such column cannot crash the run
            return [x for x in (r[k] for r in rs) if x == x and abs(x) != float("inf")]
        agg[model_name] = {k: ((statistics.fmean(_fin(k)) if _fin(k) else float("nan")),
                        (statistics.stdev(_fin(k)) if len(_fin(k)) > 1 else 0.0),
                        len(rs)) for k in keys}

    Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"grid": a.nx, "tau": a.tau, "true_brt_volume": float(gu.mean()),
               "runs": rows, "aggregate": {k: {m: list(v) for m, v in d.items()}
                                           for k, d in agg.items()}},
              open(a.out_json, "w"), indent=2)
    if a.out_md:
        L = ["# Air3D 2x2 in the MADR codebase", "",
             f"All model_names on one shared {a.nx}^3 grid at tau={a.tau} against one exact HJ solve.",
             f"System: canonical DeepReach Air3D (v={ref.evader_velocity}, "
             f"omega_max={ref.omega_max}, R={ref.goalR}, domain +/-{half}). "
             f"Exact BRT volume **{gu.mean():.4f}**.", "",
             "Capacity and learning-rate setup are matched to "
             "`scripts/configs/model_names/air3d_*.yaml`; only the two treatment factors vary.", "",
             "|model_name|seeds|IoU|MAE|FP|FN|volume|frac(V<=V*)|max(V-g)|delta(FN=0)|",
             "|---|---|---|---|---|---|---|---|---|---|"]
        for model_name in a.models:
            if model_name not in agg: continue
            g = agg[model_name]
            f = lambda k, p=4: (f"{g[k][0]:.{p}f} ± {g[k][1]:.{p}f}" if g[k][1] else f"{g[k][0]:.{p}f}")
            L.append(f"|{MODEL_LABEL.get(model_name, model_name)}|{g['iou_final'][2]}|{f('iou_final')}|{f('mae_final',5)}|"
                     f"{f('fp_final')}|{f('fn_final',5)}|{f('volume_final')}|"
                     f"{f('frac_V_le_Vstar')}|{f('max_v_minus_g')}|{f('delta_fn0',5)}|")
        L += ["", "## Per-run", "", "|run|params|model|our_loss|MPC|IoU|FN|max(V-g)|",
              "|---|---|---|---|---|---|---|---|"]
        for r in rows:
            L.append(f"|{r['run']}|{r['params']:,}|{r['deepReach_model']}|{r['our_loss']}|"
                     f"{r['use_MPC']}|{r['iou_final']:.4f}|{r['fn_final']:.4f}|{r['max_v_minus_g']:+.4f}|")
        Path(a.out_md).write_text("\n".join(L) + "\n")
        print("Wrote", a.out_md)
    print("Wrote", a.out_json)


if __name__ == "__main__":
    main()
