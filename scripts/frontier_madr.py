"""Safety-tightness frontier for the MADR-codebase Air3D 2x2.

Raw IoU compares models at whatever operating point they happen to land on, which
is not a fair comparison when the models have systematically different tube sizes:
an inflated tube scores low IoU without being wrong, if conservatism is what you
wanted. The comparison that controls for this sweeps the level-set shift delta
until each model reaches a MATCHED false-negative rate, then asks whose tube is
smaller there.

Outputs the frontier table + JSON + a figure.
"""
import argparse, json, math, os, sys
from pathlib import Path
os.environ["JAX_PLATFORMS"] = "cpu"
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
MADR = REPO / "external" / "madr"
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(MADR))
from lvfm.grids import angle_axis, linear_axis
from lvfm.hj_solvers import solve_air3d_relative
from eval_madr_models import MODEL_LABEL, MODEL_ORDER, load_madr_run, values_on

COLOR = {"vanilla": "#888888", "ours": "#1f77b4", "madr": "#ff7f0e", "madr_ours": "#d62728"}


def vol_at_fn(v, gu, target, lo=-1.0, hi=1.5, iters=60):
    for _ in range(iters):
        mid = (lo + hi) / 2
        fn = float(((v > mid) & gu).sum() / gu.sum())
        if fn <= target: hi = mid
        else: lo = mid
    tube = v <= hi
    return float(tube.mean()), float((tube & ~gu).sum() / max((~gu).sum(), 1)), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefixes", nargs="+", default=["a3", "a3hi"])
    ap.add_argument("--labels", nargs="+", default=["4096", "16384"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--nx", type=int, default=81)
    ap.add_argument("--targets", type=float, nargs="+", default=[0.05, 0.03, 0.01, 0.0])
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", default=None)
    ap.add_argument("--out-fig", default=None)
    a = ap.parse_args()
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    import dynamics.dynamics as D
    ref = D.Air3D(set_mode="avoid"); half = float(ref.state_max)
    xs = linear_axis((-half, half), a.nx); ps = angle_axis((-math.pi, math.pi), a.nx)
    X, Y, P = np.meshgrid(xs, xs, ps, indexing="ij")
    pts = np.stack([X.ravel(), Y.ravel(), P.ravel()], -1).astype(np.float32)
    gt = np.asarray(solve_air3d_relative(
        tau_steps=[1.0], xr_discretization=a.nx, yr_discretization=a.nx,
        theta_discretization=a.nx, xr_bounds=(-half, half), yr_bounds=(-half, half),
        theta_bounds=(-math.pi, math.pi), vp=ref.pursuer_velocity, ve=ref.evader_velocity,
        u_bound=ref.omega_max, d_bound=ref.omega_max, radius=ref.goalR))[-1].reshape(-1)
    gu = gt <= 0
    out = {"true_brt_volume": float(gu.mean()), "grid": a.nx, "targets": a.targets, "sets": {}}
    print(f"exact BRT volume {gu.mean():.4f}", flush=True)

    for pref, lab in zip(a.prefixes, a.labels):
        res = {}
        for model_name in MODEL_ORDER:
            per_seed = {}
            for sd in a.seeds:
                rd = MADR / "runs" / f"{pref}_{model_name}_s{sd}"
                if not (rd / "training/checkpoints/model_final.pth").exists():
                    print(f"  [{lab}] MISSING {rd.name} -- '{model_name}' aggregates fewer seeds", flush=True)
                    continue
                m, dyn, _ = load_madr_run(rd, dev)
                v, _ = values_on(m, dyn, pts, 1.0, dev)
                if not np.isfinite(v).all():
                    print(f"  [{lab}] DIVERGED {rd.name} (non-finite V) -- excluded", flush=True)
                    continue
                per_seed[sd] = [vol_at_fn(v, gu, t) for t in a.targets]
            if per_seed:
                seeds = sorted(per_seed)
                arr = np.array([per_seed[s] for s in seeds])   # (seeds, targets, 3)
                res[model_name] = {"vol": arr[:, :, 0].mean(0).tolist(),
                            "vol_std": arr[:, :, 0].std(0, ddof=1).tolist(),  # sample std, doc-wide convention
                            # Per-seed values, kept so differences can be paired by
                            # seed and re-analysed later; the mean alone cannot be
                            # (audit S1-4: the per-seed array was discarded).
                            "seeds": seeds,
                            "vol_per_seed": arr[:, :, 0].tolist(),
                            "fp": arr[:, :, 1].mean(0).tolist(),
                            "fp_per_seed": arr[:, :, 1].tolist(),
                            "delta": arr[:, :, 2].mean(0).tolist(),
                            "delta_per_seed": arr[:, :, 2].tolist(),
                            "n": int(arr.shape[0])}
                print(f"  [{lab}] {model_name:10s} vol@FN " +
                      " ".join(f"{t:.2f}:{v:.4f}" for t, v in zip(a.targets, res[model_name]['vol'])), flush=True)
        out["sets"][lab] = res

        # Paired-by-seed factor comparisons. At n=3 an eyeballed difference of
        # means is not evidence; pairing by seed removes the (large) seed effect.
        # Reported: mean/std of the per-seed difference, the fraction of seeds
        # improved, and a paired t p-value -- read the latter as descriptive at
        # this n, not as a significance stamp.
        from scipy import stats as sps
        comp = {}
        for hi_arm, lo_arm in [("ours", "vanilla"), ("madr_ours", "madr"),
                               ("madr_ours", "ours"), ("madr", "vanilla")]:
            if hi_arm not in res or lo_arm not in res: continue
            shared = sorted(set(res[hi_arm]["seeds"]) & set(res[lo_arm]["seeds"]))
            if len(shared) < 2: continue
            A = np.array([res[hi_arm]["vol_per_seed"][res[hi_arm]["seeds"].index(s)] for s in shared])
            B = np.array([res[lo_arm]["vol_per_seed"][res[lo_arm]["seeds"].index(s)] for s in shared])
            d = A - B                                  # negative = hi_arm tighter
            with np.errstate(all="ignore"):
                t = sps.ttest_rel(A, B, axis=0)
            comp[f"{hi_arm}_minus_{lo_arm}"] = {
                "seeds": shared,
                "diff_mean": d.mean(0).tolist(), "diff_std": d.std(0, ddof=1).tolist(),
                "frac_seeds_improved": (d < 0).mean(0).tolist(),
                "paired_t_p": np.atleast_1d(t.pvalue).astype(float).tolist()}
            print(f"  [{lab}] {hi_arm} - {lo_arm}: " +
                  " ".join(f"FN{t_:g}:{m:+.4f}(p={p:.3f})" for t_, m, p in
                           zip(a.targets, d.mean(0), np.atleast_1d(t.pvalue))), flush=True)
        out.setdefault("paired", {})[lab] = comp

    Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out_json, "w"), indent=2)

    if a.out_md:
        L = ["# Air3D safety-tightness frontier (MADR codebase)", "",
             "Each model's level set is shifted until it reaches a MATCHED false-negative",
             "rate; the table reports the resulting tube volume. Lower is better at equal FN.",
             f"Exact BRT volume: **{gu.mean():.4f}**.", ""]
        for lab in a.labels:
            if lab not in out["sets"]: continue
            L += [f"## collocation {lab}", "",
                  "|model_name|" + "|".join(f"vol @ FN<={t:g}" for t in a.targets) + "|",
                  "|---|" + "|".join("---" for _ in a.targets) + "|"]
            for model_name in MODEL_ORDER:
                if model_name not in out["sets"][lab]: continue
                r = out["sets"][lab][model_name]
                L.append(f"|{MODEL_LABEL[model_name]}|" + "|".join(
                    f"{v:.4f} ± {s:.4f}" for v, s in zip(r["vol"], r["vol_std"])) + "|")
            L.append("")
            if out.get("paired", {}).get(lab):
                L += ["Paired-by-seed differences (negative = first model_name tighter; "
                      "p from a paired t-test, descriptive at this n):", "",
                      "|comparison|" + "|".join(f"ΔFN<={t:g}" for t in a.targets) + "|",
                      "|---|" + "|".join("---" for _ in a.targets) + "|"]
                for name, c in out["paired"][lab].items():
                    L.append(f"|{name.replace('_minus_', ' − ')}|" + "|".join(
                        f"{m:+.4f} (p={p:.3f})" for m, p in zip(c["diff_mean"], c["paired_t_p"])) + "|")
                L.append("")
        Path(a.out_md).write_text("\n".join(L) + "\n")
        print("Wrote", a.out_md)

    if a.out_fig:
        fig, axes = plt.subplots(1, len(a.labels), figsize=(5.6*len(a.labels), 4.2), squeeze=False)
        for k, lab in enumerate(a.labels):
            ax = axes[0][k]
            for model_name in MODEL_ORDER:
                if model_name not in out["sets"].get(lab, {}): continue
                r = out["sets"][lab][model_name]
                ax.plot(a.targets, r["vol"], "o-", color=COLOR[model_name], lw=1.9, label=MODEL_LABEL[model_name])
            ax.axhline(gu.mean(), ls="--", c="k", lw=1.2)
            ax.text(max(a.targets), gu.mean(), " exact BRT volume", va="bottom", ha="right", fontsize=8)
            ax.invert_xaxis()
            ax.set_xlabel("false-negative rate (matched)"); ax.set_ylabel("tube volume")
            ax.set_title(f"collocation {lab}", loc="left", fontsize=10)
            ax.grid(alpha=.25); ax.legend(fontsize=8)
        fig.suptitle("Safety-tightness frontier: lower curve = smaller tube at equal safety", y=1.0)
        Path(a.out_fig).parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout(); fig.savefig(a.out_fig, dpi=140, bbox_inches="tight")
        print("Wrote", a.out_fig)


if __name__ == "__main__":
    main()
