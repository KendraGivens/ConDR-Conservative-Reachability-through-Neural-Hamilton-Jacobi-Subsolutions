"""Paper figures (Sep 8):
  1. Air3D BRT slices: learned tube boundary per model vs the exact level-set BRT,
     at three relative headings; missed-unsafe regions (FN) filled red.
  2. Air6D slice through the SE(2) lift: a genuinely 6D query (evader off-centre
     and rotated), exact ground truth from the 3D solve.
  3. Quadrotor 13D closed-loop rollouts (MADR-style): identical starts, no filter
     vs DeepReach-filtered vs ConDR-filtered, x-y paths around the cylinder.
Models: the published-budget seed-42 runs (a3xxh / a6xxh / q13). Everything is
read from the trained checkpoints; nothing is hand-drawn.
"""
import argparse, math, os, sys, contextlib, io
from pathlib import Path
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-paper")
import numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

REPO = Path(__file__).resolve().parents[1]
MADR = REPO / "external" / "madr"
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(MADR))
from lvfm.grids import angle_axis, linear_axis, periodic_interpolator
from lvfm.hj_solvers import solve_air3d_relative
from eval_madr_models import load_madr_run, values_on
from eval_madr_air6d import relative_state

MODELS = ["vanilla", "ours", "madr", "madr_ours"]
NAME = {"vanilla": "DeepReach", "ours": "ConDR", "madr": "MADR", "madr_ours": "ConDR-MADR"}
COL = {"vanilla": "#7f7f7f", "ours": "#1f77b4", "madr": "#ff7f0e", "madr_ours": "#d62728"}
FS = float(os.environ.get("FIG_FONT", "9"))
plt.rcParams.update({"font.size": FS, "axes.titlesize": FS + 0.5, "axes.labelsize": FS,
                     "legend.fontsize": FS - 1, "xtick.labelsize": FS - 1, "ytick.labelsize": FS - 1,
                     "pdf.fonttype": 42, "ps.fonttype": 42})


def load(name, dev):
    with contextlib.redirect_stdout(io.StringIO()):
        return load_madr_run(MADR / "runs" / name, dev)


def solve_gt3(ref, nx, half):
    cache = REPO / "docs" / "figures" / f"gt3_nx{nx}_half{half:g}.npz"
    if cache.exists():
        z = np.load(cache); gt = z["gt"]
        axes = [linear_axis((-half, half), nx), linear_axis((-half, half), nx),
                angle_axis((-math.pi, math.pi), nx)]
        return gt, axes
    gt = np.asarray(solve_air3d_relative(
        tau_steps=[1.0], xr_discretization=nx, yr_discretization=nx,
        theta_discretization=nx, xr_bounds=(-half, half), yr_bounds=(-half, half),
        theta_bounds=(-math.pi, math.pi), vp=ref.pursuer_velocity, ve=ref.evader_velocity,
        u_bound=ref.omega_max, d_bound=ref.omega_max, radius=ref.goalR))[-1]
    axes = [linear_axis((-half, half), nx), linear_axis((-half, half), nx),
            angle_axis((-math.pi, math.pi), nx)]
    cache.parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(cache, gt=gt)
    return gt, axes


def panel_slice(ax, xs, ys, gt2, V2, model_name, title, show_fn=True):
    """gt2, V2: (len(xs), len(ys)) arrays indexed [x, y]."""
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    gu = gt2 <= 0
    ax.contourf(X, Y, gu.astype(float), levels=[0.5, 1.5], colors=["#dddddd"])
    if show_fn:
        fn = gu & (V2 > 0)
        if fn.any():
            ax.scatter(X[fn], Y[fn], s=1.2, c="#ff2222", marker="s", linewidths=0, zorder=4)
    ax.contour(X, Y, gt2, levels=[0.0], colors="k", linewidths=1.3)
    ax.contour(X, Y, V2, levels=[0.0], colors=[COL[model_name]], linewidths=1.6)
    ax.set_aspect("equal"); ax.set_title(title)
    ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1])


def add_miss_inset(ax, X, Y, gt2, tp, fn, fp, half_w=0.12, loc=(0.02, 0.56, 0.42, 0.42)):
    """Zoom inset centred on the deepest missed cell (most negative exact value
    among cells the model calls safe). Draws the same three classes plus the exact
    boundary, and marks the zoom window on the parent axes."""
    if not fn.any():
        return
    k = np.argmin(np.where(fn, gt2, np.inf))
    cx, cy = X.flat[k], Y.flat[k]
    ins = ax.inset_axes(list(loc))
    for mask, col in ((tp, "#c8c8c8"), (fp, "#9ecae1"), (fn, "#e41a1c")):
        if mask.any():
            ins.contourf(X, Y, mask.astype(float), levels=[0.5, 1.5], colors=[col])
    ins.contour(X, Y, gt2, levels=[0.0], colors="k", linewidths=0.9)
    ins.set_xlim(cx - half_w, cx + half_w); ins.set_ylim(cy - half_w, cy + half_w)
    ins.set_xticks([]); ins.set_yticks([]); ins.set_aspect("equal")
    for sp in ins.spines.values(): sp.set_linewidth(0.8)
    ax.indicate_inset_zoom(ins, edgecolor="k", linewidth=0.6, alpha=0.9)


def fig_air3d(a, dev, out):
    import dynamics.dynamics as D
    ref = D.Air3D(set_mode="avoid"); half = float(ref.state_max)
    gt, axes = solve_gt3(ref, a.nx, half)
    xs, ys, ps = axes
    psis = [0.0, math.pi / 2, math.pi]
    kidx = [int(np.argmin(np.abs(np.angle(np.exp(1j * (ps - p)))))) for p in psis]
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    fig, axs = plt.subplots(len(psis), len(MODELS), figsize=(1.9 * len(MODELS) + 0.3, 1.9 * len(psis) + 0.4),
                            sharex=True, sharey=True)
    for j, model_name in enumerate(MODELS):
        model, dyn, _ = load(f"{a.prefix3}_{model_name}_s{a.seed}", dev)
        for i, (psi, k) in enumerate(zip(psis, kidx)):
            pts = np.stack([X.ravel(), Y.ravel(), np.full(X.size, ps[k])], -1).astype(np.float32)
            V, _ = values_on(model, dyn, pts, 1.0, dev)
            V2 = V.reshape(len(xs), len(ys)); gt2 = gt[:, :, k]
            fn = float(((gt2 <= 0) & (V2 > 0)).sum() / max((gt2 <= 0).sum(), 1))
            title = NAME[model_name] if i == 0 else ""
            panel_slice(axs[i, j], xs, ys, gt2, V2, model_name, title)
            axs[i, j].text(0.03, 0.03, f"FN {fn:.3f}", transform=axs[i, j].transAxes,
                           fontsize=FS - 1.5, va="bottom", ha="left")
            if j == 0:
                lab = {0.0: r"$\psi=0$", math.pi / 2: r"$\psi=\pi/2$", math.pi: r"$\psi=\pi$"}[psi]
                axs[i, j].set_ylabel(lab + "\n$y_r$")
            if i == len(psis) - 1:
                axs[i, j].set_xlabel("$x_r$")
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [Patch(facecolor="#dddddd", label="ground-truth BRT"),
               Line2D([], [], color="k", lw=1.3, label="ground-truth BRT boundary"),
               Line2D([], [], color="#444444", lw=1.6, label="learned boundary $\\{V_\\theta=0\\}$"),
               Patch(facecolor="#ff4d4d", label="false negative (unsafe, called safe)")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.01))
    if not a.no_suptitle: fig.suptitle(f"Air3D, {a.budget3} collocation points, seed {a.seed}: learned tube vs exact BRT", y=0.995)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_air3d_slices.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_air3d_slices", flush=True)


def fig_air3d_classes(a, dev, out, prefix, budget_label, tag):
    """Classification map: at the heading slice where the baselines miss the most
    volume, fill correct-unsafe (gray), missed-unsafe (red) and false-alarm (blue)."""
    import dynamics.dynamics as D
    ref = D.Air3D(set_mode="avoid"); half = float(ref.state_max)
    gt, axes = solve_gt3(ref, a.nx, half)
    xs, ys, ps = axes
    X, Y, P = np.meshgrid(xs, ys, ps, indexing="ij")
    pts = np.stack([X.ravel(), Y.ravel(), P.ravel()], -1).astype(np.float32)
    Vs = {}
    for model_name in MODELS:
        model, dyn, _ = load(f"{prefix}_{model_name}_s{a.seed}", dev)
        V, _ = values_on(model, dyn, pts, 1.0, dev)
        Vs[model_name] = V.reshape(gt.shape)
    gu = gt <= 0
    # slices ranked by missed volume of the two baselines (shared across columns)
    miss = ((gu & (Vs["vanilla"] > 0)).sum(axis=(0, 1)) + (gu & (Vs["madr"] > 0)).sum(axis=(0, 1)))
    ks = list(np.argsort(-miss)[: a.n_slices])
    fig, axs = plt.subplots(len(ks), len(MODELS), figsize=(1.9 * len(MODELS) + 0.3, 1.9 * len(ks) + 0.5),
                            sharex=True, sharey=True, squeeze=False)
    Xs, Ys = np.meshgrid(xs, ys, indexing="ij")
    for i, k in enumerate(ks):
        for j, model_name in enumerate(MODELS):
            ax = axs[i, j]; g2 = gu[:, :, k]; p2 = Vs[model_name][:, :, k] <= 0
            tp = g2 & p2; fn = g2 & ~p2; fp = ~g2 & p2
            for mask, col in ((tp, "#c8c8c8"), (fp, "#9ecae1"), (fn, "#e41a1c")):
                if mask.any():
                    ax.contourf(Xs, Ys, mask.astype(float), levels=[0.5, 1.5], colors=[col])
            ax.contour(Xs, Ys, gt[:, :, k], levels=[0.0], colors="k", linewidths=1.0)
            ax.set_aspect("equal"); ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1])
            fnv = fn.sum() / max(g2.sum(), 1); fpv = fp.sum() / max((~g2).sum(), 1)
            ax.text(0.03, 0.03, f"FN {fnv:.3f}  FP {fpv:.3f}", transform=ax.transAxes, fontsize=FS - 3)
            if i == 0: ax.set_title(NAME[model_name])
            if j == 0: ax.set_ylabel(f"$\\psi={ps[k]:.2f}$\n$y_r$")
            if i == len(ks) - 1: ax.set_xlabel("$x_r$")
            add_miss_inset(ax, Xs, Ys, gt[:, :, k], tp, fn, fp)
    from matplotlib.patches import Patch
    handles = [Patch(facecolor="#c8c8c8", label="true positive (unsafe, flagged)"),
               Patch(facecolor="#e41a1c", label="false negative (unsafe, called safe)"),
               Patch(facecolor="#9ecae1", label="false positive (safe, called unsafe)"),
               plt.Line2D([], [], color="k", lw=1.0, label="ground-truth BRT boundary")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.0))
    if not a.no_suptitle: fig.suptitle(f"Air3D, {budget_label} collocation points, seed {a.seed}: classification of the exact tube "
                 f"on the headings with the largest baseline misses", y=0.995)
    fig.tight_layout(rect=(0, 0.17, 1, 0.98))
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_air3d_classes_{tag}.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_air3d_classes", tag, flush=True)


def fig_classes_combined(a, dev, out):
    """Air3D and Air6D classification maps in one 4x4 figure (two-column width):
    rows 1-2 are Air3D heading slices, rows 3-4 Air6D pursuer-position slices,
    columns are the four models, with a single shared legend."""
    import dynamics.dynamics as D
    from matplotlib.patches import Patch
    # ---- Air3D: pick the two heading slices with the largest baseline misses
    ref3 = D.Air3D(set_mode="avoid"); half3 = float(ref3.state_max)
    gt3g, axes3g = solve_gt3(ref3, a.nx, half3)
    xs3, ys3, ps3 = axes3g
    X3g, Y3g, P3g = np.meshgrid(xs3, ys3, ps3, indexing="ij")
    pts3 = np.stack([X3g.ravel(), Y3g.ravel(), P3g.ravel()], -1).astype(np.float32)
    V3 = {}
    for model_name in MODELS:
        model, dyn, _ = load(f"{a.prefix3}_{model_name}_s{a.seed}", dev)
        V, _ = values_on(model, dyn, pts3, 1.0, dev)
        V3[model_name] = V.reshape(gt3g.shape)
    gu3 = gt3g <= 0
    miss3 = ((gu3 & (V3["vanilla"] > 0)).sum(axis=(0, 1)) + (gu3 & (V3["madr"] > 0)).sum(axis=(0, 1)))
    ks3 = list(np.argsort(-miss3)[:2])
    X3, Y3 = np.meshgrid(xs3, ys3, indexing="ij")
    # ---- Air6D: pick the two relative-heading slices with the largest baseline misses
    ref6 = D.Air6D(set_mode="avoid"); half6 = float(ref6.state_max)
    gt6ref, axes6ref = solve_gt3(ref6, a.nx, 2.0)
    interp = periodic_interpolator(axes6ref, gt6ref, periodic_dim=2, fill_value=np.nan)
    ev = np.array([0.3, -0.2, math.pi / 4], dtype=np.float32)
    n = a.nx
    xs6 = linear_axis((-half6, half6), n); ys6 = linear_axis((-half6, half6), n)
    X6, Y6 = np.meshgrid(xs6, ys6, indexing="ij")
    def slice_states(dpsi):
        thp = np.arctan2(np.sin(ev[2] + dpsi), np.cos(ev[2] + dpsi))
        return np.stack([X6.ravel(), Y6.ravel(), np.full(X6.size, thp),
                         np.full(X6.size, ev[0]), np.full(X6.size, ev[1]), np.full(X6.size, ev[2])],
                        -1).astype(np.float32)
    models6 = {model_name: load(f"{a.prefix6}_{model_name}_s{a.seed}", dev) for model_name in MODELS}
    cands = np.linspace(-math.pi, math.pi, 24, endpoint=False)
    gts6 = {}; miss6 = []
    for dpsi in cands:
        S = slice_states(dpsi); g = interp(relative_state(S)).reshape(n, n)
        g = np.where(np.isfinite(g), g, 1.0); gts6[dpsi] = g; gu = g <= 0
        m = 0
        for model_name in ("vanilla", "madr"):
            model, dyn, _ = models6[model_name]; V, _ = values_on(model, dyn, S, 1.0, dev)
            m += int((gu & (V.reshape(n, n) > 0)).sum())
        miss6.append(m)
    ks6 = [cands[i] for i in np.argsort(-np.array(miss6))[:2]]
    # ---- draw
    fig, axs = plt.subplots(4, len(MODELS), figsize=(1.78 * len(MODELS), 1.78 * 4 + 0.35),
                            squeeze=False)
    def panel(ax, XX, YY, g2, V2, gfield, row, col, ylab, marker_ev=False):
        p2 = V2 <= 0
        tp = g2 & p2; fn = g2 & ~p2; fp = ~g2 & p2
        for mask, colr in ((tp, "#c8c8c8"), (fp, "#9ecae1"), (fn, "#e41a1c")):
            if mask.any():
                ax.contourf(XX, YY, mask.astype(float), levels=[0.5, 1.5], colors=[colr])
        ax.contour(XX, YY, gfield, levels=[0.0], colors="k", linewidths=1.0)
        if marker_ev:
            ax.plot([ev[0]], [ev[1]], marker=(3, 0, math.degrees(ev[2]) - 90), color="k", ms=6)
        ax.set_aspect("equal"); ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1])
        fnv = fn.sum() / max(g2.sum(), 1); fpv = fp.sum() / max((~g2).sum(), 1)
        ax.text(0.03, 0.03, f"FN {fnv:.3f}  FP {fpv:.3f}", transform=ax.transAxes, fontsize=FS - 3.5)
        add_miss_inset(ax, XX, YY, gfield, tp, fn, fp)
        if row == 0: ax.set_title(NAME[MODELS[col]], fontsize=FS)
        if col == 0: ax.set_ylabel(ylab, fontsize=FS - 0.5)
        else: ax.set_yticklabels([])
        if row in (1, 3): ax.set_xlabel("$x_r$" if row < 2 else "$x_p$", fontsize=FS)
        else: ax.set_xticklabels([])
    for i, k in enumerate(ks3):
        for j, model_name in enumerate(MODELS):
            panel(axs[i, j], X3, Y3, gu3[:, :, k], V3[model_name][:, :, k], gt3g[:, :, k], i, j,
                  f"Air3D  $\\psi={ps3[k]:.2f}$\n$y_r$")
    for i, dpsi in enumerate(ks6):
        S = slice_states(dpsi); g2 = gts6[dpsi] <= 0
        for j, model_name in enumerate(MODELS):
            model, dyn, _ = models6[model_name]; V, _ = values_on(model, dyn, S, 1.0, dev)
            panel(axs[i + 2, j], X6, Y6, g2, V.reshape(n, n), gts6[dpsi], i + 2, j,
                  f"Air6D  $\\Delta\\theta={dpsi:.2f}$\n$y_p$", marker_ev=True)
    handles = [Patch(facecolor="#c8c8c8", label="true positive (unsafe, flagged)"),
               Patch(facecolor="#e41a1c", label="false negative (unsafe, called safe)"),
               Patch(facecolor="#9ecae1", label="false positive (safe, called unsafe)"),
               plt.Line2D([], [], color="k", lw=1.0, label="ground-truth BRT boundary"),
               plt.Line2D([], [], marker="^", color="k", ls="", label="evader, fixed (Air6D)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0),
               fontsize=FS - 1)
    fig.tight_layout(rect=(0, 0.055, 1, 0.99), h_pad=0.8)
    # dashed separator between the Air3D block (rows 0-1) and the Air6D block (rows 2-3)
    fig.canvas.draw()
    y_sep = axs[2, 0].get_position().y1 + 0.010
    x0 = axs[0, 0].get_position().x0; x1 = axs[0, -1].get_position().x1
    fig.add_artist(plt.Line2D([x0, x1], [y_sep, y_sep], transform=fig.transFigure,
                              color="0.4", lw=0.9, ls=(0, (6, 4))))
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_classes_combined.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_classes_combined", flush=True)


def fig_air6d(a, dev, out):
    import dynamics.dynamics as D
    ref = D.Air6D(set_mode="avoid"); half6 = float(ref.state_max)
    gt3, axes3 = solve_gt3(ref, a.nx, 2.0)
    interp = periodic_interpolator(axes3, gt3, periodic_dim=2, fill_value=np.nan)
    # a genuinely 6D query: evader off-centre and rotated; vary the pursuer's position
    ev = np.array([0.3, -0.2, math.pi / 4], dtype=np.float32)
    rel_heads = [0.0, math.pi / 2, math.pi]
    n = a.nx
    xs = linear_axis((-half6, half6), n); ys = linear_axis((-half6, half6), n)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    fig, axs = plt.subplots(len(rel_heads), len(MODELS), figsize=(1.9 * len(MODELS) + 0.3, 1.9 * len(rel_heads) + 0.4),
                            sharex=True, sharey=True)
    for j, model_name in enumerate(MODELS):
        model, dyn, _ = load(f"{a.prefix6}_{model_name}_s{a.seed}", dev)
        for i, dpsi in enumerate(rel_heads):
            thp = np.arctan2(np.sin(ev[2] + dpsi), np.cos(ev[2] + dpsi))
            S = np.stack([X.ravel(), Y.ravel(), np.full(X.size, thp),
                          np.full(X.size, ev[0]), np.full(X.size, ev[1]), np.full(X.size, ev[2])], -1).astype(np.float32)
            gt = interp(relative_state(S)).reshape(n, n)
            V, _ = values_on(model, dyn, S, 1.0, dev)
            V2 = V.reshape(n, n)
            ok = np.isfinite(gt)
            gt = np.where(ok, gt, 1.0)
            fn = float(((gt <= 0) & (V2 > 0)).sum() / max((gt <= 0).sum(), 1))
            panel_slice(axs[i, j], xs, ys, gt, V2, model_name, NAME[model_name] if i == 0 else "")
            axs[i, j].plot([ev[0]], [ev[1]], marker=(3, 0, math.degrees(ev[2]) - 90), color="k", ms=7)
            axs[i, j].text(0.03, 0.03, f"FN {fn:.3f}", transform=axs[i, j].transAxes, fontsize=FS - 1.5)
            if j == 0:
                lab = {0.0: r"$\theta_p-\theta_e=0$", math.pi / 2: r"$\theta_p-\theta_e=\pi/2$",
                       math.pi: r"$\theta_p-\theta_e=\pi$"}[dpsi]
                axs[i, j].set_ylabel(lab + "\n$y_p$")
            if i == len(rel_heads) - 1:
                axs[i, j].set_xlabel("$x_p$")
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [Patch(facecolor="#dddddd", label="ground-truth BRT"),
               Line2D([], [], color="k", lw=1.3, label="ground-truth BRT boundary"),
               Line2D([], [], color="#444444", lw=1.6, label="learned boundary"),
               Patch(facecolor="#ff4d4d", label="false negative (unsafe, called safe)"),
               Line2D([], [], marker="^", color="k", ls="", label="evader (fixed)")]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.01))
    if not a.no_suptitle: fig.suptitle(f"Air6D, {a.budget6} collocation points, seed {a.seed}: pursuer-position slice "
                 f"with the evader fixed at ({ev[0]:.1f}, {ev[1]:.1f}, $\\pi/4$)", y=0.995)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_air6d_slice.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_air6d_slice", flush=True)


def fig_air6d_classes(a, dev, out):
    """Air6D classification map on pursuer-position slices (evader fixed off-centre and
    rotated), at the relative headings where the baselines miss the most volume."""
    import dynamics.dynamics as D
    ref = D.Air6D(set_mode="avoid"); half6 = float(ref.state_max)
    gt3, axes3 = solve_gt3(ref, a.nx, 2.0)
    interp = periodic_interpolator(axes3, gt3, periodic_dim=2, fill_value=np.nan)
    ev = np.array([0.3, -0.2, math.pi / 4], dtype=np.float32)
    n = a.nx
    xs = linear_axis((-half6, half6), n); ys = linear_axis((-half6, half6), n)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    def slice_states(dpsi):
        thp = np.arctan2(np.sin(ev[2] + dpsi), np.cos(ev[2] + dpsi))
        return np.stack([X.ravel(), Y.ravel(), np.full(X.size, thp),
                         np.full(X.size, ev[0]), np.full(X.size, ev[1]), np.full(X.size, ev[2])], -1).astype(np.float32)
    cands = np.linspace(-math.pi, math.pi, 24, endpoint=False)
    models = {model_name: load(f"{a.prefix6}_{model_name}_s{a.seed}", dev) for model_name in MODELS}
    gts = {}; miss = []
    for dpsi in cands:
        S = slice_states(dpsi); g = interp(relative_state(S)).reshape(n, n)
        g = np.where(np.isfinite(g), g, 1.0); gts[dpsi] = g; gu = g <= 0
        m = 0
        for model_name in ("vanilla", "madr"):
            model, dyn, _ = models[model_name]; V, _ = values_on(model, dyn, S, 1.0, dev)
            m += int((gu & (V.reshape(n, n) > 0)).sum())
        miss.append(m)
    ks = [cands[i] for i in np.argsort(-np.array(miss))[: a.n_slices]]
    fig, axs = plt.subplots(len(ks), len(MODELS), figsize=(1.9 * len(MODELS) + 0.3, 1.9 * len(ks) + 0.5),
                            sharex=True, sharey=True, squeeze=False)
    for i, dpsi in enumerate(ks):
        S = slice_states(dpsi); g2 = gts[dpsi] <= 0
        for j, model_name in enumerate(MODELS):
            model, dyn, _ = models[model_name]; V, _ = values_on(model, dyn, S, 1.0, dev)
            p2 = V.reshape(n, n) <= 0; ax = axs[i, j]
            tp = g2 & p2; fn = g2 & ~p2; fp = ~g2 & p2
            for mask, col in ((tp, "#c8c8c8"), (fp, "#9ecae1"), (fn, "#e41a1c")):
                if mask.any():
                    ax.contourf(X, Y, mask.astype(float), levels=[0.5, 1.5], colors=[col])
            ax.contour(X, Y, gts[dpsi], levels=[0.0], colors="k", linewidths=1.0)
            ax.plot([ev[0]], [ev[1]], marker=(3, 0, math.degrees(ev[2]) - 90), color="k", ms=7)
            ax.set_aspect("equal"); ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1])
            fnv = fn.sum() / max(g2.sum(), 1); fpv = fp.sum() / max((~g2).sum(), 1)
            ax.text(0.03, 0.03, f"FN {fnv:.3f}  FP {fpv:.3f}", transform=ax.transAxes, fontsize=FS - 3)
            add_miss_inset(ax, X, Y, gts[dpsi], tp, fn, fp)
            if i == 0: ax.set_title(NAME[model_name])
            if j == 0: ax.set_ylabel(f"$\\Delta\\theta={dpsi:.2f}$\n$y_p$")
            if i == len(ks) - 1: ax.set_xlabel("$x_p$")
    from matplotlib.patches import Patch
    handles = [Patch(facecolor="#c8c8c8", label="true positive (unsafe, flagged)"),
               Patch(facecolor="#e41a1c", label="false negative (unsafe, called safe)"),
               Patch(facecolor="#9ecae1", label="false positive (safe, called unsafe)"),
               plt.Line2D([], [], color="k", lw=1.0, label="ground-truth BRT boundary"),
               plt.Line2D([], [], marker="^", color="k", ls="", label="evader (fixed)")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.0))
    if not a.no_suptitle: fig.suptitle(f"Air6D, {a.budget6} collocation points, seed {a.seed}: classification on pursuer-position "
                 f"slices with the largest baseline misses", y=0.995)
    fig.tight_layout(rect=(0, 0.22, 1, 0.98))
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_air6d_classes.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_air6d_classes", flush=True)


def rollout_record(model, dyn, starts, tau, dt, dev, use_filter, margin=0.0):
    """Generic-protocol rollout (safety_filter_generic.rollout) that also records
    positions. Quadrotor: no disturbance; nominal = hover trim (control_init)."""
    s = torch.tensor(starts, dtype=torch.float32, device=dev)
    n = s.shape[0]; steps = int(round(tau / dt))
    collided = np.zeros(n, dtype=bool); interv = np.zeros(n, dtype=int)
    traj = [s[:, :3].detach().cpu().numpy().copy()]
    u_nom = dyn.control_init.to(dev).float().unsqueeze(0).repeat(n, 1)
    for k in range(steps):
        t_left = max(tau * (1.0 - k / steps), 1e-3)
        sn = s.detach().cpu().numpy()
        if use_filter:
            V, G = values_on(model, dyn, sn.astype(np.float32), t_left, dev, want_grad=True)
            gt = torch.tensor(G, dtype=torch.float32, device=dev)
            unsafe = torch.tensor(V <= margin, device=dev) & torch.tensor(~collided, device=dev)
            u_opt = dyn.optimal_control(s, gt).float()
            u = torch.where(unsafe[:, None], u_opt, u_nom)
            interv += (unsafe.cpu().numpy()).astype(int)
        else:
            u = u_nom
        d = torch.zeros(n, max(dyn.disturbance_dim, 1), device=dev)
        s = s + dt * dyn.dsdt(s, u, d)
        if hasattr(dyn, "equivalent_wrapped_state"):
            try: s = dyn.equivalent_wrapped_state(s)
            except Exception: pass
        collided |= (dyn.boundary_fn(s).detach().cpu().numpy() <= 0.0)
        traj.append(s[:, :3].detach().cpu().numpy().copy())
    return np.stack(traj, 1), collided, interv / steps


def fig_quadrotor(a, dev, out):
    from safety_filter_generic import sample_starts
    m_v, dyn, opt_v = load(f"{a.prefixq}_vanilla_s{a.seed}", dev)
    m_o, dyn_o, _ = load(f"{a.prefixq}_ours_s{a.seed}", dev)
    m_m, dyn_m, _ = load(f"{a.prefixq}_madr_s{a.seed}", dev)
    m_mo, dyn_mo, _ = load(f"{a.prefixq}_madr_ours_s{a.seed}", dev)
    tau = float(getattr(opt_v, "tMax", 1.0))
    rng = np.random.default_rng(a.rollout_seed)
    S = sample_starts(dyn, 600, 0.15, rng, dev)
    # pick starts that are informative: the nominal (hover) trajectory collides for some,
    # not for others; keep a mix, drawn from the SAME pool for every panel.
    T0, c0, _ = rollout_record(m_v, dyn, S, tau, a.dt, dev, use_filter=False)
    idx_c = np.where(c0)[0][: a.n_collide]; idx_s = np.where(~c0)[0][: a.n_safe]
    idx = np.concatenate([idx_c, idx_s]); S = S[idx]
    panels = [("No filter (hover)", None, None),
              ("DeepReach filter", m_v, dyn),
              ("ConDR filter", m_o, dyn_o),
              ("MADR filter", m_m, dyn_m),
              ("ConDR-MADR filter", m_mo, dyn_mo)]
    fig, axs2 = plt.subplots(2, 3, figsize=(9.0, 6.0), sharex=True, sharey=True)
    flat = list(axs2.ravel()); lax = flat[-1]; lax.axis("off"); axs = flat[:5]
    stats = []
    for ax, (title, m, dy) in zip(axs, panels):
        if m is None:
            T, c, iv = T0[idx], c0[idx], np.zeros(len(idx))
        else:
            T, c, iv = rollout_record(m, dy, S, tau, a.dt, dev, use_filter=True)
        for i in range(len(S)):
            col = "#d62728" if c[i] else "#1f77b4"
            ax.plot(T[i, :, 0], T[i, :, 1], color=col, lw=0.9, alpha=0.9)
            ax.plot(T[i, 0, 0], T[i, 0, 1], marker="o", ms=2.5, color=col)
        ax.add_patch(Circle((0, 0), dyn.collisionR, facecolor="#bbbbbb", edgecolor="k", lw=1.0, zorder=3))
        ax.set_aspect("equal"); ax.set_xlim(-3, 3); ax.set_ylim(-3, 3)
        ax.set_title(f"{title}\n{int(c.sum())}/{len(S)} collisions, {iv.mean():.0%} interv.")
        stats.append((title, int(c.sum()), len(S), float(iv.mean())))
    for ax in axs[2:]: ax.set_xlabel("$x$ position")
    axs[2].tick_params(labelbottom=True)
    axs[0].set_ylabel("$y$ position"); axs[3].set_ylabel("$y$ position")
    from matplotlib.lines import Line2D
    lax.legend(handles=[Line2D([], [], color="#1f77b4", label="safe rollout"),
                        Line2D([], [], color="#d62728", label="collides with\nthe cylinder")],
               loc="center", frameon=False, fontsize=FS + 1, handlelength=2.5)
    if not a.no_suptitle: fig.suptitle(f"Quadrotor 13D, seed {a.seed}: closed-loop rollouts from identical starts "
                 f"(horizon {tau:g} s, cylinder radius {dyn.collisionR:g})", y=0.995)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_quadrotor_rollouts.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_quadrotor_rollouts", stats, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", nargs="+", default=["air3d", "air6d", "quad"])
    ap.add_argument("--prefix3", default="a3xxh"); ap.add_argument("--budget3", default="65,536")
    ap.add_argument("--prefix6", default="a6xxh"); ap.add_argument("--budget6", default="65,536")
    ap.add_argument("--prefixq", default="q13")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--nx", type=int, default=161)
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--rollout-seed", type=int, default=7)
    ap.add_argument("--n-collide", type=int, default=10)
    ap.add_argument("--n-safe", type=int, default=6)
    ap.add_argument("--n-slices", type=int, default=2)
    ap.add_argument("--no-suptitle", action="store_true", help="omit figure-level titles (captions carry them in the paper)")
    ap.add_argument("--out", default=str(REPO / "docs" / "figures"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if "air3d" in a.which: fig_air3d(a, dev, out)
    if "classes65k" in a.which: fig_air3d_classes(a, dev, out, a.prefix3, a.budget3, "65k")
    if "classes4k" in a.which: fig_air3d_classes(a, dev, out, "a3r", "4,096", "4k")
    if "air6d" in a.which: fig_air6d(a, dev, out)
    if "classes6d" in a.which: fig_air6d_classes(a, dev, out)
    if "combined" in a.which: fig_classes_combined(a, dev, out)
    if "quad" in a.which: fig_quadrotor(a, dev, out)


if __name__ == "__main__":
    main()
