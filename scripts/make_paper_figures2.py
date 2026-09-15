"""Paper figures, part 2 (Sep 9):
  rollouts3d : Air3D closed-loop trajectories in the relative frame under the verified-start
               pure-pursuit protocol (the protocol behind the 0/2,000 result), per model.
  rollouts6d : Air6D absolute-frame trajectories (pursuer + evader) under the generic protocol.
  rollouts9d : 9D three-vehicle trajectories under the generic protocol (no adversary).
  sweep      : margin-sweep dial, Air3D and Air6D: collision-vs-mismatch per trained eps, and
               IoU / certificate violation vs eps.  All numbers read from scripts/runs/eval/.
  shifted    : shifted-threshold control: collisions and interventions vs delta, DeepReach vs ConDR.
Nothing is hand-entered; every curve is computed from checkpoints or eval JSON on disk.
"""
import argparse, json, math, os, sys, contextlib, io, statistics as st
from pathlib import Path
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-paper")
import numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parents[1]
MADR = REPO / "external" / "madr"
E = REPO / "scripts" / "runs" / "eval"
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(MADR))
from eval_madr_models import load_madr_run, values_on

def _model_of(row):
    """Label of the model a result row belongs to.

    Rows written before the `arm` -> `model` rename carry the old key, so
    previously generated result files still load.
    """
    return row.get("model", row.get("arm"))


MODELS = ["vanilla", "ours", "madr", "madr_ours"]
NAME = {"vanilla": "DeepReach", "ours": "ConDR", "madr": "MADR", "madr_ours": "ConDR-MADR"}
COL = {"vanilla": "#7f7f7f", "ours": "#1f77b4", "madr": "#ff7f0e", "madr_ours": "#d62728"}
SAFE, CRASH = "#1f77b4", "#d62728"
FS = float(os.environ.get("FIG_FONT", "9"))
plt.rcParams.update({"font.size": FS, "axes.titlesize": FS + 0.5, "axes.labelsize": FS,
                     "legend.fontsize": FS - 1, "xtick.labelsize": FS - 1, "ytick.labelsize": FS - 1,
                     "pdf.fonttype": 42, "ps.fonttype": 42})


def load(name, dev):
    with contextlib.redirect_stdout(io.StringIO()):
        return load_madr_run(MADR / "runs" / name, dev)


def save(fig, out, stem):
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{stem}.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig); print("wrote", stem, flush=True)


def grid_2x3(n_panels=5, w=3.0, h=3.0):
    """2x3 grid: panels fill row-major, the last cell holds the legend."""
    fig, axs = plt.subplots(2, 3, figsize=(3 * w, 2 * h), sharex=True, sharey=True)
    flat = list(axs.ravel()); legend_ax = flat[-1]; legend_ax.axis("off")
    return fig, flat[:n_panels], legend_ax


# ----------------------------------------------------------------------------- Air3D
def fig_rollouts3d(a, dev, out):
    import safety_filter_rollout as R
    starts = R.sample_truly_safe_starts(600, a.safe_eps, a.rollout_seed)
    base = R.rollout(None, starts, a.dt, 0.0, use_filter=False, record=True)
    c0 = np.zeros(len(starts), bool)
    T0 = base["traj"]  # (steps+1, n, 3)
    d = np.sqrt(T0[..., 0] ** 2 + T0[..., 1] ** 2); c0 = (d <= R.RADIUS).any(0)
    idx = np.concatenate([np.where(c0)[0][: a.n_collide], np.where(~c0)[0][: a.n_safe]])
    S = starts[idx]
    panels = [("No filter (straight)", None)] + [(f"{NAME[model_name]} filter", model_name) for model_name in MODELS]
    fig, axs, lax = grid_2x3(len(panels))
    for ax, (title, model_name) in zip(axs, panels):
        if model_name is None:
            r = R.rollout(None, S, a.dt, 0.0, use_filter=False, record=True)
        else:
            vg = R.load_madr_model(f"{a.prefix3}_{model_name}_s{a.seed}", dev)
            r = R.rollout(vg, S, a.dt, 0.0, use_filter=True, record=True)
        T = r["traj"]; dist = np.sqrt(T[..., 0] ** 2 + T[..., 1] ** 2); c = (dist <= R.RADIUS).any(0)
        for i in range(len(S)):
            col = CRASH if c[i] else SAFE
            ax.plot(T[:, i, 0], T[:, i, 1], color=col, lw=0.9, alpha=0.9)
            ax.plot(T[0, i, 0], T[0, i, 1], marker="o", ms=2.5, color=col)
        ax.add_patch(Circle((0, 0), R.RADIUS, facecolor="#bbbbbb", edgecolor="k", lw=1.0, zorder=3))
        ax.set_aspect("equal"); ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
        ax.set_title(f"{title}\n{int(c.sum())}/{len(S)} collisions, {r['intervention_frac']:.0%} interv.")
    for ax in axs[3:]: ax.set_xlabel("$x_r$")
    axs[2].set_xlabel("$x_r$"); axs[2].tick_params(labelbottom=True)
    axs[0].set_ylabel("$y_r$"); axs[3].set_ylabel("$y_r$")
    lax.legend(handles=[Line2D([], [], color=SAFE, label="safe rollout"),
                        Line2D([], [], color=CRASH, label="collision\n(relative distance $\\leq 0.25$)")],
               loc="center", frameon=False, fontsize=FS + 1, handlelength=2.5)
    fig.tight_layout()
    save(fig, out, "fig_air3d_rollouts")


# ----------------------------------------------------------------------------- generic recorder
def rollout_record(model, dyn, starts, tau, dt, dev, use_filter, adv=None):
    """safety_filter_generic.rollout with position recording and the shared adversary."""
    s = torch.tensor(starts, dtype=torch.float32, device=dev); n = s.shape[0]
    steps = int(round(tau / dt)); collided = np.zeros(n, bool); interv = 0; total = 0
    traj = [s.detach().cpu().numpy().copy()]
    u_nom = dyn.control_init.to(dev).float().unsqueeze(0).repeat(n, 1) if getattr(dyn, "control_init", None) is not None \
        else torch.zeros(n, max(dyn.control_dim, 1), device=dev)
    if u_nom.shape[1] != max(dyn.control_dim, 1):
        u_nom = torch.zeros(n, max(dyn.control_dim, 1), device=dev)
    for k in range(steps):
        t_left = max(tau * (1.0 - k / steps), 1e-3); sn = s.detach().cpu().numpy().astype(np.float32)
        active = ~collided
        if use_filter:
            V, G = values_on(model, dyn, sn, t_left, dev, want_grad=True)
            gt = torch.tensor(G, dtype=torch.float32, device=dev)
            unsafe = torch.tensor((V <= 0.0) & active, device=dev)
            u = torch.where(unsafe[:, None], dyn.optimal_control(s, gt).float(), u_nom)
            interv += int(unsafe.sum());
        else:
            u = u_nom
        total += int(active.sum())
        if adv is not None and dyn.disturbance_dim > 0:
            am, ad = adv
            _, Ga = values_on(am, ad, sn, t_left, dev, want_grad=True)
            d = dyn.optimal_disturbance(s, torch.tensor(Ga, dtype=torch.float32, device=dev)).float()
            if d.ndim == 1: d = d[:, None]
        else:
            d = torch.zeros(n, max(dyn.disturbance_dim, 1), device=dev)
        s = s + dt * dyn.dsdt(s, u, d)
        if hasattr(dyn, "equivalent_wrapped_state"):
            try: s = dyn.equivalent_wrapped_state(s)
            except Exception: pass
        collided |= (dyn.boundary_fn(s).detach().cpu().numpy() <= 0.0)
        traj.append(s.detach().cpu().numpy().copy())
    return np.stack(traj, 0), collided, interv / max(total, 1)


def _pick_starts(dyn, dev, tau, dt, n_collide, n_safe, seed, adv, margin=0.15):
    from safety_filter_generic import sample_starts
    rng = np.random.default_rng(seed)
    S = sample_starts(dyn, 400, margin, rng, dev)
    _, c0, _ = rollout_record(None, dyn, S, tau, dt, dev, use_filter=False, adv=adv)
    idx = np.concatenate([np.where(c0)[0][:n_collide], np.where(~c0)[0][:n_safe]])
    return S[idx]


def fig_rollouts6d(a, dev, out):
    models = {model_name: load(f"{a.prefix6}_{model_name}_s{a.seed}", dev) for model_name in MODELS}
    dyn = models["vanilla"][1]; tau = float(getattr(models["vanilla"][2], "tMax", 1.0))
    adv = (models["vanilla"][0], models["vanilla"][1])
    S = _pick_starts(dyn, dev, tau, a.dt, a.n_collide, a.n_safe, a.rollout_seed, adv)
    panels = [("No filter", None)] + [(f"{NAME[model_name]} filter", model_name) for model_name in MODELS]
    fig, axs, lax = grid_2x3(len(panels))
    for ax, (title, model_name) in zip(axs, panels):
        m, d_, _ = models[model_name] if model_name else (None, dyn, None)
        T, c, iv = rollout_record(m, d_, S, tau, a.dt, dev, use_filter=model_name is not None, adv=adv)
        for i in range(len(S)):
            col = CRASH if c[i] else SAFE
            ax.plot(T[:, i, 3], T[:, i, 4], color=col, lw=0.9)                     # evader (ours)
            ax.plot(T[:, i, 0], T[:, i, 1], color=col, lw=0.7, ls=":", alpha=0.8)  # pursuer
            ax.plot(T[0, i, 3], T[0, i, 4], marker="o", ms=2.5, color=col)
            ax.plot(T[0, i, 0], T[0, i, 1], marker="x", ms=3, color=col)
        ax.set_aspect("equal"); ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3)
        ax.set_title(f"{title}\n{int(c.sum())}/{len(S)} collisions, {iv:.0%} interv.")
    for ax in axs[2:]: ax.set_xlabel("$x$")
    axs[2].tick_params(labelbottom=True)
    axs[0].set_ylabel("$y$"); axs[3].set_ylabel("$y$")
    lax.legend(handles=[Line2D([], [], color="k", label="evader (filtered)"),
                        Line2D([], [], color="k", ls=":", label="pursuer (worst-case,\nshared model)"),
                        Line2D([], [], color=SAFE, label="safe"), Line2D([], [], color=CRASH, label="collision")],
               loc="center", frameon=False, fontsize=FS + 1, handlelength=2.5)
    fig.tight_layout()
    save(fig, out, "fig_air6d_rollouts")


def fig_rollouts9d(a, dev, out):
    models = {model_name: load(f"{a.prefix9}_{model_name}_s{a.seed}", dev) for model_name in MODELS}
    dyn = models["vanilla"][1]; tau = float(getattr(models["vanilla"][2], "tMax", 1.0))
    S = _pick_starts(dyn, dev, tau, a.dt, a.n_collide9, a.n_safe9, a.rollout_seed, None)
    panels = [("No filter", None)] + [(f"{NAME[model_name]} filter", model_name) for model_name in MODELS]
    fig, axs, lax = grid_2x3(len(panels))
    styles = ["-", "--", ":"]
    for ax, (title, model_name) in zip(axs, panels):
        m, d_, _ = models[model_name] if model_name else (None, dyn, None)
        T, c, iv = rollout_record(m, d_, S, tau, a.dt, dev, use_filter=model_name is not None, adv=None)
        for i in range(len(S)):
            col = CRASH if c[i] else SAFE
            for v in range(3):
                ax.plot(T[:, i, 2 * v], T[:, i, 2 * v + 1], color=col, lw=0.9, ls=styles[v], alpha=0.9)
                ax.plot(T[0, i, 2 * v], T[0, i, 2 * v + 1], marker="o", ms=2.2, color=col)
        ax.set_aspect("equal"); ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3)
        ax.set_title(f"{title}\n{int(c.sum())}/{len(S)} collisions, {iv:.0%} interv.")
    for ax in axs[2:]: ax.set_xlabel("$x$ position")
    axs[2].tick_params(labelbottom=True)
    axs[0].set_ylabel("$y$ position"); axs[3].set_ylabel("$y$ position")
    lax.legend(handles=[Line2D([], [], color="k", ls=styles[v], label=f"vehicle {v+1}") for v in range(3)] +
               [Line2D([], [], color=SAFE, label="safe episode"), Line2D([], [], color=CRASH, label="collision\n(any pair $\\leq 0.25$)")],
               loc="center", frameon=False, fontsize=FS + 1, handlelength=2.5)
    fig.tight_layout()
    save(fig, out, "fig_9d_rollouts")


# ------------------------------------------------------- combined 9D + quadrotor rollouts
def fig_rollouts_combined(a, dev, out):
    """9D and quadrotor rollouts in one 2x5 figure: rows are systems, columns are
    the no-filter baseline and the four models. Saves the space of a second figure."""
    from safety_filter_generic import sample_starts
    panels = ["No filter"] + [f"{NAME[model_name]} filter" for model_name in MODELS]
    fig, axs = plt.subplots(2, 5, figsize=(1.62 * 5, 1.62 * 2 + 0.95))
    styles = ["-", "--", ":"]
    # ---- row 0: multi-vehicle 9D (three vehicles per episode)
    m9 = {model_name: load(f"{a.prefix9}_{model_name}_s{a.seed}", dev) for model_name in MODELS}
    dyn9 = m9["vanilla"][1]; tau9 = float(getattr(m9["vanilla"][2], "tMax", 1.0))
    S9 = _pick_starts(dyn9, dev, tau9, a.dt, a.n_collide9, a.n_safe9, a.rollout_seed, None)
    for j, model_name in enumerate([None] + MODELS):
        ax = axs[0, j]
        m, d_, _ = m9[model_name] if model_name else (None, dyn9, None)
        T, c, iv = rollout_record(m, d_, S9, tau9, a.dt, dev, use_filter=model_name is not None, adv=None)
        for i in range(len(S9)):
            col = CRASH if c[i] else SAFE
            for v in range(3):
                ax.plot(T[:, i, 2 * v], T[:, i, 2 * v + 1], color=col, lw=0.7, ls=styles[v], alpha=0.9)
                ax.plot(T[0, i, 2 * v], T[0, i, 2 * v + 1], marker="o", ms=1.8, color=col)
        ax.set_aspect("equal"); ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3)
        ax.set_title(f"{panels[j]}\n{int(c.sum())}/{len(S9)} coll., {iv:.0%} interv.", fontsize=FS - 1)
        ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1]); ax.tick_params(labelsize=FS - 2)
        if j: ax.set_yticklabels([])
        else: ax.set_ylabel("9D\n$y$ position", fontsize=FS - 1)
        ax.set_xticklabels([])
        print("  9D", panels[j], int(c.sum()), len(S9), round(float(iv), 4), flush=True)
    # ---- row 1: quadrotor 13D (same start pool for every panel)
    mq = {model_name: load(f"{a.prefixq}_{model_name}_s{a.seed}", dev) for model_name in MODELS}
    dynq = mq["vanilla"][1]; tauq = float(getattr(mq["vanilla"][2], "tMax", 1.0))
    rng = np.random.default_rng(a.rollout_seed)
    Sq = sample_starts(dynq, 600, 0.15, rng, dev)
    T0, c0, _ = rollout_record(None, dynq, Sq, tauq, a.dt, dev, use_filter=False)
    idx = np.concatenate([np.where(c0)[0][: a.n_collide], np.where(~c0)[0][: a.n_safe]])
    Sq = Sq[idx]
    for j, model_name in enumerate([None] + MODELS):
        ax = axs[1, j]
        if model_name is None:
            T, c, iv = T0[:, idx], c0[idx], 0.0
        else:
            m, d_, _ = mq[model_name]
            T, c, iv = rollout_record(m, d_, Sq, tauq, a.dt, dev, use_filter=True)
        for i in range(len(Sq)):
            col = CRASH if c[i] else SAFE
            ax.plot(T[:, i, 0], T[:, i, 1], color=col, lw=0.7, alpha=0.9)
            ax.plot(T[0, i, 0], T[0, i, 1], marker="o", ms=1.8, color=col)
        ax.add_patch(Circle((0, 0), dynq.collisionR, facecolor="#bbbbbb", edgecolor="k", lw=0.9, zorder=3))
        ax.set_aspect("equal"); ax.set_xlim(-3, 3); ax.set_ylim(-3, 3)
        ax.set_title(f"{int(c.sum())}/{len(Sq)} coll., {iv:.0%} interv.", fontsize=FS - 1)
        ax.set_xticks([-2, 0, 2]); ax.set_yticks([-2, 0, 2]); ax.tick_params(labelsize=FS - 2)
        if j: ax.set_yticklabels([])
        else: ax.set_ylabel("13D\n$y$ position", fontsize=FS - 1)
        ax.set_xlabel("$x$ position", fontsize=FS - 1)
        print("  13D", panels[j], int(c.sum()), len(Sq), round(float(iv), 4), flush=True)
    handles = [Line2D([], [], color=SAFE, label="safe episode"),
               Line2D([], [], color=CRASH, label="collision")] + \
              [Line2D([], [], color="k", ls=styles[v], label=f"9D vehicle {v+1}") for v in range(3)]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               fontsize=FS - 1, handlelength=2.0, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.075, 1, 1.0), h_pad=0.9, w_pad=0.35)
    save(fig, out, "fig_rollouts_combined")


# ----------------------------------------------------------------------------- frontier curves
def fig_frontier(a, dev, out):
    """Full safety-tightness frontier: for each model and seed, sweep the cutoff c over a fine grid,
    threshold the learned value at V <= c on the 81^3 evaluation grid, and record (FN, volume).
    Mean curve over seeds per model; the matched-FN table is four points on these curves."""
    import dynamics.dynamics as D
    from lvfm.grids import angle_axis, linear_axis
    from lvfm.hj_solvers import solve_air3d_relative
    ref = D.Air3D(set_mode="avoid"); half = float(ref.state_max); nx = 81
    xs = linear_axis((-half, half), nx); ps = angle_axis((-math.pi, math.pi), nx)
    X, Y, P = np.meshgrid(xs, xs, ps, indexing="ij")
    pts = np.stack([X.ravel(), Y.ravel(), P.ravel()], -1).astype(np.float32)
    gt = np.asarray(solve_air3d_relative(
        tau_steps=[1.0], xr_discretization=nx, yr_discretization=nx, theta_discretization=nx,
        xr_bounds=(-half, half), yr_bounds=(-half, half), theta_bounds=(-math.pi, math.pi),
        vp=ref.pursuer_velocity, ve=ref.evader_velocity, u_bound=ref.omega_max,
        d_bound=ref.omega_max, radius=ref.goalR))[-1].reshape(-1)
    gu = gt <= 0.0; exact_vol = float(gu.mean())
    cs = np.concatenate([np.linspace(-0.05, 0.0, 26), np.linspace(0.0, 0.12, 61)[1:]])
    seeds = [42, 43, 44, 45, 46]
    fig, ax = plt.subplots(figsize=(4.2, 3.3))
    for model_name in MODELS:
        FN = []; VOL = []
        for sd in seeds:
            model, dyn, _ = load(f"{a.prefix3}_{model_name}_s{sd}", dev)
            V, _ = values_on(model, dyn, pts, 1.0, dev)
            fn = [float(((V > c) & gu).sum() / gu.sum()) for c in cs]
            vol = [float((V <= c).mean()) for c in cs]
            FN.append(fn); VOL.append(vol)
        FN = np.mean(FN, 0); VOL = np.mean(VOL, 0)
        ax.plot(FN, VOL, color=COL[model_name], lw=1.6, label=NAME[model_name])
        k0 = int(np.argmin(np.abs(cs)))  # the c = 0 operating point
        ax.plot(FN[k0], VOL[k0], marker="o", ms=5, color=COL[model_name], mec="k", mew=0.6)
    ax.axhline(exact_vol, color="k", ls="--", lw=0.9, label=f"exact BRT volume ({exact_vol:.4f})")
    for t in (0.05, 0.03, 0.01):
        ax.axvline(t, color="#bbbbbb", lw=0.7, ls=":")
    ax.set_xscale("symlog", linthresh=1e-3); ax.set_xlim(-2e-4, 0.15)
    ax.set_xlabel("false-negative fraction (missed unsafe volume)")
    ax.set_ylabel("tube volume (fraction of box)")
    ax.set_ylim(0.075, 0.115)
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    fig.tight_layout()
    save(fig, out, "fig_frontier_air3d")


# ----------------------------------------------------------------------------- sweep dial
def _mm_file(p, M):
    """M=0 lives in mismatch0_<p>.json for the sweep prefixes and in filter_generic_<p>.json
    for the headline sets; M>0 always in mismatch<M>_<p>.json."""
    if M == 0:
        for f in (E / f"mismatch0_{p}.json", E / f"filter_generic_{p}.json"):
            if f.exists(): return f
        raise FileNotFoundError(f"no M=0 filter file for {p}")
    return E / f"mismatch{M}_{p}.json"


def _mean_coll(f, model_name):
    d = json.load(open(f)); rs = [r for r in d["results"] if _model_of(r) == model_name]
    return st.fmean([r["collision_rate"] for r in rs]), st.fmean([r["collision_rate_no_filter"] for r in rs])


def _sweep_data(system):
    Ms = [0, 0.02, 0.05, 0.1, 0.2]
    if system == "air3d":
        eps = {0.0: "a3m00", 0.02: "a3xxh", 0.05: "a3m05", 0.10: "a3m10"}
        gt = lambda p: json.load(open(E / (f"guarded_{p}_n5.json" if p == "a3xxh" else f"guarded_{p}.json")))["runs"]
        cert = lambda p: json.load(open(E / (f"certify_{p}_n5.json" if p == "a3xxh" else f"certify_{p}.json")))["rows"]
        run_key = "run"
    else:
        eps = {0.0: "a6m00", 0.02: "a6xxh", 0.05: "a6m05", 0.10: "a6m10"}
        gt = lambda p: json.load(open(E / f"air6d_{p}.json"))["runs"]
        cert = lambda p: json.load(open(E / f"certify_{p}.json"))["rows"]
        run_key = "run"
    curves = {}; nofilter = {}
    for e, p in eps.items():
        cs = []
        for M in Ms:
            f = _mm_file(p, M)
            c, nf = _mean_coll(f, "ours"); cs.append(c); nofilter[M] = nf
        curves[e] = cs
    # DeepReach curve from the a3xxh/a6xxh files
    p = eps[0.02]; dr = []
    for M in Ms:
        dr.append(_mean_coll(_mm_file(p, M), "vanilla")[0])
    iou = {}; viol = {}
    for e, p in eps.items():
        rs = [r for r in gt(p) if _model_of(r) == "ours"]; iou[e] = (st.fmean([r["iou_final"] for r in rs]), st.stdev([r["iou_final"] for r in rs]))
        cr = [r for r in cert(p) if "_ours_" in r[run_key]]; viol[e] = st.fmean([r["viol_frac"] for r in cr])
    rs = [r for r in gt(eps[0.02]) if _model_of(r) == "vanilla"]; iou_dr = st.fmean([r["iou_final"] for r in rs])
    cr = [r for r in cert(eps[0.02]) if "_vanilla_" in r[run_key]]; viol_dr = st.fmean([r["viol_frac"] for r in cr])
    return Ms, curves, dr, nofilter, iou, viol, iou_dr, viol_dr


def fig_sweep(a, out):
    fig, axs = plt.subplots(2, 2, figsize=(8.4, 6.4))
    shades = {0.0: "#9ecae1", 0.02: "#4292c6", 0.05: "#08519c", 0.10: "#08306b"}
    for row, system, label in [(0, "air3d", "Air3D"), (1, "air6d", "Air6D")]:
        Ms, curves, dr, nf, iou, viol, iou_dr, viol_dr = _sweep_data(system)
        ax = axs[row, 0]
        for e, cs in curves.items():
            ax.plot(Ms, cs, marker="o", ms=3.5, color=shades[e], label=f"ConDR $\\varepsilon={e:g}$")
        ax.plot(Ms, dr, marker="s", ms=3.5, color=COL["vanilla"], label="DeepReach")
        ax.plot(Ms, [nf[M] for M in Ms], color="k", ls="--", lw=0.9, label="no filter")
        ax.set_xlabel("model mismatch $M$"); ax.set_ylabel("collision rate (filter on)")
        ax.set_title(f"{label}: closed-loop safety vs mismatch")
        if row == 0: ax.legend(frameon=False, ncol=1, fontsize=FS - 1, loc="lower right")
        ax2 = axs[row, 1]
        es = sorted(iou); ax2.errorbar(es, [iou[e][0] for e in es], yerr=[iou[e][1] for e in es], marker="o", ms=3.5, color=COL["ours"], label="ConDR IoU", capsize=2)
        ax2.axhline(iou_dr, color=COL["vanilla"], ls="--", lw=1, label="DeepReach IoU")
        ax2.set_xlabel("trained margin $\\varepsilon$"); ax2.set_ylabel("IoU vs ground-truth BRT", color=COL["ours"])
        ax3 = ax2.twinx()
        ax3.plot(es, [viol[e] for e in es], marker="^", ms=3.5, color=CRASH, label="ConDR violation")
        ax3.axhline(viol_dr, color=CRASH, ls=":", lw=1, label="DeepReach violation")
        ax3.set_yscale("log"); ax3.set_ylabel("certificate violation fraction", color=CRASH)
        ax2.set_title(f"{label}: accuracy and certifiability vs $\\varepsilon$")
        if row == 0:
            h1, l1 = ax2.get_legend_handles_labels(); h2, l2 = ax3.get_legend_handles_labels()
            ax2.legend(h1 + h2, l1 + l2, frameon=False, fontsize=FS - 2, loc="lower left", handlelength=1.6, borderaxespad=0.3)
    fig.tight_layout()
    save(fig, out, "fig_margin_sweep")


# ----------------------------------------------------------------------------- shifted threshold
def fig_shifted(a, out):
    deltas = [0, 0.005, 0.01, 0.02, 0.03, 0.05]
    def collect3(delta):
        if delta == 0:
            rs = []
            for f in ["safety_filter_all.json", "safety_filter_s4546.json"]:
                d = json.load(open(E / f)); rs += (d["results"] if isinstance(d, dict) else d)
        else:
            rs = json.load(open(E / f"shifted_filter_a3xxh_d{delta}.json"))["results"]
            mp = E / f"shifted_filter_a3xxh_mpc_d{delta}.json"
            if mp.exists(): rs = rs + json.load(open(mp))["results"]
        return rs
    def collect6(delta):
        f = "filter_gtstart_pursuit_a6xxh.json" if delta == 0 else f"shifted_filter_a6xxh_d{delta}.json"
        return json.load(open(E / f))["results"]
    def split(rs, prefix):
        out_ = {}
        for model_name in MODELS:
            sel = [r for r in rs if (r.get("model") or "").startswith(f"{prefix}_{model_name}_s")]
            out_[model_name] = ([r["collision_rate"] for r in sel], [r["intervention_frac"] for r in sel])
        return out_
    rows = [("Air3D", {d: split(collect3(d), "a3xxh") for d in deltas}),
            ("Air6D", {d: split(collect6(d), "a6xxh") for d in deltas})]
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.4))
    for i, (title, data) in enumerate(rows):
        for ax, key, ylabel in [(axs[i, 0], 0, "collision rate"), (axs[i, 1], 1, "intervention fraction")]:
            for model_name in MODELS:
                m = [st.fmean(data[d][model_name][key]) for d in deltas]; s_ = [st.stdev(data[d][model_name][key]) for d in deltas]
                ax.errorbar(deltas, m, yerr=s_, marker="o", ms=3.5, capsize=2, color=COL[model_name], label=NAME[model_name])
            ax.axvline(0.02, color="k", ls=":", lw=0.8); ax.text(0.0205, ax.get_ylim()[1] * 0.95, "$c=\\varepsilon$", fontsize=FS - 2, va="top")
            ax.set_ylabel(ylabel)
            if i == 1: ax.set_xlabel("filter level $c$")
        axs[i, 0].set_title(title, loc="left", fontsize=FS)
    axs[0, 0].legend(frameon=False, fontsize=FS - 2, title="filter at $V_\\theta \\leq c$", title_fontsize=FS - 2, loc="center right")
    fig.tight_layout()
    save(fig, out, "fig_shifted_threshold")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", nargs="+", default=["rollouts3d", "rollouts6d", "rollouts9d", "frontier", "sweep", "shifted"])
    ap.add_argument("--prefix3", default="a3xxh"); ap.add_argument("--prefix6", default="a6xxh"); ap.add_argument("--prefix9", default="m9hi"); ap.add_argument("--prefixq", default="q13")
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--rollout-seed", type=int, default=7)
    ap.add_argument("--dt", type=float, default=0.01); ap.add_argument("--safe-eps", type=float, default=0.02)
    ap.add_argument("--n-collide", type=int, default=10); ap.add_argument("--n-safe", type=int, default=6)
    ap.add_argument("--n-collide9", type=int, default=5); ap.add_argument("--n-safe9", type=int, default=3)
    ap.add_argument("--out", default=str(REPO / "docs" / "figures"))
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if "rollouts3d" in a.which: fig_rollouts3d(a, dev, out)
    if "rollouts6d" in a.which: fig_rollouts6d(a, dev, out)
    if "rollouts9d" in a.which: fig_rollouts9d(a, dev, out)
    if "rollouts_combined" in a.which: fig_rollouts_combined(a, dev, out)
    if "frontier" in a.which: fig_frontier(a, dev, out)
    if "sweep" in a.which: fig_sweep(a, out)
    if "shifted" in a.which: fig_shifted(a, out)


if __name__ == "__main__":
    main()
