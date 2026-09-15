"""
Self-contained MPPI-style MPC label generator for the low-dim systems the
external MPC/MADR codebases do NOT support (2D linear oscillator, Air3D game).

For each sampled (state, tau) it estimates the avoid-BRT value by sampling-based
trajectory optimization (the MPC-DeepReach paradigm): sample control sequences
(and, for the game, adversarial disturbance sequences), roll out the exact
dynamics, and take the minimax of the min-over-horizon obstacle margin.

  oscillator: control_mode=min, disturbance_mode=max  -> V = min_u max_d min_t g
  air3d     : control_mode=max, disturbance_mode=min  -> V = max_u min_d min_t g

Validated against the exact HJ ground truth before use (--validate).
Output: torch file {states:[N,d], taus:[N], values:[N]} for load_mpc_data.
"""
import argparse, math, torch

def dyn_step(system, x, u, d, dt, p):
    # x:[...,sd]  u,d:[...,1]  returns x_next (explicit RK4)
    def f(x):
        if system == "linear_oscillator_2d":
            x1, x2 = x[..., 0], x[..., 1]
            base = torch.stack([x2, -(p["omega"]**2) * x1], dim=-1)
            gj = torch.zeros_like(base); gj[..., 1] = 1.0            # control jac [0,1]
            dj = torch.zeros_like(base); dj[..., 1] = 1.0            # dist jac    [0,1]
        else:  # air3d relative
            xr, yr, th = x[..., 0], x[..., 1], x[..., 2]
            base = torch.stack([-p["ve"] + p["vp"]*torch.cos(th), p["vp"]*torch.sin(th),
                                torch.zeros_like(th)], dim=-1)
            gj = torch.stack([yr, -xr, -torch.ones_like(th)], dim=-1)  # control jac
            dj = torch.zeros_like(base); dj[..., 2] = 1.0             # dist jac [0,0,1]
        return base + gj * u + dj * d
    k1 = f(x); k2 = f(x + 0.5*dt*k1); k3 = f(x + 0.5*dt*k2); k4 = f(x + dt*k3)
    return x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def obstacle(system, x, radius):
    return torch.linalg.norm(x[..., :2], dim=-1) - radius

def mpc_value(system, x0, tau, p, K, D, dt, device, rng=None):
    """x0:[B,sd] tau:[B] -> value:[B] via minimax MPPI. Chunked over control samples.

    NOTE (audit M-2): this is OPEN-LOOP random shooting. max_u min_d over sampled
    control/disturbance SEQUENCES is neither an upper nor a lower bound on the
    feedback (HJ) value -- the maximiser is handicapped by committing to a
    sequence, and min over D random sequences over-estimates the inner minimum.
    Run --validate to measure the resulting bias against the exact HJ solution
    before using these labels as a baseline."""
    B, sd = x0.shape
    Hmax = int(math.ceil(float(tau.max().item()) / dt))
    ub, db = p["u_bound"], p["d_bound"]
    cmode_max = (system == "air3d")   # control maximizes (evader); else minimizes
    # sample control [B,K,Hmax] and disturbance [B,D,Hmax] in bounds.
    # These used to bypass `rng`, so the labels were unseeded (audit P3-4).
    U = (torch.rand(B, K, Hmax, device=device, generator=rng)*2-1)*ub
    Dd = (torch.rand(B, D, Hmax, device=device, generator=rng)*2-1)*db
    # expand to [B,K,D,...]; roll out, track running min of g
    x = x0[:, None, None, :].expand(B, K, D, sd).clone()
    g_run = obstacle(system, x, p["radius"])                       # [B,K,D]
    active = torch.ones(B, K, D, device=device, dtype=torch.bool)
    for h in range(Hmax):
        still = (h < torch.ceil(tau/dt)[:, None, None]).expand(B, K, D)
        u = U[:, :, h][:, :, None].expand(B, K, D)[..., None]
        d = Dd[:, None, :, h].expand(B, K, D)[..., None]
        xn = dyn_step(system, x, u, d, dt, p)
        x = torch.where(still[..., None], xn, x)
        g = obstacle(system, x, p["radius"])
        g_run = torch.where(still, torch.minimum(g_run, g), g_run)
    # disturbance extremum (inner): air3d dist=min -> take min over D; osc dist=max -> max over D
    v_kd = g_run
    v_k = v_kd.min(dim=2).values if system == "air3d" else v_kd.max(dim=2).values   # [B,K]
    # control extremum (outer)
    v = v_k.max(dim=1).values if cmode_max else v_k.min(dim=1).values               # [B]
    return v

def sample_states(system, n, p, device, rng):
    if system == "linear_oscillator_2d":
        lo = torch.tensor(p["x_lo"], device=device); hi = torch.tensor(p["x_hi"], device=device)
        return lo + (hi-lo)*torch.rand(n, 2, generator=rng, device=device)
    xr = (torch.rand(n, generator=rng, device=device)*2-1)*p["xy"]
    yr = (torch.rand(n, generator=rng, device=device)*2-1)*p["xy"]
    th = (torch.rand(n, generator=rng, device=device)*2-1)*math.pi
    return torch.stack([xr, yr, th], dim=-1)

# Label-sampling domains MUST match the training configs' x_bounds. They used to
# be wider (air3d xy=1.5, oscillator +/-2.5 against configs' +/-1.0), so 56% of
# air3d and 84% of oscillator labels landed outside the trained domain and mapped
# to |x_net| > 1 after scaling -- states the network is never asked to be correct
# on, dominating the supervised MSE term (audit P0-2). Override with --xy.
PARAMS = {
    "linear_oscillator_2d": dict(omega=1.0, u_bound=1.0, d_bound=0.5, radius=0.25,
                                 x_lo=[-1.0,-1.0], x_hi=[1.0,1.0], xy=1.0),
    "air3d": dict(vp=0.75, ve=0.75, u_bound=3.0, d_bound=3.0, radius=0.25, xy=1.0),
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True, choices=list(PARAMS))
    ap.add_argument("--n", type=int, default=12000)
    ap.add_argument("--T", type=float, default=1.0)
    ap.add_argument("--K", type=int, default=256); ap.add_argument("--D", type=int, default=64)
    ap.add_argument("--dt", type=float, default=0.02)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--out", required=True)
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--xy", type=float, default=None,
                    help="half-width of the position domain; must match the training config's x_bounds")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p = dict(PARAMS[a.system])
    if a.xy is not None:
        p["xy"] = float(a.xy)
        p["x_lo"] = [-float(a.xy)] * 2
        p["x_hi"] = [float(a.xy)] * 2
    print(f"[domain] sampling states on +/-{p['xy']} (must match the training config x_bounds)", flush=True); rng = torch.Generator(device=dev).manual_seed(a.seed)
    states = sample_states(a.system, a.n, p, dev, rng)
    taus = torch.rand(a.n, generator=rng, device=dev)*a.T
    vals = torch.empty(a.n, device=dev)
    for s in range(0, a.n, a.chunk):
        e = min(s+a.chunk, a.n)
        vals[s:e] = mpc_value(a.system, states[s:e], taus[s:e], p, a.K, a.D, a.dt, dev, rng=rng)
        if s % (a.chunk*10) == 0: print(f"  {e}/{a.n}", flush=True)
    torch.save({"states": states.cpu(), "taus": taus.cpu(), "values": vals.cpu()}, a.out)
    print(f"saved {a.n} MPC labels -> {a.out}  (value range [{vals.min():.3f},{vals.max():.3f}])")

    if a.validate:
        # Compare the sampling-MPC labels against the exact HJ solution on the
        # SAME states/taus. This block never ran before: it unpacked a 3-tuple
        # from solvers that return one array, passed keyword names that do not
        # exist (x_discretization / x_grid vs xr_discretization / xr_bounds), and
        # passed an int where a list of times is required -- so the docstring's
        # "validated before use" was never true (audit P0-3).
        import numpy as np, sys
        sys.path.insert(0, "src")
        from lvfm import hj_solvers as H
        from lvfm.grids import linear_axis, periodic_axis, periodic_interpolator
        from scipy.interpolate import RegularGridInterpolator

        n_slices = 20
        tau_steps = list(np.linspace(0.0, a.T, n_slices + 1)[1:])
        # solve_*() prepends tau=0, so vf[k] is tau_steps[k-1] for k >= 1.
        tau_axis = np.concatenate([[0.0], np.asarray(tau_steps)])
        half = float(p["xy"])

        if a.system == "linear_oscillator_2d":
            vf = H.solve_linear_oscillator_2d(
                tau_steps, 201, 201, (-half, half), (-half, half),
                p["u_bound"], p["d_bound"], p["omega"], p["radius"])
            axes = [linear_axis((-half, half), 201), linear_axis((-half, half), 201)]
            vfn = np.asarray(vf)
            interps = [RegularGridInterpolator(tuple(axes), vfn[k], bounds_error=False,
                                               fill_value=None) for k in range(vfn.shape[0])]
        else:
            vf = H.solve_air3d_relative(
                tau_steps=tau_steps, xr_discretization=101, yr_discretization=101,
                theta_discretization=101, xr_bounds=(-half, half), yr_bounds=(-half, half),
                theta_bounds=(-math.pi, math.pi), vp=p["vp"], ve=p["ve"],
                u_bound=p["u_bound"], d_bound=p["d_bound"], radius=p["radius"],
                disturbance_mode="min")
            # psi is PERIODIC: its GT nodes have no endpoint at +pi (audit P1-1).
            axes = [linear_axis((-half, half), 101), linear_axis((-half, half), 101),
                    periodic_axis((-math.pi, math.pi), 101)]
            vfn = np.asarray(vf)
            interps = [periodic_interpolator(axes, vfn[k], periodic_dim=2, fill_value=None)
                       for k in range(vfn.shape[0])]

        st = states.cpu().numpy(); ta = taus.cpu().numpy(); mv = vals.cpu().numpy()
        # Batch by nearest tau slice instead of rebuilding an interpolator per point.
        slice_idx = np.abs(ta[:, None] - tau_axis[None, :]).argmin(axis=1)
        gt = np.empty(len(st))
        for k in np.unique(slice_idx):
            m = slice_idx == k
            gt[m] = interps[int(k)](st[m])

        err = np.abs(mv - gt)
        bias = float(np.mean(mv - gt))
        print(f"\n  VALIDATION vs exact HJ ({len(st)} states, {n_slices} tau slices):")
        print(f"    MAE={err.mean():.4f}  median={np.median(err):.4f}  "
              f"p90={np.quantile(err,0.9):.4f}  corr={np.corrcoef(mv,gt)[0,1]:.4f}")
        print(f"    signed bias (MPC - HJ) = {bias:+.4f}   "
              f"({'MPC over-estimates safety' if bias > 0 else 'MPC under-estimates safety'})")
        print(f"    sign agreement (safe/unsafe): {(np.sign(mv)==np.sign(gt)).mean():.3f}")
        unsafe = gt <= 0
        if unsafe.any():
            print(f"    missed-unsafe rate P(MPC>0 | HJ<=0): {float((mv[unsafe] > 0).mean()):.3f}")

if __name__ == "__main__":
    main()
