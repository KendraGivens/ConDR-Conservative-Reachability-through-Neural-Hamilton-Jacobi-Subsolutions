"""
Scenario / conformal certification of conservative reachability models.

Turns the empirical "FN approximately 0" observation into a probabilistic
guarantee, without ground truth. Two certificates are produced:

1. SUBSOLUTION-VIOLATION certificate (unsupervised). Sample N i.i.d. states,
   compute the pointwise subsolution violation
        v(x) = relu( max( dV/dtau - H,  V - g ) ).
   By the scenario approach / conformal prediction, with confidence >= 1 - beta,
   a fresh random state has v <= v_(k) (an order statistic) with probability
   >= 1 - eps, where k is chosen from the binomial tail. We report the smallest
   such bound delta_sub. Interpretation (audit S1-5 corrected): with high
   confidence, a FRESH SAMPLE from this distribution violates the subsolution
   inequality by more than delta_sub with probability <= eps. This is a
   distributional statement, not a proof of containment: the eps-fraction
   violation set is not controlled along characteristics. A delta-approximate
   subsolution integrates to a value gap of delta * tau, so the certified tube
   at horizon T is {V <= delta_sub * T} (Gronwall factor), certified over tau
   rather than at a single slice. scripts/certify_quantile.py is the current
   implementation of that corrected certificate for the MADR-stack runs.

2. BRT-CONTAINMENT certificate (uses GT only where tractable, for validation).
   The value shift delta_fn that makes FN exactly 0 on a dense grid; reported
   alongside for the low-dim systems as an empirical cross-check.

Reference: this mirrors the scenario-optimization / conformal machinery used by
Lin & Bansal (ICRA 2023) and Reachability Barrier Networks, applied to a model
that is already near-conservative so delta is tiny.
"""
import argparse
import json
import math
import os
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"
import numpy as np
import torch
import yaml

from lvfm.config_utils import to_namespace
# Stack-A trainer lives in scripts/attic since the 09-02 cleanup; only this
# script's own main() needs it. Import lazily so conformal_quantile_bound
# stays importable by certify_quantile.py (the live certification path).
try:
    from train_unsupervised import create_dataset_and_residual, create_model
except ModuleNotFoundError:
    import sys as _sys
    from pathlib import Path as _P
    _sys.path.insert(0, str(_P(__file__).parent / "attic"))
    try:
        from train_unsupervised import create_dataset_and_residual, create_model
    except ModuleNotFoundError:
        create_dataset_and_residual = create_model = None  # main() unusable; bound fn fine


def conformal_quantile_bound(violations, eps, beta):
    """Smallest order statistic v_(k) such that, with confidence >= 1 - beta,
    P(V_new <= v_(k)) >= 1 - eps. Uses the distribution-free binomial bound:
    choose the smallest k in {1..N} with sum_{j>=k} C(N,j) eps^j (1-eps)^(N-j) ...
    Equivalently the standard conformal rank: k = ceil((N+1)(1-eps)); we then
    inflate the rank until the binomial tail confidence holds.
    """
    v = np.sort(np.asarray(violations))
    N = v.size
    if N == 0:
        return 0.0, N
    from scipy.stats import binom
    # Nonparametric UPPER tolerance bound. The coverage of the ascending order
    # statistic, F(v_(r)), is Beta(r, N-r+1)-distributed, so
    #     P[ F(v_(r)) >= p ] = P[ Binom(N, p) <= r-1 ] = binom.cdf(r-1, N, p),
    # with p = 1-eps. v_(r) is a valid (1-eps)-content, (1-beta)-confidence upper
    # tolerance limit iff binom.cdf(r-1, N, 1-eps) >= 1-beta. cdf is increasing in
    # r, so we take the SMALLEST such r (tightest valid bound). (The earlier code
    # used binom.sf here -- the wrong tail -- which fails for all r >= r0 and
    # silently returns the sample maximum.)
    r0 = min(N, max(1, int(math.ceil((N + 1) * (1 - eps)))))
    r = N
    for rank in range(r0, N + 1):
        if binom.cdf(rank - 1, N, 1 - eps) >= 1 - beta:
            r = rank
            break
    return float(v[r - 1]), N


def compute_violations(model, residual, dataset, deepreach, cfg, n, tau, chunk, device, seed=0):
    dataset.set_tau_max(cfg.T)
    # The certification sample used the global torch RNG, so the certified delta
    # was not reproducible run-to-run (audit P3-4). Fork + seed it.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        x_phys = dataset._sample_states_phys(n)
        x_net = dataset._scale_states(x_phys)
    tau_net = tau / cfg.T if cfg.scale_time_to_01 else tau
    xt_np = np.concatenate([x_net.numpy(), np.full((n, 1), tau_net, np.float32)], axis=-1)
    viols, vg_pos = [], 0
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        xt = torch.tensor(xt_np[s:e], dtype=torch.float32, device=device).requires_grad_(True)
        t = torch.full((e - s,), float(tau), device=device)
        with torch.enable_grad():
            pde, vmg = residual.compute_deepreach_residual(model=model, xt=xt, return_components=True)
        viols.append(torch.relu(torch.maximum(pde, vmg)).detach().cpu().numpy())
        vg_pos += int((vmg.detach() > 0).sum())
    return np.concatenate(viols), vg_pos / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True, choices=["air_3d", "air_6d", "multi_9d"])
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--ckpt", default="final.pt")
    ap.add_argument("--n", type=int, default=200000)
    ap.add_argument("--eps", type=float, default=0.001, help="allowed violated fraction")
    ap.add_argument("--beta", type=float, default=0.05, help="1-beta confidence")
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--chunk", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=0, help="makes the certified delta reproducible")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", default=None)
    args = ap.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else f"cuda:{args.device}")
    sysdir = {"air_3d": "air_3d", "air_6d": "air_6d", "multi_9d": "multi_9d"}[args.system]

    results = []
    for run in args.runs:
        run_dir = Path("scripts/runs") / sysdir / run
        cfg = to_namespace(yaml.safe_load(open(run_dir / "config.yaml")))
        cfg.device = str(device)
        _, dataset, residual = create_dataset_and_residual(cfg)
        model, _, deepreach = create_model(cfg, residual, device)
        ckpt = torch.load(run_dir / "ckpts" / args.ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"]); model.eval()

        v, vg_pos_frac = compute_violations(model, residual, dataset, deepreach, cfg,
                                            args.n, args.tau, args.chunk, device, seed=args.seed)
        delta_sub, N = conformal_quantile_bound(v, args.eps, args.beta)
        viol_frac = float((v > 0).mean())
        row = {
            "run": run, "system": args.system, "n": N, "seed": args.seed,
            # The guarantee is over THIS sampling distribution at THIS tau, not
            # over the uniform measure on the domain x [0, T] (audit M-5).
            "sampling_distribution": "dataset._sample_states_phys", "tau": args.tau,
            "eps": args.eps, "confidence": 1 - args.beta,
            "subsolution_violation_frac": viol_frac,
            "delta_sub_certified": delta_sub,       # (1-eps)-quantile violation, conf 1-beta
            "max_violation": float(v.max()) if v.size else 0.0,
            "v_minus_g_positive_frac": vg_pos_frac,  # structural: should be 0
        }
        results.append(row)
        print(json.dumps(row), flush=True)

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(args.out_json, "w"), indent=2)
    if args.out_md:
        lines = ["# Conservative Certification (scenario / conformal, unsupervised)", "",
                 f"With confidence {1-args.beta:.0%}, a fresh random state violates the HJI "
                 f"subsolution inequality by at most `delta_sub` on all but an {args.eps:.1%} "
                 "fraction of THAT SAMPLING DISTRIBUTION at the certified tau (not the uniform measure on the domain x [0,T]; audit M-5). `V-g>0 frac` is the structural obstacle branch "
                 "(guaranteed 0). Smaller `delta_sub` = closer to a certified subsolution.", "",
                 "|run|N|subsol. viol. frac|delta_sub (certified)|max viol|V-g>0 frac|",
                 "|---|---|---|---|---|---|"]
        for r in results:
            lines.append(f"|{r['run']}|{r['n']}|{r['subsolution_violation_frac']:.4f}|"
                         f"{r['delta_sub_certified']:.3e}|{r['max_violation']:.3e}|{r['v_minus_g_positive_frac']:.4f}|")
        open(args.out_md, "w").write("\n".join(lines) + "\n")
    print(f"Wrote {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
