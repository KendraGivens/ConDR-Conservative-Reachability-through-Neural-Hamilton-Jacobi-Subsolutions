"""Recompute certified volumes with the obstacle-aware level.

delta_sub bounds v = max(dV/dtau - H, V - g) on a (1-eps) fraction of the domain.
The comparison-principle test function is W = V - delta*tau - delta*[obstacle branch
not exact], so the certified level is delta_sub*(T + 1) for models whose obstacle branch
can be violated (ExactBC without the sign constraint: vanilla, madr) and delta_sub*T
for the conservative construction (ours, madr_ours), where V <= g holds exactly.
delta_sub and viol_frac are unchanged; only certified_level and vol_certified move.
"""
import json, sys, numpy as np, torch
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO / "scripts"))
from certify_quantile import mc_volume, MADR
from eval_madr_models import load_madr_run
EXACT_OBSTACLE = {"ours", "madr_ours"}
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
for prefix in sys.argv[1:]:
    src = REPO / "scripts/runs/eval" / f"certify_{prefix}.json"
    d = json.load(open(src)); out_rows = []
    for row in d["rows"]:
        name = row["run"]; model_name = name.split("_", 1)[1].rsplit("_s", 1)[0]
        model, dyn, opt = load_madr_run(MADR / "runs" / name, dev)
        T = row["tmax"]; factor = T if model_name in EXACT_OBSTACLE else T + 1.0
        level = row["delta_sub"] * factor
        rng = np.random.default_rng(12345)              # same draws for every run and model
        vol = mc_volume(model, dyn, level, T, 500_000, dev, rng)
        vol_old_level = mc_volume(model, dyn, row["certified_level"], T, 500_000, dev, np.random.default_rng(12345))
        r = dict(row); r.update({"model": model_name, "obstacle_exact": model_name in EXACT_OBSTACLE,
                                 "certified_level_obs": level, "vol_certified_obs": vol,
                                 "vol_certified_oldlevel_resampled": vol_old_level})
        out_rows.append(r)
        print(f"{name:24s} {model_name:10s} dsub {row['delta_sub']:.4f} level {row['certified_level']:.4f}->{level:.4f} "
              f"vol {row['vol_certified']:.4f}->{vol:.4f} (old level resampled {vol_old_level:.4f})", flush=True)
    dst = src.with_name(f"certify_{prefix}_obs.json")
    json.dump({"config": d["config"], "note": __doc__, "rows": out_rows}, open(dst, "w"), indent=1)
    print("wrote", dst.name, flush=True)
