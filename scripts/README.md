# scripts/

Code to reproduce the experiments. See the top-level `README.md` for the full
sequence and the exact commands.

| Script | Purpose |
|---|---|
| `run_madr_air3d.sh` | Train the four models (the flag matrix is in the header) |
| `gen_mpc_labels.py` | MPC label generation for the MADR models |
| `eval_madr_models.py` | Ground-truth metrics: IoU, FN, FP, containment, max(V − g) |
| `eval_madr_air6d.py` | The same metrics on Air6D via the relative-coordinate lift |
| `certify_quantile.py` | Ground-truth-free certificate (violation fraction, δ_sub) |
| `certify_conservative.py` | Distribution-free order-statistic bound used above |
| `certify_obstacle_fix.py` | Certified volume at the obstacle-aware level |
| `safety_filter_generic.py` | Closed-loop filtering, all systems and start rules |
| `safety_filter_rollout.py` | Closed-loop filtering, Air3D verified-safe starts |
| `frontier_madr.py` | Volume at matched false-negative rate |
| `make_method_schematic.py` | Method overview figure |
| `make_paper_figures.py` | BRT classification maps, quadrotor rollouts |
| `make_paper_figures2.py` | Rollouts, margin sweep, shifted threshold |
