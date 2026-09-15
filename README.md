# ConDR: Conservative Reachability through Neural Hamilton–Jacobi Subsolutions

Code release accompanying the paper. Anonymized for double-anonymous review.

Hamilton–Jacobi (HJ) reachability characterizes the backward reachable tube
(BRT) as the zero sublevel set of a value function solving the HJ–Isaacs
variational inequality (VI). DeepReach replaces the grid with a neural value
function trained on the VI residual, but penalizes both signs of that residual
equally, so the learned value can lie *above* the true one and unsafe states
are classified as safe.

**ConDR** trains toward a *subsolution* of the HJI VI instead:

1. the obstacle branch `V ≤ g` and the terminal condition hold **by
   construction**, via `V_θ(τ,x) = g(x) − τ·softplus(NN_θ(τ,x))`;
2. a **one-sided loss** drives the residual to a safety margin `ε` below zero;
3. a **complementarity loss** keeps one branch tight, ruling out the degenerate
   subsolution that declares every state unsafe.

Training uses no ground-truth value function: only the HJI residual, the
analytic obstacle function `g`, and, in the MPC models, the baseline's
self-supervised MPC rollouts. Classical level-set solves appear only in
evaluation, and only on the two systems where they are tractable.

## Method components and where they live

| Component | Location |
|---|---|
| Conservative parameterization (`exact_cons`) | `external/patches/madr_lvfm_modifications.patch` |
| One-sided, complementarity and restoring losses | same patch, `utils/losses.py` |
| Ground-truth-free certificate | `scripts/certify_quantile.py` |
| Obstacle-aware certified level | `scripts/certify_obstacle_fix.py` |
| Closed-loop safety filter | `scripts/safety_filter_generic.py`, `scripts/safety_filter_rollout.py` |
| Ground-truth metrics | `scripts/eval_madr_models.py`, `scripts/eval_madr_air6d.py` |
| Matched-FN frontier | `scripts/frontier_madr.py` |
| Level-set ground truth, grids, helpers | `src/lvfm/` |

## The four models

All models are trained **inside the vendored baseline codebase**, sharing one
recipe per system, and differ only in the two treatment flags:

| Model | Parameterization | Residual loss | MPC labels |
|---|---|---|---|
| DeepReach | `exact` | two-sided L1 | – |
| ConDR | `exact_cons` | one-sided + complementarity + restoring | – |
| MADR | `exact` | two-sided L1 | yes |
| ConDR-MADR | `exact_cons` | one-sided + complementarity + restoring | yes |

Systems: Air3D (3D), Air6D (6D), multi-vehicle (9D), quadrotor (13D), all with
horizon `T = 1`. Air3D uses 5 seeds, every other system 3.

## Results at a glance

Air3D, 65,536 collocation points, mean over 5 seeds. FN is the fraction of truly
unsafe states classified safe; containment is the fraction of states with
`V_θ ≤ V_true`; the certified tube is the level set the ground-truth-free
certificate guarantees (exact BRT volume 0.086).

| Model | IoU ↑ | FN ↓ | Containment ↑ | Subsol. viol. ↓ | Certified vol. ↓ |
|---|---|---|---|---|---|
| DeepReach | 0.939 | 0.0022 | 0.659 | 0.25 | 0.116 |
| **ConDR** | 0.929 | **0.0009** | **0.959** | **0.01** | **0.097** |
| MADR | 0.952 | 0.0266 | 0.440 | 0.49 | 0.162 |
| ConDR-MADR | **0.962** | 0.0172 | 0.740 | 0.11 | 0.152 |

ConDR trades a little IoU, which weighs both error types equally, for an order
of magnitude fewer false negatives and a subsolution property that holds almost
everywhere. The same ordering holds on Air6D, the 9D multi-vehicle system, and
the 13D quadrotor, where no ground truth exists and the certificate is the only
containment evidence available.

As a closed-loop safety filter on Air3D from ground-truth-verified safe starts,
ConDR has no collision in 2,000 episodes against 4.6% for DeepReach and 9.0%
for MADR, at the same intervention rate.

## Setup

```bash
python -m pip install -e .            # installs the `lvfm` package and deps
```

The baselines are **not vendored as source**. Reconstruct them from the pinned
upstream commits recorded in `external/patches/`:

```bash
# upstream commit and patch provenance
cat external/patches/madr_BASE.txt
cat external/patches/deepreach_BASE.txt

# clone upstream at that commit into external/madr, then
git -C external/madr apply ../patches/madr_lvfm_modifications.patch
```

The patch adds the `exact_cons` parameterization and the ConDR loss terms to the
baseline trainer; everything else is upstream. `external/` is gitignored, so the
checkouts stay local.

## Reproducing the experiments

Training (one model, one seed; see the script header for the full flag matrix):

```bash
bash scripts/run_madr_air3d.sh          # Air3D 2x2, 3 seeds
```

The treatment flags are `--deepReach_model {exact, exact_cons}`, `--our_loss`,
`--our_loss_margin` (ε, 0.02 on Air3D/Air6D/9D and 0.085 on the quadrotor),
`--our_loss_ineq_weight 1.0`, `--our_loss_active_weight 1.2`,
`--our_loss_under_weight 0.15`, and `--not_use_MPC` for the non-MPC models.

Evaluation, in the order the paper reports it:

```bash
# ground-truth metrics: IoU, FN, FP, containment, max(V - g)
python scripts/eval_madr_models.py  --prefix a3xxh --seeds 42 43 44 45 46
python scripts/eval_madr_air6d.py --prefix a6xxh --seeds 42 43 44

# ground-truth-free certificate: violation fraction, delta_sub, certified volume
python scripts/certify_quantile.py --prefix a3xxh --seeds 42 43 44 45 46 \
    --n 200000 --eps 0.01 --beta 0.001 --gt air3d \
    --out-json results/certify_a3xxh_n5.json
python scripts/certify_obstacle_fix.py a3xxh_n5     # obstacle-aware level

# closed-loop safety filtering
python scripts/safety_filter_generic.py --prefix a3xxh --seeds 42 43 44 45 46 \
    --episodes 400 --start-mode gt3d --adversary pursuit

# matched-FN frontier
python scripts/frontier_madr.py --prefixes a3xxh --labels 65536 \
    --seeds 42 43 44 45 46
```

Figures:

```bash
python scripts/make_method_schematic.py   # method overview
python scripts/make_paper_figures.py  --which combined quad --no-suptitle
python scripts/make_paper_figures2.py --which rollouts_combined sweep shifted
```

## Outputs

Evaluation scripts write JSON and Markdown to whatever `--out-json` / `--out-md`
path you choose; the commands above use `results/`. Neither results nor model
checkpoints are included in this repository: both are large and regenerable
from the commands above.

## Tests

```bash
python -m pytest tests/ -q
```

`tests/test_audit_regressions.py` pins the findings of an internal correctness
audit: grid construction, residual scaling, MPC label domains, quaternion
sampling on S³, and the binomial tail used by the certificate.

## Notes on reproducibility

- Certificate draws are seeded and shared across models within a system, so the
  four models are compared on identical samples.
- Closed-loop start states are shared across models within a protocol.
- The quadrotor uses the released upstream recipe with an MPC label pool of
  5,000 states rather than 10,000, for GPU memory; a full-pool run reproduced
  the reported metrics.
- The 13D certificate projects the quaternion onto S³ before evaluating the
  Hamiltonian and `g`; sampling the four components in the box evaluates the
  residual off the manifold the models were trained on.

## License

Released for review. The vendored baselines remain under their upstream
licenses; see `external/patches/*_BASE.txt` for the exact upstream commits.
