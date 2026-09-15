#!/usr/bin/env bash
# Air3D 2x2 IN THE MADR CODEBASE (external/madr) -- the real published trainer.
#
#   model            not_use_MPC  deepReach_model  our_loss
#   vanilla            True         exact          False     (vanilla DeepReach)
#   ours               True         exact_cons     True      (our loss only)
#   madr               False        exact          False     (MADR)
#   madr_ours          False        exact_cons     True      (MADR + our loss)
#
# SYSTEM: the canonical DeepReach Air3D instance (collisionR 0.25, velocity 0.75,
#   omega_max 3.0, angle_alpha_factor 1.2, domain +/-1). external/madr shipped a
#   DIFFERENT instance (v=0.6, omega_max=2.0, state_max=1.5, no alpha factor);
#   dynamics.py has been aligned so this is the same system our models use.
# SETUP: the ARCHIVED RECIPE, read from
#   external/deepreach/runs/ctrl_air3d_vanilla_b4096/orig_opt.pickle -- the config
#   that produced the smooth archived BRTs (roughness 0.00342, 1 component):
#     num_nl 512, num_hl 3, lr 2e-5 CONSTANT, clip_grad 0.0 (OFF),
#     counter_end 10000 (time curriculum), pretrain 2000, numpoints 4096,
#     num_src_samples 1024, w_ineq 1.0, w_act 1.2.
#   Capacity is 791,041 params, EXACTLY matching that archived run.
#   The archived vanilla and conservative runs differ ONLY in deepreach_model,
#   so this recipe is already known to work for BOTH models of factor A.
# (superseded) previously matched to scripts/configs/models/air3d_*.yaml -- the setup our
#   own models were trained with, so the two studies are directly comparable:
#     340x2 SIREN, input_dim 4 (raw heading / (1.2*pi), NO sin-cos lift)
#       -> 233,921 parameters, IDENTICAL to our models
#     numpoints 4096, num_epochs 15000, clip_grad 10.0
#     lr 1e-4 with warmup 500 + cosine decay to 1% -- byte-for-byte the same
#       rule as train_unsupervised.py::make_lr_schedule
#     no pretraining, no time curriculum (our models use neither)
# Every other flag is identical across models, so only the two factors vary.
# 3 seeds each.
set -u
cd "$(dirname "$0")/../external/madr"
LOG=../../scripts/runs/logs
# Collocation budget and run-name prefix are overridable so the same launcher can
# produce the budget-scaling set:  NUMPOINTS=16384 PREFIX=a3hi bash <this>
NUMPOINTS="${NUMPOINTS:-4096}"
PREFIX="${PREFIX:-a3}"
# DYN/NUM_NL let the same launcher drive Air3D and the 6D lift.
DYN="${DYN:-Air3D}"
NUM_NL="${NUM_NL:-512}"
# Concurrency: high collocation budgets need fewer simultaneous runs or
# they contend for GPU memory.
CONC="${CONC:-4}"
# GPINN: gradient-enhanced PDE weight. Applied to ALL FOUR models (it is a
# generic PINN regulariser, not part of our treatment), so the 2x2 stays
# balanced and the gPINN factor is separable from the loss factor.
GPINN="${GPINN:-}"
# SEEDS: subset of "42 43 44". The 65536 budget costs ~2.4 h/run, so it is probed
# at one seed first and backfilled only if the models come out close.
SEEDS="${SEEDS:-42 43 44}"
# EPOCHS / CURRICULUM / TMAX / DYN_ARGS let the SAME launcher drive every system,
# so the four models are guaranteed to differ only in arm_flags() no matter which
# system is run. Per-system recipes live in scripts/queue_highdim.sh.
EPOCHS="${EPOCHS:-15000}"
CURRICULUM="${CURRICULUM:-10000}"
TMAX="${TMAX:-1.0}"
DYN_ARGS="${DYN_ARGS:-}"
# SET_MODE: pass empty for systems whose constructor has no set_mode argument
# (LessLinearND), where argparse would reject the flag outright.
SET_MODE="${SET_MODE-avoid}"
# CLIP: gradient clipping. 0.0 (MADR default, and the archived Air3D recipe) is
# fine for the 25x-value-scale systems, but the Quadrotor has a 106x scale and
# every model diverged to NaN within ~2000 iters without clipping. Applied to ALL
# FOUR MODELS of a system, so it never becomes an asymmetry.
CLIP="${CLIP:-0.0}"
# MPC_BATCH/MPC_NBATCH/MPC_NDATA/MPC_REFINE: MPC-supervision scale. Defaults
# (1000/3/500/10) are MADR's PUBLISHED low-dimensional command (Dubins3D in
# upstream run_experiment.sh); their published Quadrotor uses num_MPC_batches 20
# with the repo defaults for the rest (10000/5000/20) -- pass overrides there.
# SRC: num_src_samples. MADR default 3000; the archived Air3D recipe used 1024.
SRC="${SRC:-1024}"
# PRE: pretrain iters. MADR default/published 1000; archived Air3D used 2000.
PRE="${PRE:-2000}"
# EXTRA overrides the schedule/architecture block (set by the smoothness rerun).
EXTRA="${EXTRA:-}"
mkdir -p $LOG
say(){ echo; echo "=== $(date +%H:%M:%S)  $* ==="; }

COMMON="--dynamics_class $DYN ${SET_MODE:+--set_mode $SET_MODE} --experiment_class DeepReach \
--minWith target --tMax $TMAX --num_epochs $EPOCHS --numpoints $NUMPOINTS \
$DYN_ARGS \
--num_src_samples $SRC --model sine --model_mode mlp \
--dirichlet_loss_divisor 1.0 --epochs_til_ckpt 5000 --adj_rel_grads True \
--MPC_dt 0.02 --MPC_loss_type l1 --MPC_mode MPC --MPC_style direct \
--MPC_sample_mode gaussian --MPC_integration_method euler \
--MPC_decay_scheme exponential --MPC_lambda_ 0.1 --MPC_finetune_lambda 100.0 \
--MPC_importance_init 1.0 --MPC_importance_final 1.0 \
--MPC_batch_size ${MPC_BATCH:-1000} --num_MPC_batches ${MPC_NBATCH:-3} \
--num_MPC_data_samples ${MPC_NDATA:-500} \
--num_MPC_perturbation_samples 100 --num_iterative_refinement ${MPC_REFINE:-10} \
--time_till_refinement 0.2 --MPC_nbr_action_repeats 1 --MPC_receding_horizon -1 \
--our_loss_ineq_weight 1.0 --our_loss_active_weight 1.2 --not_pretrain_MPC \
${GPINN:+--gpinn_weight $GPINN} \
${EXTRA:---num_nl $NUM_NL --num_hl 3 --lr 2e-05 --clip_grad $CLIP --counter_end $CURRICULUM --pretrain --pretrain_iters $PRE}"

# --not_pretrain_MPC is required and passed to EVERY model (inert for the
# not_use_MPC models, so it cannot become an asymmetry). Without it the MPC data
# generator runs during pre-training, before experiments.py attaches
# dataset.policy = self.model, and dies with "'NoneType' object is not callable"
# in MPC.rollout_dynamics. That is almost certainly why the archived
# air3d_madr_* runs contain a config and zero checkpoints.

# OURS_FLAGS: extra flags for BOTH `ours` models only (they are the treatment, so
# this cannot unbalance the 2x2). The frozen 2026-09-01 method config is
#   --our_loss_norm l1 --cons_shift -2 --our_loss_margin 0.02 \
#   --our_loss_margin_consistent --our_loss_under_weight 0.15
# selected on Air3D@4096 seed 42 and confirmed on seeds 43/44 (round-2 sweep,
# scripts/runs/eval/cheap_round2.txt). Empty default keeps old behavior.
OURS_FLAGS="${OURS_FLAGS:-}"
arm_flags(){
  case "$1" in
    vanilla)   echo "--not_use_MPC --deepReach_model exact" ;;
    ours)      echo "--not_use_MPC --deepReach_model exact_cons --our_loss $OURS_FLAGS" ;;
    madr)      echo "--deepReach_model exact" ;;
    madr_ours) echo "--deepReach_model exact_cons --our_loss $OURS_FLAGS" ;;
  esac
}

say "$DYN 2x2 in the MADR codebase: 4 model_names x 3 seeds  (numpoints=$NUMPOINTS, num_nl=$NUM_NL, prefix=$PREFIX, conc=$CONC)"
i=0
for seed in ${SEEDS:-42 43 44}; do
  for model_name in vanilla ours madr madr_ours; do
    name="${PREFIX}_${model_name}_s${seed}"
    if [ -d "runs/$name/training/checkpoints" ] && \
       [ -f "runs/$name/training/checkpoints/model_final.pth" ]; then
      echo "  skip $name (already finished)"; continue
    fi
    # run_experiment.py VERSIONS the output dir when one already exists (writes
    # $name_v1, _v2, ...). So an interrupted run leaves a partial dir, the retry
    # lands under a different name, this skip-check never sees model_final.pth
    # under $name, and the queue retrains the same model forever. Park any partial
    # dir first so the retry reuses the canonical name. Never delete: the partial
    # holds real intermediate checkpoints.
    if [ -d "runs/$name" ]; then
      mkdir -p runs/interrupted
      mv "runs/$name" "runs/interrupted/${name}_partial_$(date +%s)"
      echo "  parked partial runs/$name before retry"
    fi
    # SLOT POOL. The old code used a barrier `wait` after every CONC launches, so
    # when the fast non-MPC models finished their GPU idled until the slow MPC model
    # cleared -- about half a GPU per wave. A plain `wait -n` pool fixes the
    # stalling but not the placement: GPU index was i%2, so a freed GPU0 could be
    # handed a job pinned to GPU1. Track pid-per-slot and reuse the slot that
    # actually became free.
    slot=""
    while [ -z "$slot" ]; do
      for k in $(seq 0 $((CONC-1))); do
        pid=$(eval echo \${SLOTPID$k:-})   # :- so an unassigned slot is empty, not a set -u abort
        if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then slot=$k; break; fi
      done
      [ -z "$slot" ] && wait -n 2>/dev/null || true
    done
    # MPC models allocate a large memory-mapped MPC buffer at startup. Three baseline
    # runs were killed mid-startup when two overlapped (13 GB free, swap full), each
    # log ending exactly at `torch.from_numpy(MPC_inputs_mmap)`. Wait for headroom
    # rather than launching into an OOM.
    case "$model_name" in
      madr|madr_ours)
        for _try in $(seq 1 60); do
          _free=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
          [ "${_free:-0}" -ge "${MIN_FREE_GB:-20}" ] && break
          echo "  [mem] ${_free}GB available < ${MIN_FREE_GB:-20}GB, waiting before $name"
          sleep 30
        done ;;
    esac
    gpu=$((slot % 2)); i=$((i+1))
    echo "[launch] $name on GPU$gpu slot$slot  ($(arm_flags $model_name))"
    CUDA_VISIBLE_DEVICES=$gpu python run_experiment.py --mode train \
      --experiment_name "$name" --experiments_dir ./runs --seed $seed \
      $COMMON $(arm_flags $model_name) > $LOG/madr_$name.log 2>&1 &
    eval "SLOTPID$slot=$!"
  done
done
wait
say "health check -- a killed run leaves no model_final, a diverged one leaves NaN"
_bad=0
for seed in ${SEEDS:-42 43 44}; do
  for model_name in vanilla ours madr madr_ours; do
    name="${PREFIX}_${model_name}_s${seed}"
    ck="runs/$name/training/checkpoints"
    if [ ! -f "$ck/model_final.pth" ]; then
      echo "  FAILED  $name  (no model_final.pth -- killed or never ran)"; _bad=$((_bad+1)); continue
    fi
    if python3 - "$ck" <<'PYEOF'
import sys, glob, re, numpy as np
d=sys.argv[1]
fs=sorted(glob.glob(d+"/train_losses_epoch_*.txt"), key=lambda p:int(re.search(r"(\d+)\.txt",p).group(1)))
sys.exit(1 if (fs and not np.isfinite(np.loadtxt(fs[-1])).all()) else 0)
PYEOF
    then :; else echo "  FAILED  $name  (non-finite loss -- diverged)"; _bad=$((_bad+1)); fi
  done
done
if [ "$_bad" -gt 0 ]; then
  say "!! $_bad RUN(S) FAILED in $PREFIX -- re-run this set before evaluating it"
else
  say "all runs in $PREFIX healthy"
fi
ls -d runs/${PREFIX}_*/ 2>/dev/null
