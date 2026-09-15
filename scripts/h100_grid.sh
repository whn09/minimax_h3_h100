#!/usr/bin/env bash
# The driver behind RESULTS.md: how fast can 8x H100 render a 480p 15 s clip?
#
#   setsid nohup bash h100_grid.sh > /opt/dlami/nvme/vdn/grid.log 2>&1 < /dev/null &
#   PHASES=c bash h100_grid.sh        # one phase
#
# Phases (PHASES, default "cbsg"):
#   c  CONTROL     768p / 14.4 s at softmax_ranks 6 -- the shape upstream publishes
#                  (8x H200 = 2.29 s/NFE), so the H100 gap is measured, not assumed
#   b  BASELINE    480p / 15 s at the H200-tuned softmax_ranks 6
#   s  SPLIT       480p / 15 s sweeping parallel.softmax_ranks: 0 (standard Ulysses),
#                  4, 5, 6, 7. The 6+2 in upstream's H200 file was tuned on a 104k-row
#                  768p sequence; 480p/15 s is 45k rows, so the balance point moves.
#   g  SCALE       the best 480p split at 1 / 2 / 4 / 8 GPUs -- the Ulysses efficiency
#                  curve, which is what says whether 8 GPUs is the right answer at all
#   n  STEPS       4 / 6 / 8 NFE at the best config (quality falls off below 8; this
#                  only measures what the step budget costs)
#
# Timing convention, matching upstream's table: `denoise_seconds / num_steps` is the
# s/NFE, measured AFTER render.warmup_steps=2 NFE have run in the same process, so the
# kernels are compiled and the second-timestep re-specialisation has happened. E2E
# additionally carries model load, VAE decode and mp4 mux; both are recorded.
set -uo pipefail

ROOT=${ROOT:-/opt/dlami/nvme/vdn}
REPO=$ROOT/vdn-minimax-h3
OUT=${OUT:-$ROOT/out}
CKPT=${CKPT:-ckpts/stage-dmd-step-250}
PROMPT=${PROMPT:-prompts/example_2.pt}
PHASES=${PHASES:-cbsg}

mkdir -p "$OUT"
cd "$REPO"
# shellcheck disable=SC1091
source .venv/bin/activate
export TOKENIZERS_PARALLELISM=false

# run <tag> <config> <gpus> [overrides...]
run() {
  local tag=$1 config=$2 gpus=$3; shift 3
  local log=$OUT/$tag.log
  if [ -s "$OUT/$tag.mp4.inference.json" ]; then echo "SKIP $tag (already done)"; return; fi
  echo "=== [$(date -u +%H:%M:%S)] $tag  (gpus=$gpus, $config, $*)"
  # A dead rank from a previous arm leaves the port bound; --standalone picks a free one,
  # but stale processes still hold HBM, so make each arm start from a clean box.
  pkill -f infer_ulysses.py 2>/dev/null; sleep 3
  CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((gpus - 1))) \
  timeout 3600 torchrun --standalone --nproc_per_node="$gpus" src/inference/infer_ulysses.py \
      --config "configs/inference/$config" \
      checkpoint="$CKPT" \
      render.prompt_file="$PROMPT" \
      render.out="$OUT/$tag.mp4" \
      render.record=true \
      "$@" > "$log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "    FAILED rc=$rc -- last lines:"; tail -6 "$log" | sed 's/^/    /'; return
  fi
  grep -E '^(canvas|Ulysses timing|branch-parallel|standard Ulysses)' "$log" | sed 's/^/    /'
  nvidia-smi --query-gpu=memory.used --format=csv,noheader | paste -sd' ' - | sed 's/^/    HBM after: /'
}

C480=8nfe_480p_15s_ulysses_h100.yaml
C768=8nfe_768p_144s_ulysses_h100.yaml

case $PHASES in *c*)
  echo "##### PHASE c: control, 768p / 14.4 s (upstream's published shape)"
  run c_768p_144s_r6 $C768 8
;; esac

case $PHASES in *b*)
  echo "##### PHASE b: baseline, 480p / 15 s at the H200-tuned 6+2 split"
  run b_480p_15s_r6 $C480 8
;; esac

case $PHASES in *s*)
  echo "##### PHASE s: softmax_ranks sweep at 480p / 15 s"
  for r in 0 4 5 7; do
    run "s_480p_15s_r$r" $C480 8 parallel.softmax_ranks=$r
  done
;; esac

case $PHASES in *g*)
  echo "##### PHASE g: GPU-count curve at 480p / 15 s"
  # softmax_ranks must stay < world_size; on 1 GPU only standard Ulysses (0) is legal.
  run g_480p_15s_g1 $C480 1 parallel.softmax_ranks=0
  run g_480p_15s_g2 $C480 2 parallel.softmax_ranks=0
  run g_480p_15s_g2_r1 $C480 2 parallel.softmax_ranks=1
  run g_480p_15s_g4 $C480 4 parallel.softmax_ranks=0
  run g_480p_15s_g4_r3 $C480 4 parallel.softmax_ranks=3
;; esac

case $PHASES in *n*)
  echo "##### PHASE n: step budget at 480p / 15 s (8 GPUs)"
  for n in 4 6; do
    run "n_480p_15s_s$n" $C480 8 render.num_steps=$n
  done
;; esac

echo "=== [$(date -u +%H:%M:%S)] grid done"
python "$REPO/../summarize.py" "$OUT" 2>/dev/null || true
