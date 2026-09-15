#!/usr/bin/env bash
# The driver behind RESULTS.md: how fast can 8x H100 render a 480p clip?
#
# Clip length is 345 frames (14.375 s) everywhere except one arm, so the comparison with
# upstream's published 768p number moves ONE variable -- the canvas. 362 frames (15.083 s)
# is the smallest clip that actually reaches 15 s and gets its own arm.
#
#   setsid nohup bash h100_grid.sh > /opt/dlami/nvme/vdn/grid.log 2>&1 < /dev/null &
#   PHASES=c bash h100_grid.sh        # one phase
#
# Phases (PHASES, default "cbsg"):
#   c  CONTROL     768p / 345 frames, the shape upstream publishes (8x H200 =
#                  2.29 s/NFE), at both splits -- so the H100 gap is measured, not
#                  assumed. Needs patch 0003 as well: rank 0 is the only rank carrying
#                  decoders, and at 768p those ~11 GiB are what it does not have.
#   b  BASELINE    480p / 345 frames at the H200-tuned 6+2, plus the 362-frame
#                  (15.083 s) variant -- the literal "15 second" answer
#   s  SPLIT       480p / 345 frames sweeping parallel.softmax_ranks: 0 (standard Ulysses),
#                  4, 5, 6, 7. The 6+2 in upstream's H200 file was tuned on a 104k-row
#                  768p sequence; 480p/345f is 44k rows, so the balance point moves.
#                  Measured: it does -- standard Ulysses wins at this size.
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
  pkill -f 'infer_ulysses\.py' 2>/dev/null; sleep 3
  # OMP_NUM_THREADS: torchrun's own default is 1, which is right for the render but makes
  # patch 2's host-side assembly (LoRA merge + fp8 quantise on the CPU) single-threaded.
  # Measured on this box: 1 thread was still going after 6 minutes; 192/nproc finishes the
  # whole assembly in ~200 s. The render itself is unaffected -- it is all on the device.
  CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((gpus - 1))) \
  OMP_NUM_THREADS=$((192 / gpus)) MKL_NUM_THREADS=$((192 / gpus)) \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
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

C480=8nfe_480p_345f_ulysses_h100.yaml    # 14.375 s, upstream's clip length
C480L=8nfe_480p_362f_ulysses_h100.yaml   # 15.083 s, the literal 15-second ask
C768=8nfe_768p_345f_ulysses_h100.yaml    # the control

case $PHASES in *c*)
  echo "##### PHASE c: control, 768p / 345 frames (upstream's published shape)"
  run c_768p_345f_r6 $C768 8 parallel.softmax_ranks=6
  run c_768p_345f_r0 $C768 8 parallel.softmax_ranks=0
;; esac

case $PHASES in *b*)
  echo "##### PHASE b: baseline, 480p / 345 frames at the H200-tuned 6+2 split"
  run b_480p_345f_r6 $C480 8 parallel.softmax_ranks=6
  run b_480p_362f_r0 $C480L 8 parallel.softmax_ranks=0
;; esac

case $PHASES in *s*)
  echo "##### PHASE s: softmax_ranks sweep at 480p / 345 frames"
  for r in 0 4 5 7; do
    run "s_480p_345f_r$r" $C480 8 parallel.softmax_ranks=$r
  done
;; esac

case $PHASES in *g*)
  echo "##### PHASE g: GPU-count curve at 480p / 345 frames"
  # softmax_ranks must stay < world_size; on 1 GPU only standard Ulysses (0) is legal.
  run g_480p_345f_g1 $C480 1 parallel.softmax_ranks=0
  run g_480p_345f_g2 $C480 2 parallel.softmax_ranks=0
  run g_480p_345f_g2_r1 $C480 2 parallel.softmax_ranks=1
  run g_480p_345f_g4 $C480 4 parallel.softmax_ranks=0
  run g_480p_345f_g4_r3 $C480 4 parallel.softmax_ranks=3
;; esac

case $PHASES in *n*)
  echo "##### PHASE n: step budget at 480p / 345 frames (8 GPUs)"
  for n in 4 6; do
    run "n_480p_345f_s$n" $C480 8 render.num_steps=$n
  done
;; esac

echo "=== [$(date -u +%H:%M:%S)] grid done"
python "$REPO/../summarize.py" "$OUT" 2>/dev/null || true
