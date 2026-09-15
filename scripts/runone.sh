#!/usr/bin/env bash
# One arm, detached, with the environment this box needs.
#
#   bash runone.sh <tag> <config> <gpus> [overrides...]
#
# Lives on the box rather than being pasted into ssh on purpose: a `pkill -f torchrun`
# typed inline matches the ssh command line that carries it and kills the caller.
#
# OMP_NUM_THREADS: torchrun defaults it to 1, which is right for a GPU render but makes
# patch 2's host-side assembly (LoRA merge + fp8 quantise on the CPU) single-threaded --
# measured at 6+ minutes and still going. 24 = 192 cores / 8 ranks.
#
# expandable_segments: the render's own allocation pattern is stable, but the move of the
# quantised model onto the card fragments the arena otherwise.
set -uo pipefail

ROOT=${ROOT:-/opt/dlami/nvme/vdn}
REPO=$ROOT/vdn-minimax-h3
OUT=${OUT:-$ROOT/out}
CKPT=${CKPT:-ckpts/stage-dmd-step-250}
PROMPT=${PROMPT:-prompts/example_2.pt}

tag=$1 config=$2 gpus=$3; shift 3
mkdir -p "$OUT"
cd "$REPO"
# shellcheck disable=SC1091
source .venv/bin/activate

pkill -f 'infer_ulysses\.py' 2>/dev/null
sleep 4

CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((gpus - 1))) \
OMP_NUM_THREADS=$((192 / gpus)) \
MKL_NUM_THREADS=$((192 / gpus)) \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
TOKENIZERS_PARALLELISM=false \
exec torchrun --standalone --nproc_per_node="$gpus" src/inference/infer_ulysses.py \
    --config "configs/inference/$config" \
    checkpoint="$CKPT" \
    render.prompt_file="$PROMPT" \
    render.out="$OUT/$tag.mp4" \
    render.record=true \
    "$@"
