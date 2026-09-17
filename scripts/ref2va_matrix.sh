#!/usr/bin/env bash
# The 3x2 attribution matrix for the customer's "融化", all at 768p, one variable per cell.
#
#   bash ref2va_matrix.sh            # ~40 min: 4 server starts, 6 renders
#
# WHY THESE SIX CELLS. The complaint is a face that collapses at t1.8 s and hands that collapse at
# t4.6 s in the two delivered clips. Three things differ between those clips and a clean render, and
# until each is varied alone none of them is attributable:
#
#            | refedge 2048 (SGLang default) | refedge 1024 (the latency patch)
#   base 50  | B50_2048 -- ground truth      | B50_1024 -- refedge alone
#   base  8  | B08_2048 -- step count alone  | B08_1024
#   LoRA  8  | L08_2048 -- the LoRA alone    | L08_1024 == what was delivered
#
# B50_2048 is what the model can do; L08_1024 is what the customer got; every other cell removes
# exactly one of the three differences. Read it as a 2x3 table, not as six numbers.
#
# WHY 768p ONLY. The 8-step ref2v checkpoint is `..._v1.0_768p_bf16.safetensors` -- 768p is its
# training resolution, so a 480p cell would confound "off-distribution" with everything else. The
# 480p question is already answered separately: at 480p the base model and the 4-step v0.1 LoRA both
# render ten correct fingers where the 8-step 768p LoRA renders pink stumps.
#
# WHY fp8 EVERYWHERE, including the base cells. fp8 is the serving configuration, so the base cells
# have to be fp8 too or the comparison smuggles in a precision change. It also buys the 8-way
# Ulysses (31 GB of DiT) instead of bf16's TP=2, which is why 50 steps at 768p is affordable at all.
#
# WHY 345 FRAMES AND SEED 42. Both delivered clips are 345 f (14.375 s) at seed 42 with the prompt in
# sglang_ref2va.py. The melt lands at t1.8 s, but the trajectory that gets there depends on the whole
# clip length, so a shorter render would not be the same moment.
#
# ONE RESTART PER refedge, NOT PER CELL. MINIMAX_H3_REFERENCE_IMAGE_SHORT_EDGE is a module constant
# patched before the workers fork, so it needs a fresh server; num_inference_steps is per request, so
# the two base cells at one refedge share a server. Hence 4 starts for 6 renders.
set -uo pipefail

VDN=${VDN:-/opt/dlami/nvme/vdn}
IMG=${IMG:-minimax-h3:local}
REF=${REF:-$VDN/ref/subject.png}
MERGED_CKPT=${MERGED_CKPT:-$VDN/ref2v_turbo_bf16}
LOGDIR=${LOGDIR:-/opt/dlami/nvme/sglang/logs}
cd "$VDN" || exit 1

# "Uvicorn running" is the readiness line; the warmup at 768p/345f is minutes, so poll generously and
# bail loudly rather than firing a request at a server that died during warmup.
up() {
  local log=$1 i
  for i in $(seq 1 240); do
    grep -q "Uvicorn running" "$log" 2>/dev/null && { echo "  ready after ${i}0s"; return 0; }
    if ! docker ps --format '{{.Names}}' | grep -q '^h3-ref2va$'; then
      echo "  CONTAINER GONE -- last lines of $log:"; tail -25 "$log"; return 1
    fi
    sleep 10
  done
  echo "  TIMEOUT waiting for $log"; tail -25 "$log"; return 1
}

# serve <logtag> ; the caller exports MERGED/REFEDGE around it. QUANT=fp8 for every cell.
serve() {
  # Two statements, not one `local tag=$1 log=...$tag...`: bash expands every argument to `local`
  # before it assigns any of them, so the log path would interpolate an unset tag (and die under -u).
  local tag=$1
  local log="$LOGDIR/serve_ref2va_768p_$tag.log"
  echo "=== serving $tag  (MERGED='${MERGED-}' REFEDGE='${REFEDGE-2048 default}')"
  rm -f "$log"
  IMAGE=$IMG LOGTAG=$tag bash docker/h3.sh serve ref2va 768 >/dev/null
  up "$log"
}

render() {                      # render <tag> <steps>
  echo "--- render $1 at $2 steps"
  IMAGE=$IMG bash docker/h3.sh exec sglang_ref2va.py "ref=$REF" "tag=$1" "768:$2:345"
}

stop() { bash docker/h3.sh stop >/dev/null 2>&1; sleep 8; }

export QUANT=fp8

# Cell order puts the two cells that could change the recommendation first: if the LoRA is clean at
# refedge 2048, that is the fix, and the base cells then only calibrate how clean "clean" is.
export MERGED=$MERGED_CKPT
unset REFEDGE          || true
serve L08_2048 && render L08_2048 8
stop

export REFEDGE=1024
serve L08_1024 && render L08_1024 8
stop

unset MERGED           || true
export REFEDGE=1024
serve B_1024 && { render B08_1024 8; render B50_1024 50; }
stop

unset REFEDGE          || true
serve B_2048 && { render B08_2048 8; render B50_2048 50; }
stop

echo "=== matrix done"
ls -la "$VDN"/pull/ref2va/ref2va_{L08,B08,B50}_*768p*/ 2>/dev/null
