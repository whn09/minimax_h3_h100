#!/usr/bin/env bash
# The customer's two cases on ONE H100 (p5.4xlarge), base MiniMax-H3, fp8 and bf16.
#
#   bash case_p54x.sh                      # 4 server starts, 4 renders
#   EDGE=480 bash case_p54x.sh             # same grid at 480p
#
# WHAT THIS RUNS AND WHY IT IS FOUR STARTS, NOT ONE. Two cases and two precisions is a 2x2, and
# every one of the four cells needs its own server process:
#   * t2va and ref2va are different WEIGHT PARTITIONS selected by --model-variant, so they cannot
#     share a process (and on one card they cannot even overlap in time: bf16 DiT alone is 61.73 GB).
#   * --quantization is a server argument too.
# Only num_inference_steps, resolution, frame count and the prompt are per-request. Hence:
#   t2va fp8 -> ref2va fp8 -> t2va bf16 -> ref2va bf16
# BOTH fp8 cells first, not fp8/bf16 per task: fp8 is the configuration that is known to fit on one
# card, so running both of them first proves the case file, the reference image, the client and the
# 121-frame length end to end before the precision that needs offload -- and the one that can still
# OOM -- gets a turn. If bf16 turns out not to fit at all, the deliverable is still two of four.
#
# NO LoRA ANYWHERE. The request was 原版 -- the base model. The 8-step ref2v Turbo LoRA is what
# REF2VA.md's correction section pins the customer's melting on, so it stays out of this grid, and
# with it goes the 8-step schedule: 25 steps on base weights is a schedule the base model can
# actually denoise on (its own default is 50), which is the point of asking for 25.
#
# THE TEXT ENCODER IS THE SINGLE-GPU PROBLEM, NOT THE PRECISION. Measured, on the first attempt at
# this grid, at fp8 -- the cell that "obviously fits":
#     Loaded text_encoder: MiniMaxH3Qwen3VLEncoder. model size: 48.09 GB. avail GPU mem: 29.91 GB
#     Loading transformer ... torch.OutOfMemoryError: Tried to allocate 498.00 MiB
# Qwen3-VL is 48.09 GB in bf16 and `--encoder-parallel auto` had been HIDING it: on eight cards it
# shards to 7.40 GB per rank, which is the number every log in this repo records and the reason the
# encoder never looked like a memory item. On one card there is nothing to shard onto, so 48.09 GB
# is resident and the fp8 DiT's 31 GB does not fit in the 29.91 GB left. bf16 would have failed for
# the same reason 30 GB earlier. So this is not "bf16 needs help": both precisions do, and the
# offload flags below go on ALL FOUR cells -- which is also what keeps the precision comparison
# clean, since the memory policy is then identical across it.
#
# AND MOVING THE ENCODER OFF THE CARD IS STILL NOT ENOUGH, because the second wall is ACTIVATIONS,
# not weights. With the encoder host-resident and the fp8 DiT loaded (30.93 GB, 46.01 GB free) the
# server came up and then died inside its own warmup:
#     server warmup req (1344x768x124f, 2/50 steps) processing failed:
#     CUDA out of memory. Tried to allocate 250.00 MiB ... 77.63 GiB is allocated by PyTorch
# 77.63 - 30.93 = ~47 GB of activations for one 121-frame 768p sequence, and that is the number that
# makes a single card the wrong shape for this job rather than merely a tight one. 1344x768 is
# 84x48 = 4032 latent tokens per latent frame and 121 frames compress to ~31 of them, so the
# denoiser attends over ~125k tokens. On eight cards --ulysses-degree 8 splits that sequence eight
# ways, which is why every 768p run in this repo fits without anything special; on one card there is
# no split, and 46 GB of headroom is not enough for the unsplit sequence.
# So the DiT has to come off the card too -- LAYERWISE, one layer at a time, which is what
# --performance-mode memory selects for H3 (its pipeline config declares
# dit_layerwise_offload_modes=("auto", "memory")). That frees ~31 GB (fp8) or ~62 GB (bf16) of
# resident weights for activations and costs one pass of the DiT's weights over PCIe per STEP, i.e.
# 25 times per request, prefetch-overlapped with compute. It is the slow configuration on purpose:
# the point of these four renders is the customer's two clips at two precisions, not a latency
# number, and any latency measured here says more about PCIe than about H3. The 8-card numbers in
# RESULTS.md remain the latency record.
#
# 121 FRAMES, NOT 120. "5 秒" has to be a legal length: the server validates
# duration_seconds in [4, 15] and the model wants frames = 1 mod 8, so 5 s is 121 f = 5.04 s.
# FRAMES is also exported so --warmup-num-frames matches the request; warming up at the default 345
# would allocate for a 14 s clip and can be the difference between fitting and not.
set -uo pipefail

VDN=${VDN:-/opt/dlami/nvme/vdn}
IMG=${IMG:-lmsysorg/sglang:dev}
CASE=${CASE:-$VDN/case.txt}
EDGE=${EDGE:-768}
STEPS=${STEPS:-25}
LOGDIR=${LOGDIR:-/opt/dlami/nvme/sglang/logs}
export FRAMES=${FRAMES:-121}
export GPUS=1                    # p5.4xlarge: one H100. GPUS = TP * ULYSSES, so both are 1.
export TP=1                      # explicit: sglang_ref2va_arm.sh defaults bf16 to TP=2
export ULYSSES=1
# Every cell, both precisions: see the header. `memory` mode already implies the encoder and VAE
# offload, but they are spelled out so the log's server_args line answers "was this on?" without
# having to reason about what the auto-tuner decided.
OFFLOAD=(--performance-mode memory --text-encoder-cpu-offload true --vae-cpu-offload true)
# Last resort if layerwise still will not fit: also evict the DiT wholesale between stages, which
# buys back whatever the layerwise resident window keeps on the card during the VAE decode. Quality
# is untouched either way -- both are placement, not numerics -- so unlike the refedge lever in
# REF2VA.md this fallback costs only time, and the run reports when it fires.
FALLBACK_FLAGS=("${OFFLOAD[@]}" --dit-cpu-offload true)

cd "$VDN" || exit 1

# `up <container> <log>`: wait for readiness, and bail loudly the moment it cannot come.
#
# THE READINESS LINE IS *NOT* "Uvicorn running", and using it cost a whole grid run. On this nightly
# http_server.py runs the warmup in _run_server_warmup_after_http_live -- i.e. Uvicorn announces the
# port, answers GET /liveness with 200, and only *then* warms up. So "Uvicorn running" means "the
# socket is open", the client fires a real request into a server that is mid-warmup, and when the
# warmup OOMs the server shuts down while the client sits on its 60 s socket timeout. The line that
# actually means ready is server_warmup.py:389.
#
# AND WARMUP FAILURE HAS TO BE MATCHED EXPLICITLY, because it does not always kill the container
# fast enough for the `docker ps` check below to notice on the same pass. There is also nothing to
# wait for after it: SGLang's own OOM-degrade path is broken in this build --
#     server_warmup.py:145 _degrade_after_oom -> warmup_request_builder.py:337
#     ValueError: field num_frames is declared with init=False, it cannot be specified with replace()
# so a warmup OOM cannot retry at fewer frames the way upstream intends; it aborts startup.
up() {
  local name=$1 log=$2 i
  for i in $(seq 1 180); do
    grep -q "fired up and ready to roll" "$log" 2>/dev/null && { echo "  ready after ${i}0s"; return 0; }
    if grep -q "warmup failed\|Server warmup failed" "$log" 2>/dev/null; then
      echo "  WARMUP FAILED -- tail of $log:"; tail -30 "$log"; return 1
    fi
    if ! docker ps --format '{{.Names}}' | grep -q "^${name}$"; then
      echo "  CONTAINER GONE -- tail of $log:"; tail -30 "$log"; return 1
    fi
    sleep 10
  done
  echo "  TIMEOUT waiting for $log"; tail -30 "$log"; return 1
}

stop() { bash docker/h3.sh stop >/dev/null 2>&1; sleep 8; }

# cell <arm> <tag> <quant> [extra sglang flags...]
#   arm  = t2va|ref2va          (which arm script, i.e. which --model-variant and which port)
#   quant= fp8 | "" for bf16    (empty is meaningful; h3.sh forwards set-but-empty deliberately)
cell() {
  local arm=$1 tag=$2 quant=$3; shift 3
  local name="h3-$arm"
  # CELLS="ref2va_FP8 t2va_BF16" reruns a subset. Each cell is a ~6-minute server start plus a
  # ~2-minute render, so after a partial failure re-running the whole grid throws away 25 minutes
  # of work that is already on disk -- and the mp4s are named per cell, so nothing is ambiguous.
  if [ -n "${CELLS-}" ] && [[ " $CELLS " != *" ${arm}_${tag} "* ]]; then
    echo "=== skip ${arm}_${tag} (not in CELLS)"; return 0
  fi
  local base=serve_base; [ "$arm" = ref2va ] && base=serve_ref2va
  local log="$LOGDIR/${base}_${EDGE}p_$tag.log"
  echo "=== $tag: $arm, quant='${quant:-bf16}', ${EDGE}p, $FRAMES f, $STEPS steps"
  rm -f "$log"
  QUANT="$quant" LOGTAG="$tag" IMAGE=$IMG bash docker/h3.sh serve "$arm" "$EDGE" "$@" >/dev/null
  if ! up "$name" "$log"; then
    # FALLBACK is set by the caller, not passed as a command prefix: `ARR=(x) somefunc` is not a
    # valid bash prefix assignment (arrays cannot be one), and writing it that way makes bash treat
    # the whole thing as a command name lookup and fail with a syntax error at parse time.
    if [ -n "${FALLBACK+x}" ]; then
      echo "  retrying $tag with ${FALLBACK[*]}"
      rm -f "$log"
      QUANT="$quant" LOGTAG="$tag" IMAGE=$IMG bash docker/h3.sh serve "$arm" "$EDGE" \
        "${FALLBACK[@]}" >/dev/null
      up "$name" "$log" || { stop; return 1; }
    else
      stop; return 1
    fi
  fi
  IMAGE=$IMG bash docker/h3.sh exec sglang_case.py "case=$CASE" "task=$arm" "tag=$tag" \
    "$EDGE:$STEPS:$FRAMES"
  stop
}

stop
FALLBACK=("${FALLBACK_FLAGS[@]}")
cell t2va   FP8  fp8 "${OFFLOAD[@]}"
cell ref2va FP8  fp8 "${OFFLOAD[@]}"
cell t2va   BF16 ""  "${OFFLOAD[@]}"
cell ref2va BF16 ""  "${OFFLOAD[@]}"

echo "=== case grid done"
ls -la "$VDN"/pull/case/*/ 2>/dev/null
