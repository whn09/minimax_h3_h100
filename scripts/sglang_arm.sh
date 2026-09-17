#!/usr/bin/env bash
# One SGLang arm: serve, or measure, at a canvas -- against the reference stack's
# 11.44 s (480p/345f) and 33.21 s (768p/345f) on the same eight cards.
#
#   bash sglang_arm.sh serve 768 [extra serve flags...]     # then, in another shell:
#   bash sglang_arm.sh bench 768 [nreq]                     # default 10 requests
#   bash sglang_arm.sh stop
#
# THE COMPARISON IS NOT SYMMETRIC, and it is asymmetric against sglang: its latency
# includes text encoding inside the request (the Qwen3-VL conditioner folded across idle
# Ulysses ranks), while every number in RESULTS.md excludes it -- the reference stack
# torch.loads an offline prompt cache. So sglang is carrying a stage the baseline does not
# run at all. If it still wins, it wins twice; if it loses by less than the encode, it is
# ahead on the part that is comparable. `text encoding` is broken out separately in the
# bench output for exactly this reason -- read that line before the totals.
#
# Everything below is transcribed from the cookbook and has NEVER been run on this box.
# `sglang serve --help` is the authority on flag names, not this file.
set -uo pipefail

ROOT=${ROOT:-/opt/dlami/nvme/sglang}
VDNROOT=${VDNROOT:-/opt/dlami/nvme/vdn}
MODEL=${MODEL:-OpenVDN/vdn-minimax-h3}
PORT=${PORT:-30010}
GPUS=${GPUS:-8}
export HF_HOME=${HF_HOME:-$VDNROOT/hf}
export SGLANG_DIFFUSION_CACHE_ROOT=${SGLANG_DIFFUSION_CACHE_ROOT:-$ROOT/cache}

mode=${1:?serve|bench|stop}
# shellcheck source=_env.sh
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

if [ "$mode" = stop ]; then
  # The [s] is not decoration. `pkill -f 'sglang.*serve'` matches any shell whose command line
  # merely mentions both words -- including `ssh box 'pkill -f sglang.*serve; bash sglang_arm.sh
  # serve 480'`, which kills the session issuing it and returns 255 with nothing started. The
  # bracketed class makes the pattern unable to match its own text. Also stops a stale
  # bench_serving, which otherwise sits polling a dead port for the rest of the day.
  pkill -f '[s]glang.*serve' ; pkill -f '[b]ench_serving'
  sleep 5; echo stopped
  exit 0
fi

edge=${2:-768}
case "$edge" in
  768) canvas=1344x768 ;;                # the paper workload; compare 33.21 s
  480) canvas=864x480  ;;                # our target;          compare 11.44 s
  *) echo "short edge must be 768 or 480 (the two canvases RESULTS.md measured)"; exit 1 ;;
esac
FRAMES=${FRAMES:-345}                    # 345 @ 24 fps = 14.375 s, the readme workload
SECONDS_OUT=$(python -c "print($FRAMES/24)")

if [ "$mode" = serve ]; then
  shift 2
  mkdir -p "$ROOT/logs"
  # --quantization fp8: on SM90 this is base H3's per-channel fp8 path (the cookbook's
  #   online mxfp8 default is SM100+ only), and it is the arm whose B200 row is 0.98
  #   s/NFE against the reference stack's 1.40 -- i.e. the H100-relevant 1.43x.
  # --attention-backend hybrid_window_attn_h3: REQUIRED. The cookbook is explicit that a
  #   dense backend silently skips the linear branch and the gates, which does not error,
  #   it just serves a different model.
  # --warmup-*: without them the first forward pays 2-3 s of allocator growth, which is
  #   precisely the request-1-vs-steady gap this study measures separately.
  # --performance-mode speed keeps components resident. 8x B200 Ulysses8 peaked at
  #   79,972 MB/GPU, which is over what this card gives PyTorch, so if load or warmup
  #   OOMs, that flag is the first thing to trade -- see RUNBOOK section 1c. It did NOT OOM:
  #   768p peaked at 62.1 GB with everything resident, so the ladder was never used.
  # NOT `exec ... | tee`: exec inside a pipeline replaces only that subshell, so when the
  # server exits the parent shell falls straight through into the bench section below and
  # starts polling a port nothing is listening on. Run the pipeline, then exit on the
  # server's status, not tee's.
  set -x
  sglang serve \
    --model-path "$MODEL" \
    --num-gpus "$GPUS" \
    --ulysses-degree "$GPUS" \
    --quantization fp8 \
    --attention-backend hybrid_window_attn_h3 \
    --encoder-parallel auto \
    --performance-mode speed \
    --warmup-num-frames "$FRAMES" \
    --warmup-resolutions "$canvas" \
    --host 127.0.0.1 --port "$PORT" \
    "$@" 2>&1 | tee "$ROOT/logs/serve_${edge}p.log"
  exit "${PIPESTATUS[0]}"
fi

# ---------------------------------------------------------------------------- bench
nreq=${3:-10}
tag=sg_${edge}p_${FRAMES}f_rep${nreq}
# NOTE on the body: `seconds` is NOT sent, even though the cookbook's example carries it.
# VideoGenerationsRequest types it as an int, and 345 frames at 24 fps is 14.375 s, so the
# server answered every request with 400 and pydantic's int_from_float -- ten instant failures
# and a benchmark that reported 0/10 in 0.01 s rather than an error. target.duration_seconds
# takes the float, and the accepted response echoes size 864x480 / seconds 14.375, so the
# fractional duration survives; it is only that one integer field that cannot express it.
# --warmup-requests 1 then nreq measured, --max-concurrency 1: post-warmup single-request
# latency, the same metric as `steady (2-10)` in the reference stack's logs. Ten of them
# because a mean of one is what the first version of this study reported and had to retract.
set -x
python3 -m sglang.multimodal_gen.benchmarks.bench_serving \
  --host 127.0.0.1 --port "$PORT" \
  --model "$MODEL" \
  --dataset vbench --task text-to-video \
  --num-prompts "$nreq" --max-concurrency 1 --warmup-requests 1 \
  --extra-body "{\"task\":\"t2va\",\"conditions\":[],\"target\":{\"short_edge\":$edge,\"aspect_ratio\":\"16:9\",\"duration_seconds\":$SECONDS_OUT},\"flow_shift\":12.0,\"audio_flow_shift\":3.0}" \
  2>&1 | tee "$ROOT/logs/$tag.log"
