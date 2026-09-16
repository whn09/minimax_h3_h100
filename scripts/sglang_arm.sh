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
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"
# This box has no /usr/local/cuda and deep_gemm's find_cuda_home asserts rather than falling
# back, which kills `import sglang.multimodal_gen` before any of the above matters. The venv
# carries a full pip CUDA (the VDN delta-factors kernel JITs against nvcc through
# apache-tvm-ffi), so point at that -- and find it by looking for bin/nvcc rather than by
# guessing the directory, because it is nvidia/cu13/, not the per-component nvidia/cuda_nvcc/
# layout the older wheels used, and a CUDA_HOME that merely exists satisfies deep_gemm's
# assert while leaving the JIT to fail later with something far less obvious.
export CUDA_HOME=${CUDA_HOME:-$(python - <<'PY'
import pathlib, sysconfig
nv = pathlib.Path(sysconfig.get_paths()["purelib"]) / "nvidia"
print(next((str(p.parent.parent) for p in sorted(nv.glob("*/bin/nvcc"))), ""))
PY
)}
[ -x "$CUDA_HOME/bin/nvcc" ] || echo "warning: no nvcc under CUDA_HOME=$CUDA_HOME; the VDN delta-factors JIT will fail"

# H3's pipeline hard-requires both binaries and raises before it touches a GPU -- every rank
# dies with "missing executables: ffmpeg, ffprobe" and the parent shows only an EOFError from
# the pipe, which reads like a crash rather than a missing package. Checked here so the message
# is the message. `apt-get install ffmpeg`; on this box that first needed the broken
# developer.download.nvidia.com cuda-ubuntu2604 source moved out of sources.list.d.
for b in ffmpeg ffprobe; do
  command -v "$b" >/dev/null 2>&1 || { echo "missing $b -- sglang H3 refuses to start without it"; exit 2; }
done

# The JIT link line is `c++ ... -L$CUDA_HOME/lib64 -lcudart`, which assumes a system CUDA
# install. The pip wheel ships neither: the directory is lib/, not lib64/, and it carries
# only the runtime soname libcudart.so.13, not the libcudart.so a -l flag resolves. nvcc
# compiles fine and then ld says "cannot find -lcudart", every rank dies, and the parent
# reports EOFError. Two symlinks inside the venv close it -- cheaper than a system CUDA
# install, and scoped to this venv so the reference stack is untouched.
if [ -d "$CUDA_HOME/lib" ]; then
  [ -e "$CUDA_HOME/lib64" ] || ln -sfn lib "$CUDA_HOME/lib64"
  for so in "$CUDA_HOME"/lib/lib*.so.[0-9]*; do
    base=${so%%.so.*}.so
    [ -e "$base" ] || ln -sfn "$(basename "$so")" "$base"
  done
fi

# NCCL_NET_PLUGIN=none, on an AWS box specifically. The DLAMI puts the EFA/OFI plugin on the
# system loader path (/etc/ld.so.conf.d/100_ofinccl.conf), so NCCL dlopens
# /opt/amazon/ofi-nccl/lib/libnccl-net.so during comm init. deep_ep's check_nccl_so() then
# scans /proc/self/maps for anything matching "libnccl", sees that plugin next to the venv's
# libnccl.so.2, and asserts "Duplicate NCCL runtime found" -- it is a plugin, not a second
# runtime, so the check is simply wrong here. It is fatal because sglang guards the deep_ep
# import with `except ImportError` and this is an AssertionError, so the MoE token dispatcher
# takes down a diffusion worker: the visible symptom is
# "Model architectures ['MiniMaxH3Qwen3VLEncoder'] failed to be inspected", two import layers
# away from the cause. Nothing here is multi-node; the plugin buys nothing on eight NVLinked
# cards in one box. Drop this line before running anything across nodes.
export NCCL_NET_PLUGIN=${NCCL_NET_PLUGIN:-none}

# expandable_segments, because --quantization fp8 here is ONLINE quantization: the loader
# reads the 65.65 GiB bf16 checkpoint (measured from the safetensors headers: 1426 BF16
# tensors + 13 F32) onto the card and casts afterwards, so every bf16 block it frees leaves a
# hole the fp8 weights cannot reuse. First 480p attempt died with 51.89 GiB allocated and
# 18.70 GiB reserved-but-unallocated -- the fit was never 12 GiB short, it was fragmented by
# that much. Expandable segments let the allocator give the holes back instead of hoarding
# them. This is the cheapest of the memory levers and the only one that costs no latency, so
# it goes before --performance-mode auto and before any offload.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

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
  #   OOMs, that flag is the first thing to trade -- see RUNBOOK arm A.
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
