#!/usr/bin/env bash
# Base MiniMax-H3 on the same eight cards, so "VDN is fast" has a same-machine denominator.
#
#   bash sglang_base_arm.sh serve 480 [extra flags...]
#   bash sglang_base_arm.sh stop
#
# THREE MODELS IN ONE FAMILY, and only the first is what people mean by "H3":
#   MiniMaxAI/MiniMax-H3   MiniMaxH3SamplingParams.num_inference_steps = 50   (configs/sample/minimax_h3.py:41)
#   FastH3                 asserts exactly 5 sigma points, i.e. 4 DiT forwards, t2va only
#   OpenVDN/vdn-minimax-h3 asserts exactly 9 sigma points, i.e. the 8 NFE this repo measures
# So the honest base number is 50 steps, and 8 steps on base weights is a step-cost probe, not a
# render anyone would ship -- the schedule it was trained for is 50.
#
# WHAT CHANGES FROM sglang_arm.sh, and why:
#   * no --attention-backend hybrid_window_attn_h3. That backend IS the VDN architecture (linear
#     branch + gates). Base H3 is dense softmax attention; passing it would either error or serve
#     a model these weights are not.
#   * --lora-path is accepted here (runtime/pipelines_core/lora/pipeline.py reads
#     server_args.lora_path), which is how the g7e project's Turbo LoRA gets onto base weights.
#     It does NOT work together with --quantization fp8 on this build, and not for the reason g7e
#     recorded. Their failure was a runtime "merge" doing an in-place add on [out, in] against an
#     fp8 weight stored transposed (their fc1 reported 21504 vs 5376); upstream has since fixed
#     that -- _should_merge_lora_for_layers sees can_merge_base_weight == False on a quantized
#     layer and falls back to dynamic LoRA by itself. The dynamic wrapper is then not
#     quantization-aware, and warmup dies one layer later:
#         AttributeError: 'RowParallelLinearWithLoRA' object has no attribute 'quant_method'
#     So the LoRA arm runs with QUANT= (bf16), where merge mode auto merges the adapter into the
#     weights at load and costs nothing at forward time. Two consequences, both deliberate:
#       - that arm's latency is bf16 latency and is not comparable to the fp8 arms. It does not
#         need to be: a merged LoRA changes weight values, not shapes or the graph, so the fp8
#         Turbo-at-8-steps latency IS the fp8 base-H3-at-8-steps latency already measured.
#       - the offline route (scripts/lora_merge_h3.py) is kept but is blocked on a key-layout
#         translation: this LoRA is named for the *native* checkpoint layout
#         (blocks.N.attn.qkv_proj / mlp.fc1), every t2va tree on disk -- including VDN's h3-base --
#         is named for the *diffusers* layout (transformer_blocks.N.attn.to_{q,k,v} /
#         ff.net.0.proj), and only FL2VA/ and Ref2VA/ ship native names. That is why g7e merged
#         against FL2VA/transformer and hit 259/259, and why the same script hits 0/259 here.
#         This blocks *this* LoRA on a *t2va* tree and nothing more: lightx2v's H3 Turbo LoRAs are
#         diffusers-named, and MiniMaxAI/MiniMax-H3 ships transformer_ref/ diffusers-named as well,
#         so the ref2va merge needs no translation at all. See REF2VA.md and sglang_ref2va_arm.sh.
# Everything else -- CUDA_HOME discovery, the lib64/-lcudart symlinks, NCCL_NET_PLUGIN=none,
# expandable_segments, the ffmpeg gate -- is identical and carries the same reasons as
# sglang_arm.sh. Read that file for them.
set -uo pipefail

ROOT=${ROOT:-/opt/dlami/nvme/sglang}
VDNROOT=${VDNROOT:-/opt/dlami/nvme/vdn}
MODEL=${MODEL:-MiniMaxAI/MiniMax-H3}
PORT=${PORT:-30011}                      # not 30010: leaves the VDN server's port free
# QUANT=fp8 for every latency arm. QUANT= (empty) serves bf16, which is the only way --lora-path
# runs at all -- see the LoRA note below. A bf16 arm's latency is NOT comparable to the fp8 arms.
QUANT=${QUANT-fp8}
GPUS=${GPUS:-8}
export HF_HOME=${HF_HOME:-$VDNROOT/hf}
export SGLANG_DIFFUSION_CACHE_ROOT=${SGLANG_DIFFUSION_CACHE_ROOT:-$ROOT/cache}

mode=${1:?serve|stop}
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"
export CUDA_HOME=${CUDA_HOME:-$(python - <<'PY'
import pathlib, sysconfig
nv = pathlib.Path(sysconfig.get_paths()["purelib"]) / "nvidia"
print(next((str(p.parent.parent) for p in sorted(nv.glob("*/bin/nvcc"))), ""))
PY
)}
if [ -d "$CUDA_HOME/lib" ]; then
  [ -e "$CUDA_HOME/lib64" ] || ln -sfn lib "$CUDA_HOME/lib64"
  for so in "$CUDA_HOME"/lib/lib*.so.[0-9]*; do
    base=${so%%.so.*}.so
    [ -e "$base" ] || ln -sfn "$(basename "$so")" "$base"
  done
fi
export NCCL_NET_PLUGIN=${NCCL_NET_PLUGIN:-none}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

if [ "$mode" = stop ]; then
  pkill -f '[s]glang.*serve'
  sleep 5; echo stopped
  exit 0
fi

edge=${2:-480}
case "$edge" in
  768) canvas=1344x768 ;;
  480) canvas=864x480  ;;
  *) echo "short edge must be 768 or 480"; exit 1 ;;
esac
FRAMES=${FRAMES:-345}
# Consume mode and edge before "$@" forwards the rest to sglang: without this the server is
# launched with `serve 480` appended and dies on "unrecognized arguments".
shift; [ $# -gt 0 ] && shift
quant=()
[ -n "$QUANT" ] && quant=(--quantization "$QUANT")
# LOGTAG keeps a second arm at the same short edge from overwriting the first arm's server log --
# the fp8 and bf16 480p runs both want serve_base_480p.log otherwise.
log="$ROOT/logs/serve_base_${edge}p${LOGTAG:+_$LOGTAG}.log"
mkdir -p "$ROOT/logs"
set -x
sglang serve \
  --model-path "$MODEL" \
  --num-gpus "$GPUS" \
  --ulysses-degree "$GPUS" \
  "${quant[@]}" \
  --encoder-parallel auto \
  --performance-mode speed \
  --warmup-num-frames "$FRAMES" \
  --warmup-resolutions "$canvas" \
  --host 127.0.0.1 --port "$PORT" \
  "$@" 2>&1 | tee "$log"
exit "${PIPESTATUS[0]}"
