#!/usr/bin/env bash
# Base MiniMax-H3 served as ref2va, with or without lightx2v's Turbo LoRA. REF2VA.md is the why.
#
#   bash sglang_ref2va_arm.sh serve 480                                   # arm A, the control
#   LORA=$L/ref2v_8step.safetensors bash sglang_ref2va_arm.sh serve 480   # arm B, alpha from file
#   LORA=... LORA_ALPHA=128 bash sglang_ref2va_arm.sh serve 480           # arm C, 16x overdrive
#   MERGED=/opt/dlami/nvme/vdn/ref2v_turbo_bf16 bash sglang_ref2va_arm.sh serve 480   # fp8, merged
#   bash sglang_ref2va_arm.sh refedge 1024      # arm F, and the biggest latency lever in ref2va
#   bash sglang_ref2va_arm.sh refedge restore
#   bash sglang_ref2va_arm.sh stop
#
# WHY THIS IS BASE H3 AND NOT VDN. vdn-minimax-h3 ships the fl2va partition only; upstream never
# trained ref2va for it, and the server says so. So ref2va means base weights, which means 50 steps
# unless a distill LoRA supplies a shorter schedule -- which is exactly what lightx2v's ref2v Turbo
# LoRA is for. That is the whole reason this arm exists.
#
# QUANT DEFAULTS TO EMPTY (bf16) HERE, the opposite of sglang_base_arm.sh. Not a preference:
# --lora-path does not survive --quantization fp8 on this build (AttributeError:
# 'RowParallelLinearWithLoRA' object has no attribute 'quant_method', during warmup), and every
# LoRA arm in REF2VA.md is a *quality* arm where bf16 is fine and fp8 is not available. Set
# QUANT=fp8 only with MERGED= (a merged checkpoint has no LoRA wrapper to trip over) or with no
# adapter at all. bf16 is also ~62 GiB of DiT per rank under Ulysses, so if 768p or a large
# reference OOMs, that is the first thing to blame -- try --tp-size 2 --ulysses-degree 4.
#
# ALPHA IS THE ONE FLAG TO GET RIGHT. lightx2v's files are all rank 128 and declare alpha 8
# (scale 0.0625) or 128 (scale 1.0) depending on the file; both ref2v LoRAs are 8. SGLang reads
# that from the safetensors metadata by itself (peft_adapter.py:25 _SAFETENSORS_ALPHA_KEYS), so
# leaving LORA_ALPHA unset is CORRECT and copying lightx2v's README `--lora-alpha 128` onto a ref2v
# LoRA is a 16x overdrive. LORA_ALPHA is here to reproduce that error on purpose (arm C).
#
# Everything else -- CUDA_HOME discovery, the lib64/-lcudart symlinks, NCCL_NET_PLUGIN=none,
# expandable_segments -- is identical to sglang_arm.sh and carries the same reasons.
set -uo pipefail

ROOT=${ROOT:-/opt/dlami/nvme/sglang}
VDNROOT=${VDNROOT:-/opt/dlami/nvme/vdn}
MODEL=${MODEL:-MiniMaxAI/MiniMax-H3}
PORT=${PORT:-30012}                      # 30010 VDN, 30011 base t2va, 30012 ref2va
QUANT=${QUANT-}                          # empty = bf16; see the note above before setting fp8
GPUS=${GPUS:-8}
LORA=${LORA-}
LORA_ALPHA=${LORA_ALPHA-}
MERGED=${MERGED-}
export HF_HOME=${HF_HOME:-$VDNROOT/hf}
export SGLANG_DIFFUSION_CACHE_ROOT=${SGLANG_DIFFUSION_CACHE_ROOT:-$ROOT/cache}

mode=${1:?serve|stop|refedge}
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"

# The reference-image short edge is a module constant, not a flag: reference_encoding.py:47
#   MINIMAX_H3_REFERENCE_IMAGE_SHORT_EDGE = 2048
# with allow_upscale: True. That is MiniMax's own recipe, and it is NOT what lightx2v trained
# their LoRA on -- they use `match` = min(1, sqrt(target_area/ref_area)), which never upscales.
# So this patch is both a quality hypothesis (REF2VA.md cause 3) and the largest cost lever in
# ref2va (one 16:9 reference is 7,296 rows at 2048 and 1,824 at 1024; rows go as the square).
# Editing the installed file is deliberate: workers are separate processes, so a monkeypatch in
# the client would not reach them.
if [ "$mode" = refedge ]; then
  # Python rather than sed: it locates the installed module by import, keeps one .orig, and
  # re-reads the file to prove the constant actually changed. A silent no-match here would send
  # a whole arm to the wrong conclusion.
  WANT=${2:?a short edge, or the word restore} python - <<'PY'
import os, re, shutil
import sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.reference_encoding as m

f, want = m.__file__, os.environ["WANT"]
orig = f + ".orig"
if want == "restore":
    if not os.path.isfile(orig):
        raise SystemExit(f"no {orig} to restore")
    shutil.copy(orig, f)
else:
    edge = int(want)
    if edge % 32:
        raise SystemExit(f"{edge} is not a multiple of 32 (MINIMAX_H3_REFERENCE_IMAGE_MULTIPLE)")
    if not os.path.isfile(orig):
        shutil.copy(f, orig)
    src = open(orig).read()
    pat = re.compile(r"^(MINIMAX_H3_REFERENCE_IMAGE_SHORT_EDGE = )\d+$", re.M)
    new, n = pat.subn(rf"\g<1>{edge}", src)
    if n != 1:
        raise SystemExit(f"expected exactly 1 match in {f}, got {n} -- upstream moved the constant")
    open(f, "w").write(new)
print(f, [l for l in open(f) if l.startswith("MINIMAX_H3_REFERENCE_IMAGE_SHORT_EDGE")])
PY
  echo "restart the server for this to take effect"
  exit 0
fi

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
shift; [ $# -gt 0 ] && shift

extra=()
[ -n "$QUANT" ] && extra+=(--quantization "$QUANT")
[ -n "$LORA" ] && extra+=(--lora-path "$LORA")
[ -n "$LORA_ALPHA" ] && extra+=(--lora-alpha "$LORA_ALPHA")
# MERGED replaces the transformer with an already-merged bf16 tree. --model-variant hybrid is the
# variant that accepts it and refuses to start without it (minimax_h3_pipeline.py:97); it takes the
# rest of its components from the Ref2VA partition, which is what ref2va needs anyway.
if [ -n "$MERGED" ]; then
  extra+=(--model-variant hybrid --component-weights-paths.transformer "$MERGED")
else
  extra+=(--model-variant ref2va)
fi
log="$ROOT/logs/serve_ref2va_${edge}p${LOGTAG:+_$LOGTAG}.log"
mkdir -p "$ROOT/logs"
set -x
sglang serve \
  --model-path "$MODEL" \
  --num-gpus "$GPUS" \
  --ulysses-degree "$GPUS" \
  --encoder-parallel auto \
  --performance-mode speed \
  --warmup-num-frames "$FRAMES" \
  --warmup-resolutions "$canvas" \
  "${extra[@]}" \
  --host 127.0.0.1 --port "$PORT" \
  "$@" 2>&1 | tee "$log"
exit "${PIPESTATUS[0]}"
