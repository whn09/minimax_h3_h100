#!/usr/bin/env bash
# Base MiniMax-H3 served as ref2va, with or without lightx2v's Turbo LoRA. REF2VA.md is the why.
#
#   bash sglang_ref2va_arm.sh serve 480                                   # arm A, the control
#                                                 (bf16 => TP=2 ULYSSES=4 by default; see below)
#   LORA=$L/ref2v_8step.safetensors bash sglang_ref2va_arm.sh serve 480   # arm B, alpha from file
#   LORA=... LORA_ALPHA=128 bash sglang_ref2va_arm.sh serve 480           # arm C, 16x overdrive
#   MERGED=/opt/dlami/nvme/vdn/ref2v_turbo_bf16 bash sglang_ref2va_arm.sh serve 480   # fp8, merged
#   bash sglang_ref2va_arm.sh refedge 1024      # arm F, and the biggest latency lever in ref2va
#   bash sglang_ref2va_arm.sh refedge restore
#   bash sglang_ref2va_arm.sh mxfp8_guard      # applied automatically whenever LORA is set
#   bash sglang_ref2va_arm.sh stop
#
# WHY THIS IS BASE H3 AND NOT VDN. vdn-minimax-h3 ships the fl2va partition only; upstream never
# trained ref2va for it, and the server says so. So ref2va means base weights, which means 50 steps
# unless a distill LoRA supplies a shorter schedule -- which is exactly what lightx2v's ref2v Turbo
# LoRA is for. That is the whole reason this arm exists.
#
# QUANT DEFAULTS TO EMPTY (bf16) HERE, the opposite of sglang_base_arm.sh, and the reason recorded
# in this comment for one commit was wrong. It said "--lora-path does not survive --quantization
# fp8 (AttributeError: 'RowParallelLinearWithLoRA' object has no attribute 'quant_method')". That
# AttributeError has nothing to do with fp8: it fires in bf16 too, on the first request, and the
# mxfp8_guard() note below is what it actually is. bf16 remains the default here because it is the
# only precision the *merge* path is defined for, but fp8 + LoRA was never tested on its own terms
# and the failure that was attributed to it was this bug. Set QUANT=fp8 with MERGED= (a merged
# checkpoint has no LoRA wrapper at all) or with no adapter.
#
# TP=2 IS NOT OPTIONAL FOR bf16, and this is measured, not precautionary. Ulysses is sequence
# parallelism: it splits the *tokens* and replicates the weights, so every one of the eight ranks
# holds the whole DiT. At bf16 that is 61.73 GB on a card with 79.18 GB usable, and the log reads
#     Loaded text_encoder ... 7.40 GB. avail GPU mem: 68.23 GB
#     Loaded transformer  ... 61.73 GB. avail GPU mem:  5.54 GB
#     Loaded audio_vae    ...  0.56 GB. avail GPU mem:  5.22 GB
#     torch.OutOfMemoryError: ... Tried to allocate 48.00 MiB ... 4.06 MiB is free
#     RuntimeError: Failed to load customized video_vae; native fallback is disabled
# i.e. it dies loading the 5.2 GB video VAE into 5.22 GB, at *startup*, before any request. The 48
# MiB in the message makes this look like a fragmentation problem; it is not, it is 5.2 into 5.22.
# TP shards the DiT's linear layers instead, so TP=2 ULYSSES=4 costs one all-reduce per layer and
# brings the resident DiT to ~31 GB, which fits with room for the reference tower. Hence the
# defaults below: bf16 (no QUANT) implies TP=2 ULYSSES=4, and fp8 -- 31 GB of DiT -- keeps the
# 8-way Ulysses that RESULTS.md measured. Override either explicitly.
#
# ALPHA IS THE ONE FLAG TO GET RIGHT. lightx2v's files are all rank 128 and declare alpha 8
# (scale 0.0625) or 128 (scale 1.0) depending on the file; both ref2v LoRAs are 8. SGLang reads
# that from the safetensors metadata by itself (peft_adapter.py:25 _SAFETENSORS_ALPHA_KEYS), so
# leaving LORA_ALPHA unset is CORRECT and copying lightx2v's README `--lora-alpha 128` onto a ref2v
# LoRA is a 16x overdrive. LORA_ALPHA is here to reproduce that error on purpose (arm C).
#
# Everything else -- CUDA_HOME discovery, the lib64/-lcudart symlinks, NCCL_NET_PLUGIN=none,
# expandable_segments -- lives in _env.sh, which every arm sources and which reads the same in the
# venv and in the container.
set -uo pipefail

ROOT=${ROOT:-/opt/dlami/nvme/sglang}
VDNROOT=${VDNROOT:-/opt/dlami/nvme/vdn}
MODEL=${MODEL:-MiniMaxAI/MiniMax-H3}
PORT=${PORT:-30012}                      # 30010 VDN, 30011 base t2va, 30012 ref2va
QUANT=${QUANT-}                          # empty = bf16; see the note above before setting fp8
GPUS=${GPUS:-8}
# Derived from QUANT rather than fixed, so that neither route needs to be remembered: bf16 does not
# start at all without the shard (see above), and fp8 does not need it. --num-gpus stays $GPUS
# either way, since GPUS = TP * ULYSSES.
if [ -n "${QUANT-}" ]; then TP=${TP:-1};              ULYSSES=${ULYSSES:-$GPUS}
else                        TP=${TP:-2}; ULYSSES=${ULYSSES:-$((GPUS / TP))}; fi
LORA=${LORA-}
LORA_ALPHA=${LORA_ALPHA-}
MERGED=${MERGED-}
export HF_HOME=${HF_HOME:-$VDNROOT/hf}
export SGLANG_DIFFUSION_CACHE_ROOT=${SGLANG_DIFFUSION_CACHE_ROOT:-$ROOT/cache}

mode=${1:?serve|stop|refedge|mxfp8_guard}
# shellcheck source=_env.sh
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

# The reference-image short edge is a module constant, not a flag: reference_encoding.py:47
#   MINIMAX_H3_REFERENCE_IMAGE_SHORT_EDGE = 2048
# with allow_upscale: True. That is MiniMax's own recipe, and it is NOT what lightx2v trained
# their LoRA on -- they use `match` = min(1, sqrt(target_area/ref_area)), which never upscales.
# So this patch is both a quality hypothesis (REF2VA.md cause 3) and the largest cost lever in
# ref2va (one 16:9 reference is 7,296 rows at 2048 and 1,824 at 1024; rows go as the square).
# Editing the installed file is deliberate: workers are separate processes, so a monkeypatch in
# the client would not reach them.
refedge() {
  # Python rather than sed: it locates the installed module by import, keeps one .orig, and
  # re-reads the file to prove the constant actually changed. A silent no-match here would send
  # a whole arm to the wrong conclusion.
  WANT=${1:?a short edge, or the word restore} python - <<'PY'
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
}

# EVERY LoRA ARM DIES WITHOUT THIS, in bf16, at warmup, with a message that names quantization:
#     AttributeError: 'RowParallelLinearWithLoRA' object has no attribute 'quant_method'
# The g7e project and this repo both filed that under "--lora-path does not survive fp8". It is not
# an fp8 problem. models/dits/minimax_h3.py:373 is
#     def _accepts_mxfp8_input(linear): return linear.quant_method is not None and ...
# a fast-path guard asking whether a Linear can take mxfp8 input directly, and it reads the
# attribute rather than getattr-ing it. Loading a LoRA replaces the targeted Linears with
# BaseLayerWithLoRA wrappers, and those forward .weight and .bias to the base layer but NOT
# .quant_method (runtime/layers/lora/linear.py:110-125), so the plain access raises. The path that
# raises is the *token refiner* MLP's fc2 -- reached on every request, in bf16, with no quantization
# anywhere -- and it raises whether the adapter was merged or left dynamic, because merging writes
# into base_layer.weight and leaves the wrapper in the module tree either way. Two consequences
# worth writing down: the "fp8" story was wrong, and merge_mode is not a workaround.
# The fix is to treat "no quant_method" as "not an mxfp8 layer", which is what an unquantized layer
# is. Conservative on purpose: it only ever declines a fast path, never takes one, so it cannot
# change the numerics of a run that was already working.
mxfp8_guard() {
  python - <<'PY'
import re, shutil
import sglang.multimodal_gen.runtime.models.dits.minimax_h3 as m

f = m.__file__
orig = f + ".orig"
src = open(orig).read() if __import__("os").path.isfile(orig) else open(f).read()
if not __import__("os").path.isfile(orig):
    shutil.copy(f, orig)
pat = re.compile(
    r"def _accepts_mxfp8_input\(linear: nn\.Module\) -> bool:\n"
    r"    return linear\.quant_method is not None and linear\.quant_method\.accepts_mxfp8_input\(\n"
    r"        linear\n    \)\n")
new, n = pat.subn(
    'def _accepts_mxfp8_input(linear: nn.Module) -> bool:\n'
    '    # getattr, not attribute access: LoRA wrappers do not expose quant_method.\n'
    '    qm = getattr(linear, "quant_method", None)\n'
    '    return qm is not None and qm.accepts_mxfp8_input(linear)\n', src)
if n != 1:
    raise SystemExit(f"expected exactly 1 match in {f}, got {n} -- upstream changed the guard; "
                     "re-read the traceback before assuming this patch is still needed")
open(f, "w").write(new)
print(f"patched _accepts_mxfp8_input in {f}")
PY
}

if [ "$mode" = mxfp8_guard ]; then
  mxfp8_guard
  exit 0
fi

if [ "$mode" = refedge ]; then
  refedge "${2:?a short edge, or the word restore}"
  echo "restart the server for this to take effect"
  exit 0
fi


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
# Only when sharding: --tp-size 1 is the default and passing it needlessly would show up in every
# log line and invite the question of whether the fp8 arms were sharded too. They were not.
[ "$TP" -gt 1 ] && extra+=(--tp-size "$TP")
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
# REFEDGE=1024 patches the reference short edge in *this* process tree before the server starts,
# which is the only form that works in a container: the patch rewrites an installed module, and a
# `refedge` run in a throwaway container is discarded with that container's writable layer. In a
# venv either form works and `refedge` alone is the same thing. Arm F is REFEDGE=1024.
if [ -n "${REFEDGE-}" ]; then
  refedge "$REFEDGE"
fi
# Applied in this process tree, before the workers fork, for the same reason REFEDGE is: the workers
# are separate processes and a container's writable layer is discarded, so a patch run in a
# throwaway container would not reach the server that needs it.
if [ -n "$LORA" ]; then
  mxfp8_guard
fi
log="$ROOT/logs/serve_ref2va_${edge}p${LOGTAG:+_$LOGTAG}.log"
mkdir -p "$ROOT/logs"
set -x
sglang serve \
  --model-path "$MODEL" \
  --num-gpus "$GPUS" \
  --ulysses-degree "$ULYSSES" \
  --encoder-parallel auto \
  --performance-mode speed \
  --warmup-num-frames "$FRAMES" \
  --warmup-resolutions "$canvas" \
  "${extra[@]}" \
  --host 127.0.0.1 --port "$PORT" \
  "$@" 2>&1 | tee "$log"
exit "${PIPESTATUS[0]}"
