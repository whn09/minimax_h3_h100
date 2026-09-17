#!/usr/bin/env bash
# The checkpoint download, on its own so both routes call the same code:
#   sglang_bringup.sh (the venv route) and docker/h3.sh weights (inside the image).
#
#   WEIGHTS=t2va,ref2va bash fetch_weights.sh
#
# WHY THIS IS A SELECTOR AND NOT `hf download MiniMaxAI/MiniMax-H3`. Unfiltered that is 464 GiB,
# because the repo ships the same 62 GiB DiT under four names: transformer/ (t2va, diffusers-named),
# transformer_ref/ (ref2va, diffusers-named), and the self-contained FL2VA/ and Ref2VA/ partitions
# (134 GiB each, native-named, each carrying its own copy of the 62 GiB Qwen3-VL text encoder). No
# arm needs more than two of them. /opt/dlami/nvme is instance store, so this is paid again after
# every stop, which makes download time the only cost that matters on a fresh box.
#
# sglang owns the checkpoint-directory mapping (cookbook section 2: "do not point --model-path at a
# manually downloaded subdirectory"), so it is given the repo ID and reads what lands in HF_HOME --
# it cannot reuse a --local-dir tree, which has no cache layout.
#
# THE TEXT ENCODER IS NOT OPTIONAL for any sglang arm, which is the one trap here: the reference
# stack torch.loads offline prompt caches (the "text encoding is not in any of these numbers"
# section of RESULTS.md), so a checkpoint tree that worked for it can be missing text_encoder/ and
# processor/ entirely. A server that takes a prompt over HTTP needs both.
set -euo pipefail

WEIGHTS=${WEIGHTS:-${1:-vdn,t2va}}
export HF_HOME=${HF_HOME:-/opt/dlami/nvme/vdn/hf}
mkdir -p "$HF_HOME"
command -v hf >/dev/null 2>&1 || { echo "no \`hf\` on PATH: pip install huggingface_hub"; exit 2; }

# Which accelerated-download knob exists depends on the hub version, and setting the wrong one
# costs real minutes on 200 GiB. hub 1.x: hf_transfer is deleted, hf-xet is the backend and a
# default dependency, and HF_HUB_ENABLE_HF_TRANSFER=1 only earns a FutureWarning telling you to use
# HF_XET_HIGH_PERFORMANCE. hub 0.x: the reverse, and `huggingface_hub[hf_transfer]` is how you get
# it -- on 1.x that extra warns "does not provide the extra 'hf-transfer'" and silently leaves you
# on the slow path. Ask the installed package rather than guess.
if python -c "import hf_transfer" 2>/dev/null; then
  export HF_HUB_ENABLE_HF_TRANSFER=1
  echo "fast download: hf_transfer"
elif python -c "import hf_xet" 2>/dev/null; then
  export HF_XET_HIGH_PERFORMANCE=1
  echo "fast download: hf-xet (HF_XET_HIGH_PERFORMANCE)"
else
  echo "fast download: NEITHER hf_transfer nor hf_xet -- expect this to take much longer"
fi

# EVERY PATTERN NEEDS ITS OWN --include. `hf` is click-based from huggingface_hub 1.x and --include
# takes exactly one value per occurrence, so `--include 'a/*' 'b/*'` sends b/* to the FILENAMES
# positional and dies on "File not found in repository: .../b/%2A". Verified, not assumed. The
# patterns are fnmatch against the full relative path and `*` crosses `/`, which is why '*.json' is
# enough to bring every config and index file in the repo (all tiny).
base() {  # base <pattern>...  -- the small shared files plus whatever this arm needs
  local inc=(--include '*.json' --include 'tokenizer/*' --include 'processor/*'
             --include 'scheduler/*' --include 'audio_scheduler/*')
  local p; for p in "$@"; do inc+=(--include "$p"); done
  hf download MiniMaxAI/MiniMax-H3 "${inc[@]}" > /dev/null
}

for w in ${WEIGHTS//,/ }; do
  echo "=== [$(date -u +%H:%M:%S)] $w"
  case "$w" in
    none) ;;
    # VDN's 8-step distill. t2va + fl2va only; it has no ref2va partition.        82 GB
    vdn)  hf download OpenVDN/vdn-minimax-h3 > /dev/null ;;
    # base H3 t2va/fl2va-over-t2va: the 50-step denominator and the 8-step probe. 134 GiB
    t2va) base 'transformer/*' 'text_encoder/*' 'vae/*' 'audio_vae/*' ;;
    # the ref2va partition, native-named, self-contained. --model-variant ref2va. 134 GiB
    ref2va) base 'Ref2VA/*' ;;
    # the fl2va partition, native-named. Only needed to merge a *native* LoRA.    134 GiB
    fl2va) base 'FL2VA/*' ;;
    # ref2va DiT under diffusers names -- the tree a lightx2v LoRA merges into with no key
    # translation at all. Needs `ref2va` too, for everything else.                 62 GiB
    transformer_ref) base 'transformer_ref/*' ;;
    all)  hf download MiniMaxAI/MiniMax-H3 > /dev/null ;;
    *) echo "unknown WEIGHTS entry '$w'; pick from vdn,t2va,ref2va,fl2va,transformer_ref,all,none"
       exit 2 ;;
  esac
  echo "  $w: done"
done
du -sh "$HF_HOME" 2>/dev/null || true
