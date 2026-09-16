#!/usr/bin/env bash
# SGLang Diffusion as an alternative to the patched reference stack, on the same box.
#
#   setsid nohup bash sglang_bringup.sh > /opt/dlami/nvme/sglang/bringup.log 2>&1 < /dev/null &
#
# WHY: sglang's cookbook (docs/cookbook/diffusion/MiniMax/MiniMax-H3.mdx, section 7) now
# serves OpenVDN/vdn-minimax-h3 directly, and on 8x B200 it measures 0.88 s/NFE against
# the reference stack's published 1.40 -- 1.60x, or 1.43x on the per-channel fp8 path that
# an H100 (SM90) would take. The cookbook attributes the gap to parallel efficiency
# (86-91% from 2 to 8 cards, against the reference stack's 56-64%), not to Blackwell
# precision, so it should carry to H100. It also ships, as flags, three things this repo
# implements as patches: encoder folding across idle ranks, VAE residency/offload policy,
# and layerwise DiT placement.
#
# WHAT IS NOT KNOWN: the cookbook has NO H100 numbers for VDN. Its only VDN tables are
# B200 and RTX PRO 6000, and its H100 rows are 4-card base-H3. 8x H100 Ulysses8 for VDN
# is legal but unverified, and 79,972 MB peak/GPU on 8x B200 is above what an 80 GB card
# gives PyTorch here (65.26 GiB after a 13.92 GiB non-PyTorch floor), so a memory fit is
# the first thing the arm settles, not an assumption it rests on.
#
# Separate root, separate venv, nothing shared with $VDNROOT but the HF cache. The
# reference stack's .venv (torch 2.13.0+cu129, patched diffusers) must keep measuring
# 11.44 s at 480p while this is installed, because that is the number sglang is compared
# against; sglang pins its own torch and would otherwise overwrite it.
set -euo pipefail

ROOT=${ROOT:-/opt/dlami/nvme/sglang}
VDNROOT=${VDNROOT:-/opt/dlami/nvme/vdn}
export HF_HOME=${HF_HOME:-$VDNROOT/hf}
export UV_CACHE_DIR=${UV_CACHE_DIR:-$VDNROOT/uvcache}
export UV_PYTHON_INSTALL_DIR=${UV_PYTHON_INSTALL_DIR:-$VDNROOT/uvpython}
# The overlay prefuses both adapters into the transformer on first launch -- a 62 GB
# write. The cookbook asks for >=90 GB free on the same filesystem as the HF cache.
export SGLANG_DIFFUSION_CACHE_ROOT=${SGLANG_DIFFUSION_CACHE_ROOT:-$ROOT/cache}

mkdir -p "$ROOT" "$SGLANG_DIFFUSION_CACHE_ROOT" "$HF_HOME"
step() { echo "=== [$(date -u +%H:%M:%S)] $*"; }

step "uv"
command -v uv >/dev/null 2>&1 || \
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=$HOME/.local/bin sh
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.12

step "venv + sglang[diffusion]"
[ -d "$ROOT/.venv" ] || uv venv --python 3.12 "$ROOT/.venv"
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"
# --prerelease=allow is the cookbook's own instruction (flash-attn-4 / cutlass-dsl again).
# No --index-url here: unlike the reference stack, sglang resolves its own torch build,
# and forcing cu129 on it is how you get a wheel mismatch it never asked for.
uv pip install -q "sglang[diffusion]" --prerelease=allow
uv pip install -q 'huggingface_hub[hf_transfer]'

step "does this build actually have VDN?"
# The cookbook is the main branch; the PyPI release may predate section 7. These two
# strings are the gate -- the backend name and the model id are what section 7 adds.
# If either is missing, install from source instead of guessing:
#   uv pip install -q --prerelease=allow "sglang[diffusion] @ git+https://github.com/sgl-project/sglang.git#subdirectory=python"
python -c "import sglang; print('sglang', sglang.__version__)"
if sglang serve --help 2>&1 | grep -q hybrid_window_attn_h3; then
  echo "hybrid_window_attn_h3: present"
else
  echo "hybrid_window_attn_h3: MISSING -- this build predates VDN support; see the"
  echo "  git+ line above, and check whether the flag names in sglang_arm.sh moved too."
  exit 2
fi

step "weights"
export HF_HUB_ENABLE_HF_TRANSFER=1
# sglang owns the checkpoint-directory mapping (cookbook section 2: "do not point
# --model-path at a manually downloaded subdirectory"), so it is given the repo ID and
# fetches into HF_HOME itself -- it cannot reuse $VDNROOT/ckpts, which was pulled with
# --local-dir and has no cache layout. It also hard-links the Qwen3-VL conditioner and
# the VAEs out of MiniMaxAI/MiniMax-H3, so that repo comes down too: ~110 GB on top of
# the 82 GB it re-fetches, plus the 62 GB fused overlay. ~250 GB, minutes on the NVMe.
hf download OpenVDN/vdn-minimax-h3 > /dev/null
hf download MiniMaxAI/MiniMax-H3 > /dev/null

step "done"
df -h --output=avail "$ROOT" | tail -1
du -sh "$HF_HOME" "$SGLANG_DIFFUSION_CACHE_ROOT" 2>/dev/null
echo "next: bash $ROOT/../vdn/sglang_arm.sh serve 768   (see RUNBOOK section 3, arm A)"
