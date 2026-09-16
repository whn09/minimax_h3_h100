#!/usr/bin/env bash
# Bring up VDN-Minimax-H3 on a p5.48xlarge (8xH100 80GB, Ubuntu 26.04, NVLink NV18).
#
#   setsid nohup bash h100_bringup.sh > /opt/dlami/nvme/vdn/bringup.log 2>&1 < /dev/null &
#
# Everything lands on the 27 TB ephemeral NVMe (/opt/dlami/nvme); / is only 484 GB and
# the DLAMI's own venv (/opt/pytorch, python 3.13 + torch 2.13+cu130) is NOT usable:
# the repo pins `requires-python >=3.12,<3.13` and torch from the cu129 index, because
# flash-attn-4's dependency tree pulls the cu130 default otherwise.  So we build a
# private uv-managed python 3.12.
set -euo pipefail

ROOT=${ROOT:-/opt/dlami/nvme/vdn}
REPO=$ROOT/vdn-minimax-h3
CKPTS=$ROOT/ckpts
export HF_HOME=${HF_HOME:-$ROOT/hf}
export UV_CACHE_DIR=${UV_CACHE_DIR:-$ROOT/uvcache}
export UV_PYTHON_INSTALL_DIR=${UV_PYTHON_INSTALL_DIR:-$ROOT/uvpython}

mkdir -p "$ROOT" "$CKPTS" "$HF_HOME"
step() { echo "=== [$(date -u +%H:%M:%S)] $*"; }

# ---------------------------------------------------------------- 1. uv + python 3.12
step "uv"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=$HOME/.local/bin sh
fi
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.12

# ---------------------------------------------------------------- 2. repo
step "clone"
[ -d "$REPO/.git" ] || git clone https://github.com/OpenVDN/vdn-minimax-h3.git "$REPO"
cd "$REPO"
# ckpts/ is resolved relative to the repo root (src/paths.py), so point it at the NVMe
[ -e "$REPO/ckpts" ] || ln -s "$CKPTS" "$REPO/ckpts"

# ---------------------------------------------------------------- 3. venv + torch
step "venv + torch 2.13.0 cu129"
[ -d "$REPO/.venv" ] || uv venv --python 3.12 "$REPO/.venv"
# shellcheck disable=SC1091
source "$REPO/.venv/bin/activate"
# torchvision has to be named here too, and from the same index. pyproject pins
# `torchvision==0.28.0` with no local version, so if it is left to `uv pip install -e .`
# below, uv takes the PyPI default -- which is the cu130 build -- and every import of
# diffusers.loaders.peft then dies with "PyTorch has CUDA Version=12.9 and torchvision has
# CUDA Version=13.0". Worse, `uv pip install torchvision==0.28.0 --index-url .../cu129`
# afterwards is a no-op: 0.28.0 already satisfies 0.28.0, so it never fetches 0.28.0+cu129.
# Getting both from the cu129 index up front is the only ordering that works.
uv pip install -q torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu129

step "deps (flash-attn-4 needs --prerelease=allow for nvidia-cutlass-dsl)"
uv pip install -q --prerelease=allow -e .

step "patched diffusers"
bash scripts/setup_diffusers.sh

# ---------------------------------------------------------------- 4. weights (82 GB)
step "weights -> $CKPTS"
uv pip install -q 'huggingface_hub[hf_transfer]'
export HF_HUB_ENABLE_HF_TRANSFER=1
hf download OpenVDN/vdn-minimax-h3 --local-dir "$CKPTS"

step "done"
python -c "import torch, diffusers; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpus', torch.cuda.device_count()); print('diffusers', diffusers.__version__)"
du -sh "$CKPTS"/*
