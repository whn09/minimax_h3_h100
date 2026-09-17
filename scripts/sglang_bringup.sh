#!/usr/bin/env bash
# SGLang Diffusion as an alternative to the patched reference stack, on the same box.
#
#   WEIGHTS=t2va,ref2va setsid nohup bash sglang_bringup.sh \
#       > /opt/dlami/nvme/sglang/bringup.log 2>&1 < /dev/null &
#
# WEIGHTS is a comma list picking which checkpoints to fetch; see the `weights` step for the
# sizes and what each arm needs. It defaults to what this repo's headline numbers were
# measured with (vdn,t2va). WEIGHTS=none installs the environment and downloads nothing.
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
WEIGHTS=${WEIGHTS:-vdn,t2va}
export HF_HOME=${HF_HOME:-$VDNROOT/hf}
export UV_CACHE_DIR=${UV_CACHE_DIR:-$VDNROOT/uvcache}
export UV_PYTHON_INSTALL_DIR=${UV_PYTHON_INSTALL_DIR:-$VDNROOT/uvpython}
# The overlay prefuses both adapters into the transformer on first launch -- a 62 GB
# write. The cookbook asks for >=90 GB free on the same filesystem as the HF cache.
export SGLANG_DIFFUSION_CACHE_ROOT=${SGLANG_DIFFUSION_CACHE_ROOT:-$ROOT/cache}

mkdir -p "$ROOT" "$SGLANG_DIFFUSION_CACHE_ROOT" "$HF_HOME"
step() { echo "=== [$(date -u +%H:%M:%S)] $*"; }

# Refuse to run inside an activated environment. Installing into the DLAMI's /opt/pytorch
# fails, and the error blames the wrong thing: sglang pins an outlines_core 0.1.x, whose
# newest wheel is cp312, so python 3.13 has to build it from source and dies on
# "can't find Rust compiler". Installing Rust makes that one line pass and leaves you on
# 3.13 for the next gap -- and it would also overwrite /opt/pytorch's torch for everyone.
if [ -n "${VIRTUAL_ENV:-}" ]; then
  echo "deactivate first: \$VIRTUAL_ENV=$VIRTUAL_ENV. This script builds its own python 3.12."
  exit 2
fi

step "ffmpeg"
# H3's pipeline validates output delivery through ffmpeg/ffprobe and raises at startup if
# either is missing -- checked before the 250 GB download, not after. On this box apt was
# broken first: /etc/apt/sources.list.d/cuda-ubuntu2604-x86_64.list serves a malformed
# Packages file ("Encountered a section with no Package: header"), and no amount of clearing
# /var/lib/apt/lists fixes it because apt re-fetches the same bad file. Moving that one
# source out of the way and re-running update is what worked; nothing here installs CUDA
# from apt anyway (the venv carries its own).
if ! command -v ffmpeg >/dev/null 2>&1 || ! command -v ffprobe >/dev/null 2>&1; then
  sudo -n apt-get update -qq && sudo -n DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ffmpeg
  command -v ffprobe >/dev/null 2>&1 || { echo "install ffmpeg+ffprobe first; see the apt note above"; exit 2; }
fi

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
python -V | grep -q '3\.12' || { echo "venv is not python 3.12; see the 3.13 note above"; exit 2; }
# FROM GIT, NOT PyPI. Checked on this box: the latest release, 0.5.19, has base MiniMax-H3
# (pipeline_configs/minimax_h3.py, the two VAE contracts, --model-variant fl2va/ref2va) but
# NOT VDN -- `hybrid_window_attn_h3` and the string `vdn` appear in zero files. On main VDN
# is a whole subsystem: its own pipeline, DiT, attention backend and Triton/CUDA kernels
# (multimodal_gen/{configs,runtime}/**/minimax_h3_vdn*.py, backends/hybrid_window_attn_h3.py,
# kernels/ops/diffusion/attention/vdn_*). Retry PyPI once 0.5.20 is out; until then this is
# the only source of it. No CUDA build: sglang-kernel==0.4.7 ships a cp310-abi3 x86_64 wheel
# that installs on 3.12, and the VDN delta-factors kernel JITs through apache-tvm-ffi with
# nvidia-cuda-nvcc as a pip dependency, so the missing /usr/local/cuda does not matter.
#
# SGLANG_BUILD_RUST_EXTS=none is not optional. main's setup.py shells out to `cargo` just to
# *discover* the Rust extension modules under rust/ (the LLM router), so with no toolchain the
# build fails in get_requires_for_build_wheel, before any Python is compiled. The escape hatch
# is the error message's own suggestion; nothing in the diffusion path uses those modules.
# Installing rustup instead would work and would also spend ten minutes building a router this
# never calls.
export SGLANG_BUILD_RUST_EXTS=none
uv pip install -q --prerelease=allow \
  "sglang[diffusion] @ git+https://github.com/sgl-project/sglang.git#subdirectory=python"
# NOT 'huggingface_hub[hf_transfer]'. As of huggingface_hub 1.x that extra no longer exists
# ("does not provide the extra 'hf-transfer'") and hf_transfer is gone from the codebase: the
# fast path is hf-xet, a default dependency, and the knob is HF_XET_HIGH_PERFORMANCE. Asking
# for the old extra installs plain hub and silently leaves the download on the slow path.
uv pip install -q huggingface_hub

step "does this build actually have VDN?"
# Import the module, do not grep --help. `sglang serve --help` adds the diffusion flags
# only after it has resolved --model-path, so on an LLM-only build it prints the LLM flag
# set and greps for --ulysses-degree or hybrid_window_attn_h3 come back empty on a build
# that does have diffusion. The import is the fact.
python -c "import sglang; print('sglang', sglang.__version__)"
if python -c "import sglang.multimodal_gen.configs.pipeline_configs.minimax_h3_vdn" 2>/dev/null; then
  echo "VDN pipeline: present"
else
  echo "VDN pipeline: MISSING even from this install -- check whether the module moved,"
  echo "  and whether the flag names in sglang_arm.sh moved with it. sglang --help first."
  exit 2
fi
# deep_gemm asserts on import if CUDA_HOME is unset (its find_cuda_home has no fallback),
# and this box has no /usr/local/cuda. Point it at the pip CUDA the venv already carries --
# found by locating bin/nvcc, since the wheel layout is nvidia/cu13/ here. sglang_arm.sh does
# the same thing; this only reports it.
python - <<'PY'
import pathlib, sysconfig
nv = pathlib.Path(sysconfig.get_paths()["purelib"]) / "nvidia"
hit = next((p.parent.parent for p in sorted(nv.glob("*/bin/nvcc"))), None)
print(f"CUDA_HOME={hit}" if hit else "CUDA_HOME: no pip nvcc found, set it by hand")
PY

step "weights: $WEIGHTS"
# One selector, shared with docker/h3.sh so both routes fetch identically. Read fetch_weights.sh
# for the per-set sizes, why it is a selector rather than `hf download <repo>`, and the two
# huggingface_hub 1.x behaviour changes it works around.
WEIGHTS="$WEIGHTS" bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/fetch_weights.sh"

step "done"
df -h --output=avail "$ROOT" | tail -1
du -sh "$HF_HOME" "$SGLANG_DIFFUSION_CACHE_ROOT" 2>/dev/null
echo "next, whichever matches WEIGHTS=$WEIGHTS:"
echo "  vdn    -> bash $VDNROOT/sglang_arm.sh serve 480          (RUNBOOK 1c)"
echo "  t2va   -> bash $VDNROOT/sglang_base_arm.sh serve 480     (RUNBOOK 1f)"
echo "  ref2va -> bash $VDNROOT/sglang_ref2va_arm.sh serve 768   (RUNBOOK 1g)"
