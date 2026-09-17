#!/usr/bin/env bash
# Sourced by every sglang_*_arm.sh. Nothing here is arm-specific; it is the set of things that
# have to be true before `import sglang.multimodal_gen` survives on this box, each of which cost
# a debugging session to find. Sourced, not executed -- it exports into the caller's shell.
#
# IT MUST WORK IN BOTH WORLDS. Two ways the server gets served now:
#   * the venv at $ROOT/.venv, built by sglang_bringup.sh          (RUNBOOK 1b-venv)
#   * inside lmsysorg/sglang, built by docker/Dockerfile           (RUNBOOK 1b, the default)
# In the container there is no .venv, there IS a real /usr/local/cuda, and ffmpeg is already
# installed -- so every step below is a conditional, and a container run is expected to take the
# short branch of all of them and change nothing.

# --- the interpreter -------------------------------------------------------------------------
# Activate the venv only if there is one. In the container sglang is already on PATH and the
# image's own /opt/sglang venv is active; sourcing a missing activate under `set -u` would abort.
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f "${ROOT:-/opt/dlami/nvme/sglang}/.venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "${ROOT:-/opt/dlami/nvme/sglang}/.venv/bin/activate"
fi
command -v sglang >/dev/null 2>&1 || {
  echo "no sglang on PATH, and no venv at ${ROOT:-/opt/dlami/nvme/sglang}/.venv."
  echo "either run docker/h3.sh (RUNBOOK 1b) or sglang_bringup.sh (RUNBOOK 1b-venv) first."
  exit 2
}

# --- CUDA_HOME -------------------------------------------------------------------------------
# deep_gemm's find_cuda_home asserts rather than falling back, which kills
# `import sglang.multimodal_gen` before any flag matters. Prefer a real system CUDA (the
# container's nvidia/cuda base image has one); on the bare DLAMI there is none, and the venv
# carries a full pip CUDA instead -- the VDN delta-factors kernel JITs against nvcc through
# apache-tvm-ffi, so it needs a real nvcc, not just a directory that exists. Find it by looking
# for bin/nvcc rather than by guessing the path: the current wheel layout is nvidia/cu13/, not
# the per-component nvidia/cuda_nvcc/ of the older ones, and a CUDA_HOME that merely exists
# satisfies deep_gemm's assert while leaving the JIT to fail later with something far less
# obvious.
if [ -z "${CUDA_HOME:-}" ]; then
  if [ -x /usr/local/cuda/bin/nvcc ]; then
    export CUDA_HOME=/usr/local/cuda
  else
    export CUDA_HOME=$(python - <<'PY'
import pathlib, sysconfig
nv = pathlib.Path(sysconfig.get_paths()["purelib"]) / "nvidia"
print(next((str(p.parent.parent) for p in sorted(nv.glob("*/bin/nvcc"))), ""))
PY
)
  fi
fi
[ -x "$CUDA_HOME/bin/nvcc" ] || echo "warning: no nvcc under CUDA_HOME=$CUDA_HOME; a JIT kernel will fail"

# The JIT link line is `c++ ... -L$CUDA_HOME/lib64 -lcudart`, which assumes a system CUDA
# install. A pip wheel ships neither: the directory is lib/, not lib64/, and it carries only the
# runtime soname libcudart.so.13, not the libcudart.so a -l flag resolves. nvcc compiles fine and
# then ld says "cannot find -lcudart", every rank dies, and the parent reports EOFError. Two
# symlinks close it -- cheaper than a system CUDA install, and scoped to the venv so the
# reference stack is untouched. A system CUDA already has lib64/, so this is a no-op there.
if [ -d "$CUDA_HOME/lib" ] && [ ! -d "$CUDA_HOME/lib64" ]; then
  ln -sfn lib "$CUDA_HOME/lib64"
  for so in "$CUDA_HOME"/lib/lib*.so.[0-9]*; do
    base=${so%%.so.*}.so
    [ -e "$base" ] || ln -sfn "$(basename "$so")" "$base"
  done
fi

# --- ffmpeg ----------------------------------------------------------------------------------
# H3's pipeline hard-requires both binaries and raises before it touches a GPU -- every rank dies
# with "missing executables: ffmpeg, ffprobe" and the parent shows only an EOFError from the pipe,
# which reads like a crash rather than a missing package. Checked here so the message is the
# message. The container has them (upstream's Dockerfile installs ffmpeg); on the bare DLAMI
# `apt-get install ffmpeg` first needs the broken developer.download.nvidia.com cuda-ubuntu2604
# source moved out of sources.list.d -- see RUNBOOK section 1.
for b in ffmpeg ffprobe; do
  command -v "$b" >/dev/null 2>&1 || { echo "missing $b -- sglang H3 refuses to start without it"; exit 2; }
done

# --- NCCL ------------------------------------------------------------------------------------
# NCCL_NET_PLUGIN=none, on an AWS box specifically. The DLAMI puts the EFA/OFI plugin on the
# system loader path (/etc/ld.so.conf.d/100_ofinccl.conf), so NCCL dlopens
# /opt/amazon/ofi-nccl/lib/libnccl-net.so during comm init. deep_ep's check_nccl_so() then scans
# /proc/self/maps for anything matching "libnccl", sees that plugin next to the venv's
# libnccl.so.2, and asserts "Duplicate NCCL runtime found" -- it is a plugin, not a second
# runtime, so the check is simply wrong here. It is fatal because sglang guards the deep_ep
# import with `except ImportError` and this is an AssertionError, so the MoE token dispatcher
# takes down a diffusion worker: the visible symptom is "Model architectures
# ['MiniMaxH3Qwen3VLEncoder'] failed to be inspected", two import layers away from the cause.
# Nothing here is multi-node; the plugin buys nothing on eight NVLinked cards in one box. Drop
# this before running anything across nodes. (A container without -v /opt/amazon does not see the
# plugin at all, but setting this costs nothing and keeps the two routes identical.)
export NCCL_NET_PLUGIN=${NCCL_NET_PLUGIN:-none}

# --- the allocator ---------------------------------------------------------------------------
# expandable_segments, because --quantization fp8 here is ONLINE quantization: the loader reads
# the 65.65 GiB bf16 checkpoint (measured from the safetensors headers: 1426 BF16 tensors + 13
# F32) onto the card and casts afterwards, so every bf16 block it frees leaves a hole the fp8
# weights cannot reuse. The first 480p attempt died with 51.89 GiB allocated and 18.70 GiB
# reserved-but-unallocated -- the fit was never 12 GiB short, it was fragmented by that much.
# Expandable segments let the allocator give the holes back instead of hoarding them. This is the
# cheapest memory lever and the only one that costs no latency, so it goes before
# --performance-mode auto and before any offload.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
