# MiniMax-H3 on 8x H100, as an image. Eleven lines of real content, because upstream already
# ships the hard parts -- see docker/h3.sh for the wrapper and RUNBOOK section 1 for the flow.
#
# WHY THIS IS A THIN OVERLAY AND NOT A Dockerfile OF OUR OWN.
# sglang's own docker/Dockerfile is 795 lines and builds CUDA 13.0.3 + python 3.12 + torch +
# sglang-kernel + flashinfer cubin caches + a Rust toolchain + ffmpeg, and `lmsysorg/sglang`
# publishes the result. Every trap sglang_bringup.sh works around is a trap that only exists
# because it installs into a bare DLAMI:
#   * "can't find Rust compiler" / SGLANG_BUILD_RUST_EXTS=none  -> the image has cargo
#   * deep_gemm's find_cuda_home assert, the lib64/-lcudart symlinks
#                                                              -> the image has real /usr/local/cuda
#   * apt is broken, so ffmpeg/ffprobe cannot be installed      -> the image has ffmpeg
#   * python 3.13 vs the cp312-only outlines_core wheel         -> the image is 3.12
# So the only thing left for us to do is the one thing the published image does NOT have.
#
# WHAT THE PUBLISHED IMAGE DOES NOT HAVE: this model. The image bundles the sglang *source* at
# /sgl-workspace/sglang but installs the LLM extras only, which is why upstream's own generated
# H3 command (docs/src/snippets/configs/MiniMaxAI/minimax-h3.jsx, `dockerRunCommand`) is
#     bash -lc 'python -m pip install -e "/sgl-workspace/sglang/python[diffusion]" && exec sglang serve "$@"' --
# i.e. it pip-installs the diffusion extra at *every container start*. That is a couple of
# minutes per launch and a different resolution each time. This image does exactly that, once,
# at build time -- plus the checkout below, because a published tag's bundled revision is
# whatever it was when the tag was built, and VDN is younger than that.
#
# WHY WE STILL PIN A SHA. Release 0.5.19 on PyPI has base MiniMax-H3 but zero occurrences of
# `vdn` or `hybrid_window_attn_h3`; on main VDN is a whole subsystem. RESULTS.md's numbers were
# measured against main at 3f8eb35e, and main moves dozens of commits a day, so an unpinned
# `git+main` means two boxes disagree and no number is comparable to the last one. The image is
# where that pin belongs.

ARG BASE=lmsysorg/sglang:dev
FROM ${BASE}

# The revision RESULTS.md was measured at. SGLANG_REV=keep uses the image's bundled tree
# instead, which is the right choice only when you are deliberately testing a newer main.
ARG SGLANG_REV=3f8eb35eadfb29ad98d7900910f19fafcbac5ccb
WORKDIR /sgl-workspace/sglang
RUN set -eux; \
    if [ "$SGLANG_REV" != keep ]; then \
      git fetch --depth 1 origin "$SGLANG_REV" || git fetch origin; \
      git checkout --detach "$SGLANG_REV"; \
    fi; \
    git log -1 --format='sglang source: %H %cd'

# `|| SGLANG_BUILD_RUST_EXTS=none`: setup.py shells out to cargo just to *discover* the Rust
# router extensions, so on an image whose final stage dropped the toolchain the install dies in
# get_requires_for_build_wheel before any Python compiles. Nothing in the diffusion path uses
# those modules. Upstream's own command omits the variable, so try it their way first and only
# fall back -- if the fallback is what runs, the build log says so.
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install -e "python[diffusion]" \
    || { echo "retrying without the Rust extensions"; \
         SGLANG_BUILD_RUST_EXTS=none python -m pip install -e "python[diffusion]"; }

# Fail the BUILD, not the 8-GPU launch 20 minutes later. The import is the fact: `sglang serve
# --help` adds the diffusion flags only after it has resolved --model-path, so grepping its help
# for --ulysses-degree comes back empty on a build that does have diffusion.
RUN python - <<'PY'
import importlib, sglang
for m in ("sglang.multimodal_gen.configs.pipeline_configs.minimax_h3",
          "sglang.multimodal_gen.configs.pipeline_configs.minimax_h3_vdn",
          "sglang.multimodal_gen.runtime.pipelines.minimax_h3_pipeline"):
    importlib.import_module(m)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3 import (
    reference_encoding as re_, constants as c)
print("sglang", sglang.__version__)
print("t2va/fl2va/ref2va + VDN: present")
print("reference short edge", re_.MINIMAX_H3_REFERENCE_IMAGE_SHORT_EDGE, "fps", c.MINIMAX_H3_SUPPORTED_FPS)
PY

# Everything the arms need at runtime is bind-mounted at its host path (see h3.sh), so nothing
# model-shaped is baked in: no weights -- a 200 GiB image is slower to pull than the weights are
# to download, and /opt/dlami/nvme is instance store either way.
ENV HF_HOME=/opt/dlami/nvme/vdn/hf \
    SGLANG_DIFFUSION_CACHE_ROOT=/opt/dlami/nvme/sglang/cache \
    NCCL_NET_PLUGIN=none \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
WORKDIR /opt/dlami/nvme/vdn
