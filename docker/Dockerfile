# MiniMax-H3 on 8x H100. THIS FILE IS AN ESCAPE HATCH, NOT THE WAY IN: the default route builds
# nothing and runs upstream's nightly image directly. `bash h3.sh probe` first, then serve.
#
# WHY THERE IS ALMOST NOTHING HERE. This started as an overlay that pip-installed the diffusion
# extra, on the belief that the published image carries the LLM extras only. That belief was
# wrong, and checking it is what deleted the layer:
#   * .github/workflows/release-docker-dev.yml runs on `cron: "0 0 * * *"` -- the `dev` tag IS
#     the x86 nightly, rebuilt daily from main, and it also publishes the immutable dated aliases
#     `nightly-dev-{date}-{short_sha}`. (`nightly-cu134` is a different thing: arm64-only Rubin.)
#   * both docker workflows pass `--build-arg BUILD_TYPE=all`, python/pyproject.toml's
#     `all = ["sglang[diffusion]", "sglang[http2]", "sglang[tracing]"]`, and stage `torch_deps`
#     installs `".[all]"` *with* dependencies (Dockerfile:216). So diffusers==0.37.0, av,
#     cache-dit, st_attn, vsa, opencv-python-headless, moviepy, nvidia-modelopt are all in there.
#   * `--build-arg BRANCH_TYPE=local` makes framework_final `COPY . /src` into
#     /sgl-workspace/sglang, and .dockerignore excludes only .gitignore -- so the image carries
#     the full checkout *including .git*, at exactly the sha in its tag, editable-installed with
#     `pip install --no-deps -e "python[all]"` (Dockerfile:584).
# A nightly is therefore already a pinned, diffusion-capable, VDN-capable image. Upstream's own
# H3 recipe still pip-installs `[diffusion]` at every container start
# (docs/src/snippets/configs/MiniMaxAI/minimax-h3.jsx) -- against a nightly that is redundant.
#
# SO WHAT IS THIS FOR. Exactly one thing the nightly cannot give you: a revision that is not a
# midnight one. The nightlies pin main's head at 00:00 UTC; RESULTS.md was measured at 3f8eb35e,
# a mid-day commit on 2026-09-16, which no tag names. Build this only when an exact sha matters
# (bisecting a regression, or reproducing a number against the commit it was measured at).
# Otherwise pin the base by digest and skip the build -- `h3.sh probe` prints the digest.

ARG BASE=lmsysorg/sglang:dev
FROM ${BASE}

# `keep` = use the image's own tree, i.e. add only the assertion below. Pass a sha to move:
#   SGLANG_REV=3f8eb35eadfb29ad98d7900910f19fafcbac5ccb bash h3.sh build
ARG SGLANG_REV=keep
WORKDIR /sgl-workspace/sglang
RUN set -eux; \
    if [ "$SGLANG_REV" != keep ]; then \
      git fetch --depth 1 origin "$SGLANG_REV" || git fetch origin; \
      git checkout --detach "$SGLANG_REV"; \
      # --no-deps is upstream's own form for this install (Dockerfile:584) and it matters: the
      # dependency set is already resolved and frozen into /sgl-workspace/constraints.txt, so a
      # plain `pip install -e` re-resolves ~60 pins against PyPI and can quietly swap torch or
      # diffusers under a checkpoint that was measured on the old ones. If the checked-out
      # revision *added* a dependency, the import assertion below is what tells you -- rerun
      # with `python -m pip install -e "python[all]"` then, deliberately.
      python -m pip install --no-deps -e "python[all]"; \
    fi; \
    git log -1 --format='sglang source: %H %cd'

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
