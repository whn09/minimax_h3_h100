# VDN-MiniMax-H3 on 8x H100 80GB — 480P, 15 s, minimum latency

The question: **how long does one 480P 15-second clip take on a `p5.48xlarge` (8x H100
80GB), using the VDN-tuned model** [`OpenVDN/vdn-minimax-h3`](https://huggingface.co/OpenVDN/vdn-minimax-h3)?

## Which stack this is — not SGLang

This is **not** the SGLang path used in `../minimax_h3_h200/` and `../minimax_h3_g7e/`.
Those serve **stock MiniMax-H3** plus the community
[`larryvrh/MiniMax-H3-Turbo-Lora`](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora).
The VDN checkpoint adds a **second attention branch** (frame-wise linear attention
alongside the window softmax) that SGLang has no code for, so SGLang cannot load it at
all — `linear_branch/model.safetensors` has nowhere to go.

The stack here is VDN's own research repo,
[`OpenVDN/vdn-minimax-h3`](https://github.com/OpenVDN/vdn-minimax-h3) (Apache-2.0, weights
under the MiniMax H3 Community License), which uses **patched diffusers** for the
transformer / VAE / scheduler classes but implements its own sampler, hybrid attention,
fp8 kernels and Ulysses sequence parallelism:

| path | entrypoint | what it gets you |
|---|---|---|
| **VDN Ulysses (used here)** | `src/inference/infer_ulysses.py` | the tuned kernels + fp8 + 8-GPU branch-parallel Ulysses. Upstream's published numbers are this path |
| VDN single GPU | `src/inference/infer.py` | same kernels, one GPU |
| plain diffusers | `src/inference/infer_diffusers.py` | `ModularPipeline`, one GPU, no tuned kernels — a "does it render" entrypoint |
| SGLang | — | **cannot load this checkpoint** |

The model is **`ckpts/stage-dmd-step-250`** = VDN-H3-8-step, the Stage-DMD distilled
`turbo` adapter. That is the fastest tier upstream ships and the one behind their headline
(768p / 14.4 s: **18.3 s on 8x H200**, 11.23 s on 8x B200, both at 8 NFE).

## What had to change for this box

Two patches, in `patches/`. Both are needed; neither is upstream.

| # | patch | why |
|---|---|---|
| 1 | `0001-vdn-render-resolution-as-a-config-field.patch` | **480P does not exist upstream.** `src/inference/render.py` hardcodes `LATENT_H, LATENT_W = 48, 84` — the 768x1344 canvas — as a module constant. Everything downstream is already resolution-agnostic (the softmax window is per *frame*, the layout carries its own spatial grid, Ulysses shards by row), so this only lifts the constant into `render.height` / `render.width` and threads it through both entrypoints, with a `latent_grid()` that rejects anything not a multiple of 32 (16x VAE, then the 2x2 transformer patch) |
| 2 | `0002-vdn-assemble-and-quantise-on-the-host.patch` | **fp8 assembly does not fit 80 GB.** Ulysses shards the sequence, not the weights, so every rank replicates the whole DiT: 63 GiB in bf16, ~78 GiB once the hybrid branch and the two `stage-dmd-step-250` LoRAs are merged. `convert_linear_to_fp8` then needs one Linear's bf16 weight *and* its fp8 copy alive together — about 1 GiB more than an H100 has. It fits a 141 GiB H200, which is why upstream never hit it. The patch assembles and quantises **on the host** and moves only the finished model to the card. Order is untouched (transform → branch → LoRA → fp8) and every step is elementwise or a small matmul, so the weights come out identical; only the device the arithmetic ran on differs. Without fp8 there is nothing to gain, so the released bf16 path is left exactly as it was |

The failure patch 2 fixes, for the record:

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 294.00 MiB. GPU 5 has a
total capacity of 79.18 GiB of which 24.06 MiB is free.
  File "src/models/ops/fp8_linear.py", line 263, in __init__
```

**H100 is sm90, the same compute capability as H200**, so every *kernel* choice in
upstream's H200 config carries over untouched: `softmax_backend: flex` is the FA4-CuTe
static variant either way, and fp8 e4m3 uses rowwise scale granularity on both. What
differs is HBM (80 vs 141 GB) and bandwidth (3.35 vs 4.8 TB/s) — the first is patch 2,
the second is what the control arm measures.

## Geometry

`align_num_frames(n, 17, 5)` snaps the frame count to the VAE's 17-frames-to-5-latents
chunking, so **15 s is `render.num_frames: 360` → 362 frames → 15.083 s** at 24 fps.

| | canvas | latent | tokens/frame | latent frames | video rows |
|---|---|---|---|---|---|
| upstream headline | 1344x768 | 84x48 | 1008 | 102 (14.375 s) | 102,816 |
| **this workload** | **864x480** | **54x30** | **405** | **107 (15.083 s)** | **43,335** |

2.37x fewer rows. The window softmax should gain more than that — its cost goes as
rows x tokens-per-frame-window, and 480p cuts both — while the linear branch is linear in
rows. So the 480p speedup should exceed the sequence-length ratio, and the arm mix in
`scripts/h100_grid.sh` is there to say by how much rather than guess.

## Layout

| path | what |
|---|---|
| `scripts/h100_bringup.sh` | bare-metal prep: uv + python 3.12, torch 2.13.0 **cu129**, flash-attn-4, patched diffusers, the 82 GB checkpoint onto the NVMe |
| `scripts/h100_grid.sh` | the measurement driver: control / baseline / `softmax_ranks` split sweep / GPU-count curve / step budget |
| `scripts/summarize.py` | the `*.inference.json` records → one markdown table |
| `scripts/p5.sh` | ssh/scp helper for the box |
| `configs/8nfe_480p_15s_ulysses_h100.yaml` | **the deliverable**: 480p, 362 frames, 8 NFE, fp8, 8 GPUs |
| `configs/8nfe_768p_144s_ulysses_h100.yaml` | the control: upstream's published 768p / 14.4 s shape, so the H100↔H200 gap is measured |
| `patches/` | the two patches above + `BASE.txt` (the upstream commit they apply to) |

## Traps found on this box

1. **`torchvision` must come from the cu129 index too.** `pyproject.toml` pins
   `torchvision==0.28.0` but deliberately does not pin torch, so `uv pip install -e .`
   takes torchvision from PyPI — which is cu130 — and every `import transformers` then
   dies with `PyTorch and torchvision were compiled with different CUDA major versions`.
   `scripts/h100_bringup.sh` installs it explicitly from `download.pytorch.org/whl/cu129`.
2. **Do not use the DLAMI's `/opt/pytorch`.** It is python 3.13 + torch 2.13.0+cu130;
   the repo requires `>=3.12,<3.13` and the cu129 wheels.
3. **Everything goes on `/opt/dlami/nvme`** (27 TB). `/` is 484 GB and the checkpoint
   alone is 82 GB.
4. **Host RAM is a real constraint with patch 2.** Assembling on the CPU means 8 ranks
   each hold ~78 GiB of bf16 weights at once — ~620 GB. Fine on `p5.48xlarge` (2 TB),
   but it would not fit a smaller box, and there the alternative is to stream the fp8
   conversion through the host one Linear at a time instead.

## Licence

The weights are under the **MiniMax H3 Community License Agreement**, whose applicable
territory **excludes the United States** (also the EU, UK and South Korea). This box is
`us-east-2`. That is a licence question for whoever ships this, not a technical one, but
it should not be discovered late — the benchmark numbers are unaffected either way.
