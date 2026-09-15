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

Nine patches, in `patches/`. All are needed; none is upstream. They fall into four groups.
Patches 1 and 6 are about the **workload** — 480P does not exist upstream, in the renderer
or in the keyframe encoder. Patches 2 and 3 are the same underlying fact showing up at two
different moments: **80 GiB is not 141 GiB**, and Ulysses replicates the whole DiT on every
rank. Patches 4, 5, 7 and 9 are about the **other half of the latency** — once the denoise is
fast the decode is the bigger number, and each one attacks whatever the previous one left
largest: measure the stages, then the video VAE on one rank, then the colour conversion in
swscale, then libx264 itself. Patch 8 is a **test**, and it is here rather than folded into
patch 5 because finding out *why* the obvious test could not work is most of what it says.

| # | patch | why |
|---|---|---|
| 1 | `0001-vdn-render-resolution-as-a-config-field.patch` | **480P does not exist upstream.** `src/inference/render.py` hardcodes `LATENT_H, LATENT_W = 48, 84` — the 768x1344 canvas — as a module constant. Everything downstream is already resolution-agnostic (the softmax window is per *frame*, the layout carries its own spatial grid, Ulysses shards by row), so this only lifts the constant into `render.height` / `render.width` and threads it through both entrypoints, with a `latent_grid()` that rejects anything not a multiple of 32 (16x VAE, then the 2x2 transformer patch) |
| 2 | `0002-vdn-assemble-and-quantise-on-the-host-so-fp8-fits-an.patch` | **fp8 assembly does not fit 80 GB.** Ulysses shards the sequence, not the weights, so every rank replicates the whole DiT: 63 GiB in bf16, ~78 GiB once the hybrid branch and the two `stage-dmd-step-250` LoRAs are merged. `convert_linear_to_fp8` then needs one Linear's bf16 weight *and* its fp8 copy alive together — about 1 GiB more than an H100 has. It fits a 141 GiB H200, which is why upstream never hit it. The patch assembles and quantises **on the host** and moves only the finished model to the card. Order is untouched (transform → branch → LoRA → fp8) and every step is elementwise or a small matmul, so the weights come out identical; only the device the arithmetic ran on differs. Without fp8 there is nothing to gain, so the released bf16 path is left exactly as it was |
| 3 | `0003-vdn-never-keep-the-DiT-and-the-decoders-on-the-card-.patch` | **the decoders do not fit next to a 768p render.** `install_ulysses` gives every rank the same transformer (55.5 GiB in fp8 at 768p) but only the main rank loads the video and audio VAEs, and those are ~11 GiB. That leaves rank 0 about 13 GiB for activations where a block wants 10.33 GiB plus everything already live — so **rank 0 dies and ranks 1–7 finish**. The decoders are idle during the loop and the DiT is idle during the decode, so the decoders arrive *after* the loop and — at 768p only, behind `parallel.offload_transformer_before_decode` — the DiT leaves *before* the decode. Both moves are timed separately (`decoder_load_seconds`, `transformer_offload_seconds`) so neither hides in the denoise. The offload is ~27 s of PCIe, which is why it is a switch: 480p has the headroom and leaves it off |
| 4 | `0004-vdn-time-the-decode-stages-separately.patch` | **"decode" was one 14-second number.** At 480p the decode outweighs the denoise it follows, so it needs a breakdown before it can be attacked: video VAE / audio VAE / frames-to-host / mux, each device-synchronised. It also prints the s/NFE as soon as the loop ends rather than only in the final summary, so a render that dies in the decode still reports the number it was run for |
| 5 | `0005-vdn-data-parallel-video-VAE-decode-across-the-Ulysse.patch` | **the video VAE ran on rank 0 while seven cards waited.** Only the *encoder* is a causal 3D CNN; the **decoder is a non-causal ViT** (36 layers, 32x64 heads, hidden 2048), so no conv cache is threaded between temporal chunks and `_decode_clip` is a pure function of its 7-latent-frame slice. The only coupling anywhere is `_blend`, a linear cross-fade in pixel space after every forward has finished. That makes it **DP, not TP** — see below — and the patch spreads the 20 temporal chunks over the ranks for one `all_gather`, bit-identical output, behind `parallel.parallel_vae_decode` |
| 6 | `0006-vdn-let-encode_keyframes.py-target-a-canvas-other-th.patch` | **fl2va could not be measured at 480p.** A keyframe cache is only usable at the canvas it was encoded for — `condition_latents` come out at that canvas's latent size and the layout reserves rows to match, so the released `example_fl2va.pt` (`(1, 24, 1, 48, 84)`) fits a 1344x768 render and nothing else. `encode_keyframes.py` derives the canvas from the first keyframe's aspect ratio under the released rule (short edge 768, `768*1344` max pixels, both edges a multiple of 32), with no way to ask for another one; those stay the default and `--height` / `--width` override them. Note that exposing the *rule*'s parameters instead would not have worked: at short edge 480 the pixel cap scales to `768*1344*(480/768)² = 403,200`, below `480*864 = 414,720`, so the cap would have quietly produced an 832-wide canvas rather than 864 |

| 7 | `0007-vdn-convert-RGB-to-YUV420p-on-the-GPU-instead-of-in-.patch` | **the mux became the biggest single item.** Once patch 5 took the video VAE to 2.15 s, the 2.70 s mp4 mux was the largest thing left in a 480p render, and it is entirely host-side. Two things in it are avoidable: swscale's `rgb24→yuv420p` conversion, and the fact that PyAV leaves `thread_count` at 1. The pixels are already on the card as float32 in [0,1] and the conversion is a 3×3 matrix plus a 2×2 average, so it happens there; that also halves the host copy, yuv420p being 1.5 bytes/px against rgb24's 3 — measured 0.42 → 0.27 s. Matched-config, the mux goes 2.70 → 1.85 s at 480p and 4.32 → 2.39 s at 768p. **Of that 0.85 s, ~0.40 is the thread setting and ~0.45 the conversion** — an earlier version of this table said the conversion alone was 2.03 s, which came from timing `frame.reformat()` per frame in a loop (a fresh frame allocated each call); the real remainder, 1.85 s, is libx264, which is patch 9's problem. It re-implements a lossy conversion, so it sits behind `render.gpu_color_convert` and is verified numerically rather than assumed — `mux_bench.py --verify` reports Y within 1 level (71.42 dB) and chroma at 57.40/58.79 dB against swscale's own output on real render frames |
| 8 | `0008-vdn-test-the-parallel-VAE-decode-against-the-serial-.patch` | **the correctness test for patch 5 was measuring the wrong thing.** The grid rendered a prompt twice, serial and parallel, and `cmp`'d the mp4s; it reported `MISMATCH`. But this pipeline does not reproduce run to run — two renders at the same seed and config sit ~17 dB apart with no bit-identical frame — so that diff compares two denoise trajectories and fails whatever the decoder does. The patch tests the decode where it lives: one process, one latent tensor, both paths back to back |
| 9 | `0009-vdn-encode-the-mp4-in-parallel-segments-instead-of-o.patch` | **libx264 does not parallelise itself here.** With the conversion gone the mux *is* x264, and x264's own frame threading buys 1.12× on this canvas — 345 frames of 864×480 go 251 fps at `thread_count=0` against 224 at 1, on a box with 192 vCPUs — because the serial part is the per-frame Python plane writes, which hold the GIL. So the clip is split into `render.encode_segments` contiguous ranges, encoded concurrently in threads, and the bitstreams concatenated by copy: each segment opens with an IDR and references nothing outside itself, and mp4 carries one SPS/PPS in `avcC`, which is asserted identical rather than assumed. **4 segments is 3.2× (1.37 → 0.43 s) for +2.5 % bits and −0.38 dB.** Not free, but a better trade than the alternative of a faster preset, which is 1.9× for −2.8 dB *and* fewer bits. `scripts/clipinfo.py` is the structural check — every frame decoded, timestamps a clean run, audio present — because a PSNR against another render cannot see a concatenation bug through the run-to-run divergence |

The two OOM failures, for the record. Patch 2's, during assembly:

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 294.00 MiB. GPU 5 has a
total capacity of 79.18 GiB of which 24.06 MiB is free.
  File "src/models/ops/fp8_linear.py", line 263, in __init__
```

Patch 3's, at 768p, on rank 0 only, after patch 2 had already done its job:

```
[rank0]: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 10.33 GiB. GPU 0
has a total capacity of 79.18 GiB of which 8.69 GiB is free.
  File "src/models/ops/fused_block.py", line 106, in fast_block_forward
```

480p needs only patches 1 and 2 to *run* — it has the headroom. 768p needs 3 as well.
Patches 4 and 5 are about latency, not fitting, and they matter most at 480p.

**H100 is sm90, the same compute capability as H200**, so every *kernel* choice in
upstream's H200 config carries over untouched: `softmax_backend: flex` is the FA4-CuTe
static variant either way, and fp8 e4m3 uses rowwise scale granularity on both. What
differs is HBM (80 vs 141 GB) and bandwidth (3.35 vs 4.8 TB/s) — the first is patch 2,
the second is what the control arm measures.

## Geometry

`align_num_frames(n, 17, 5)` snaps the frame count to the VAE's 17-frames-to-5-latents
chunking, so the reachable clip lengths are **5 + 17k frames** — and around 15 s that is
345 (14.375 s) or 362 (15.083 s), with **nothing in between**. 15.000 s is not a length
this model can produce.

Both are measured. 345 is the one to quote against upstream's published number, because
it holds the clip length at their 14.375 s and moves only the canvas; 362 is the answer
when the requirement is literally "15 seconds".

| | canvas | latent | tokens/frame | frames | latent frames | video rows |
|---|---|---|---|---|---|---|
| upstream headline | 1344x768 | 84x48 | 1008 | 345 (14.375 s) | 102 | 102,816 |
| **primary** | **864x480** | **54x30** | **405** | **345 (14.375 s)** | **102** | **41,310** |
| also measured | 864x480 | 54x30 | 405 | 362 (15.083 s) | 107 | 43,335 |

2.49x fewer video rows at the same clip length, 2.41x counting the 1299 text and 1150 audio
rows that do not shrink.

It was tempting to predict *more* than 2.41x. The window softmax costs
`O(rows x window_frames x tokens_per_frame)` and 480p cuts both factors, so that branch
alone should gain ~6x; the linear branch and the MLPs are `O(rows)`. **Measured, it is
less, not more**: 2.702 → 1.302 s/NFE at standard Ulysses is **2.08x**, below the row
ratio. Each canvas at its own best split — 2.470 (768p, 5+3) → 1.090 (480p, 3+5) — is
**2.27x**, still below 2.41x. The softmax branch is simply not the dominant term at these
sizes, and what is left over is per-NFE fixed cost — the per-block Ulysses all-to-alls, the
kernel launches, the scheduler step — which shrinks with the canvas much more slowly, if at
all. Predicting the shape of a curve from its asymptotics
is how you get this wrong; `RESULTS.md` has the measurements.

The branch split is the one place the canvas shows up *qualitatively* rather than as a
factor: 480p's optimum is **3** softmax ranks and 768p's is **5**. That is the softmax
branch's extra `tokens_per_frame` factor being visible in exactly the way the cost model
says it should be — 768p has 2.49x more softmax work per row to spread, so it wants more
ranks to spread it over — even though the *total* time does not follow the cost model.

## What the latency actually is

The number asked for is **post-warmup end-to-end**: with the process already up and the
kernels compiled, how long from "go" to a finished mp4. That is not the denoise. Upstream's
published 18.3 s is denoise only (`denoise_seconds / num_steps x num_steps`), and it is the
right thing to publish for a kernel comparison, but it is roughly half of what a request
costs. The four stages, at 480p / 345 f / 8 NFE / 8 GPUs:

| stage | on how many ranks | 480p, before patch 5 | 480p, after | 768p (standard) |
|---|---|---|---|---|
| denoise, 8 NFE | 8 | 8.72 s | 8.72 s | 21.62 s |
| transformer offload to host | 1 | 0 (off) | 0 (off) | ~27 s |
| decoder load, NVMe → GPU | 1 | 2.53 s | 3.65 s | 2.52 s |
| video VAE | 1 → **8** | ~11.23 s | **2.97 s** | 25.08 s (with the rest) |
| audio VAE + frames-to-host + mux | 1 | (in the 14.38 s) | 3.18 s | ↑ |
| **post-warmup E2E** | | **25.63 s** | **18.53 s** | **49.21 s** |
| **steady state** (decoders resident) | | **23.10 s** | **14.88 s** | **46.69 s** |

Decoder load is once per **process**, not once per request, so the steady-state
per-request figure excludes it. Everything else is per request. (The 768p row shows no
offload because that control arm predates the offload existing; the ~27 s is what the arms
that do carry it measure.) `RESULTS.md` carries the full table; the point here is the shape:

**At 480p the decode was bigger than the denoise, and none of it was parallel.** Eight cards
spent 8.7 s working and then one card spent 14.4 s working while seven sat at a barrier.
Patch 5 takes the video VAE — the large majority of that — to all eight ranks, and the
steady-state request drops from 23.1 s to **14.9 s against a 14.375 s clip, i.e. real
time**.

What is left is more interesting than what was fixed: the biggest single item after the
denoise is now the **H.264 mux at 2.66 s**, which is host-side libx264 and touches no GPU.
The GPU-side decode is 2.97 s. Attacking the decode further means attacking a video encoder,
not a model.

## The VAE decoder: DP, not TP

The instinct with a big module and eight idle cards is tensor parallelism. That is the wrong
answer here, and the reason is in the checkpoint's own structure.

`AutoencoderKLMiniMaxH3` is **asymmetric**. The encoder is a causal 3D CNN — it *does* thread
a conv cache from one temporal chunk to the next, and it would be genuinely serial. The
**decoder is a non-causal ViT**: 36 layers, 32 heads x 64 dim, hidden 2048, no cache, no
recurrence. So `_decode_clip(z[:, :, start : start + tokens_chunk_size + token_overlap])` is
a **pure function of its slice**. The spatial tiles inside a clip are independent too. The
only thing that couples the pieces is `_blend`, a linear cross-fade in pixel space, applied
*after* every forward has finished.

So the decode is not one big op, it is a grid of independent ones. At 345 frames, with the
derived geometry (`clip_length` 17, `token_drop` 3 → `tokens_chunk_size` 5,
`token_overlap` 2, `frame_overlap` 5, `chunk_num_frames` 20) and tiling on by default
(256x256 px tiles, 64 px minimum overlap):

| canvas | chunks | tiles/chunk | decoder calls | tile area / frame area |
|---|---|---|---|---|
| 480p | 20 | 15 | **300** | 2.37x |
| 768p | 20 | 28 | **560** | 1.78x |

The call ratio 560/300 = 1.87x against the measured decode-time ratio 25.1/13.8 = **1.82x**
— the cost tracks the call count, which is what says the grid is the right model of it.

Given that:

* **TP** would shard a 2048-wide ViT and pay an all-reduce *inside each of 300 tiles*.
* **DP** pays **one** all-gather for the whole decode.

DP, then. The unit is the **chunk**, not the (chunk, tile) pair: chunks give 20/⌈20/8⌉ =
6.67x of ideal parallelism against pairs' 300/⌈300/8⌉ = 7.89x, but pairs ship overlapping
tiles — 2.37x the pixels at 480p — for about a fifth of a second, and chunks leave upstream's
tile stitching completely untouched. Output is bit-identical: the same slices through the
same `_decode_clip`, then upstream's own blend chain on rank 0 — checked by
`scripts/decode_parity.py`, which decodes one latent tensor both ways in one process. It
cannot be checked by diffing two renders; see below for why.

Worth naming the redundancy while it is on the table: at 480p the tiles cover **2.37x** the
frame area, because a 864x480 frame is 4x2 tiles of 256x256 with 64 px of forced overlap on
a canvas that barely needs tiling at all. Turning tiling off at 480p is a separate, larger
win than parallelising it, and it is orthogonal to patch 5.

## The same seed does not give the same video, so identity is tested on latents

Worth knowing before trusting any before/after comparison of *output* on this pipeline.
Rendering the same prompt twice at the same seed, same config and same split gives **~17 dB
PSNR between the two clips** and not one bit-identical frame out of 345 (480p 3+5 twice:
17.55 dB; 768p standard Ulysses twice: 16.69 dB).

The seeding is fine — an explicit `torch.Generator(device).manual_seed(seed)`, three draws in
a fixed order, the same on every rank. **One GPU is exactly reproducible**: two `infer.py`
runs give byte-identical mp4s, which rules out the fp8 GEMMs, the kernels and the sampler,
since the single-GPU path uses all of them too.

The cause is upstream's asynchronous frame mean,
`UlyssesRuntime.video_frame_mean_async` (`src/inference/utils/ulysses_runtime.py:497-520`, not
touched by any patch here), which the multi-rank path calls and `infer.py` never does. It
accumulates with `index_add_` — CUDA atomics, so the addition order varies — and then sums
across ranks with `dist.all_reduce`, whose order follows NCCL's algorithm choice for that run.
Both give ULP differences in a value that normalises the linear branch for *every* frame, and
eight sampler steps amplify them. Ulysses' other collectives are all-to-alls, which are
permutations and therefore exact, which is why this is the only candidate.

It is nonetheless the **same video**: the temporal cross-correlation peaks at shift 0 with a
sharp peak, and 48× downsampled PSNR reaches only 24.90 dB where independent high-frequency
noise would have gained ~33 dB. Scene, composition and motion agree; the detail differs
everywhere.

The consequence is procedural. `scripts/h100_grid.sh` phase `v` used to `cmp` a serial render
against a parallel one and call a mismatch a decode bug — that test could only ever fail,
because it compares two denoise trajectories. It is replaced by `scripts/decode_parity.py`
(patch 8): one process, one latent tensor, decoded both ways back to back. Run that way the
parallel decode is **bit-identical at both canvases**, max absolute difference 0.0, so patch
5's claim holds and the `MISMATCH` was the test. The same reasoning is why patch 7's colour
conversion is verified by `mux_bench.py --verify` against swscale on *fixed frames* rather
than by diffing two renders, and why patch 9's segmented encode is checked two ways that do
not involve a second render — PSNR against the input planes, and `scripts/clipinfo.py` for
the structural properties a PSNR cannot see (frame count, monotone evenly-spaced timestamps,
audio present).

If reproducibility ever matters more than the ~0.5 % of a step that the async frame mean
saves, the fix is in that one function: replace `index_add_` with a fixed-order segment sum
(the video rows of a frame are contiguous, so it is a reshape and a sum, not a scatter) and
the `all_reduce` with a gather plus a local sum in rank order.

## Text encoding is not in any of these numbers

Neither here nor upstream. `src/inference/encode_prompt.py` runs the conditioner **once,
offline, on one GPU** and writes `prompt_embeds` to a `.pt`; `load_prompt()` just
`torch.load`s it. Upstream's 18.3 s excludes it for the same reason. A prompt-to-video
service cannot, so the size of the thing matters:

* `Qwen3VLForConditionalGeneration`, 64 text layers, hidden 5120, intermediate 25600,
  64 heads / 8 KV, vocab 151936, vision depth 27 → **~31.5 B params = 63 GB in bf16**.
* VDN reads **`hidden_states[50]`**, which HF fills with the *input* to layer 50. So layers
  52–64 (13 of 64) and the 5120x151936 LM head never contribute: **~49 GB is what has to be
  resident**, not 63.
* **Host offload is not viable.** Patch 3 measures the real rate for this kind of move:
  45.2 GiB in 26.69 s = **1.7 GB/s**, far off PCIe Gen5, because a state dict is thousands
  of separate unpinned tensors. ~49 GB round-tripping per request is ~30 s on top of a 25 s
  render. It more than doubles the latency.
* **Sharding is.** 8-way, the needed weights are **6.1 GiB per rank**, which sits next to the
  45.2 GiB fp8 DiT at 51.3 of 79.2 GiB. The forward is ~59 TFLOP for a ~1300-token prompt —
  about 0.04 s spread over eight cards. It is not the compute that is the problem, it is
  only ever the residency.

`scripts/text_encoder_bench.py` measures all of the above on the box rather than asserting
it.

## 2K: not a super-resolution model, and not open

Asked directly, because it changes the answer completely. From MiniMax's own model card:

> For H3's 2K-resolution output, instead of using a conventional dedicated super-resolution
> module, we use the H3 base model to regenerate its own low-resolution result through an
> in-context manner.

> **Due to the complexity of the system, this module is not yet open-sourced.**

So: **not** a separate SR network, **not** any built-in GPU capability (no DLSS-style
anything), and **not** runnable here. The full system is three modules — H3-Context-IR →
H3-Base (768p) → H3-Regenerate-2K — and only H3-Base is released. 2K is reachable only
through the `/video-generation-v2-regeneration` API. The open checkpoint is additionally
hard-capped at `canvas_max_pixels = 768*1344` with a 768 short edge.

**2K here means 2560x1440.** Not inferred — MiniMax publish reference outputs at both
resolutions (`assets/t2va_2k.mp4`, `assets/t2va.mp4`); probed, they are 2560x1440 and
1344x768, both 243 frames at 24 fps. Note that this is not a pure rescale: 1344x768 is
1.750:1 and 2560x1440 is 1.778:1, so the regeneration slightly recomposes the frame too.

That gives latent 160x90 → 80x45 patched = **3600 tokens per frame, 3.571x** the 768p
canvas's 1008, and 3.571x the pixels for the VAE to decode. What can be measured on this
box is what **one denoising pass over a 2560x1440 canvas** costs on this DiT
(`configs/8nfe_2k_ulysses_h100.yaml`, phase `k`). Treat it as a **lower bound** on the
regeneration pass, for two reasons: regeneration additionally carries the 768p clip's
~103k rows as context, and the arm is out of distribution so its output is noise — only the
timing means anything.

Measured, and then stopped there, since the real module cannot be run:

| frames | result |
|---|---|
| 345 | **OOM in the denoise** — rank 0 asks for **63.61 GiB in one allocation** with 11.94 GiB free |
| 243 | **OOM** the same way (ranks 6/7 first, 1.68 GiB short) |
| 90 | fits: **3.58 s/NFE**, denoise 28.63 s, decode+encode 4.27 s |

The wall is in the **denoise, not the decode**, so patch 5 — the thing that fixed the 480p
and 768p latency — is not what 2K needs. And 3.58 s/NFE for 90 frames means the denoise for
3.75 s of video is already 3.3x the whole 480p/345f denoise. Phase `k` is left in the driver
with these numbers written into it, and out of every default `PHASES`.

## Conditioning: t2va vs fl2va vs ref2va

Everything above is **t2va** — text to video+audio, `prompts/example_2.pt`.

**fl2va** (first/last-frame) and **i2va** run on the same fast path. The conditioning rows a
keyframe cache adds sit *outside* the layout's video span, so they shard, gather and attend
as ordinary global rows and no collective changes shape. **Measured, fl2va costs ~9 % more
than t2va — +8.9 % at 768p, +8.7 % at 480p** (`RESULTS.md` has the table).

The cost is not just the extra rows, and the two effects pull in opposite directions:

* fl2va adds only **+4.0 %** rows (768p: 105,265 → 109,467; 480p: 43,759 → 45,549), but
  costs +8.9 %. The amplification is 2.2x because the text and condition rows are *global* —
  attended densely in both directions by the window softmax, skipped entirely by the linear
  branch — and 1299 → 4295 dense rows is **3.3x**.
* But the overhead is **proportional, not additive**: the vision rows scale with the canvas
  (2016 → 814) and so do the condition latents (2016 → 810). So the 2.2x amplification is
  canvas-invariant, and the smaller canvas does *not* pay relatively more. I predicted it
  would and was wrong — see `RESULTS.md`.

Rendering fl2va at 480p needs a keyframe cache encoded at 864x480, since
`condition_latents` come out at the canvas's latent size. That is patch 6.

### `reference_image_short_edge` is the biggest knob in ref2va, and it is 2048

This only affects **ref2va**. It is worth stating explicitly because the two paths look
similar and are not:

| path | rule the image is resized by | rows per image |
|---|---|---|
| fl2va / i2va | the **canvas** rule — short edge 768, area capped at `768*1344`, both edges a multiple of 32. The same rule the output video follows | 768p: 1,008. 480p: **405** (verified: patch 6 writes `condition_latents` of `(1, 24, 1, 30, 54)` at 864x480) |
| ref2va | `reference_image_short_edge`, **2048** for the released checkpoint — its own short edge, **no area cap, upscaling included** (the code says so in as many words) | 16:9 → 3648x2048 → **7,296** |

So one ref2va reference costs **18x** what an fl2va keyframe costs at 480p. Rows go as the
*square* of the short edge (16x VAE, then the 2x2 patch), so halving it saves 4x:

| reference | short edge 2048 | short edge 1024 |
|---|---|---|
| 16:9 (1920x1080) | 3648x2048 → **7,296 rows** | 1824x1024 → **1,824** |
| 1:1 | 2048x2048 → **4,096** | 1024x1024 → **1,024** |
| 4:1 (the aspect limit) | 8192x2048 → **16,384** | 4096x1024 → **4,096** |

Against a 480p/345f t2va sequence of 43,759 rows:

| references | rows added at 2048 | at 2048 | at 1024 |
|---|---|---|---|
| 1 x 16:9 | 7,296 | **+16.7 %** | +4.2 % |
| 3 | 21,888 | **+50 %** | +12.5 % |
| 12 (`max_references`) | 87,552 | **+200 %** | +50 % |

**Twelve 2048 references are 87,552 rows — more than twice the 41,310 rows of the 480p video
itself.** And these are not ordinary rows: like fl2va's condition rows they sit outside the
video span, so the window softmax attends them densely both ways while the linear branch
skips them. The measured fl2va amplification is **2.1x** time per row, which extrapolates to
roughly +35 % denoise for a single 2048 reference at 480p versus +9 % at 1024.

Note the asymmetry that makes this *worse* at 480p than at 768p: reference rows do not scale
with the target canvas — `reference_image_short_edge` is independent of `canvas_short_edge` —
while the video rows drop 2.5x. So the references occupy a larger share of a 480p sequence.
Lowering the short edge to 1024 is the single largest ref2va lever, and it is a `ConfigSpec`
override, not a code change.

The row arithmetic above is exact, read off the resize code and confirmed against a real
480p encode. The *times* are extrapolations and are not measured, because:

**ref2va** does **not** run on this path. It exists in the patched diffusers
(`MiniMaxH3Ref2VASetupStep`, `Ref2VATextEncoderStep`, `Ref2VAReferenceEncoderStep`,
`Ref2VAPrepareLayoutStep`, `Ref2VAPrepareLatentsStep`, `Ref2VADenoiseStep`) but VDN's
Ulysses entrypoint inlines a sampler that only covers fl2va-style conditioning rows —
ref2va has its own denoise step with its own layout. It would need implementing, not
configuring. `Ref2VADenoiseStep` is one forward per step (no CFG doubling), so the
arithmetic per row is t2va's and the reference rows tabulated above are the whole story —
they are the dominant term, not the sampler.

## Layout

| path | what |
|---|---|
| `scripts/h100_bringup.sh` | bare-metal prep: uv + python 3.12, torch 2.13.0 **cu129**, flash-attn-4, patched diffusers, the 82 GB checkpoint onto the NVMe |
| `scripts/h100_grid.sh` | the measurement driver: control / baseline / `softmax_ranks` split sweep (both canvases) / GPU-count curve / step budget / parallel decode A-B / fl2va vs t2va / 2K |
| `scripts/runone.sh` | one arm, detached, with the environment this box needs — see trap 5 |
| `scripts/summarize.py` | the `*.inference.json` records → one markdown table |
| `scripts/text_encoder_bench.py` | what the Qwen3-VL conditioner would cost if it were in the request path |
| `scripts/p5.sh` | ssh/scp helper for the box |
| `scripts/sync_box.sh` | push the patched sources + configs + driver onto the box — see trap 7 |
| `configs/8nfe_480p_345f_ulysses_h100.yaml` | **the deliverable**: 480p, 345 frames (14.375 s), 8 NFE, fp8, 8 GPUs, 3+5 split, parallel decode |
| `configs/8nfe_480p_362f_ulysses_h100.yaml` | the same at 362 frames (15.083 s), the literal "15 second" ask |
| `configs/8nfe_768p_345f_ulysses_h100.yaml` | the control: upstream's published 768p shape, so the H100↔H200 gap is measured |
| `configs/8nfe_2k_ulysses_h100.yaml` | 2560x1440 — a lower bound on H3-Regenerate-2K, which is not open-sourced |
| `patches/` | the nine patches above + `BASE.txt` (the upstream commit they apply to) |
| `RESULTS.md` | the measured numbers |
| `samples/` | the renders the numbers came from, video+audio muxed, all t2va from `prompts/example_2.pt`. The current best config, with all nine patches: **`n_480p_seg4.mp4`** (864x480, 345 f), **`n_480p_362f_seg4.mp4`** (the literal 15 s, 362 f) and **`n_768p_seg4.mp4`** (1344x768, 345 f) — these three are the segmented-encode output, checked with `clipinfo.py`. Earlier renders kept for comparison: `vdn_*` (pre-patch-7 swscale mux), `z_*` (patches 1–6), `y_768p_345f_r5` (the 768p split winner), `f_*` (fl2va) |

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
5. **`torchrun` sets `OMP_NUM_THREADS=1`**, and says so in its own output. That is right
   for a GPU render, but patch 2's host-side LoRA merge and fp8 quantise are CPU work:
   at 1 thread all 8 ranks sat at 99% of a single core and were still going after six
   minutes. At `192 / nproc_per_node` threads the whole assembly finishes in ~200 s.
   `scripts/runone.sh` and `scripts/h100_grid.sh` both set it.
6. **Do not put `pkill -f torchrun` in an ssh command line.** `-f` matches the full
   command line, which includes the ssh command carrying the pkill, so it kills the
   caller — it comes back as exit 255 with no other explanation. `scripts/runone.sh`
   exists partly so the pattern never has to be typed remotely.
7. **The box's working tree is not a branch.** Patches 1–3 were applied there by hand before
   the series existed as commits, so `git am` refuses on top of them. `scripts/sync_box.sh`
   therefore syncs the **file set** the series touches, copied out of the patch workspace,
   which makes the box byte-identical to it rather than "probably up to date" — the only
   property worth having when `RESULTS.md` comes off that box.
8. **A stale record looks exactly like a fresh one.** The `c_*` arms have no
   `transformer_offload_seconds` key at all (they predate the offload), and the `b_*` arms
   have it at ~27 s because the offload was unconditional before it became a config field —
   480p now leaves it off. Their **denoise is comparable and their E2E is not**, which is not
   visible from the tag. That is why phases `f` and `z` re-run arms that already have
   records instead of reusing them, and why `summarize.py` prints `offload` as its own
   column rather than burying it in a total.

## Licence

The weights are under the **MiniMax H3 Community License Agreement**, whose applicable
territory **excludes the United States** (also the EU, UK and South Korea). This box is
`us-east-2`. That is a licence question for whoever ships this, not a technical one, but
it should not be discovered late — the benchmark numbers are unaffected either way.
