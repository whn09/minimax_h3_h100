# VDN-MiniMax-H3 on 8x H100 80GB — 480P, 15 s, minimum latency

The question: **how long does one 480P 15-second clip take on a `p5.48xlarge` (8x H100
80GB), using the VDN-tuned model** [`OpenVDN/vdn-minimax-h3`](https://huggingface.co/OpenVDN/vdn-minimax-h3)?

## Which stack this is — and the SGLang option, which has changed

> **Correction, and it is a load-bearing one.** This section used to read "SGLang cannot load
> this checkpoint — `linear_branch/model.safetensors` has nowhere to go." **That is no longer
> true.** SGLang's cookbook now has a
> [VDN-H3 section](https://github.com/sgl-project/sglang/blob/main/docs/cookbook/diffusion/MiniMax/MiniMax-H3.mdx#7-vdn-h3-hybrid-attention-8-step-distill):
> `--model-path OpenVDN/vdn-minimax-h3` with `--attention-backend hybrid_window_attn_h3`, which
> implements both branches and the gates, prefuses the linear branch and the 8-step DMD2 LoRA
> into the transformer on first launch, and hard-links the conditioner and VAEs from
> `MiniMaxAI/MiniMax-H3`. It serves `t2va` and `fl2va` and rejects `ref2va`, which matches this
> repo's own reading of the checkpoint.
>
> **And it is measured against this stack, on the same workload.** On 8× B200 at 345 frames /
> 1344×768, SGLang runs **0.88 s/NFE against the reference stack's published 1.40** — the
> cookbook's "OpenVDN reference" rows are `8nfe_tuned_fp8.yaml` + `infer_ulysses.py` with
> `parallel.softmax_ranks` swept, i.e. exactly what is measured below. On the per-channel fp8
> path an H100 would take, **0.98**. It credits the gap to parallel efficiency (86–91 % from 2
> to 8 cards against 56–64 %), with the two stacks within 4 % on a **single** card — so it is a
> scaling result, not a Blackwell-precision one, and it ought to carry here.
>
> **No H100 VDN number existed on either side of that comparison. Now one does — it is measured
> here, and SGLang wins.** On these eight H100s, ten requests each at 345 frames:
> **480p 8.02 s median** against this repo's 11.45 (**1.43x**) and **768p 19.04 s median** against
> 33.21 (**1.74x**), with text encoding *inside* SGLang's number and *outside* this repo's. The
> 768p memory worry was unfounded: peak 62.1 GB/GPU with the DiT, both VAEs and the conditioner
> all resident, so none of the offload ladder was needed — but
> `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` was, because online fp8 quantization of a
> 65.65 GiB bf16 checkpoint fragments the allocator by 18.7 GiB. Details, and the five unrelated
> errors between a fresh box and a working server, in
> [`RESULTS.md`](RESULTS.md#sglang-diffusion-is-faster-than-all-of-it-on-the-same-eight-cards).
>
> So the conclusion the correction was reaching for is the right one: the twelve patches below
> were the right way to learn *where the time goes* and the wrong way to *serve* it.

The stack measured in the rest of this document is **not** the SGLang path used in
`../minimax_h3_h200/` and `../minimax_h3_g7e/`. Those serve **stock MiniMax-H3** plus the
community [`larryvrh/MiniMax-H3-Turbo-Lora`](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora),
which is a different model, not a different runtime for this one.

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
| **SGLang Diffusion (fastest)** | `sglang serve --attention-backend hybrid_window_attn_h3` | an HTTP server, resident conditioner, and **8.02 s at 480p / 19.04 s at 768p measured on these 8x H100** — 1.43x / 1.74x over this table's stack, text encoding included. `scripts/sglang_arm.sh` |

The model is **`ckpts/stage-dmd-step-250`** = VDN-H3-8-step, the Stage-DMD distilled
`turbo` adapter. That is the fastest tier upstream ships and the one behind their headline
(768p / 14.4 s: **18.3 s on 8x H200**, 11.23 s on 8x B200, both at 8 NFE — and both
**denoise only**, by their own Results caveat; see "What the latency actually is").

## What had to change for this box

Twelve patches, in `patches/`. All are needed; none is upstream. They fall into five groups.
Patches 1 and 6 are about the **workload** — 480P does not exist upstream, in the renderer
or in the keyframe encoder. Patches 2 and 3 are the same underlying fact showing up at two
different moments: **80 GiB is not 141 GiB**, and Ulysses replicates the whole DiT on every
rank. Patches 4, 5, 7 and 9 are about the **other half of the latency** — once the denoise is
fast the decode is the bigger number, and each one attacks whatever the previous one left
largest: measure the stages, then the video VAE on one rank, then the colour conversion in
swscale, then libx264 itself. Patch 8 is a **test**, and it is here rather than folded into
patch 5 because finding out *why* the obvious test could not work is most of what it says. Patches 11 and 12 are about **the request after this one** — a one-shot render cannot tell you what a server costs, and when we finally measured ten requests in one process it corrected the published 480p figure by 1.75 s and falsified the 768p one outright.

| # | patch | why |
|---|---|---|
| 1 | `0001-vdn-render-resolution-as-a-config-field.patch` | **480P does not exist upstream.** `src/inference/render.py` hardcodes `LATENT_H, LATENT_W = 48, 84` — the 768x1344 canvas — as a module constant. Everything downstream is already resolution-agnostic (the softmax window is per *frame*, the layout carries its own spatial grid, Ulysses shards by row), so this only lifts the constant into `render.height` / `render.width` and threads it through both entrypoints, with a `latent_grid()` that rejects anything not a multiple of 32 (16x VAE, then the 2x2 transformer patch) |
| 2 | `0002-vdn-assemble-and-quantise-on-the-host-so-fp8-fits-an.patch` | **fp8 assembly does not fit 80 GB.** Ulysses shards the sequence, not the weights, so every rank replicates the whole DiT: 63 GiB in bf16, ~78 GiB once the hybrid branch and the two `stage-dmd-step-250` LoRAs are merged. `convert_linear_to_fp8` then needs one Linear's bf16 weight *and* its fp8 copy alive together — about 1 GiB more than an H100 has. It fits a 141 GiB H200, which is why upstream never hit it. The patch assembles and quantises **on the host** and moves only the finished model to the card. Order is untouched (transform → branch → LoRA → fp8) and every step is elementwise or a small matmul, so the weights come out identical; only the device the arithmetic ran on differs. Without fp8 there is nothing to gain, so the released bf16 path is left exactly as it was |
| 3 | `0003-vdn-never-keep-the-DiT-and-the-decoders-on-the-card-.patch` | **the decoders do not fit next to a 768p render.** `install_ulysses` gives every rank the same transformer (55.5 GiB in fp8 at 768p) but only the main rank loads the video and audio VAEs, and those are ~11 GiB. That leaves rank 0 about 13 GiB for activations where a block wants 10.33 GiB plus everything already live — so **rank 0 dies and ranks 1–7 finish**. The decoders are idle during the loop and the DiT is idle during the decode, so the decoders arrive *after* the loop and — at 768p only, behind `parallel.transformer_before_decode` — the DiT leaves *before* the decode. Both moves are timed separately (`decoder_load_seconds`, `transformer_release_seconds`) so neither hides in the denoise. Patch 10 makes the eviction cheap; this patch is what establishes that it has to happen at all, and 480p has the headroom and leaves it off |
| 4 | `0004-vdn-time-the-decode-stages-separately.patch` | **"decode" was one 14-second number.** At 480p the decode outweighs the denoise it follows, so it needs a breakdown before it can be attacked: video VAE / audio VAE / frames-to-host / mux, each device-synchronised. It also prints the s/NFE as soon as the loop ends rather than only in the final summary, so a render that dies in the decode still reports the number it was run for |
| 5 | `0005-vdn-data-parallel-video-VAE-decode-across-the-Ulysse.patch` | **the video VAE ran on rank 0 while seven cards waited.** Only the *encoder* is a causal 3D CNN; the **decoder is a non-causal ViT** (36 layers, 32x64 heads, hidden 2048), so no conv cache is threaded between temporal chunks and `_decode_clip` is a pure function of its 7-latent-frame slice. The only coupling anywhere is `_blend`, a linear cross-fade in pixel space after every forward has finished. That makes it **DP, not TP** — see below — and the patch spreads the 20 temporal chunks over the ranks for one `all_gather`, bit-identical output, behind `parallel.parallel_vae_decode` |
| 6 | `0006-vdn-let-encode_keyframes.py-target-a-canvas-other-th.patch` | **fl2va could not be measured at 480p.** A keyframe cache is only usable at the canvas it was encoded for — `condition_latents` come out at that canvas's latent size and the layout reserves rows to match, so the released `example_fl2va.pt` (`(1, 24, 1, 48, 84)`) fits a 1344x768 render and nothing else. `encode_keyframes.py` derives the canvas from the first keyframe's aspect ratio under the released rule (short edge 768, `768*1344` max pixels, both edges a multiple of 32), with no way to ask for another one; those stay the default and `--height` / `--width` override them. Note that exposing the *rule*'s parameters instead would not have worked: at short edge 480 the pixel cap scales to `768*1344*(480/768)² = 403,200`, below `480*864 = 414,720`, so the cap would have quietly produced an 832-wide canvas rather than 864 |

| 7 | `0007-vdn-convert-RGB-to-YUV420p-on-the-GPU-instead-of-in-.patch` | **the mux became the biggest single item.** Once patch 5 took the video VAE to 2.15 s, the 2.70 s mp4 mux was the largest thing left in a 480p render, and it is entirely host-side. Two things in it are avoidable: swscale's `rgb24→yuv420p` conversion, and the fact that PyAV leaves `thread_count` at 1. The pixels are already on the card as float32 in [0,1] and the conversion is a 3×3 matrix plus a 2×2 average, so it happens there; that also halves the host copy, yuv420p being 1.5 bytes/px against rgb24's 3 — measured 0.42 → 0.27 s. Matched-config, the mux goes 2.70 → 1.85 s at 480p and 4.32 → 2.39 s at 768p. **Of that 0.85 s, ~0.40 is the thread setting and ~0.45 the conversion** — an earlier version of this table said the conversion alone was 2.03 s, which came from timing `frame.reformat()` per frame in a loop (a fresh frame allocated each call); the real remainder, 1.85 s, is libx264, which is patch 9's problem. It re-implements a lossy conversion, so it sits behind `render.gpu_color_convert` and is verified numerically rather than assumed — `mux_bench.py --verify` reports Y within 1 level (71.42 dB) and chroma at 57.40/58.79 dB against swscale's own output on real render frames |
| 8 | `0008-vdn-test-the-parallel-VAE-decode-against-the-serial-.patch` | **the correctness test for patch 5 was measuring the wrong thing.** The grid rendered a prompt twice, serial and parallel, and `cmp`'d the mp4s; it reported `MISMATCH`. But this pipeline does not reproduce run to run — two renders at the same seed and config sit ~17 dB apart with no bit-identical frame — so that diff compares two denoise trajectories and fails whatever the decoder does. The patch tests the decode where it lives: one process, one latent tensor, both paths back to back |
| 9 | `0009-vdn-encode-the-mp4-in-parallel-segments-instead-of-o.patch` | **libx264 does not parallelise itself here.** With the conversion gone the mux *is* x264, and x264's own frame threading buys 1.12× on this canvas — 345 frames of 864×480 go 251 fps at `thread_count=0` against 224 at 1, on a box with 192 vCPUs — because the serial part is the per-frame Python plane writes, which hold the GIL. So the clip is split into `render.encode_segments` contiguous ranges, encoded concurrently in threads, and the bitstreams concatenated by copy: each segment opens with an IDR and references nothing outside itself, and mp4 carries one SPS/PPS in `avcC`, which is asserted identical rather than assumed. **4 segments is 3.2× (1.37 → 0.43 s) for +2.5 % bits and −0.38 dB.** Not free, but a better trade than the alternative of a faster preset, which is 1.9× for −2.8 dB *and* fewer bits. `scripts/clipinfo.py` is the structural check — every frame decoded, timestamps a clean run, audio present — because a PSNR against another render cannot see a concatenation bug through the run-to-run divergence |
| 10 | `0010-vdn-free-the-DiT-before-the-decode-instead-of-copyin.patch` | **the 768p eviction cost 26.33 s and bought nothing.** Patch 3 evicts the DiT before the decode because the two do not fit on an 80 GiB card — measured, `keep` OOMs 102 MiB short in the assembly. But it evicted by *copying to the host*, and `.to("cpu")` is thousands of separate pageable allocations at ~1.7 GB/s. Both modes free the same GPU memory, and nothing downstream of the denoise reads the weights, so the copy is pure cost. `parallel.offload_transformer_before_decode` (bool) becomes `parallel.transformer_before_decode` = `keep | free | offload`; `free` uses `to_empty(device="meta")` rather than `del`, since the sampler closure and the Ulysses hooks still hold references and dropping the attribute would free nothing. **2.81 s against 26.33 s**, taking the 768p request 56.29 → 32.32 s post-warmup and 53.10 → 29.50 s by E2E-minus-decoder-load, 1.80x, denoise unchanged |
| 11 | `0011-vdn-keep-the-DiT-resident-at-768p-by-never-building-.patch` | **the eviction was never the fix; rank 0's decode peak was.** Patch 3 evicts the DiT at 768p because `keep` OOMs by 102 MiB, and patch 10 only made the eviction cheap. Two allocations account for the peak and neither is necessary: the assembly's full-canvas fp16 RGB output (1.99 GiB at 345 f of 1344x768) plus its fp32 denormalisation (3.98 GiB), when the destination format is yuv420p at 1.5 bytes per pixel; and the all-gather destination, 24 pieces x 3 x 22 x 768 x 1344 = 3.045 GiB gathered at once. `parallel.stream_yuv_assembly` turns each chunk into its final planes as it leaves the assembly, and the gather goes **one slot at a time** (0.38 GiB) — which costs nothing, because slot `s` across the ranks in rank order *is* chunks `8s..8s+7`, exactly the order the blend consumes them in, so the gather is pipelined rather than merely split. Both bit-identical, asserted by `scripts/decode_parity.py` at both canvases (0 differing of 1,068,318,720 at 768p). 768p decode peak → **64.5 GiB reserved**, and the DiT stays |
| 12 | `0012-vdn-serve-N-requests-from-one-process-and-cycle-the-.patch` | **every latency figure here was one render with the decoder load subtracted off, and that is wrong in both directions.** `render.repeat=N` serves N complete requests from one warm process. It found that 480p steady state is **11.45 s ± 0.040, not 13.20** — a first request is slower in stages that have nothing to do with the decoder load (video VAE 2.12 → 1.55 s, tail 1.56 → 1.18 s) — and that patch 11's 768p residency **does not survive a second request**: the decoders load *after* the first denoise, so only later ones coexist with them, and request 2 dies 444–468 MiB short on the three linear-branch ranks. So one side has to cycle per request at 768p, and `parallel.vae_after_decode` picks the cheaper one: the video VAE is 9.70 GiB per rank against the DiT's 45.24, **4.7x less data**. Ten requests at 768p: 33.21 s ± 0.850, memory growth **+0.000 GiB** |

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
kernels compiled, how long from "go" to a finished mp4. **That is not what upstream
publishes**, and the difference is not small. Their Results section is explicit about it:

> We report steady-state **denoising** speed on the 768p, 14.4-second video generation
> workload [...] We **exclude model loading, warm-up, VAE decoding, and MP4 encoding**. For a
> live setup, we recommend running the text prompt rewriter, VAE decoding, and MP4
> conversion on separate machines, so the eight GPUs only denoise.

So `18.3 s` is `2.29 s/NFE x 8`, nothing else — the column header `8 NFE
(VDN-H3-8-step)` names the step count and the checkpoint, not a metric. That is the right
thing to publish for a kernel comparison, and it is the number to compare a `s/NFE` against.
It is also **about half of what one request costs**. Note the terminology collision too:
upstream's "steady state" means the denoise loop after warm-up, while "steady state" below
means the whole request with the decoders already resident.

480p / 345 f / 8 NFE / 8 GPUs / 3+5, as the patches landed:

| stage | on how many ranks | before patch 5 | after patch 5 | now (7, 9, 10) |
|---|---|---|---|---|
| denoise, 8 NFE | 8 | 8.72 s | 8.72 s | 8.55 s |
| release the DiT | 1 | 0 (`keep`) | 0 (`keep`) | 0 (`keep`) |
| decoder load, NVMe → GPU | 1 | 2.53 s | 3.65 s | 3.87 s |
| video VAE | 1 → **8** | ~11.23 s | **2.97 s** | 3.10 s |
| audio VAE + frames-to-host + mux | 1 | (in the 14.38 s) | 3.18 s | **1.55 s** |
| **post-warmup E2E** (request 1 of the process) | | **25.63 s** | 18.53 s | **17.07 s** |

Decoder load is once per **process**, not once per request, so a second request does not pay
it. Everything else is per request. That makes "steady state" a different number, and the
honest way to get it is to serve ten requests from one process (`render.repeat`, patch 12)
rather than to subtract the load from a single render — subtraction was how the earlier
**13.20 s** figure in this table came about, and it was 1.75 s too slow, because a first
request is systematically slower than a warm one in the decode and the tail as well:

| 480p, 345 f, requests 2–10 of one process | mean | sd | min | max |
|---|---:|---:|---:|---:|
| **request, end to end** | **11.45 s** | 0.040 | 11.40 | 11.51 |
| denoise, 8 NFE | 8.70 | 0.032 | 8.66 | 8.74 |
| video VAE, 8 ranks | 1.57 | 0.039 | 1.55 | 1.67 |
| audio VAE + frames-to-host + mux | 1.18 | 0.019 | 1.15 | 1.22 |

Ten requests, ±40 ms, and `reserved` grew 0.254 GiB in total over all ten and then stopped —
so it repeats. `RESULTS.md` carries the full tables and the 768p equivalent (**33.21 s ± 0.85**
steady, which also only became honest under `render.repeat`: the one-shot run implied 26.79 s
and the second request OOMed). The point here is the shape, and it changed twice:

**First, the decode was bigger than the denoise and none of it was parallel.** Eight cards
spent 8.7 s working and then one card spent 14.4 s working while seven sat at a barrier.
Patch 5 takes the video VAE to all eight ranks and the request drops to 14.88 s.

**Then the biggest item after the denoise was an H.264 encoder**, host-side libx264 touching
no GPU. Patches 7 and 9 take the tail 3.18 → **1.55 s** by doing the colour conversion on the
card and running four independent encoders instead of one. What is left in the tail is
0.26 s of AAC, 0.28 s of PCIe and 1.02 s of x264.

Following upstream's own advice — decode and mux on another machine — would take the tail off
these eight GPUs and raise their *throughput*, since the tail would overlap the next
request's denoise. It would make the *latency* asked for here worse, not better, because a
finished clip then has to cross a network before it exists.

At `p5.48xlarge` on a **3-year no-upfront Instance Savings Plan** — $23.77728/instance-hour,
$0.00660480/instance-second — those steady-state latencies price out at **$0.005261 per
finished video second at 480p** ($0.0756/clip, $18.94 per hour of video) and **$0.015259 at
768p** ($0.219/clip, $54.93/hour). 480p renders for *less* than the instance costs to run,
0.80x an instance-second per finished second. This is a latency-optimised price and roughly
2x the cheapest way to buy the same seconds — see RESULTS.md for the throughput alternative.
(The 768p figure is 13 % worse than the $0.013554 published here earlier, and that is not a
regression in the code: the earlier number came from a process that only ever served one
request, and 768p needs 6.87 s per request of decoder cycling to serve a second one.)

### What that means for an API, since `free` keeps no copy

`parallel.transformer_before_decode: free` is right for a one-shot 768p render and wrong for a
server, because `to_empty(device="meta")` leaves the weights nowhere: the next request has to
get 45.24 GiB back onto the card. Measured (`scripts/reload_bench.py`, restores verified
bit-identical over all 1802 tensors):

| getting the DiT back | seconds |
|---|---:|
| from a **pinned** host copy | **4.49** (10.08 GiB/s) |
| from a **pageable** host copy — what `offload` leaves | **10.53** (4.30 GiB/s) |
| rebuild from the checkpoint, one process | **138.4** |
| rebuild from the checkpoint, in the 8-rank job | **217–232** |

So `offload` is **strictly dominated**: 26.67 s to write a pageable copy plus 10.53 s to read
it back is 37.2 s, against 0.41 + 4.49 = **4.9 s** for `free` plus a retained pinned copy. And
even the good path is not worth wanting — ~5 s per request and 362 GiB of page-locked host
memory across 8 ranks, to recover the **102 MiB** the 768p decode was short of. For a server
the fix is rank 0's decode peak, not the DiT, and patches 11 and 12 are that fix: streaming
the assembly into yuv420p planes and gathering the parallel decode one round of chunks at a
time take the 768p peak to 64.5 GiB reserved, so `transformer_before_decode: keep` holds at
both canvases and nothing reloads 45.24 GiB ever again.

What patch 11 alone does **not** do is let 768p keep everything — an earlier draft of this
paragraph claimed it would, and `render.repeat=10` falsified that. The decoders are loaded
*after* the first denoise, so they never coexist with it, and every denoise after the first
one does: request 2 dies 444–468 MiB short on the three linear-branch ranks. Something has to
cycle per request at 768p, and the choice is settled by size — the video VAE is 9.70 GiB per
rank against the DiT's 45.24, so `parallel.vae_after_decode: free` cycles **4.7x less data**
than reloading the weights would. It costs 5.05 s of decoder load plus 1.82 s of release, and
a pinned host copy of those 9.70 GiB would restore in ~0.96 s at the measured 10.08 GiB/s —
768p at ~28 s instead of 33.21. That patch is designed and not built.

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
  64 heads / 8 KV, vocab 151936, vision depth 27 → **~31.5 B params, 62.1 GiB in bf16**.
* VDN reads **`hidden_states[50]`**, which HF fills with the *input* to layer 50. So layers
  52–64 (13 of 64) and the 5120x151936 LM head never contribute: dropping them leaves
  **48.9 GiB resident**, measured, and takes the forward 169 → **135 ms** at 1300 tokens.
* **The forward is free; only the residency is a problem.** 135 ms against an 11.45 s render.
* **Host offload is not viable, and the *unload* is what kills it.** 48.9 GiB to the host
  takes **29.6 s (1.65 GiB/s)** and 7.6 s (6.43 GiB/s) to come back — a 37 s round trip,
  three times the render. The slow direction is the one allocating pageable destinations,
  which is the same 1.7 GB/s patch 3 measured on the DiT's `to("cpu")`, and for the same
  reason: a state dict is thousands of separate unpinned tensors.
* **Sharding is.** 8-way, the needed weights are **6.1 GiB per rank**. That is not enough on
  its own to make it resident, though: the 480p pipeline already peaks at **63.1 of the
  65.3 GiB** a rank can give PyTorch (DiT 45.24 + video VAE 9.70 + audio VAE 0.56 +
  activations), so there are only ~2 GiB spare and the shard needs 6.1. The cheapest fit is
  to keep the shard in **pinned** host memory and upload it per request — 6.1 GiB at the
  measured 10.08 GiB/s is 0.61 s, so ~0.8 s including the forward, **+7 %** — because the
  expensive direction (the 1.65 GiB/s trip *to* the host) then happens once at startup
  instead of every request. Derived, not measured; RESULTS.md gives the alternatives.

Measured by `scripts/text_encoder_bench.py`; the table is in RESULTS.md.

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
| `scripts/reload_bench.py` | how long the DiT takes to come **back** after patch 10 frees it — the question a long-lived API has and a one-shot render does not |
| `scripts/decode_parity.py` | asserts patch 11's two memory changes are **bit-identical** to `vae.decode`, at both canvases, with fixed latents in one process — multi-rank denoise is not reproducible, so a whole-render A-B could not have shown this |
| `scripts/box_check.sh` | run **on the box**: is it in the state these numbers were measured in? Every line is something that has silently been wrong once — the two `+cu129` suffixes most of all |
| `scripts/sglang_bringup.sh` | the alternative runtime, in its own venv beside the reference stack, gated on whether the installed build actually has `hybrid_window_attn_h3` |
| `scripts/sglang_arm.sh` | `serve` / `bench` / `stop` one SGLang arm at 480 or 768, 345 frames, 1 warmup + 10 measured — the same post-warmup metric as `steady (2-10)` |
| `scripts/p5.sh` | ssh/scp helper for the box |
| `scripts/sync_box.sh` | push the patched sources + configs + driver onto the box — see trap 7 |
| `configs/8nfe_480p_345f_ulysses_h100.yaml` | **the deliverable**: 480p, 345 frames (14.375 s), 8 NFE, fp8, 8 GPUs, 3+5 split, parallel decode |
| `configs/8nfe_480p_362f_ulysses_h100.yaml` | the same at 362 frames (15.083 s), the literal "15 second" ask |
| `configs/8nfe_768p_345f_ulysses_h100.yaml` | the control: upstream's published 768p shape, so the H100↔H200 gap is measured. Also the only config that cycles anything per request (`vae_after_decode: free`) |
| `configs/8nfe_2k_ulysses_h100.yaml` | 2560x1440 — a lower bound on H3-Regenerate-2K, which is not open-sourced |
| `patches/` | the twelve patches above + `BASE.txt` (the upstream commit they apply to) |
| `RESULTS.md` | the measured numbers |
| `RUNBOOK.md` | how to run this on the box. **Section 1 is SGLang** — a wiped box to the 8.02 s / 19.04 s numbers, plus fl2va. Section 2 rebuilds the patched reference stack as the *control* that makes those a measured ratio rather than a claim |
| `samples/` | the renders the numbers came from, video+audio muxed, all t2va from `prompts/example_2.pt`. The current best config, with all ten patches: **`n_480p_seg4.mp4`** (864x480, 345 f), **`n_480p_362f_seg4.mp4`** (the literal 15 s, 362 f) and **`p_768p_free.mp4`** (1344x768, 345 f) — all `clipinfo.py`-checked. The patch-11/12 renders (`r2_768p_keep_yuv`, `s2_480p_rep10`, `s5_480p_362f_rep10`, `s4_768p_rep10`) are **not** in the repo — they are the same prompt at the same canvas as the clips above and the patches change no pixels, which `scripts/decode_parity.py` asserts bit-exactly, so the mp4s carry no information the tracked ones do not. Earlier renders kept for comparison: `vdn_*` (pre-patch-7 swscale mux), `z_*` (patches 1–6), `n_768p_seg4` (768p with patch 3's host offload, before patch 10), `y_768p_345f_r5` (the 768p split winner), `f_*` (fl2va) |

## Traps found on this box

1. **`torchvision` must come from the cu129 index too.** `pyproject.toml` pins
   `torchvision==0.28.0` but deliberately does not pin torch, so `uv pip install -e .`
   takes torchvision from PyPI — which is cu130 — and every `import transformers` then
   dies with `PyTorch and torchvision were compiled with different CUDA major versions` —
   surfacing, unhelpfully, as `Could not import module 'BloomPreTrainedModel'`.
   `scripts/h100_bringup.sh` installs it explicitly from `download.pytorch.org/whl/cu129`.
   It did **not** until the box was rebuilt, and the repair is not the obvious one:
   `uv pip install torchvision==0.28.0 --index-url .../cu129` is a no-op, because `0.28.0`
   already satisfies `0.28.0` and uv never fetches `0.28.0+cu129`. It needs
   `--reinstall-package torchvision`, or — better — both wheels named from that index before
   `-e .` ever resolves. Check with `torchvision.__version__`: it must read `+cu129`.
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
