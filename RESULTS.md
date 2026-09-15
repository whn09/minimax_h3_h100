# Measured: 480P / 15 s on 8x H100 80GB

Box `p5.48xlarge`, 8x H100 SXM 80 GB, us-east-2. Model `ckpts/stage-dmd-step-250`
(VDN-H3-8-step, the Stage-DMD distilled turbo adapter), fp8 e4m3 rowwise,
`softmax_backend: flex`, `inference_kernels: true`, 8 NFE, `warmup_steps: 2`, t2va
(`prompts/example_2.pt`). Every number here comes from a `*.inference.json` record written
by the run itself; `scripts/summarize.py` regenerates the tables from those records.

## The answer

**480P, 8 NFE, 8x H100, branch-parallel Ulysses at 3 softmax + 5 linear ranks, video VAE
data-parallel over all eight ranks.** Both reachable lengths near 15 s, since they cost
almost the same, and every per-request figure is a ten-request measurement:

| | 345 f / 14.375 s | 362 f / 15.083 s |
|---|---|---|
| denoise, 8 NFE | **8.70** (1.088 s/NFE) | **9.02** (1.128 s/NFE) |
| video VAE, 8 ranks | 1.57 | 1.56 |
| audio VAE + frames-to-host + mux | 1.18 | 1.19 |
| **steady state, per request** | **11.45 ± 0.040** | **11.78 ± 0.035** |
| steady vs clip length | 1.26x realtime | **1.28x realtime** |
| decoder load, once per process | 4.81 | 4.60 |
| post-warmup end-to-end, request 1 | 17.16 | 17.18 |

Every steady-state figure is the **mean over requests 2–10 of a ten-request process**
(`render.repeat=10`, arms `s2_480p_rep10` and `s5_480p_362f_rep10`) — not a single render with
the decoder load subtracted off, which is how the earlier and wrong **13.20 / 13.10** figures
in this table were produced. Full distributions in "Ten requests in one process" below; the
short version is sd 35–40 ms over nine requests at both lengths, so this is not a figure that
needs an error bar wider than its own last digit.

**15.000 s is not a reachable length** — `align_num_frames(n, 17, 5)` snaps to `5 + 17k`,
so the grid near 15 s is 345 (14.375 s) then 362 (15.083 s), with nothing between. 362 is
the literal answer to "15 seconds", and at 11.78 s steady it renders **1.28x faster than the
clip plays** — and it is also the cheaper of the two per finished second, $0.005158 against
$0.005261, because the extra 17 frames cost 0.33 s and buy 0.71 s of clip.

The extra 17 frames cost **nothing in the decode**: 1.56 s against 1.57 s. That is the
`ceil(chunks/8)` prediction landing to a millisecond — 362 frames is 21 temporal VAE chunks
against 345's 20, and `ceil(21/8) = ceil(20/8) = 3`, so the parallel decode's critical path is
three chunks either way. Parallel decode time is a function of `ceil(chunks/8)`, not of frame
count; it is flat across a whole band of clip lengths and steps only when the chunk count
crosses a multiple of 8. The whole 0.33 s difference between the two columns is denoise, where
the cost really is proportional to tokens.

One thing the repeat runs retire: the **video VAE stage used to look like the noisy one**,
2.11–3.82 s across seven single-render arms at matched canvas (median 2.97, σ 0.61), and this
table used to carry a ±0.6 s caveat because of it. Inside one warm process its sd is **1 ms**.
The variance was never in the decode — it was between processes, and a per-request latency
figure should never have been carrying it.

Two framings of the same run, because they get quoted for different things:

* **1.088 s/NFE** is the number comparable with upstream's published table (2.29 on
  8x H200, 1.40 on 8x B200 — all at 768p, and all **denoise only**: their Results section
  excludes "model loading, warm-up, VAE decoding, and MP4 encoding", so their `18.3 s` is
  just `2.29 x 8`). Nine warm requests put it at 1.088 ± 0.004, which agrees with the split
  sweep's 1.090 and retires the 1.069 this file used to quote — that came from one process's
  single render, and it was the fastest of the several that measured this arm.
* **11.45 s steady-state end-to-end** is what a request costs on a warm server, measured over
  nine consecutive requests. Post-warmup E2E including the one-time decoder load is 17.2 s.

Two corrections to earlier versions of this file, both the same mistake — comparing two
numbers measured at **different branch splits**:

* 362 frames was reported as "+25 % denoise (1.367 s/NFE)". That measurement was at standard
  Ulysses while 345's was at 3+5. Like-for-like it is **+4.2 %** (1.136 vs 1.090), which
  tracks the +4.9 % row count. Clip length is linear here; nothing about 362 is special.
* The H100↔H200 gap was reported as 1.18x from H100 `r0` against H200 `r6`. At the same
  split it is **1.13x**. See the 768p section.

## What a finished second costs

`p5.48xlarge`, **EC2 3-Year No-Upfront Instance Savings Plan**: **$23.77728 / instance-hour**
= $0.00660480 per instance-second, $2.9722 per GPU-hour. (`gpu-public-pricing.csv`, field
`isp3Year`. The file carries only the us-east-1 row; the box is us-east-2, where p5 list
pricing is the same, but that is an assumption and not from the file.)

All eight GPUs work on one clip, so cost per clip is just wall clock x the instance rate:

| | clip | steady state | $/clip | **$/finished video-second** | $ per video-hour |
|---|---|---|---|---|---|
| **480p, 345 f** | 14.375 s | 11.45 s | $0.0756 | **$0.005261** | $18.94 |
| **480p, 362 f** | 15.083 s | 11.78 s | $0.0778 | **$0.005158** | $18.57 |
| **768p, 345 f** | 14.375 s | 33.21 s | $0.2193 | **$0.015259** | $54.93 |

Per 1000 clips: **$76** at 480p, **$219** at 768p. Using post-warmup E2E instead of steady
state — i.e. charging every request for the one-time decoder load, which is only honest for a
cold process serving a single clip — 480p is $0.007883/s ($113 per 1000) and 768p is
$0.015296/s ($220 per 1000; at 768p the two are nearly equal, because the decoder load has
stopped being a one-time cost and is now inside every request).

Every steady-state figure here is a ten-request measurement, which moved all three rows:
480p got 13 % cheaper (the old figures charged a warm request for a cold request's decode) and
768p got 13 % dearer (the old figure was from a process that could not have served a second
request at all). Directionally opposite errors from the same methodological cause.

Three things this table is and is not:

* **480p renders for less than the instance costs to run.** At 1.26x realtime the finished
  second costs 0.80x an instance-second; the 362-frame arm at 1.28x realtime costs 0.78x.
  768p costs 2.31x an instance-second. Of the 2.9x gap to 480p, 2.4x is the row count in the
  denoise and the rest is 768p's per-request decoder cycling.
* **The one-time build is not in here.** 138 s in a single process, 217–232 s in the 8-rank
  job (`setup`), = $0.91–$1.53 of instance time per process lifetime. Over a few hundred
  requests it rounds away; over three it doubles the bill. It is an argument for long-lived
  workers, and against any per-request DiT reload — a 768p server that freed and restored the
  weights each request would add ~5 s to the 33.21 and pay **$0.0175 / video-second, +15 %**,
  which is precisely why `vae_after_decode: free` cycles the 9.70 GiB decoders instead of the
  45.24 GiB of weights.
* **This is a latency-optimised price, and it is roughly 2x the cheapest way to buy these
  seconds.** 8 GPUs give a 4.03x denoise speedup over 1 GPU (4.39 -> 1.088 s/NFE at 480p), so
  eight independent single-GPU renders would produce about **1.98x** the clips per hour on the
  same instance — call it ~$0.003 / video-second — at 3-4x the per-clip latency. A single card
  holds the DiT and the decoders together at 480p (45.2 + ~11 GiB of 80), so that
  configuration is plausible, but the number is denoise-only arithmetic and has not been run
  end to end. It is the trade the brief chose against: the ask was minimum latency.

## 480p: the branch split matters more than anything else

`parallel.softmax_ranks: n` gives n ranks the window-softmax branch and `8-n` the linear
branch; `0` is standard Ulysses, where every rank owns heads of both. Upstream's H200 file
ships `6`, tuned on a 105k-row 768p sequence. 480p/345f is 43,759 rows, and the balance
point moves — the full sweep, 8 GPUs, 480p, 345 frames:

| softmax_ranks | split | s/NFE | denoise | vs best |
|---|---|---|---|---|
| 0 | standard Ulysses | 1.302 | 10.42 s | +19 % |
| 1 | 1+7 | 2.096 | 16.77 s | +92 % |
| 2 | 2+6 | 1.266 | 10.12 s | +16 % |
| **3** | **3+5** | **1.090** | **8.72 s** | — |
| 4 | 4+4 | 1.110 | 8.88 s | +2 % |
| 5 | 5+3 | 1.230 | 9.84 s | +13 % |
| 6 | 6+2 (upstream's H200 default) | 1.356 | 10.85 s | +24 % |
| 7 | 7+1 | 1.969 | 15.75 s | +81 % |

Three things worth reading off it:

1. **The minimum is interior, and it is not where upstream put it.** Taking upstream's
   `6` unchanged costs 24 %. This is the single cheapest win in the whole exercise: one
   config field.
2. **The curve is not smooth.** `r1` (2.096) is worse than `r0`, `r2` and `r3`; `r7`
   likewise. Starving a branch of ranks is much worse than balancing it badly. Sampling
   `0/4/5/6/7` alone — as the first pass did — finds the minimum at 4 and looks conclusive;
   it takes `1/2/3` to see that the real minimum is 3 and that the curve has a spike at 1.
3. **480p wants fewer softmax ranks than 768p.** The window softmax costs
   `O(rows x window_frames x tokens_per_frame)` while the linear branch and the MLPs cost
   `O(rows)`, and 480p has 405 tokens per frame against 768p's 1008. Less softmax work to
   spread, so fewer ranks to spread it over. The 768p sweep below confirms the direction.

## Scaling: is 8 GPUs the right answer?

Each GPU count at *its own* best split, not at 8 GPUs' best split — otherwise the
comparison flatters whichever count the split was tuned for:

| GPUs | best split | s/NFE | denoise | speedup vs 2 | efficiency |
|---|---|---|---|---|---|
| 2 | 1+1 | 2.895 | 23.16 s | 1.00x | — |
| 4 | 2+2 | 1.620 | 12.96 s | 1.79x | 89 % |
| 8 | 3+5 | 1.090 | 8.72 s | 2.66x | 66 % |

For contrast, standard Ulysses at the same counts: 3.212 / 1.942 / 1.302 s/NFE. Note that
at 4 GPUs the *even* split (2+2, 1.620) beats both standard Ulysses (1.942) and the lopsided
3+1 (2.188) — the same shape as at 8 GPUs, where 3+5 and 4+4 beat 7+1 badly.

**8 GPUs is still the right answer for latency**, but the marginal return is falling: 2→4
buys 1.79x, 4→8 buys 1.49x. At 480p the sequence is only 43,759 rows, so at 8 ranks each
rank owns ~5,500 rows and the per-block all-to-alls stop amortising. If the goal were
throughput rather than latency, two 4-GPU replicas would beat one 8-GPU render (2 x 1.620
vs 1.090, i.e. 1.34x the clips per second).

## Step budget

Per-NFE cost is flat in the step count, so the step budget is a pure multiplier — quality
is the only reason to pick 8:

| NFE | s/NFE | denoise | post-warmup E2E |
|---|---|---|---|
| 4 | 1.291 | 5.17 s | 20.59 s |
| 6 | 1.305 | 7.83 s | 23.42 s |
| 8 | 1.302 | 10.42 s | 25.87 s |

(All at standard Ulysses, so comparable with each other and with `r0` above, not with the
3+5 headline.) Note how little the E2E moves: 4 NFE halves the denoise and takes 20 % off
the request, because the decode does not care how many steps produced the latents.

## 768p control: the H100↔H200 gap

Same shape as upstream's published headline (768p, 345 frames, 8 NFE) so the gap is
measured rather than assumed. H100 and H200 are both sm90, so every kernel choice is
identical; what differs is HBM (80 vs 141 GB) and bandwidth (3.35 vs 4.8 TB/s).

The 768p sweep, so the control has its own best split rather than borrowing 480p's:

| softmax_ranks | split | s/NFE | denoise | vs best |
|---|---|---|---|---|
| 0 | standard Ulysses | 2.702 | 21.62 s | +9 % |
| 3 | 3+5 (480p's best) | 3.331 | 26.65 s | +35 % |
| 4 | 4+4 | 2.755 | 22.04 s | +12 % |
| **5** | **5+3** | **2.470** | **19.78 s** | — |
| 6 | 6+2 (upstream's H200 default) | 2.590 | 20.74 s | +5 % |

**768p's optimum is 5, 480p's is 3.** That is the prediction from the cost model coming out
right: the window softmax is `O(rows x window_frames x tokens_per_frame)` and 768p has 1008
tokens per frame against 480p's 405, so 768p has 2.49x more softmax work per row to spread
and wants more ranks to spread it over. Upstream's `6`, tuned on H200, is only 5 % off the
H100 optimum here — it is a bad default at 480p (+24 %), not at 768p.

Note also that 480p's best split is the *worst* of the five at 768p. The split is not a
property of the machine; it has to be re-tuned per canvas.

| | split | s/NFE | denoise |
|---|---|---|---|
| 8x H100 | 5+3 (its best) | 2.470 | 19.78 s |
| 8x H100 | 6+2 | 2.590 | 20.74 s |
| 8x H200 (upstream published) | 6+2 | 2.29 | 18.3 s |
| 8x B200 (upstream published) | 6+2 | 1.40 | 11.23 s |

The like-for-like comparison is the `6+2` row, since that is the split upstream published
at: **H100 is 1.13x slower than H200** (2.590 / 2.29). Best-measured H100 against
published H200 is 1.08x, but that is not a fair fight — H200 was never swept, and if `5`
wins there too its own number would come down.

A purely bandwidth-bound render would be 1.43x slower (4.8 / 3.35 TB/s), so 1.13x says
most of the work is fp8 compute, where the two parts are bit-for-bit identical kernels
(both sm90, both rowwise scale granularity), and only the memory-bound remainder pays the
HBM difference. **1.13x is the factor to apply** when comparing any number here against a
published H200 one — and it is small enough that the branch split, worth up to 35 %, matters
more than the choice of card.

480p vs 768p at the same clip length, standard Ulysses both: 1.302 vs 2.702 s/NFE =
**2.08x**, against a row ratio of 2.41x. Sublinear, not superlinear — see the README for
why the asymptotic argument gives the wrong answer here.

### 768p end to end: half the request is one PCIe transfer

The denoise is the fair comparison above, but it is not the request. Full 768p breakdown,
arm `n_768p_seg4`, same nine patches and same parallel decode as the 480p headline:

| stage | 768p, patch 10 | 768p, patch 3's offload | 480p / 345 f |
|---|---|---|---|
| denoise, 8 NFE | 19.77 (2.471 s/NFE) | 19.78 (2.473) | 8.55 (1.069) |
| **release the DiT** | **2.81** (free) | **26.33** (copy to host) | **0** (not needed) |
| decoder load (once per process) | 2.81 | 3.19 | 3.87 |
| video VAE, 8 ranks | 4.47 | 4.78 | 3.10 |
| audio VAE + frames-to-host + mux | 2.47 | 2.21 | 1.55 |
| **post-warmup end-to-end** | **32.32** | 56.29 | **17.07** |
| E2E minus the decoder load | 29.50 | 53.10 | 13.20 |
| vs clip length | 2.05x slower than realtime | 3.69x slower | 1.09x faster |

(Arms `p_768p_free` and `n_768p_seg4`. Both outputs `clipinfo.py`-clean.)

The second-to-last row used to be labelled "steady state", and that was the wrong name for a
subtraction: **neither of these configurations has a steady state**, because both throw the
weights away and a second request would rebuild them for 217–232 s. The measured steady states
are 11.45 s at 480p and 33.21 s at 768p, both from ten-request processes — see the last section
of this file. The subtraction is kept here only because it is what makes the patch-10 vs
patch-3 comparison in this table like-for-like.

768p used to be **4.0x** the 480p request against a 2.41x row ratio, and almost all of the
excess was one line: `model.transformer.to("cpu")`, 45.2 GiB back to the host at 1.7 GB/s.
**Patch 10 deletes 23.5 s of that by not making the host copy.** Both modes free exactly the
same GPU memory — the decode needs the space, not the eviction — and the 26.33 s is the
host-side allocation, thousands of separate pageable tensors, not the release. Releasing
without the copy costs 2.81 s (`to_empty(device="meta")` plus `empty_cache()` over a 45.2 GiB
arena; not instant, but 9.4x cheaper). Nothing downstream of the denoise reads the weights,
so the copy bought nothing this path uses.

That takes 768p to **2.2x** the 480p request, which is finally in line with the 2.41x row
ratio, and the denoise is untouched at 2.471 s/NFE. `parallel.transformer_before_decode` is
now `keep | free | offload`; 480p uses `keep`, 768p uses `free`, and `offload` survives only
as the measured comparison — the section below shows it is strictly dominated as a way to get
the weights *back*, which was the only thing it could have been for.

**Releasing the weights at all is load-bearing, and that was tested rather than assumed.**
Patch 3 introduced the eviction when the decode was still serial on rank 0; patch 5 made the
decode 8-way, which changes where the peak lives, so the arm was re-run with `keep`:

```
denoise 19.77s over 8 NFE = 2.472s/NFE      <- fine
torch.OutOfMemoryError: Tried to allocate 102.00 MiB. GPU 0 has a total capacity of
79.18 GiB of which 69.88 MiB is free ... this process has 79.10 GiB memory in use.
  in parallel_vae.py:148 -> vae._blend -> torch.cat
```

Rank 0 dies in the assembly, not the chunk decode, and it dies **102 MiB short of finishing**
— which says the eviction is buying a couple of GiB, not tens. That margin is also why the
remaining 2.81 s is worth attacking rather than accepting: if rank 0's peak came down by a
few GiB the DiT could simply stay on the card and the whole stage would disappear.

**The cheapest possible explanation was tested first, and it is wrong.** `empty_cache()` was
only ever called on the eviction path, never in `keep` mode, so the theory was that the
denoise's cached arena had simply never been handed back and the *contiguous* 102 MiB was a
fragmentation artefact. Calling it unconditionally reproduces the failure **to the byte** —
same 102 MiB request, same 69.88 MiB free, same 79.10 GiB in use (arm `q2_768p_keep_ec`).
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is already reusing those blocks, so there
was nothing to hand back. The call was reverted rather than left in, since in `keep` mode it
would add ~0.5 s to every 480p render for nothing. What the arm did buy is the budget, which
no earlier run had printed:

| rank 0 at 768p, `keep` | GiB |
|---|---:|
| allocated by PyTorch at the OOM | 64.97 |
| reserved by PyTorch, unallocated | 0.50 |
| **non-PyTorch** — CUDA context, NCCL, cuBLAS/flash-attn workspaces | **14.1** |
| total in use | 79.10 |
| card | 79.18 |

That 14.1 GiB is 18 % of the card and is the least examined number in this file. It is not
reachable from the config, but it is the reason the margin is 102 MiB instead of several GiB:
of a nominal 80 GiB, the decode is really working against ~65.

A second arm (`q_768p_keep_ec`) fails differently and is worth recording as a distinct
result: run **without** `parallel_vae_decode`, rank 0 falls back to the serial diffusers
`_decode`, and the OOM moves to `autoencoder_kl_minimax_h3.py:830` asking for **1.99 GiB with
1.73 GiB free**. 1.99 GiB is exactly `345 x 1344 x 768 x 3 ch x 2 B` — the whole clip in fp16
RGB — so the serial path is short by ~260 MiB rather than 102, and both paths die building
the same full-canvas tensor. Where those GiB go, from the tensor shapes at 768p (3.10 MB per
frame at 3 channels fp16, 1344x768):

* `gathered`, the all-gather destination — 24 slots × 22 frames, since every slot carries its
  chunk's 17 main frames plus its 5-frame decode overlap: **3.045 GiB**.
* `dec`, the concatenated clip: 345 × 1344 × 768 × 3 × 2 B = **1.99 GiB**.
* the fp32 pixel denormalisation downstream of it, at 12 bytes per pixel: **3.98 GiB**.

So rank 0 holds the whole clip three times over, at 6, 6 and 12 bytes per pixel, while
patch 7 already knows how to represent a finished frame in **1.5**.

### It does stay resident, and it took two changes rather than one

Both were written and measured (patch 11); the first alone was **not** enough, which is worth
recording because the estimate said it would be.

1. **Never materialise `dec` or the fp32 clip.** The assembly already yields one blended chunk
   at a time and `torch.cat` existed only to hand `decode_and_save` a single tensor. Each
   chunk now becomes its final yuv420p planes as it leaves the loop, written into a
   preallocated plane set at 1.5 bytes per pixel — `parallel.stream_yuv_assembly`. This
   removes 1.99 + 3.98 GiB, which is 60x the 102 MiB deficit, so on the arithmetic it should
   have been the end of it. It was not: free memory went from 69.88 MiB to **2.84 GiB** and
   the run died anyway, because with the full-canvas buffers gone the largest single
   allocation left is `gathered` and it wants a contiguous **3.05 GiB**.

2. **Gather one round of chunks at a time.** Chunk *c* lives on rank *c* mod 8 in slot
   *c* div 8, so slot *s* across the ranks in rank order *is* chunks 8*s* … 8*s*+7 — exactly
   the order the assembly consumes them in. So the all-gather does not have to be one
   collective over the whole decode: it can be one per slot, three at 345 frames, each holding
   8 pieces instead of 24. Peak falls 3.045 → **0.38 GiB**, the gather pipelines with the
   blend, and the only new cost is cloning the 5-frame carried overlap at each slot boundary
   (31 MiB) because it is a view into a buffer the next gather overwrites.

The candidate this file previously ranked second — convert to YUV *before* the all-gather —
was **not** built, and (2) is the better trade anyway: it returns more (3.045 vs ~2.29 GiB)
and it stays exactly bit-identical, whereas converting before the blend cannot be, since
`_blend` cross-fades in float and a linear ramp over quantised uint8 chroma is not the
quantisation of the ramp.

Measured, arm `r2_768p_keep_yuv`, matched against `p_768p_free` (`softmax_ranks: 5`, 8-way
parallel decode, 345 f, fp8, t2va):

| stage | `free` | `keep` + streaming YUV + per-slot gather |
|---|---:|---:|
| denoise, 8 NFE | 19.77 (2.472 s/NFE) | **19.75 (2.469)** |
| DiT release | 2.81 | **0.00** |
| video VAE, 8 ranks | 4.47 | 4.71 |
| audio VAE + frames-to-host + mux | 2.47 | 2.33 |
| E2E minus the decoder load | 29.52 | **26.79** |
| decoder load (once per process) | 2.81 | 5.45 |
| post-warmup E2E | 32.33 | 32.24 |

By that subtraction the change is worth 2.71 s (−9.2 %), and post-warmup E2E does not move —
the 2.81 s saved on the release is spent on a decoder load that is 2.64 s slower, plausibly
because the decoders now allocate against a card that already holds 45.2 GiB of weights.
**Neither 26.79 nor 29.52 is a steady state**, which is what the next section is about; the
real figure with the DiT resident is 33.21 s, and the subtraction is off by 24 %.

Rank 0's budget with the weights resident, from the new instrumentation:

| rank 0 at 768p, `keep`, streaming | GiB |
|---|---:|
| reserved before the decode (unchanged across it) | 62.1 |
| decode peak, allocated | 63.8 |
| decode peak, reserved | 64.5 |
| non-PyTorch floor, from the OOM arm above | ~13.6 |
| **implied headroom** | **~1.1** |

**And that is where a single render stops being able to tell you anything, because the run
above does not survive a second request.** See the next section: this table is a one-shot
result and was published as a steady-state one for about an hour.

At 480p there was never anything to evict, and the point of running it through the changed
code was only to show the primary result did not regress. Arm `r2_480p_yuv`: denoise 8.68 s
(1.085 s/NFE), video VAE 2.50 s, audio+host+mux 1.69 s — unchanged within the between-process
spread, so the 480p configs are left as they were.

## Ten requests in one process, which is the only test that means anything

Everything above measures **one** render per process and calls the leftover "steady state" by
subtracting the decoder load from it. That subtraction is wrong in two directions at once, and
`render.repeat` — serve N complete requests from one warm process, report the distribution —
shows both. Re-running the *process* N times would not have: it re-measures the 220 s build
every time and never lets two requests share a resident model, which is the entire question.

**480p, 345 f, 10 requests** (arm `s2_480p_rep10`, published config, DiT and decoders `keep`):

```
 req   denoise    vae x8   dec+enc   dload    total  alloc_end  resv_end
   1      8.67      2.12      1.56    4.81    17.16       58.0      62.9
   2      8.74      1.55      1.18    0.00    11.47       58.0      62.9
   3      8.66      1.56      1.19    0.00    11.41       58.0      62.9
   4      8.66      1.67      1.16    0.00    11.49       58.0      63.1
   5      8.70      1.55      1.17    0.00    11.43       58.0      63.1
   6      8.73      1.55      1.18    0.00    11.47       58.0      63.1
   7      8.69      1.55      1.15    0.00    11.40       58.0      63.1
   8      8.67      1.55      1.17    0.00    11.40       58.0      63.1
   9      8.68      1.56      1.20    0.00    11.44       58.0      63.1
  10      8.73      1.55      1.22    0.00    11.51       58.0      63.1
```

| 480p steady, requests 2–10 | mean | sd | min | max |
|---|---:|---:|---:|---:|
| **request total** | **11.45** | **0.040** | 11.40 | 11.51 |
| denoise, 8 NFE | 8.70 | 0.032 | 8.66 | 8.74 |
| video VAE ×8 | 1.57 | 0.039 | 1.55 | 1.67 |
| decode+encode | 1.18 | 0.019 | 1.15 | 1.22 |

**The real 480p steady state is 11.45 s, not the 13.20 s this file reported**, and the
correction is not noise — it is 1.75 s, and it is systematic. A process's *first* request is
slower than its later ones in stages that have nothing to do with the decoder load: the video
VAE runs 2.12 s then 1.55 s forever after, and the tail 1.56 s then 1.18 s. Allocator arena,
x264's thread pool, cuBLAS and cuDNN workspace selection — all first-request costs, all
invisible to a method that renders once and subtracts.

The other correction goes the other way and is the more embarrassing one: this file quoted
**13.2 ± 0.6 s**, with the ±0.6 attributed to the video VAE's 2.11–3.82 s spread "across seven
arms". Nine consecutive requests in one process put that stage at **1.57 ± 0.039**. The 0.6 was
never run-to-run variance in a warm process; it was variance *between processes*, i.e. mostly
first-request effects plus config differences. Steady state is an order of magnitude tighter
than advertised: **sd 40 ms on an 11.45 s request.**

Memory is flat, which is the leak check `repeat` exists for: `allocated` is 58.0 GiB after
every one of the ten requests, and `reserved` settles at 63.1 by request 4 and does not move
again (+0.254 GiB total, all of it arena growth in requests 3–4).

**480p, 362 f, 10 requests** (arm `s5_480p_362f_rep10`) — the same, at the literal-15-second
length:

```
 req   denoise    vae x8   dec+enc   dload    total  alloc_end  resv_end
   1      9.02      2.00      1.56    4.60    17.18       58.1      62.9
   2      9.05      1.56      1.17    0.00    11.78       58.1      63.0
   3      9.01      1.56      1.18    0.00    11.76       58.1      63.2
   4      9.00      1.56      1.24    0.00    11.81       58.1      63.2
   5      9.01      1.56      1.16    0.00    11.73       58.1      63.2
   6      9.04      1.56      1.19    0.00    11.80       58.1      63.2
   7      8.99      1.56      1.18    0.00    11.73       58.1      63.2
   8      9.04      1.56      1.21    0.00    11.82       58.1      63.2
   9      9.00      1.57      1.17    0.00    11.75       58.1      63.2
  10      9.04      1.56      1.21    0.00    11.81       58.1      63.2
```

| 480p / 362 f steady, requests 2–10 | mean | sd | min | max |
|---|---:|---:|---:|---:|
| **request total** | **11.78** | **0.035** | 11.73 | 11.82 |
| denoise, 8 NFE | 9.02 | 0.022 | 8.99 | 9.05 |
| video VAE ×8 | 1.56 | **0.001** | 1.56 | 1.57 |
| decode+encode | 1.19 | 0.027 | 1.16 | 1.24 |

Reserved: +0.234 GiB over ten requests, settled by request 3. Two things fall out of putting
this next to the 345 f table:

* **the parallel decode's sd is 1 ms.** Not 0.6 s, not 39 ms — one millisecond, over nine
  requests, and the *same* 1.56 s at both clip lengths. This is the strongest confirmation of
  the `ceil(chunks/8)` model in the file: 21 chunks and 20 chunks both take three rounds, and
  three rounds take 1.56 s whichever they are. The stage that used to look like the noisiest
  part of the request is the most deterministic thing in it.
* **the subtraction had the two lengths in the wrong order.** It said 362 f was *faster* than
  345 f (13.10 vs 13.20); measured, it is 0.33 s slower, which is +3.7 % on a +4.9 % row count
  and lands where the denoise scaling says it should. The old ordering was an artifact of two
  different processes' first-request decode noise, and it inverted a real effect.

### 768p: residency survived one request and died on the second

Same test at 768p, `keep` + streaming YUV + per-slot gather — the configuration the section
above declared repeatable. Request 1 completed normally (denoise 19.75 s). **Request 2 died in
the denoise**, on ranks 5, 6 and 7:

```
--- request 2/10
[rank7]: OutOfMemoryError: Tried to allocate 444.00 MiB ... 439.88 MiB free, 64.74 GiB allocated
[rank5]: OutOfMemoryError: Tried to allocate 468.00 MiB ... 119.88 MiB free, 65.06 GiB allocated
  one_request -> sample -> generate_latents -> linear_attention/branch.py:313
```

Ranks 5–7 are exactly the three **linear-branch** ranks at `softmax_ranks: 5`. And the cause is
the ordering that made the one-shot run work: **the decoders are loaded after the first
denoise, so that denoise is the only one that never coexists with them.** Every subsequent
request denoises with the video VAE already resident on all eight ranks. Measured directly:

| resident per rank at 768p | GiB |
|---|---:|
| DiT (fp8) | 45.24 |
| video VAE — **every** rank, because the decode is data-parallel | **9.70** |
| audio VAE — rank 0 only | 0.56 |
| non-PyTorch floor | ~13.6 |
| subtotal, DiT + video VAE + floor | **68.5** |
| left for the denoise | 10.6 |
| what the denoise actually needs | **~16.9** |

Short by ~6 GiB. Not a margin that a smaller buffer somewhere recovers — at 768p one of the
two big residents has to leave on every request, and the choice is settled by size:

| cycle this every request | data moved per rank | restore from a pinned host copy |
|---|---:|---:|
| **video VAE** | **9.70 GiB** | ~0.96 s at the measured 10.08 GiB/s |
| DiT | 45.24 GiB | 4.49 s (measured) |

So the answer to "can the DiT stay resident at 768p" is **yes, but only if the decoders do
not** — and that is the good trade, because cycling the decoders moves 4.7x less data than
cycling the weights. `parallel.vae_after_decode: free` implements it: the decoders are dropped
after the frames are out and reloaded before the next decode, which the existing
`model.vae is None` guard already knew how to do. Measured, arm `s4_768p_rep10`:

```
 req   denoise    vae x8   dec+enc   dload    total  alloc_end  resv_end
   1     19.76      3.65      2.54    5.09    33.29       57.2      64.5
   2     21.55      3.23      1.63    6.21    34.38       56.4      63.6
   3     20.94      3.82      1.74    5.36    33.71       56.4      63.6
   4     20.74      3.26      1.65    5.40    32.22       56.4      63.6
   5     21.56      4.10      1.67    4.63    34.03       56.4      63.6
   6     20.63      3.65      1.68    5.22    32.36       56.4      63.6
   7     21.64      3.40      1.65    5.19    34.07       56.4      63.6
   8     20.52      3.69      1.76    4.72    33.02       56.4      63.6
   9     20.45      4.07      1.67    4.49    32.76       56.4      63.6
  10     20.69      4.04      1.65    4.19    32.30       56.4      63.6
```

| 768p steady, requests 2–10 | mean | sd | min | max |
|---|---:|---:|---:|---:|
| **request total** | **33.21** | **0.850** | 32.22 | 34.38 |
| denoise, 8 NFE | 20.97 | 0.483 | 20.45 | 21.64 |
| video VAE ×8 | 3.70 | 0.343 | 3.23 | 4.10 |
| decode+encode | 1.68 | 0.043 | 1.63 | 1.76 |
| decoder load (now **per request**) | 5.05 | 0.607 | 4.19 | 6.21 |
| decoder release | 1.82 | 0.415 | 1.17 | 2.34 |

Ten requests, `reserved` identical after request 2 and request 10 to **+0.000 GiB**. So 768p is
now repeatable — and it costs **33.21 s**, not the 26.79 s the one-shot run implied and not the
29.50 s before that. Both earlier figures were arithmetic on a process that only ever served
one request.

Two things this table says that are worth not glossing over:

* **6.87 s of the 33.21 is pure cycling overhead** — 5.05 s reloading the decoders from disk
  plus 1.82 s releasing them. Both are attackable and neither is fundamental: a pinned host
  copy of the 9.70 GiB video VAE restores at the measured 10.08 GiB/s in **~0.96 s**, and the
  release is mostly `empty_cache()` over a large arena. That is the next patch, and it puts
  768p at roughly **28 s** on arithmetic that is now grounded in measurements rather than
  guesses. It is not built here.
* **the denoise itself is 1.2 s slower in steady state** — 19.76 s on request 1, 20.97 ± 0.48
  after. Two candidate causes and this run cannot separate them: the cycling churns 9.70 GiB
  through the allocator every request and `empty_cache()` hands cached blocks back to the
  driver that the next denoise has to re-acquire; or eight H100s at 768p for five minutes are
  power/thermally limited in a way a single render never sees. 480p's denoise moves only
  +0.03 s over ten requests, but 480p never cycles anything, so it is a control for the first
  hypothesis and not for the second.

The 480p configs stay on `keep` at both ends — 10 requests, no growth, nothing to cycle.

Worth stating explicitly, since it is the obvious alternative: **tensor parallelism would
also solve this, and it is the expensive way to.** TP=2 x Ulysses=4 halves the resident
weights to ~22.6 GiB and frees 22.6 GiB where the OOM needed 102 MiB, so the eviction goes
away entirely. But nothing in this tree shards a weight — Ulysses shards the *sequence*
inside the attention modules — so it means column/row-parallel projections, an all-reduce per
block, making patch 2's host-side fp8 assembly shard-aware, and re-deriving the branch split
on a 4-way grid where 768p's measured optimum of 5-of-8 is not expressible (4-of-8 costs
+12 %, 6-of-8 costs +5 %). Against that: a mode string, and 23.5 s recovered.

### What `free` costs a server: how long the DiT takes to come back

Everything above measures a one-shot render, where `free` is unambiguously right because
nothing after the denoise reads the weights and the process exits. An API process denoises
again, and `to_empty(device="meta")` leaves **no copy anywhere** — the storages are gone and
every parameter points at meta. So "how long to reload" is a real question with four
different answers depending on what you kept. Measured on one H100 at the 768p config's fp8
numerics, `scripts/reload_bench.py`, 45.24 GiB over 1075 parameters + 727 buffers:

| getting the weights back onto the card | seconds | rate |
|---|---:|---|
| from a **pinned** host copy | **4.49** (1.08 alloc + 3.41 copy) | 10.08 GiB/s |
| from a **pageable** host copy — what `offload` leaves you | **10.53** (0.48 + 10.04) | 4.30 GiB/s |
| rebuild from the checkpoint, 1 process, 192 threads | **138.36** | — |
| rebuild from the checkpoint, inside the 8-rank job | **217–232** | — |

and the costs on the way out, same run:

| | seconds |
|---|---:|
| `to_empty(device="meta")` + `empty_cache()`, clean arena | 0.41 / 0.89 |
| the same after a denoise, fragmented arena (in-render) | 2.81 |
| `.to("cpu")` — the pageable snapshot `offload` makes | 26.67 |
| pinning 45.24 GiB of host memory (once, at build) | 47.65 |

Both restores were checked tensor by tensor against the host copy they came from: **0 of 1802
bit-identical**. That is the right check and an end-to-end diff would not be — the multi-rank
render is not reproducible run to run, but a weight copy is, so a copy either landed or it
did not.

Four things fall out of this.

**The naive answer is a non-starter.** `free` with nothing retained means the next request
pays a full rebuild: 138 s in a single process, and the grid logs' `setup` of 217–232 s is
what it actually costs in the 8-rank job, where eight processes contend for the same 192
vCPUs doing the same LoRA merge and the same 363-Linear fp8 quantisation. Either number is
4–7x the entire 32.32 s request it is supposed to be helping.

**`offload` is strictly dominated, and now provably so.** It pays 26.67 s to write a
*pageable* copy, and pageable is exactly the copy that restores slowest: 26.67 + 10.53 =
**37.2 s** round trip. Keeping a pinned copy instead is 0.41 + 4.49 = **4.9 s** round trip,
7.6x cheaper on both legs at once. The 2.3x gap between the two restores is the same
mechanism as everything else in this file — a pinned copy DMAs, a pageable one is staged
through a bounce buffer — so the rule is: if you keep a host copy, pin it.

**Even the good path is not cheap enough to want.** Release plus pinned restore is ~5 s
(~7 s using the in-render 2.81 s release), added to a 33.21 s steady-state 768p request, for
+15–21 %. And it wants 45.24 GiB of *page-locked* host memory per rank — **362 GiB across 8
ranks**, which fits the box's 2 TiB but is locked away from the page cache and from the
frames-to-host staging in patch 7. All of that to free the 102 MiB the OOM was short of.

**So for a server the answer is not to cycle the weights at all.** At 480p — the primary
target — this whole question is void: `keep` is the default, nothing is evicted, and the
17.16 s / 11.45 s figures already are the API's numbers, measured over ten requests. At 768p
the fix is the decode's peak, not the DiT, and patches 11 and 12 are that fix: streaming the
assembly and gathering per slot let `keep` hold at 64.5 GiB reserved, and what cycles instead
is the **9.70 GiB video VAE**, 4.7x less data than the weights. Cycling 45.24 GiB twice per
request is the wrong shape of fix; it is only in the tree because it was the cheapest thing
that made a one-shot 768p render finish.

The correction the repeat runs forced on the paragraph above: it used to end "and then 768p
also runs `keep` and drops to ~26.7 s with no reload cost on any request", and the second half
of that was wrong. `keep` on *both* sides survives exactly one request. Something has to cycle
at 768p — the arithmetic is 45.24 + 9.70 + 13.6 = 68.5 GiB resident against a denoise that
wants ~16.9 GiB of activations in the 10.6 that are left — and the only real choice is which
side. 768p steady is 33.21 s, not 26.7.

`scripts/reload_bench.py` takes no arguments beyond the config and prints the JSON above, so
the numbers can be re-derived on other hardware — the pinned restore rate is a PCIe property
and will differ on a box with a different topology.

## The decode was the other half

At 480p, upstream's decode takes **14.38 s** against the denoise's 8.72 s, and all of it
runs on rank 0 while seven cards wait at a barrier. Patch 5 spreads the video VAE's temporal
chunks over the ranks. Measured at 480p / 345 f, 8 GPUs, 3+5:

| stage | ranks | before | after |
|---|---|---|---|
| video VAE | 1 → 8 | ~11.23 (derived) | **2.97** |
| audio VAE | 1 | 0.26 | 0.26 |
| frames → host | 1 | 0.23 | 0.23 |
| mp4 mux (H.264, on the host) | 1 | 2.66 | 2.66 |
| **decode total** | | **14.38** | **6.15** |
| post-warmup E2E | | 25.63 | **18.53** |
| steady state | | 23.10 | **14.88** |

**The video VAE goes 3.78x faster on 8 ranks**, the whole decode 2.34x, the request 1.38x,
and the steady-state request 1.55x. The "before" video VAE figure is the 14.38 s total minus
the three stages that patch 5 does not touch, since the pre-patch record has no stage
breakdown; the three subtracted stages are measured, not modelled.

3.78x against a ceiling of 6.67x — 20 chunks over 8 ranks is a critical path of
`ceil(20/8) = 3`, not `20/8 = 2.5` — so 57 % of the achievable parallel efficiency. What is
left is per-chunk fixed cost, the single all-gather, and the fact that four ranks decode
three chunks while four decode two and then wait.

At 768p the same patch takes decode+encode from **25.14 s to 5.87 s, 4.28×** — a larger
factor than 480p's, because 768p's chunks are 1.87× the decoder calls each, so the fixed
per-chunk cost that limits 480p is a smaller share of the total.

**That left the H.264 mux as the largest single item, at 2.70 s** — more than the eight-way
video VAE decode it follows, and host-side with no GPU involvement at all. The next section
is what it decomposes into.

## The tail: two changes, and one wrong diagnosis in between

Patches 7 and 9 go after the three stages that patch 5 does not touch. Both arms below are
the same config with one field flipped, so the difference is the change and nothing else
(480p, 345 f, 8 GPUs, 3+5):

| stage | swscale, 1 encoder | GPU convert, 1 encoder | GPU convert, 4 segments |
|---|---|---|---|
| audio VAE | 0.263 | 0.260 | 0.255 |
| frames → host | 0.417 | **0.273** | 0.276 |
| mp4 mux | 2.697 | 1.852 | **1.018** |
| **tail total** | **3.409** | **2.385** | **1.548** |
| vs upstream | — | 1.43× | **2.20×** |
| output size | 2.859 MB | 2.814 MB | 2.926 MB |

At 768p the same two changes take the mux **4.32 → 2.39 → 1.39 s** and the tail 3.21 → 2.21 s.
The conversion is worth more there (1.81× against 480p's 1.46×) because there are 2.4× the
pixels to convert against the same fixed cost.

The `frames → host` row is the prediction landing: yuv420p is 1.5 bytes per pixel against
rgb24's 3, so the transfer is 215 MB instead of 430 MB, and the GPU-side conversion itself is
free at this scale — it is inside that 0.273 s.

**The mux was not swscale, and I said it was.** An earlier version of this file attributed
2.03 s of the 2.66 s mux to swscale's colour conversion, from a micro-benchmark that called
`frame.reformat()` per frame in a loop. That allocates a fresh frame every call and overstates
what the encoder's internal conversion costs. The matched measurement above says the whole
mux only moved 0.85 s, and `mux_bench.py`'s thread-only arm accounts for 1.17× of that, so:

| | seconds |
|---|---|
| upstream: swscale, `thread_count` unset (1) | 2.70 |
| `thread_count=0`, still swscale | ~2.30 |
| planes from the GPU, `thread_count=0` | 1.85 |

≈0.40 s for the thread setting and ≈0.45 s for the conversion. **The rest, 1.85 s, is libx264
doing real work**, which is what patch 9 addresses.

### libx264 does not parallelise itself, so encode in segments

x264's own frame threading is nearly worthless on this canvas: 345 frames of 864×480 encode at
251 fps with `thread_count=0` against 224 at 1, a **1.12×** on a box with 192 vCPUs. The serial
part is the per-frame Python plane writes, which hold the GIL. So patch 9 splits the clip into
contiguous ranges, encodes them concurrently in threads and concatenates the bitstreams by
copy. Measured on fixed frames, PSNR against the input planes:

| | seconds | size | Y PSNR vs input |
|---|---|---|---|
| 1 encoder, `thread_count=0` | 1.37 | 2.36 MB | 42.72 dB |
| 2 segments × 8 threads | 0.69 | 2.38 MB | 42.61 dB |
| **4 segments × 8 threads** | **0.43** | 2.42 MB | 42.34 dB |
| 8 segments × 8 threads | 0.34 | 2.51 MB | 41.63 dB |
| 16 segments × 8 threads | 0.33 | 2.65 MB | 40.68 dB |

**4 segments is 3.2× for +2.5 % bits and −0.38 dB**, and that is the default. In the render
the mux stage goes **1.873 → 1.018 s, 1.84×** rather than 3.2×: the stage also carries the AAC
audio encode and the container write, which are ~0.6 s and unaffected. The absolute saving,
0.86 s, is what the standalone measurement predicted (1.37 − 0.43 = 0.94 s).

Two things make 4 the right stopping point. Past 8 segments the time stops falling (0.34 → 0.33 s) while the
bits keep climbing, because each segment costs one extra IDR. And the obvious alternative —
a faster x264 preset — is a worse trade, which is worth showing because the file sizes make it
look free at first glance:

| preset | seconds | size | PSNR vs source |
|---|---|---|---|
| medium (upstream's default) | 1.54 | 2.33 MB | **38.88 dB** |
| veryfast | 0.80 | 2.01 MB | 36.07 dB |
| ultrafast | 0.41 | 5.92 MB | 36.54 dB |

`veryfast` is 1.9× faster and its file is *smaller*, which is the tell: it is spending fewer
bits **and** losing 2.8 dB, a real rate–distortion loss rather than a boundary cost. So the
preset stays at medium.

**Hardware encoding is not an option on this box.** `h264_nvenc` fails to open on H100 —
GH100 ships NVDEC and NVJPEG but no NVENC silicon. Confirmed by running it, not inferred:
`scripts/mux_bench.py --nvenc` reports the arm as FAILED.

## The same seed does not give the same video

This has to come before any correctness claim about the decode, because it invalidates the
obvious way of checking one. **This pipeline does not reproduce run to run.** Two renders at
the same seed, same config, same split, same decode path:

| pair | canvas | split | PSNR between the two renders |
|---|---|---|---|
| `z_480p_345f_best` vs `f_480p_t2va_r3` | 480p | 3+5 | **17.55 dB** |
| `f_768p_t2va_r0` vs `v_768p_345f_serial` | 768p | standard Ulysses | **16.69 dB** |

Not one frame of 345 is bit-identical in either pair. The seeding is not the cause — it is an
explicit `torch.Generator(device).manual_seed(seed)`, drawn in a fixed order for the three
draws, identical on every rank.

**It is the parallel path, and one GPU is exactly reproducible.** Two `infer.py` runs at
480p/345f came out **byte-identical mp4s** (phase `g`), which rules out the fp8 GEMMs, the
kernels and the sampler — all of which the single-GPU path also uses. The divergence is
introduced by something only the multi-rank path executes, and it is not the Ulysses
all-to-alls: those are permutations, and permutations are bitwise exact.

It is upstream's **asynchronous frame mean**, `UlyssesRuntime.video_frame_mean_async`
(`src/inference/utils/ulysses_runtime.py:497-520`, untouched by any patch here), which has
two nondeterministic steps and no third candidate between them:

```python
sums.index_add_(0, frames, local_x[is_video].float())   # CUDA atomics, order varies
...
work = dist.all_reduce(sums, async_op=True)             # cross-rank float sum
```

`index_add_` on CUDA accumulates with atomics, so the order of up to `tokens_per_frame`
additions per (frame, channel) is not fixed run to run; `all_reduce` then sums across ranks in
whatever order NCCL's algorithm choice gives that run. Both produce ULP-level differences in
a quantity that feeds the linear branch's normalisation for *every* frame, and eight sampler
steps amplify that into visible detail differences. This also explains the shape of the
evidence: it appears at standard Ulysses (no branch split) because the frame mean is computed
either way, and it disappears on one GPU because `infer.py` never builds a runtime to call it.

**It is the same video, though.** Two measurements say so, and both matter:

* a temporal cross-correlation peaks at **shift 0** and falls 5 dB by four frames either
  side, so the motion is in phase — the runs are not re-timings of each other;
* PSNR under progressive downsampling climbs from 17.99 dB at 1× to only **24.90 dB at 48×**.
  Independent high-frequency noise would have gained ~33 dB over that reduction. It gained
  7. So the difference is low-frequency and structured, and per-frame nearest-neighbour
  matching puts every frame's best match at its own index.

Scene, composition and motion agree; the detail differs everywhere. For a video model that
is a nuisance rather than a defect, but it does mean **no end-to-end diff can validate
anything downstream of the denoise**, which is exactly what the parallel-decode check was
trying to do.

### So the decode is tested on fixed latents instead

The grid used to render the same prompt twice, once serial and once parallel, and `cmp` the
mp4s. It reported `MISMATCH` at both canvases, and **the test was wrong, not the decode** — it
was comparing two denoise trajectories. `scripts/decode_parity.py` (patch 8) replaces it: one
process, one latent tensor, decoded both ways back to back, nothing upstream in the
comparison. Run that way:

```
480p: OK -- bit-identical. The eight-rank decode is upstream's decode.
768p: OK -- bit-identical. The eight-rank decode is upstream's decode.
```

Max absolute difference **0.0** at both canvases, on the pixel-denormalised float output. So
patch 5's claim holds exactly, and the earlier `MISMATCH` was an artefact of the test.

## Text encoding is not in any of these numbers

Neither here nor upstream — the Qwen3-VL conditioner runs once, offline, and the render
`torch.load`s a cached `prompt_embeds`. Upstream's published 18.3 s excludes it for the same
reason. A prompt-to-video service cannot, so `scripts/text_encoder_bench.py` measures what
putting it in the request path would cost. On this box, 1300-token prompt:

| | |
|---|---|
| load, checkpoint → GPU, bf16 | **8.3 s** for 62.1 GiB (7.51 GiB/s) |
| full 64-layer forward | 169 ms → `hidden_states[50]`, (1, 1300, 5120) |
| resident after dropping layers 52–64 and the LM head | **48.9 GiB** (was 62.1, saved 13.3) |
| trimmed 51-layer forward | **135 ms** |
| offload 48.9 GiB to host, then back | **29.6 s** (1.65 GiB/s) + 7.6 s (6.43 GiB/s) |
| sharded over 8 ranks | **6.1 GiB per rank**, alongside the 45.2 GiB fp8 DiT = 51.3 of 79.2 |

Three things fall out of that:

1. **The forward is free. Everything else is memory movement.** 135 ms against an 11.45 s
   render is 1.2 %. VDN reads `hidden_states[50]`, which HF fills with the *input* to layer 50,
   so layers 52–64 — 13 of 64 — and the 5120×151936 LM head never contribute; dropping them
   is 13.3 GiB off what has to be resident, for free.
2. **Per-request host offload is not an option, and it is the *unload* that kills it.** The
   round trip is 37 s, three times the render, and 29.6 of it is the trip *to* the host at
   1.65 GiB/s — the same 1.7 GiB/s patch 3 measured on the DiT's `to("cpu")`, and for the
   same reason: a state dict is thousands of separate unpinned tensors, so the direction that
   allocates pageable destinations is the slow one. Keeping the conditioner permanently in
   host memory and only paying the 7.6 s upload would be tolerable-ish; paying to evict it
   every request is not.
3. **The 8-way shard is the answer, and it needs no offloading at all.** 6.1 GiB per rank
   next to the fp8 DiT leaves 28 GiB of headroom on each card, so the conditioner can simply
   stay resident and the request path pays only the 135 ms forward.

One number to read sceptically: the 7.51 GiB/s "NVMe → GPU" load is almost certainly served
from the page cache — the box has 2 TiB of RAM and the checkpoint had just been downloaded —
so it is a host-memory read wearing a filesystem's clothes, which is why it lands next to the
6.43 GiB/s host→device figure rather than above it. A genuinely cold load would be slower.
It does not change the conclusion, since the conclusion is not to move the weights at all.

## Conditioning: what fl2va costs over t2va

Same canvas, same split, same clip length, same step count — only the prompt file differs.
The 480p pair needs a keyframe cache encoded at 864x480, which is what patch 6 is for.

| canvas | mode | rows | s/NFE | denoise | vs t2va |
|---|---|---|---|---|---|
| 768p, standard | t2va | 105,265 | 2.704 | 21.63 s | — |
| 768p, standard | fl2va | 109,467 (+4.0 %) | 2.945 | 23.56 s | **+8.9 %** |
| 480p, 3+5 | t2va | 43,759 | 1.079 | 8.63 s | — |
| 480p, 3+5 | fl2va | 45,549 (+4.1 %) | 1.173 | 9.38 s | **+8.7 %** |

**fl2va costs about 9 % more than t2va, at either canvas.** That is the practical answer:
first/last-frame conditioning is not a different performance regime.

The row counts close exactly, which is worth writing down because it is what makes the
next paragraph a mechanism rather than a guess:

| | text+vision | condition | audio | video | total |
|---|---|---|---|---|---|
| 768p t2va | 1,299 | — | 1,150 | 102,816 | 105,265 |
| 768p fl2va | 3,485 | 2,016 | 1,150 | 102,816 | 109,467 |
| 480p t2va | 1,299 | — | 1,150 | 41,310 | 43,759 |
| 480p fl2va | 2,279 | 810 | 1,150 | 41,310 | 45,549 |

Two things follow, one of which corrects a prediction I had written into the driver.

1. **9 % of time for 4 % of rows** — a 2.2x amplification, the same at both canvases. The
   conditioning rows are not ordinary rows: text and condition rows sit *outside* the video
   span, so the window softmax attends them densely in both directions while the linear
   branch skips them entirely. Dense rows go 1,299 → 4,295 at 768p, a 3.3x increase, and
   that is where the extra 5 points come from.
2. **The relative cost does NOT grow on the smaller canvas.** I had predicted it would, on
   the grounds that conditioning adds a fixed number of rows to a shorter sequence. It does
   not add a fixed number: the vision rows scale with the canvas (2,016 → 814) and so do the
   condition latents (2,016 → 810, being `(30x54)/(2x2) x 2` keyframes against
   `(48x84)/(2x2) x 2`). fl2va's overhead is **proportional, not additive**, which is why
   +8.9 % and +8.7 % come out the same. The prediction was wrong for a reason that only
   shows up once you read what the encoder actually writes.

ref2va is not implemented on VDN's Ulysses fast path — see the README.

## 2K

MiniMax's 2K is **H3-Regenerate-2K**: the 768p result fed back into the base model
in-context, not a super-resolution network and not any GPU feature — and **not
open-sourced** (MiniMax's model page says so explicitly; it is API-only). 2K is 2560x1440
(probed from MiniMax's own reference outputs), 3.571x the 768p canvas's tokens per frame.

Because the module is not released, nothing here can measure it. What was measured is the
weaker question — what a 2560x1440 canvas costs *this* DiT, which is a lower bound, since
regeneration additionally carries the 768p clip's ~103k rows as context:

| frames | result |
|---|---|
| 345 | **OOM, in the denoise** — rank 0 asks for 63.61 GiB in one allocation with 11.94 GiB free |
| 243 | **OOM** the same way (ranks 6/7 first, 1.68 GiB short) |
| 90 | fits: **3.58 s/NFE**, denoise 28.63 s, decode+encode 4.27 s |

Two things worth taking from that. The wall is in the **denoise, not the decode** — the thing
patch 5 fixed is not what limits 2K. And at 3.58 s/NFE the denoise for 90 frames (3.75 s of
video) is already 3.3x the whole 480p/345f denoise, so even the lower bound is far from a
15-second clip on this box. Not pursued further; the real module is not available to run.

## Reproducing

```bash
# on the box, after scripts/h100_bringup.sh
setsid nohup bash h100_grid.sh > grid.log 2>&1 < /dev/null &
PHASES=s bash h100_grid.sh          # one phase
python3 summarize.py out            # the tables above
```

The headline arm on its own:

```bash
OMP_NUM_THREADS=24 MKL_NUM_THREADS=24 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --standalone --nproc_per_node=8 src/inference/infer_ulysses.py \
    --config configs/inference/8nfe_480p_345f_ulysses_h100.yaml \
    checkpoint=ckpts/stage-dmd-step-250 \
    render.prompt_file=prompts/example_2.pt \
    render.out=results/480p_345f.mp4 render.record=true
```

`OMP_NUM_THREADS` is not optional — see README trap 5.
