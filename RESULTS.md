# Measured: 480P / 15 s on 8x H100 80GB

Box `p5.48xlarge`, 8x H100 SXM 80 GB, us-east-2. Model `ckpts/stage-dmd-step-250`
(VDN-H3-8-step, the Stage-DMD distilled turbo adapter), fp8 e4m3 rowwise,
`softmax_backend: flex`, `inference_kernels: true`, 8 NFE, `warmup_steps: 2`, t2va
(`prompts/example_2.pt`). Every number here comes from a `*.inference.json` record written
by the run itself; `scripts/summarize.py` regenerates the tables from those records.

## The answer

**480P, 8 NFE, 8x H100, branch-parallel Ulysses at 3 softmax + 5 linear ranks, video VAE
data-parallel over all eight ranks.** Both reachable lengths near 15 s, since they cost
almost the same:

| | 345 f / 14.375 s | 362 f / 15.083 s |
|---|---|---|
| denoise, 8 NFE | **8.55** (1.069 s/NFE) | **9.02** (1.128 s/NFE) |
| video VAE, 8 ranks | 3.10 | 2.46 |
| audio VAE + frames-to-host + mux | 1.55 | 1.61 |
| decoder load (once per process, not per request) | 3.87 | 4.47 |
| **post-warmup end-to-end** | **17.07** | **17.57** |
| **steady state, decoders already resident** | **13.20** | **13.10** |
| steady vs clip length | 1.09x realtime | **1.15x realtime** |

(Arms `n_480p_seg4` and `n_480p_362f_seg4`. One caveat on how precisely to read these: the
**video VAE stage is the noisy one**, 2.11–3.82 s across seven arms at matched canvas, split
and decode path — median 2.97, σ 0.61. The denoise is stable to ±0.02 s/NFE and the tail to
±0.06 s, so treat the steady-state figure as **13.2 ± 0.6 s**, with the uncertainty living
entirely in the decode.)

**15.000 s is not a reachable length** — `align_num_frames(n, 17, 5)` snaps to `5 + 17k`,
so the grid near 15 s is 345 (14.375 s) then 362 (15.083 s), with nothing between. 362 is
the literal answer to "15 seconds", and at 13.10 s steady it renders **1.15x faster than the
clip plays**.

The two lengths cost the same for a reason worth stating: 362 frames is 21 temporal VAE
chunks against 345's 20, and `ceil(21/8) = ceil(20/8) = 3`, so the parallel decode's
critical path is three chunks either way. Parallel decode time is a function of
`ceil(chunks/8)`, not of frame count — it is flat across a whole band of clip lengths and
steps only when the chunk count crosses a multiple of 8.

Two framings of the same run, because they get quoted for different things:

* **1.069 s/NFE** is the number comparable with upstream's published table (2.29 on
  8x H200, 1.40 on 8x B200 — all at 768p, and all **denoise only**: their Results section
  excludes "model loading, warm-up, VAE decoding, and MP4 encoding", so their `18.3 s` is
  just `2.29 x 8`).
* **13.2 s steady-state end-to-end** is what a request costs on a warm server. Post-warmup
  E2E including the one-time decoder load is 17.1 s.

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
| **480p, 345 f** | 14.375 s | 13.20 s | $0.0872 | **$0.006065** | $21.83 |
| **480p, 362 f** | 15.083 s | 13.10 s | $0.0865 | **$0.005736** | $20.65 |
| **768p, 345 f** | 14.375 s | 29.50 s | $0.1948 | **$0.013554** | $48.80 |

Per 1000 clips: **$87** at 480p, **$195** at 768p. Using post-warmup E2E instead of steady
state — i.e. charging every request for the one-time decoder load, which is only honest for a
cold process serving a single clip — 480p is $0.007843/s ($113 per 1000) and 768p is
$0.014850/s ($213 per 1000).

Three things this table is and is not:

* **480p renders for less than the instance costs to run.** At 1.09x realtime the finished
  second costs 0.92x an instance-second; the 362-frame arm at 1.15x realtime costs 0.87x.
  768p costs 2.05x an instance-second, and the whole 2.2x gap to 480p is the row count.
* **The one-time build is not in here.** 138 s in a single process, 217–232 s in the 8-rank
  job (`setup`), = $0.91–$1.53 of instance time per process lifetime. Over a few hundred
  requests it rounds away; over three it doubles the bill. It is an argument for long-lived
  workers, and against any per-request DiT reload — a 768p server that freed and restored the
  weights each request would run 34.4–36.8 s and pay **$0.0158–0.0169 / video-second, +17 to
  +25 %**.
* **This is a latency-optimised price, and it is roughly 2x the cheapest way to buy these
  seconds.** 8 GPUs give a 4.03x denoise speedup over 1 GPU (4.39 -> 1.069 s/NFE at 480p), so
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
| **steady state** | **29.50** | 53.10 | **13.20** |
| vs clip length | 2.05x slower than realtime | 3.69x slower | 1.09x faster |

(Arms `p_768p_free` and `n_768p_seg4`. Both outputs `clipinfo.py`-clean.)

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
the same full-canvas tensor. Where those GiB go, from the tensor shapes at 768p (6.19 MB per
frame at 3 channels fp16):

* `gathered`, the all-gather destination — 24 slots of ~34 frames each, since every slot
  carries its chunk's decode overlap: **~5 GiB**.
* `dec`, the concatenated clip that `_blend` builds on top of it: **2.1 GiB**.

So rank 0 holds the whole clip roughly twice, in fp16 RGB at 6 bytes per pixel, while
patch 7 already knows how to represent a finished frame in **1.5** bytes per pixel.

**So yes, the DiT can stay resident at 768p — but it needs a patch, not a knob.** Two
candidates, both measured against the same 102 MiB deficit:

1. **Never materialise `dec`.** The assembly loop already yields one blended chunk at a time,
   and `torch.cat` exists only to hand `decode_and_save` a single tensor. Converting each
   chunk to YUV as it comes out of the loop and writing it into a preallocated plane set
   removes the 1.99 GiB entirely — **20x the margin needed** — without touching the
   all-gather, the NCCL traffic, or `_blend`'s arithmetic, which keeps running in fp16 RGB per
   chunk pair exactly as it does now. The awkward part is the trailing `pad_frames` trim,
   which has to be computed up front so the output can be sized before the loop rather than
   after it.
2. **YUV before the all-gather.** Cuts `gathered` *and* `dec` 4x — ~5.4 GiB back and 4x less
   NCCL traffic — but this is the harder one: `_blend` ramps across chunk boundaries in float
   and consecutive chunks live on different ranks by the round-robin, so the overlap frames
   would have to stay float while the interior went to uint8.

(1) is strictly smaller and already sufficient, so it is the one to write; (2) only becomes
worth it if the all-gather's 5 GiB or its NCCL time start mattering on their own. Either way
768p then runs `keep`, the release stage disappears, and the request goes 29.50 → **~26.7 s**
steady with no reload cost on any request — which is the number an API wants, per the reload
section above. At 480p there is nothing to fix, because there is nothing to evict.

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
(~7 s using the in-render 2.81 s release), added to a 29.50 s steady-state 768p request, for
+17–24 %. And it wants 45.24 GiB of *page-locked* host memory per rank — **362 GiB across 8
ranks**, which fits the box's 2 TiB but is locked away from the page cache and from the
frames-to-host staging in patch 7. All of that to free the 102 MiB the OOM was short of.

**So for a server the answer is not to cycle the weights at all.** At 480p — the primary
target — this whole question is void: `keep` is the default, nothing is evicted, and the
17.07 s / 13.20 s figures already are the API's numbers. At 768p the fix is the decode's
peak, not the DiT: the YUV-before-all-gather patch sketched above returns ~5.4 GiB on rank 0,
which is 50x the margin needed, and then 768p also runs `keep` and drops to ~26.7 s with no
reload cost on any request. Cycling 45.24 GiB twice per request to recover 102 MiB is the
wrong shape of fix; it is only in the tree because it was the cheapest thing that made a
one-shot 768p render finish.

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

1. **The forward is free. Everything else is memory movement.** 135 ms against a 13.2 s
   render is 1 %. VDN reads `hidden_states[50]`, which HF fills with the *input* to layer 50,
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
