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
| denoise, 8 NFE | **8.72** (1.090 s/NFE) | **9.09** (1.136 s/NFE) |
| video VAE, 8 ranks | 2.97 | 2.55 |
| audio VAE + frames-to-host + mux | 3.18 | 3.27 |
| decoder load (once per process, not per request) | 3.65 | 5.41 |
| **post-warmup end-to-end** | **18.53** | **20.32** |
| **steady state, decoders already resident** | **14.88** | **14.91** |
| steady vs clip length | 0.97x realtime | **1.01x realtime** |

**15.000 s is not a reachable length** — `align_num_frames(n, 17, 5)` snaps to `5 + 17k`,
so the grid near 15 s is 345 (14.375 s) then 362 (15.083 s), with nothing between. 362 is
the literal answer to "15 seconds", and at 14.91 s steady it renders **slightly faster than
the clip plays**.

The two lengths cost the same for a reason worth stating: 362 frames is 21 temporal VAE
chunks against 345's 20, and `ceil(21/8) = ceil(20/8) = 3`, so the parallel decode's
critical path is three chunks either way. Parallel decode time is a function of
`ceil(chunks/8)`, not of frame count — it is flat across a whole band of clip lengths and
steps only when the chunk count crosses a multiple of 8.

Two framings of the same run, because they get quoted for different things:

* **1.090 s/NFE** is the number comparable with upstream's published table (2.29 on
  8x H200, 1.40 on 8x B200 — all at 768p, all denoise-only).
* **14.9 s steady-state end-to-end** is what a request costs on a warm server. Post-warmup
  E2E including the one-time decoder load is 18.5 s.

Two corrections to earlier versions of this file, both the same mistake — comparing two
numbers measured at **different branch splits**:

* 362 frames was reported as "+25 % denoise (1.367 s/NFE)". That measurement was at standard
  Ulysses while 345's was at 3+5. Like-for-like it is **+4.2 %** (1.136 vs 1.090), which
  tracks the +4.9 % row count. Clip length is linear here; nothing about 362 is special.
* The H100↔H200 gap was reported as 1.18x from H100 `r0` against H200 `r6`. At the same
  split it is **1.13x**. See the 768p section.

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

**The bottleneck is now the H.264 mux, at 2.66 s** — more than the eight-way video VAE
decode it follows, and it is host-side with no GPU involvement at all. See the next section:
it turned out not to be libx264.

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
draws, identical on every rank. Nor is it the parallelism: the 768p pair is at standard
Ulysses with no branch split, and Ulysses' collectives are all-to-alls, which are
permutations and bitwise exact. What is left is the fp8 GEMMs, whose split-k reductions
accumulate with atomics, amplified over eight sampler steps.

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
mp4s. It reported `MISMATCH` at both canvases, and **the test was wrong, not necessarily the
decode** — it was comparing two denoise trajectories. `scripts/decode_parity.py` (patch 8)
replaces it: one process, one latent tensor, decoded both ways back to back, nothing upstream
in the comparison.

_Pending: the parity result from the box._

## Text encoding is not in any of these numbers

Neither here nor upstream — the Qwen3-VL conditioner runs once, offline, and the render
`torch.load`s a cached `prompt_embeds`. See the README for the sizing (63 GB checkpoint,
~49 GB actually needed, host offload ruled out at 1.7 GB/s measured, 8-way shard at 6.1 GiB
per rank) and `scripts/text_encoder_bench.py` for the measurement.

_Pending: bench results from the box._

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
