# Runbook — everything here runs **on the box**

Every command below is meant to be typed in a shell on the p5.48xlarge. The two exceptions are
marked **[on your Mac]**: getting in, and pulling results out. Nothing needs this repo to be
checked out on the box — the box already has what it needs, and section 2 says how to get it
back if it doesn't.

Sections: **0** get in · **1** check the box · **2** rebuild a fresh box · **3** the queue of
arms still worth running · **4** results · **5** traps.

---

## 0. Getting in **[on your Mac]**

```bash
ssh -i /Users/henanwan/Documents/account/579019700964/henanwan/henanwan-us-east-2.pem \
    -o StrictHostKeyChecking=no ubuntu@18.189.225.222
```

**The IP, not `ec2-18-189-225-222.us-east-2.compute.amazonaws.com`.** After the stop/start the
public DNS record stopped resolving while the address stayed put. If the instance is ever
replaced, the address changes and `scripts/p5.sh` on the Mac needs its `HOST=` updated too.

Once in, everything lives under one directory, and it is on the **ephemeral** NVMe:

```
/opt/dlami/nvme/vdn/
├── vdn-minimax-h3/          the code. upstream repo at 2f740c9 + the 12 patches as commits
│   ├── .venv/               python 3.12, torch 2.13.0+cu129 -- runone.sh sources it for you
│   ├── src/  scripts/       patched sources; scripts/ holds the benches
│   ├── configs/inference/   the four 8nfe_*_h100.yaml
│   ├── prompts/             example_{0,1,2}.pt -- prompt caches, shipped in the repo
│   └── ckpts -> ../ckpts    82 GB of weights
├── runone.sh h100_grid.sh summarize.py box_check.sh   the drivers, one level UP from the repo
├── patches/                 the 12 .patch files, for rebuilding the tree
├── out/                     renders + their .inference.json records
└── *.log                    one per arm
```

`/opt/dlami/nvme` is an **instance store: a stop/start wipes it.** `/` is 484 GB and the
checkpoint alone is 82 GB, so nothing can move there. Assume everything above is gone after a
stop and that section 2 is the price of restarting.

Long arms take 6–15 minutes. Use `tmux` (installed) rather than hoping the ssh session holds:

```bash
tmux new -s vdn          # then ctrl-b d to detach, `tmux a -t vdn` to come back
```

---

## 1. Check the box before trusting it

```bash
bash /opt/dlami/nvme/vdn/box_check.sh
```

```
repo        : HEAD 5f4406b, 12 commits past 2f740c9, 0 tracked files modified
              -> series as commits, tree clean
configs     : 4/4 h100 yamls
weights     : 82G (expect 82G)
prompts     : 3 caches
drivers     : 3/3 in /opt/dlami/nvme/vdn
free on nvme: 27T
gpus busy   : 0 processes
python env  : torch 2.13.0+cu129  torchvision 0.28.0+cu129  diffusers 0.40.0.dev0  gpus 8
              OK
```

**Both `+cu129` suffixes matter more than anything else on that list** — see trap 1. If the
script says `NOT SET UP`, go to section 2. If `gpus busy` is not 0, something is still running:

```bash
nvidia-smi                                  # what
pkill -f 'infer_ulysses\.py'; sleep 4       # stop it (runone.sh does this itself)
```

---

## 2. Rebuilding a fresh box

Needed after any stop/start. ~10 minutes, most of it download.

### 2a. Get the two files the box cannot fetch by itself **[on your Mac]**

The bringup script and the patch series live in this private repo, and the box has no GitHub
credentials. One copy, and it is the only Mac-side step in the rebuild:

```bash
cd /Users/henanwan/Documents/workspace/bytedance/minimax_h3_h100
bash scripts/p5.sh 'mkdir -p /opt/dlami/nvme/vdn/staging'
bash scripts/p5.sh --put scripts/h100_bringup.sh /opt/dlami/nvme/vdn/h100_bringup.sh
bash scripts/p5.sh --put scripts/box_check.sh    /opt/dlami/nvme/vdn/box_check.sh
for f in patches/*.patch configs/*.yaml scripts/{runone,h100_grid,sglang_bringup,sglang_arm}.sh \
         scripts/{summarize,text_encoder_bench,mux_bench,reload_bench,clipinfo,vidcmp,vidscale,vidshift}.py; do
  bash scripts/p5.sh --put "$f" "/opt/dlami/nvme/vdn/staging/$(basename "$f")"
done
```

Everything lands in one `staging/` directory, which is what section 2b copies out of. (The box
currently has these files in `patches/` from the last rebuild; either name works as long as the
`cp` lines below match.)

### 2b. Everything else, on the box

```bash
cd /opt/dlami/nvme/vdn

# --- 1. uv + python 3.12, torch/torchvision 2.13.0+cu129, flash-attn-4, patched diffusers,
#        and the 82 GB OpenVDN checkpoint. ~10 min; the download is ~4 of it.
setsid nohup bash h100_bringup.sh > bringup.log 2>&1 < /dev/null &
tail -f bringup.log                          # ends with "=== [..] done"

# --- 2. PIN THE REPO, then rebuild diffusers, IN THAT ORDER. The clone lands on upstream
#        HEAD; the patch series is based on 2f740c9 and upstream has moved past it (e02ff07),
#        and that newer commit DELETES one of the diffusers patches. Pinning after installing
#        diffusers silently leaves you a patch short.
cd vdn-minimax-h3
git checkout -q --detach 2f740c9 && git log --oneline -1
export PATH=$HOME/.local/bin:$PATH UV_CACHE_DIR=/opt/dlami/nvme/vdn/uvcache
source .venv/bin/activate
rm -rf diffusers && bash scripts/setup_diffusers.sh

# --- 3. apply the 12 patches as real commits. `git am` needs an identity; any will do.
git config user.email vdn@p5.local && git config user.name p5
git am --3way /opt/dlami/nvme/vdn/staging/00*.patch
git log --oneline 2f740c9..HEAD | wc -l      # 12

# --- 4. the configs and benches, which are NOT in the patch series
cp /opt/dlami/nvme/vdn/staging/8nfe_*_h100.yaml configs/inference/
cp /opt/dlami/nvme/vdn/staging/{text_encoder_bench,mux_bench,reload_bench,clipinfo,vidcmp,vidscale,vidshift}.py scripts/
cp /opt/dlami/nvme/vdn/staging/{runone.sh,h100_grid.sh,summarize.py} /opt/dlami/nvme/vdn/

# --- 5. verify, then run the control (B0) before trusting any new number
bash /opt/dlami/nvme/vdn/box_check.sh
```

`scripts/decode_parity.py`, `src/inference/utils/parallel_vae.py` and
`src/inference/utils/yuv.py` come from the patch series, not from step 4 — if they are missing,
`git am` did not run.

`git am --3way` on the 12 patches has been verified clean against a fresh 2f740c9 on this box,
and the tree it produces is byte-identical to the one the 11.44 s measurement came off.

---

## 3. The queue

> **A is done, and it won: 480p 8.02 s median / 768p 19.04 s median, ten requests each, nothing
> offloaded, peak 62.1 GB/GPU at 768p** (2026-09-16, sglang main `3f8eb35e`). That is 1.43× and
> 1.74× the reference stack, with text encoding inside SGLang's number and outside the reference
> stack's. **A0 and A1 below are still the instructions to reproduce it**, with two additions the
> run needed and this file did not predict: `ffmpeg`/`ffprobe` must exist before startup, and
> `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is what makes it fit — the offload ladder in
> A1 was never used. `scripts/sglang_arm.sh` carries all five fixes with the symptom each produced.
> Numbers and the full account: `RESULTS.md`, "SGLang Diffusion is faster than all of it".
>
> This makes **B1–B5 optional**, and B4 (pricing a conditioner in the request path) moot: the
> conditioner is resident in the winning configuration and its cost is already inside the 8.02 s.

**Arm A comes first, and it may make most of B obsolete.** SGLang Diffusion now serves this
exact checkpoint, and on 8× B200 it beats the stack measured here by 1.43–1.60× at the same GPU
count. Run A1 before spending time on B1–B5.

The B arms are the patched reference stack. Ordered so each is interpretable given the ones above
it. Times include the ~220 s model build, which every arm pays once, and all of them run from
`/opt/dlami/nvme/vdn`.

The pattern is always the same:

```bash
cd /opt/dlami/nvme/vdn
setsid nohup bash runone.sh <tag> <config> 8 [key=value ...] > <tag>.log 2>&1 < /dev/null &
tail -f <tag>.log
```

`runone.sh` sources the venv, sets `OMP_NUM_THREADS=24` (trap 2), `expandable_segments`, kills
any previous `infer_ulysses.py`, and passes `render.record=true`. Overrides are OmegaConf
dotlist, so anything in `src/config/inference.py` can go on the end.

### A0 — install SGLang Diffusion beside the reference stack. **~15 min, ~250 GB.**

```bash
mkdir -p /opt/dlami/nvme/sglang
cp /opt/dlami/nvme/vdn/staging/sglang_{bringup,arm}.sh /opt/dlami/nvme/vdn/
setsid nohup bash /opt/dlami/nvme/vdn/sglang_bringup.sh \
    > /opt/dlami/nvme/sglang/bringup.log 2>&1 < /dev/null &
tail -f /opt/dlami/nvme/sglang/bringup.log
```

**Why this is now the top of the queue.** SGLang's cookbook
([`docs/cookbook/diffusion/MiniMax/MiniMax-H3.mdx`](https://github.com/sgl-project/sglang/blob/main/docs/cookbook/diffusion/MiniMax/MiniMax-H3.mdx),
section 7) serves `OpenVDN/vdn-minimax-h3` directly, on the same 345-frame 1344×768 workload,
and publishes a head-to-head against *this* stack — its "OpenVDN reference" rows are
`8nfe_tuned_fp8.yaml` + `infer_ulysses.py` with `parallel.softmax_ranks` swept, i.e. the thing
B0–B2 measure. On 8× B200: **0.88 s/NFE against the reference stack's 1.40**, and **0.98** on
the per-channel fp8 path, which is the one an H100 (SM90) takes. It also implements as *flags*
three things this repo carries as patches — encoder folding across idle ranks, VAE
residency/offload policy, layerwise DiT placement — and it is an HTTP server, which is the shape
the eventual deployment needs anyway.

The cookbook credits the gap to parallel efficiency, not Blackwell precision: 86–91 % from 2 to
8 cards against the reference stack's 56–64 %, with the two stacks within 4 % of each other on
**one** card (6.03 vs 6.25 s/NFE fp8). That is the claim to test here, because it is the claim
that should carry to H100 — and H100's NVLink is roughly half B200's, so the all-to-all it
attributes the win to is *relatively more* expensive on this box, not less.

**What the cookbook does not have: any H100 VDN number.** Its VDN tables are B200 and
RTX PRO 6000; its H100 rows are 4-card *base* H3 (TP2+Ulysses2, 13.25 s). 8× H100 Ulysses8 for
VDN is legal but unverified, and 480p is off the released 768 target (it appears only in the
consumer sections, at 864×480 — our exact canvas). So A1 is a measurement, not a lookup.

Two failure modes to expect at install time. The PyPI wheel may predate section 7 — the script
gates on `sglang serve --help | grep hybrid_window_attn_h3` and tells you the `git+` line if it
is missing. And disk: it cannot reuse `/opt/dlami/nvme/vdn/ckpts` (that was a `--local-dir`
download with no cache layout), it hard-links the conditioner and VAEs out of
`MiniMaxAI/MiniMax-H3` so that repo comes down too, and the first launch prefuses both adapters
into the transformer as a **62 GB write**. ~250 GB on the NVMe, which has 27 TB.

### A1 — SGLang on eight H100s, both canvases. **~10 min each after A0.**

```bash
cd /opt/dlami/nvme/vdn
setsid nohup bash sglang_arm.sh serve 480 > /dev/null 2>&1 < /dev/null &
tail -f /opt/dlami/nvme/sglang/logs/serve_480p.log      # wait for the warmup to finish
bash sglang_arm.sh bench 480 10                          # 1 warmup + 10 measured
bash sglang_arm.sh stop
```

Then the same with `768`. **Compare against B0's 11.44 s and B1's 33.21 s** — and note the
comparison is loaded *against* SGLang: its request latency includes text encoding, which every
number in RESULTS.md excludes, because the reference stack `torch.load`s an offline prompt cache.
Read the broken-out `text encoding` line before the totals. On 8× B200 that stage costs ~0.2 s
folded across ranks, so if it costs similarly here it does not decide anything; on one RTX 5090
it was 4.2 s, so it is not free by construction.

**Expect, and this is a prediction rather than a reading:** if the parallel-efficiency claim
carries, 768p denoise goes 20.97 → **~14–15 s** and 480p 8.71 → **~6 s**, putting 480p
end-to-end near **8 s** against 11.44. If instead it lands within noise of the reference stack,
the 1.43× was Blackwell-specific and the patched stack stays.

**The first thing this settles is memory, not speed.** 8× B200 Ulysses8 peaked at
**79,972 MB/GPU**. This card has 81,559 MiB total and gives PyTorch **65.26 GiB** after the
measured 13.92 GiB non-PyTorch floor, so `--performance-mode speed` at 768p may simply not fit.
In order, the things to trade:

```bash
bash sglang_arm.sh serve 768 --performance-mode auto           # 120 GiB residency threshold
bash sglang_arm.sh serve 768 --layerwise-offload-components text_encoder
bash sglang_arm.sh serve 768 --layerwise-offload-components dit,text_encoder \
                             --dit-layerwise-resident-layers 14
```

Each of those costs latency, so record which one was needed: "SGLang is faster" and "SGLang is
faster while offloading the DiT" are different findings. A 480p fit is much more likely than a
768p one — 480p is a quarter of the packed rows (43,759 vs 105,265) — which is why A1 runs 480p
first, and 480p is the target anyway.

If A1 wins, the honest conclusion is that this repo's twelve patches were the right way to find
out *where the time goes* (the 3+5 branch split, the decode's 8.1 GiB, the fp8 host assembly) and
the wrong way to *serve* it, and B3–B5 should be dropped in favour of re-running fl2va and the
conditioner questions inside SGLang, where both are already flags.

### B0 — the control: 480p/345f, ten requests. **~7 min. Nothing means anything without it.**

```bash
setsid nohup bash runone.sh f1_480p_rep10 8nfe_480p_345f_ulysses_h100.yaml 8 \
    render.repeat=10 > f1_480p_rep10.log 2>&1 < /dev/null &
```

**Why:** a rebuilt environment is not the same environment until it is shown to be. Everything
else here is a comparison against numbers measured on a machine that no longer exists.

**Expect**, requests 2–10: total **11.44 ± 0.05 s**, denoise 8.71, video VAE 1.56,
decode+encode 1.17; request 1 ≈ **16.1 s** (decoder load ~3.6 inside it); decode peak
~62.3 GiB reserved; reserved growth over the ten +0.27 GiB. **Already done on this box** — it
reproduced the old 11.447 ± 0.040 to 0.06 %. Re-run it after any environment change and
nowhere else. Outside ±2 % on the mean, stop and check the two `+cu129` suffixes.

Read the result with:

```bash
grep -E "^(steady|reserved after| req|  [0-9])" f1_480p_rep10.log
```

### B1 — the 768p control. **~10 min.**

```bash
setsid nohup bash runone.sh f2_768p_rep10 8nfe_768p_345f_ulysses_h100.yaml 8 \
    render.repeat=10 > f2_768p_rep10.log 2>&1 < /dev/null &
```

**Expect:** **33.21 ± 0.85 s** over requests 2–10, with a decoder load ~5.0 s and release
~1.8 s *inside each request* (that config cycles the decoders on purpose), and peak reserved
63.59 GiB **identical at request 2 and request 10**. Zero memory growth is as much the point of
this arm as the mean is.

### B2 — reproduce the OOM the whole 768p design rests on. **~6 min, expected to fail.**

```bash
setsid nohup bash runone.sh f3_768p_keepkeep 8nfe_768p_345f_ulysses_h100.yaml 8 \
    render.repeat=2 parallel.vae_after_decode=keep > f3_768p_keepkeep.log 2>&1 < /dev/null &
```

**Why this one first if you only run one:** `vae_after_decode: free`, the 4.7x argument, and the
corrected memory table in RESULTS.md all rest on **one** OOM message from **one** run. It is
the load-bearing negative result in the study, it has never been reproduced, and it is the
cheapest arm here.

**Expect:** request 1 completes (denoise ~19.8 s); request 2 dies in the denoise on ranks
**5, 6, 7** — exactly the three linear-branch ranks at `softmax_ranks: 5` — asking for
444–468 MiB with 120–440 MiB free and 64.7–65.1 GiB already allocated by PyTorch.

```bash
grep -E "OutOfMemory|is allocated by PyTorch|total capacity" f3_768p_keepkeep.log | sort -u
```

**What it settles:** the non-PyTorch floor on this newer driver (595.91.07). RESULTS.md derives
13.92 GiB from the old log, and the 10.32 GiB of denoise headroom is that number's complement.
If the floor moved, the 768p config's justification needs re-stating, not just re-measuring.

### B3a — fl2va end to end at 768p. **~11 min, no download, nothing to prepare.**

The only fl2va numbers so far are **denoise-only**: +8.9 % at 768p, +8.7 % at 480p. What a
first/last-frame request actually costs a caller has never been measured, and it has to come out
*lower* than the per-NFE figure, because the decode and the tail do not scale with conditioning
rows — only the denoise does.

The repo ships a ready 768p keyframe cache, `prompts/image/example_fl2va.pt` (3,485 embedding
rows, anchors `first`+`last`, `condition_latents` 2 × (1,24,1,48,84), encoded from
`prompts/image/{first,last}.png` at 768×1344), so this arm needs **no conditioner download and
no encoding step**:

```bash
cd /opt/dlami/nvme/vdn
setsid nohup env PROMPT=prompts/image/example_fl2va.pt bash runone.sh f4_768p_fl2va_rep10 \
    8nfe_768p_345f_ulysses_h100.yaml 8 render.repeat=10 \
    > f4_768p_fl2va_rep10.log 2>&1 < /dev/null &
```

**Expect** requests 2–10: denoise 20.97 × 1.089 ≈ **22.8 s**, so a request total of
**~35.1 s against B1's 33.21 — about +5.6 %**, not +8.9 %. If it lands near +8.9 % instead, the
premise above is wrong and something *does* scale with the conditioning rows outside the
denoise, which is worth knowing. Run B1 in the same session or immediately before, so the
comparison is same-process-generation.

Note this cache carries a **different prompt** from `prompts/example_2.pt` (same 1,299 text
rows, different words). That is fine for a cost comparison — the row counts are recorded and
they are what the time tracks — but it is not a same-prompt A/B, so do not compare the *pixels*.

Check the row split before reading the timings; it is what makes the result a mechanism rather
than a number:

```bash
cd /opt/dlami/nvme/vdn/vdn-minimax-h3 && .venv/bin/python -c "import json; \
print(json.load(open('/opt/dlami/nvme/vdn/out/f4_768p_fl2va_rep10.mp4.inference.json'))['sequence_splits'])"
# expect text+vision 3485, condition 2016, audio 1150, video 102816 -> 109467
```

### B3b — fl2va at 480p. **~15 min, plus a 62 GiB download.**

Only worth doing after B3a, and only if you need 480p specifically: it tests whether fl2va's
overhead really is **proportional rather than additive** (the claim that explains why +8.9 % and
+8.7 % came out the same at two canvases). **A keyframe cache is only valid at the canvas it was
encoded for** — `condition_latents` come out at that canvas's latent size — so 480p needs its
own, which is what patch 6 is for. Encoding needs the **Qwen3-VL conditioner**: 62 GiB from
`MiniMaxAI/MiniMax-H3`, which is *not* in the OpenVDN checkpoint.

Reuse the shipped cache's own prompt and images so B3a and B3b differ only in canvas:

```bash
cd /opt/dlami/nvme/vdn/vdn-minimax-h3 && source .venv/bin/activate
export HF_HOME=/opt/dlami/nvme/vdn/hf HF_HUB_ENABLE_HF_TRANSFER=1

TEXT=$(python -c "import torch; print(torch.load('prompts/image/example_fl2va.pt', \
    weights_only=True)['prompt'], end='')")
python src/inference/encode_keyframes.py --prompt "$TEXT" \
    --first prompts/image/first.png --last prompts/image/last.png \
    --height 480 --width 864 --out prompts/fl2va_480p.pt
# prints: wrote ... N tokens (980 vision rows), 2 keyframes [(1, 24, 1, 30, 54), ...]

cd /opt/dlami/nvme/vdn
setsid nohup env PROMPT=prompts/fl2va_480p.pt bash runone.sh f5_480p_fl2va_rep10 \
    8nfe_480p_345f_ulysses_h100.yaml 8 render.repeat=10 \
    > f5_480p_fl2va_rep10.log 2>&1 < /dev/null &
```

**Expect:** rows `2279 / 810 / 1150 / 41310 = 45549`, and 11.44 → **~12.2 s (+6.5 %)**. The
prediction that matters is that B3b's percentage and B3a's are again the *same*, because the
vision rows and the condition latents both scale with the canvas.

### B4 — pricing a conditioner in the request path. **~5 min, needs no weights at all.**

```bash
cd /opt/dlami/nvme/vdn/vdn-minimax-h3 && source .venv/bin/activate
torchrun --standalone --nproc_per_node=8 scripts/text_encoder_bench.py --transfer 6.1
```

**Why:** the recommendation in RESULTS.md — keep the trimmed conditioner as a **pinned** host
copy and upload 6.1 GiB per rank per request, ~0.8 s, +7 % — rests on a **single-rank**
10.08 GiB/s figure applied to eight simultaneous uploads. Eight H2D streams share host memory
bandwidth and NUMA paths. If the per-rank rate collapses to 4 GiB/s the answer is 1.5 s and
+13 %, which changes the recommendation. The arm prints the single-rank transfer next to the
eight-rank one, so the contention is measured instead of assumed. It needs no checkpoint, so it
is nearly free.

The same numbers price the unbuilt **pinned host copy of the video VAE** (9.70 GiB/rank), which
would take 768p from 33.21 to ~28 s by replacing the 5.0 s disk load. Run B4 and that becomes
arithmetic.

The model-side numbers, if the conditioner is downloaded anyway (single GPU, ~10 min):

```bash
python scripts/text_encoder_bench.py --tokens 1300      # load, forward, trim, offload
```

Then, and only if the transfer holds up, the actual implementation: because VDN reads
`hidden_states[50]`, only **layers 0–50** are needed, so a **pipeline** shard is enough — rank
*r* owns a contiguous slice of the 51 layers and hands on a 1300×5120 bf16 = **13 MB** tensor,
seven hops over NVLink. No tensor parallelism, no custom kernels. Designed, **not implemented**;
`--transfer` prices its dominant cost first on purpose.

### B5 — cu130, in a separate venv. **~20 min.**

Worth knowing, not worth mixing — so build a second venv and leave the control intact:

```bash
cd /opt/dlami/nvme/vdn/vdn-minimax-h3
export PATH=$HOME/.local/bin:$PATH UV_CACHE_DIR=/opt/dlami/nvme/vdn/uvcache
uv venv --python 3.12 .venv130 && source .venv130/bin/activate
uv pip install -q torch==2.13.0 torchvision==0.28.0          # cu130 IS the PyPI default
uv pip install -q --prerelease=allow -e .
DIFFUSERS_DIR=diffusers130 bash scripts/setup_diffusers.sh
```

(That cu130 is the default with no index specified is exactly why trap 1 exists.) Then re-run
**B0 inside it** and compare against 11.44 ± 0.05 — `runone.sh` hardcodes `.venv`, so either
edit it or run `torchrun` directly.

**Expect little.** cu129 is upstream VDN's own pin (their README requires `2.13.0+cu129`,
because flash-attn-4's `nvidia-cutlass-dsl` otherwise drags in the cu130 default), the hot path
is FA4's CuteDSL kernels plus this repo's Triton kernels and `torch.compile` rather than cuBLAS,
and sm90 is not where cu130's work went. It may also simply not build — that is a real risk,
not a formality. If it *is* faster by more than the control's ±2 %, the whole table has to be
re-measured there, and that cost is what the gain has to beat.

### Still open, no code yet

* **Pinned host copy of the video VAE** — 768p 33.21 → ~28 s. B4 gives the transfer rate.
  Superseded if A1 wins: SGLang's `--component-residency vae=resident` is the same idea as a
  flag, and its cookbook already reports 4.8 GiB/GPU for it on a 2×H100 CI recipe.
* **Why 768p's steady denoise is *slower* than its own first request** (19.77 → 20.97 s) while
  480p's is flat. Investigate with `parallel.profile=true` appended to B1. Worth doing either
  way — it is a property of the reference stack that no SGLang number explains.
* **fl2va and the conditioner inside SGLang.** Both are flags there (`task: "fl2va"` with
  keyframe conditions; `--encoder-parallel auto`), which is what B3 and B4 exist to hand-build.
  If A1 lands, re-ask them there instead.

---

## 4. Results

Every arm writes an mp4 and, because `runone.sh` passes `render.record=true`, a JSON record
next to it:

```bash
ls -la /opt/dlami/nvme/vdn/out/
cd /opt/dlami/nvme/vdn/vdn-minimax-h3 && .venv/bin/python scripts/clipinfo.py \
    /opt/dlami/nvme/vdn/out/<tag>.mp4                      # canvas, frames, duration, audio
python /opt/dlami/nvme/vdn/summarize.py /opt/dlami/nvme/vdn/out   # the RESULTS.md tables
```

`<tag>.mp4.inference.json` holds the checkpoint identity, the fully resolved config, the actual
kernel state, per-NFE timings, per-request timings under `requests`, and `sequence_splits`.
**Copy the JSON off the box even for a run that failed** — it is the only durable record, and
the previous instance was terminated with an mp4 transfer half-finished.

**[on your Mac]**, to pull things back:

```bash
cd /Users/henanwan/Documents/workspace/bytedance/minimax_h3_h100
bash scripts/p5.sh --get '/opt/dlami/nvme/vdn/out/<tag>.mp4.inference.json' rescue/
bash scripts/p5.sh --get '/opt/dlami/nvme/vdn/out/<tag>.mp4' samples/
```

---

## 5. Traps

1. **`torchvision` must be `+cu129`.** `pyproject.toml` pins `torchvision==0.28.0` with no
   local version, so if it is left to `uv pip install -e .` the resolver takes the PyPI
   default — **cu130** — and then every `import diffusers.loaders.peft` dies with `PyTorch has
   CUDA Version=12.9 and torchvision has CUDA Version=13.0`, surfacing unhelpfully as
   `Could not import module 'BloomPreTrainedModel'`. `h100_bringup.sh` now names both wheels
   from the cu129 index up front, which is the only ordering with no repair step, because the
   obvious repair does not work: `uv pip install torchvision==0.28.0 --index-url .../cu129` is
   a **no-op** (0.28.0 already satisfies 0.28.0, so `0.28.0+cu129` is never fetched). If you
   are already in that state:

   ```bash
   cd /opt/dlami/nvme/vdn/vdn-minimax-h3 && source .venv/bin/activate
   uv pip install --reinstall-package torchvision torchvision==0.28.0 \
       --index-url https://download.pytorch.org/whl/cu129
   ```

2. **`torchrun` sets `OMP_NUM_THREADS=1`** and says so in its own output. Right for a GPU
   render, wrong for patch 2's host-side LoRA merge and fp8 quantise: at 1 thread all 8 ranks
   sit at 99 % of a single core and are still going after six minutes. `runone.sh` sets
   `192 / nproc_per_node`, which brings the assembly in at ~200 s. If you invoke `torchrun`
   by hand, set it by hand.

3. **Do not use the DLAMI's `/opt/pytorch`.** Python 3.13 + cu130; the repo requires
   `>=3.12,<3.13` and the cu129 wheels. **SGLang fails there too, and the error names the
   wrong culprit:** `sglang[diffusion]` pins an `outlines_core` 0.1.x whose newest wheel is
   **cp312** (checked on PyPI: every 0.1.x stops at cp312; cp313 first appears in 0.2.9), so
   under 3.13 pip has no wheel, falls back to the sdist, and dies on
   `error: can't find Rust compiler`. Installing Rust is the wrong fix — it makes that one
   package build, leaves you on 3.13 for the next gap, and installing into `/opt/pytorch` at
   all would overwrite the shared DLAMI torch. Build a private 3.12 instead, which is what
   `sglang_bringup.sh` does; it now refuses to run with `$VIRTUAL_ENV` set and asserts the
   venv is 3.12 before installing anything.

4. **Everything on `/opt/dlami/nvme`.** `/` is 484 GB, the checkpoint is 82 GB, and the NVMe
   is 27 TB — but it is an instance store and a stop/start wipes it.

5. **Host RAM is a real constraint.** Patch 2 assembles on the CPU, so 8 ranks each hold
   ~78 GiB of bf16 weights at once, ~620 GB. Fine on this box's 2 TB; it would not fit a
   smaller one.

6. **`git checkout .` / `git stash` / `git reset --hard` inside the repo.** Harmless in the
   current state (the 12 patches are commits, `box_check.sh` says `tree clean`), destructive in
   the older one where they were file copies. `box_check.sh` tells you which state you are in.

7. **Multi-rank denoise is not bit-reproducible** — `index_add_` atomics plus `all_reduce`
   reorder floating-point work, so the same seed across two runs gives ~17 dB PSNR, not
   identity. Any parity claim about the decode has to be made in one process with fixed
   latents, which is what `scripts/decode_parity.py` does.
