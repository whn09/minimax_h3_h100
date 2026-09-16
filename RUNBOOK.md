# Runbook: rebuild the box, and the arms still worth running

Written for operating this by hand. Two halves: **A** brings a bare p5.48xlarge back to the
state every number in RESULTS.md was measured in, **B** is the queue of measurements that are
still open, each with why it is worth the GPU time and what result would settle what.

Everything assumes `scripts/p5.sh` points at the box. After a stop/start the **public DNS name
stops resolving while the address stays put**, so `p5.sh` now names the IP; if the instance is
replaced, edit `HOST=` there (and nothing else).

```bash
HOST=ubuntu@<new-ip> bash scripts/p5.sh 'nvidia-smi -L'      # one-off override
```

---

## A. Bringing a fresh box up (~10 minutes, most of it download)

The ephemeral NVMe (`/opt/dlami/nvme`, 27 TB) is **wiped by a stop/start**. `/` is 484 GB and
too small for the checkpoint, so everything — python, venv, repo, weights — is rebuilt there
every time. Nothing in `$HOME` matters.

```bash
# 1. push the bringup script and run it detached (uv + python 3.12, torch 2.13.0+cu129,
#    flash-attn-4, patched diffusers, the 82 GB OpenVDN checkpoint)
bash scripts/p5.sh 'mkdir -p /opt/dlami/nvme/vdn'
bash scripts/p5.sh --put scripts/h100_bringup.sh /opt/dlami/nvme/vdn/h100_bringup.sh
bash scripts/p5.sh 'cd /opt/dlami/nvme/vdn && setsid nohup bash h100_bringup.sh \
    > bringup.log 2>&1 < /dev/null &'
bash scripts/p5.sh 'tail -f /opt/dlami/nvme/vdn/bringup.log'   # ends with "=== done"

# 2. PIN THE REPO. The clone lands on upstream HEAD; the patch series is based on 2f740c9
#    and upstream has moved past it (e02ff07 at the time of writing, which also DELETES one
#    of the diffusers patches). Pin first, then rebuild diffusers, in that order.
bash scripts/p5.sh 'cd /opt/dlami/nvme/vdn/vdn-minimax-h3 && git checkout -q --detach 2f740c9 \
    && git log --oneline -1'
bash scripts/p5.sh 'cd /opt/dlami/nvme/vdn/vdn-minimax-h3 \
    && export PATH=$HOME/.local/bin:$PATH UV_CACHE_DIR=/opt/dlami/nvme/vdn/uvcache \
    && source .venv/bin/activate && rm -rf diffusers && bash scripts/setup_diffusers.sh'

# 3. push the patched sources, the four configs and the drivers
bash scripts/sync_box.sh

# 4. prove the environment before trusting any number out of it (see B0)
bash scripts/p5.sh 'cd /opt/dlami/nvme/vdn && setsid nohup bash runone.sh \
    f1_480p_rep10 8nfe_480p_345f_ulysses_h100.yaml 8 render.repeat=10 \
    > f1_480p_rep10.log 2>&1 < /dev/null &'
```

### Two traps that bit on the rebuild

1. **`torchvision` must come from the cu129 index, named explicitly, *before* `-e .`.**
   `pyproject.toml` pins `torchvision==0.28.0` with no local version, so `uv pip install -e .`
   resolves it from PyPI — which is the **cu130** build — and then *every* import of
   `diffusers.loaders.peft` dies with `PyTorch has CUDA Version=12.9 and torchvision has CUDA
   Version=13.0`, surfacing confusingly as `Could not import module 'BloomPreTrainedModel'`.
   Repairing it afterwards is not enough either: `uv pip install torchvision==0.28.0
   --index-url .../cu129` is a **no-op**, because 0.28.0 already satisfies 0.28.0 and uv never
   fetches `0.28.0+cu129`. It needs `--reinstall-package torchvision`. `h100_bringup.sh` now
   installs both from cu129 up front, which is the only ordering with no repair step. (The
   README documented this trap before the script implemented it.)
2. **cu129 is upstream's pin, not a preference.** VDN's own README says the code requires
   `2.13.0+cu129` and installs it from that index specifically because flash-attn-4's
   dependency tree (`nvidia-cutlass-dsl`) otherwise pulls the cu130 default. The DLAMI's
   `/opt/pytorch` is python 3.13 + cu130 and is unusable for a second reason: the repo pins
   `requires-python >=3.12,<3.13`. A cu130 arm is a legitimate experiment (B5) but it is a
   **different environment**, so it invalidates comparison against every table here and
   against upstream's H200/B200 figures until the control arm is re-run inside it.

### Verifying, before running anything long

```bash
bash scripts/p5.sh 'cd /opt/dlami/nvme/vdn/vdn-minimax-h3 && source .venv/bin/activate && python -c "
import torch, torchvision, diffusers, transformers, flash_attn
import diffusers.loaders.peft
print(torch.__version__, torchvision.__version__, diffusers.__version__, torch.cuda.device_count())"'
# expect: 2.13.0+cu129 0.28.0+cu129 0.40.0.dev0 8
```

Both local versions must read `+cu129`. If `torchvision` reads bare `0.28.0`, trap 1 has
happened.

---

## B. The queue

Ordered so that each arm's result is interpretable given the ones above it. Times are GPU
wall-clock including the ~220 s model build, which every arm pays once.

### B0 — the control. 480p/345f, ten requests. **~7 min. Run this first, always.**

```bash
bash scripts/p5.sh 'cd /opt/dlami/nvme/vdn && setsid nohup bash runone.sh \
    f1_480p_rep10 8nfe_480p_345f_ulysses_h100.yaml 8 render.repeat=10 \
    > f1_480p_rep10.log 2>&1 < /dev/null &'
```

**Why:** a rebuilt environment is not the same environment until it is shown to be. Everything
else in this file is a comparison against a number measured on the box that no longer exists.

**Expect** (requests 2–10): total **11.447 ± 0.040 s**, denoise 8.695, video VAE 1.568,
decode+encode 1.181; request 1 **17.16** (decoder load 4.81 inside it); peak reserved
**62.89 → 63.14 GiB**. Anything outside about ±2 % on the mean means the environment differs
and the rest of the queue is not comparable — check the two local versions first.

### B1 — the 768p control. **~10 min.**

```bash
bash scripts/p5.sh 'cd /opt/dlami/nvme/vdn && setsid nohup bash runone.sh \
    f2_768p_rep10 8nfe_768p_345f_ulysses_h100.yaml 8 render.repeat=10 \
    > f2_768p_rep10.log 2>&1 < /dev/null &'
```

**Expect:** **33.206 ± 0.850 s** over requests 2–10, decoder load 5.05 + release 1.82 inside
each, peak reserved 63.59 **identical at request 2 and request 10** (zero growth is the point
of the arm as much as the mean is).

### B2 — reproduce the OOM the whole 768p design rests on. **~6 min, expected to fail.**

```bash
bash scripts/p5.sh 'cd /opt/dlami/nvme/vdn && setsid nohup bash runone.sh \
    f3_768p_keepkeep 8nfe_768p_345f_ulysses_h100.yaml 8 \
    render.repeat=2 parallel.vae_after_decode=keep \
    > f3_768p_keepkeep.log 2>&1 < /dev/null &'
```

**Why:** `vae_after_decode: free`, the 4.7x argument, and the corrected memory table in
RESULTS.md all rest on **one** OOM message from **one** run. It is the load-bearing negative
result in the whole study and it has never been reproduced. It is also the cheapest arm here.

**Expect:** request 1 completes (denoise ~19.8 s), request 2 dies in the denoise on ranks
**5, 6, 7** — the three linear-branch ranks at `softmax_ranks: 5` — asking for 444–468 MiB
with 120–440 MiB free and 64.7–65.1 GiB already allocated by PyTorch. Grep it out with:

```bash
bash scripts/p5.sh 'grep -E "OutOfMemory|GiB is allocated|total capacity" \
    /opt/dlami/nvme/vdn/f3_768p_keepkeep.log | sort -u'
```

**What it settles:** the non-PyTorch floor (total capacity − in use + free arithmetic) on a
freshly booted driver, and whether the margin is still a few hundred MiB. If the floor moved —
driver 595.91.07 here — the 10.32 GiB of headroom moves with it and the 768p config's
justification needs re-stating, not just re-measuring.

### B3 — fl2va end to end. **~15 min including the conditioner download.**

The only fl2va numbers so far are **denoise-only** (+8.9 % at 768p, +8.7 % at 480p). The
end-to-end question — what a first/last-frame request actually costs a caller — has never been
measured, because the decode and the tail do not scale with the conditioning rows and so the
percentage must come out *lower* end to end than it does per NFE. Predicted: 480p goes
11.447 → **~12.2 s (+6.5 %)**, not +8.7 %, because only the 8.695 s denoise grows.

Needs a keyframe cache, and that needs the **62 GiB Qwen3-VL conditioner** from
`MiniMaxAI/MiniMax-H3` (the OpenVDN checkpoint does not contain it) plus two images. A cache is
only valid at the canvas it was encoded for, so 480p needs its own:

```bash
bash scripts/p5.sh 'cd /opt/dlami/nvme/vdn/vdn-minimax-h3 && source .venv/bin/activate \
    && HF_HOME=/opt/dlami/nvme/vdn/hf HF_HUB_ENABLE_HF_TRANSFER=1 \
       python src/inference/encode_keyframes.py \
         --prompt "<the same prompt as prompts/example_2.pt>" \
         --first prompts/image/<a>.png --last prompts/image/<b>.png \
         --height 480 --width 864 --out prompts/fl2va_480p.pt'
bash scripts/p5.sh 'cd /opt/dlami/nvme/vdn && setsid nohup env \
    PROMPT=prompts/fl2va_480p.pt bash runone.sh f4_480p_fl2va_rep10 \
    8nfe_480p_345f_ulysses_h100.yaml 8 render.repeat=10 \
    > f4_480p_fl2va_rep10.log 2>&1 < /dev/null &'
```

Check the row split in the record before reading the timings — it is what makes the result a
mechanism rather than a number:

```bash
bash scripts/p5.sh 'python3 -c "
import json; r=json.load(open(\"/opt/dlami/nvme/vdn/out/f4_480p_fl2va_rep10.mp4.inference.json\"))
print(r[\"sequence_splits\"])"'
# expect text+vision 2279, condition 810, audio 1150, video 41310 -> 45549
```

### B4 — the conditioner in the request path. **~5 min, no checkpoint needed.**

```bash
bash scripts/p5.sh 'cd /opt/dlami/nvme/vdn/vdn-minimax-h3 && source .venv/bin/activate \
    && torchrun --standalone --nproc_per_node=8 scripts/text_encoder_bench.py --transfer 6.1'
```

**Why:** the recommendation in RESULTS.md — keep the trimmed conditioner as a **pinned** host
copy, upload 6.1 GiB per rank per request, ~0.8 s, +7 % — rests on a **single-rank** 10.08
GiB/s figure applied to eight concurrent uploads. Eight H2D streams share host memory
bandwidth and NUMA paths; if the per-rank rate collapses to 4 GiB/s the recommendation is 1.5 s
and +13 %, which changes the answer. This arm needs no weights and no model, so it is nearly
free, and it prints the single-rank arm next to the eight-rank one so the contention is the
measurement rather than an assumption.

Then, and only if the transfer holds up, the actual work: `hidden_states[50]` means only
**layers 0–50** are needed, so a **pipeline** shard (rank *r* owns a contiguous slice of the
51 layers, hidden state handed rank→rank as a 1300×5120 bf16 = 13 MB tensor over NVLink) is
enough — no tensor parallelism, no custom kernels, seven hops of 13 MB. That is designed but
**not implemented**; `--transfer` prices its dominant cost first on purpose.

### B5 — cu130, as a separate environment. **~20 min.**

Worth knowing, not worth mixing. Build a *second* venv rather than upgrading the first, so the
control stays intact:

```bash
bash scripts/p5.sh 'cd /opt/dlami/nvme/vdn/vdn-minimax-h3 \
    && export PATH=$HOME/.local/bin:$PATH UV_CACHE_DIR=/opt/dlami/nvme/vdn/uvcache \
    && uv venv --python 3.12 .venv130 && source .venv130/bin/activate \
    && uv pip install -q torch==2.13.0 torchvision==0.28.0 \
    && uv pip install -q --prerelease=allow -e . && DIFFUSERS_DIR=diffusers130 \
       bash scripts/setup_diffusers.sh'
```

(cu130 is the PyPI default, so it needs no index at all — which is exactly why trap 1 exists.)
Then re-run **B0 inside it** and compare against 11.447 ± 0.040. Expect little: the hot path is
flash-attn-4's CuteDSL kernels, this repo's Triton kernels and `torch.compile`, not cuBLAS, and
sm90 is not where cu130's work went. If it is faster by more than the control's ±2 %, the whole
table has to be re-measured there, which is the cost to weigh against the gain. Note it may
simply not build — upstream pins cu129 because of the FA4 tree, and that is a real risk, not a
formality.

### Still on the list, unbuilt

* **A pinned host copy of the video VAE** so 768p stops paying 5.05 s of disk load per request:
  9.70 GiB at 10.08 GiB/s is ~0.96 s, taking 33.21 → **~28 s**. Needs 77.6 GiB of page-locked
  host memory across 8 ranks. B4's numbers apply directly to it — same transfer, same
  concurrency — so run B4 first and this becomes arithmetic.
* **Why 768p's steady denoise is *slower* than its first** (19.77 → 20.97 s) while 480p's is
  flat. Cheap to investigate with `parallel.profile: true` on B1.

---

## Getting results back

```bash
bash scripts/p5.sh 'ls -la /opt/dlami/nvme/vdn/out/'
bash scripts/p5.sh --get '/opt/dlami/nvme/vdn/out/f1_480p_rep10.mp4.inference.json' /tmp/
bash scripts/p5.sh --get '/opt/dlami/nvme/vdn/out/f1_480p_rep10.mp4' samples/
python3 scripts/summarize.py <dir-of-json>          # the RESULTS.md tables
```

`render.record=true` (which `runone.sh` always passes) writes `<out>.inference.json` next to
the mp4: checkpoint identity, the fully resolved config, the actual kernel state, per-NFE
timings, per-request timings under `requests`, and `sequence_splits`. **Pull the JSON even for
a run that failed** — it is the only durable record, and the last box was terminated with the
mp4 transfer half-finished.
