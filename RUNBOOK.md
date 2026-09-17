# Runbook — everything here runs **on the box**

Every command below is meant to be typed in a shell on the p5.48xlarge. The exceptions are marked
**[on your Mac]**: getting in, copying scripts up, and pulling results out.

> **Use SGLang.** It is measured on these eight cards at **480p/345f 8.02 s** and
> **768p/345f 19.04 s** post-warmup end-to-end, nothing offloaded, and it serves fl2va from the
> same process. That is 1.43× and 1.74× the patched reference stack, and it is an HTTP server,
> which is the shape the deployment needs. **Section 1 is the whole path from a wiped box to those
> numbers.**
>
> **Section 2 is the reference stack, and it is kept on purpose.** It is not an alternative
> deployment — it is the *control*. Its 11.44 s and 33.21 s are what SGLang is 1.43×/1.74× faster
> than, and both were measured here rather than read off someone's table. Rebuild it when you need
> to re-establish a baseline on a new box, driver or CUDA; otherwise skip it.

Sections: **0** get in · **1** SGLang, the path · **2** the reference stack, the control ·
**3** what is still open · **4** results · **5** traps.

---

## 0. Getting in **[on your Mac]**

```bash
export P5=<the new instance's public IP>          # 18.189.225.222 was the previous box
export KEY=/Users/henanwan/Documents/account/579019700964/henanwan/henanwan-us-east-2.pem
ssh -i "$KEY" -o StrictHostKeyChecking=no ubuntu@"$P5"
```

**The IP, not `ec2-<a-b-c-d>.us-east-2.compute.amazonaws.com`.** On the previous box the public
DNS record stopped resolving after a stop/start while the address stayed put. A *replaced*
instance gets a new address, so `scripts/p5.sh` needs it as well — it reads `HOST` from the
environment: `HOST=ubuntu@$P5 bash scripts/p5.sh --put …`.

Two independent trees, both on the **ephemeral** NVMe:

```
/opt/dlami/nvme/
├── sglang/                  section 1 -- the one you want
│   ├── .venv/               python 3.12, sglang main, its own torch. NOT shared with vdn/
│   ├── cache/               the 62 GB fused overlay the first launch writes
│   └── logs/                serve_*.log, sg_*_rep10.log, cond.log
└── vdn/                     section 2 -- the control, plus the shared HF cache
    ├── hf/                  HF_HOME. BOTH stacks read it; size is whatever WEIGHTS asked for (1b).
    ├── vdn-minimax-h3/      upstream at 2f740c9 + the 12 patches as commits, own .venv
    ├── outputs/             where the sglang server writes its mp4s (relative to its cwd)
    ├── keyframes/           first_{480,768}.png, last_{480,768}.png for the fl2va arm
    ├── out/                 the reference stack's renders + .inference.json records
    └── *.sh *.py            the drivers, staged from this repo
```

`/opt/dlami/nvme` is an **instance store: a stop/start wipes it.** `/` is 484 GB and the HF cache
alone was 407 GB on the last box, so nothing can move there. Assume all of the above is gone after a
stop — the weights and both venvs are rebuilt every time, which is why §1b takes a `WEIGHTS=` list
instead of pulling everything.

Long steps take 6–15 minutes. Use `tmux` (installed) rather than hoping the ssh session holds:

```bash
tmux new -s vdn          # then ctrl-b d to detach, `tmux a -t vdn` to come back
```

---

## 1. SGLang — install, serve, measure

### 1a. Copy the scripts up **[on your Mac]**

The box has no GitHub credentials, so this private repo cannot be cloned there.

```bash
cd /Users/henanwan/Documents/workspace/bytedance/minimax_h3_h100
export HOST=ubuntu@$P5                            # p5.sh has no default; see section 0
bash scripts/p5.sh 'mkdir -p /opt/dlami/nvme/vdn/docker /opt/dlami/nvme/sglang'
for f in scripts/_env.sh scripts/fetch_weights.sh scripts/sglang_bringup.sh scripts/sglang_arm.sh \
         scripts/sglang_cond.py scripts/sglang_parity.py scripts/sglang_base_arm.sh \
         scripts/sglang_base_steps.py scripts/lora_merge_h3.py scripts/sglang_ref2va_arm.sh \
         scripts/sglang_ref2va.py scripts/melt_metrics.py; do
  bash scripts/p5.sh --put "$f" "/opt/dlami/nvme/vdn/$(basename "$f")"
done
for f in docker/Dockerfile docker/h3.sh; do
  bash scripts/p5.sh --put "$f" "/opt/dlami/nvme/vdn/docker/$(basename "$f")"
done
```

`_env.sh` is not optional — the three arm scripts source it for `CUDA_HOME`, the `-lcudart`
symlinks, `NCCL_NET_PLUGIN`, the allocator flag and the ffmpeg gate, and it is what makes them run
unchanged both in a venv and inside the container.

### 1b. Install, the container route. **No install and no build. ~3 min pull, then the download you asked for.**

**This is the recommended route, and it exists because almost every trap in this runbook is a
consequence of installing into a bare DLAMI, not of the model.** `lmsysorg/sglang:dev` is
upstream's **x86 nightly** — `.github/workflows/release-docker-dev.yml` runs on
`cron: "0 0 * * *"`, builds `docker/Dockerfile` on `nvidia/cuda:13.0.3-cudnn-devel-ubuntu24.04` for
amd64+arm64, and publishes `dev` plus the immutable dated aliases
`nightly-dev-{date}-{short_sha}`. (`nightly-cu134` is a different image: arm64-only Rubin, useless
here.) Four traps in this runbook are gone because the image has python 3.12, torch,
`sglang-kernel`, the flashinfer cubin cache, cargo, a real `/usr/local/cuda` and `ffmpeg`: the
`can't find Rust compiler` build failure, `deep_gemm`'s `find_cuda_home` assert plus the
`lib64`/`-lcudart` symlinks, the broken apt source that blocks `ffmpeg`, and python 3.13 vs the
cp312-only `outlines_core` wheel.

```bash
cd /opt/dlami/nvme/vdn/docker
bash h3.sh probe            # ~3 min. Pull the nightly and ask it what it has
bash h3.sh weights t2va,ref2va
bash h3.sh serve ref2va 768 ; bash h3.sh logs ref2va
```

**There is no build step, and the reason is worth writing down, because this runbook said the
opposite for one commit.** The claim was that the image installs the LLM extras only — the way
upstream's own generated H3 command
(`docs/src/snippets/configs/MiniMaxAI/minimax-h3.jsx`, `dockerRunCommand`) still implies, since it
runs `pip install -e "/sgl-workspace/sglang/python[diffusion]"` at **every container start**. Read
the build instead of the docs and it is false:

* both docker workflows pass **`--build-arg BUILD_TYPE=all`**, `python/pyproject.toml` has
  `all = ["sglang[diffusion]", "sglang[http2]", "sglang[tracing]"]`, and stage `torch_deps`
  installs `".[all]"` **with** dependencies (`docker/Dockerfile:216`) — so `diffusers==0.37.0`,
  `av`, `cache-dit`, `st_attn`, `vsa`, `opencv-python-headless`, `moviepy`, `nvidia-modelopt` are
  already installed;
* `--build-arg BRANCH_TYPE=local` makes `framework_final` `COPY . /src` into
  `/sgl-workspace/sglang`, and `.dockerignore` excludes only `.gitignore` — so the image carries
  the **whole checkout including `.git`**, at exactly the sha in its tag, editable-installed with
  `pip install --no-deps -e "python[all]"` (`docker/Dockerfile:584`).

A nightly is therefore already diffusion-capable, already VDN-capable, and already pinned. `probe`
is the step that proves it on the box; if every line says `have`, stop there.

**Measured, on a fresh `p5.48xlarge` in us-west-1 (2026-09-17).** `probe` took ~6 min, almost all of
it the pull — the image is **51.8 GB** — and **every line came back `have`**: python 3.12.3,
torch 2.13.0+cu130, `minimax_h3` + `minimax_h3_vdn` + `minimax_h3_pipeline`, all eight diffusion
packages (`diffusers av cv2 cache_dit st_attn vsa moviepy imageio_ffmpeg`), `ffmpeg ffprobe nvcc
cargo git`, `CUDA_HOME=/usr/local/cuda`. Nothing needed installing. For the record:

```
sglang 0.0.0.dev1+g408d2334c   torch 2.13.0+cu130   python 3.12.3
bundled source: 408d2334c34d387a36a26398dff9a8f004328344  Wed Sep 16 17:11:55 2026 -0700
BASE=lmsysorg/sglang@sha256:6bcaa47db52f78ce0d67863b8b2431221b79bc23204a80cad757fa819d00e921
```

That box also had docker 29.8.0, the `nvidia` runtime registered, `ubuntu` in the `docker` group and
27 T on `/opt/dlami/nvme` — nothing to install there either. **Note the bundled sha is not the one
this runbook predicted** (`46ae84df`, main's head at 23:54 UTC): the scheduled build actually ran at
00:11 UTC and picked up a later commit. That is the argument for pinning by digest rather than by
reconstructing a `nightly-dev-{date}-{sha}` tag from commit timestamps.

**Pin it by digest anyway.** `:dev` is rebuilt every night, and "measured on the nightly" is not a
reproducible statement. The last line `probe` prints is the pin:

```bash
BASE=lmsysorg/sglang@sha256:… bash h3.sh serve vdn 480     # record this in RESULTS.md
```

`docker/Dockerfile` survives as an **escape hatch for one case only**: a revision no nightly names.
Nightlies pin main's head at 00:00 UTC; RESULTS.md's numbers were measured at `3f8eb35e`, a mid-day
commit on 2026-09-16 that sits between the `20260916` and `20260917` nightlies. If an exact sha
matters — bisecting, or reproducing a number against the commit it came from:

```bash
SGLANG_REV=3f8eb35eadfb29ad98d7900910f19fafcbac5ccb bash h3.sh build   # ~7 min, measured
IMAGE=minimax-h3:local bash h3.sh serve vdn 480                        # nothing picks it up implicitly
```

**This has been run.** It works and costs ~7 min and 6 GB (57.8 GB image against the base's 51.8):
5.5 min of it is the editable rebuild, 30 s the import assertion, 77 s the layer export. The result
reports `sglang 0.0.0.dev1+g3f8eb35ea`, `source: 3f8eb35eadfb… Wed Sep 16 12:44:49 2026 +0800`,
`torch 2.13.0+cu130 cuda True 8`, with `sglang hf ffmpeg ffprobe nvcc` all on PATH and the bind
mount visible at the same path — i.e. the arms can run on it unchanged. Either way the build asserts
the H3, VDN and pipeline imports, so a bad pin fails in `docker build` and not 20 minutes into an
8-GPU launch.

> **Trap: the image's `.git` cannot fetch as shipped, and the error blames the wrong thing.**
> `SGLANG_REV=…` first failed with
> `fatal: could not read Username for 'https://github.com': No such device or address`. sglang is a
> public repo, so nothing here needs a credential — what happened is that `actions/checkout` left
> `http.https://github.com/.extraheader=AUTHORIZATION: basic <token>` in `.git/config`, the token is
> dead outside that runner, GitHub answers 401, and git falls back to prompting for a username. The
> Dockerfile now unsets that key and sets `GIT_TERMINAL_PROMPT=0`. The clone is also **depth 1**
> (`git rev-list --count HEAD` == 1), so the target commit really is absent and must be fetched;
> `git fetch --depth 1 origin <sha>` works against GitHub, and `--unshallow` is the fallback because
> a plain `git fetch` on a shallow clone brings the new tip and still not the wanted commit.

`h3.sh` bind-mounts `/opt/dlami/nvme` **at the same path inside the container**, so every absolute
path in this runbook — the HF cache, the fused overlay, reference images, output mp4s, LoRA files —
means the same thing in both worlds and no request needs a rewritten URI. It also sets
`--ipc=host --shm-size 32g --network host` (upstream's own H3 numbers; the 64 MB docker default is
where a multi-rank diffusion worker dies with a bare `Bus error`).

> **`h3.sh stop`, never `sglang_arm.sh stop`, when serving in a container.** That stop path is a
> `pkill`, and without `--pid=host` a second container has its own PID namespace: the pkill matches
> nothing, reports success, and the server keeps all eight cards.

Nothing here contains weights, and that is deliberate: a ~200 GiB image is slower to move than the
weights are to download, and `/opt/dlami/nvme` is instance store, so a fresh box pays the download
either way. **What the container buys is the ~10 minutes of pip, the four traps, and a revision
that is a digest instead of a date — not the ~25-minute download, which is the real cost on a fresh
box and which no image can carry more cheaply than the CDN.**

**Sections 1c–1g below are written for the venv route.** The container equivalents are mechanical,
because the scripts and every path are the same either way:

| §1c–1g says | in the container |
|---|---|
| `bash sglang_arm.sh serve 480` | `bash h3.sh serve vdn 480` |
| `bash sglang_base_arm.sh serve 480` | `bash h3.sh serve t2va 480` |
| `LORA=… bash sglang_ref2va_arm.sh serve 768` | `LORA=… bash h3.sh serve ref2va 768` |
| `python sglang_ref2va.py 768:8 ref=…` | `bash h3.sh exec sglang_ref2va.py 768:8 ref=…` |
| `bash sglang_ref2va_arm.sh refedge 1024` then serve | `REFEDGE=1024 bash h3.sh serve ref2va 768` |

**`REFEDGE=1024` is a melting arm, not a production setting.** It is what makes the ref2v Turbo LoRA
collapse hands and faces — see REF2VA.md, "Correction: what actually melts". Serve with the constant
left alone (2048); the patch is here to reproduce the fault, like `LORA_ALPHA=128`.

| `… stop` | `bash h3.sh stop` |

`h3.sh` forwards `QUANT LORA LORA_ALPHA MERGED REFEDGE GPUS FRAMES SEED PORT LOGTAG MODEL OUTDIR
PROMPT` when they are set, so an arm reads the same either way. **`REFEDGE=` is the container form
of arm F and not just a shorthand:** the patch rewrites an installed module, so applying it in a
throwaway container discards it with that container's writable layer. `REFEDGE=` applies it inside
the serving process tree, immediately before `sglang serve` starts.

### 1b-weights. Which checkpoints — the same selector in either route.

`WEIGHTS` is a comma list and it is the only thing to decide in either route. **`hf download
MiniMaxAI/MiniMax-H3` with no filter is 464 GiB**, because the repo ships the same 62 GiB DiT under
four names — `transformer/` (t2va, diffusers-named), `transformer_ref/` (ref2va, diffusers-named),
and the self-contained `FL2VA/` and `Ref2VA/` partitions (134 GiB each, native-named, each carrying
its own copy of the 62 GiB Qwen3-VL text encoder). No arm needs more than two of them.

| `WEIGHTS=` | size | what it unlocks |
|---|---|---|
| `vdn` | 82 GB | §1c–1e. The 8-step distill, **already measured** — 480p 8.02 s, 768p 19.04 s |
| `t2va` | 134 GiB | §1f. Base H3 at 50 and 8 steps, i.e. the denominator |
| `ref2va` | 134 GiB | §1g. `--model-variant ref2va`, the customer's actual ask |
| `fl2va` | 134 GiB | only to merge a *native*-named LoRA. Not needed for lightx2v's |
| `transformer_ref` | 62 GiB | §1g merge route: lightx2v's diffusers LoRA folds in with no key translation |
| `all` | 464 GiB | don't |
| `none` | 0 | environment only |

`t2va` and `ref2va` share every small file and the `text_encoder/`, so `t2va,ref2va` is **~200 GiB
on disk, not 268** — the two 134 GiB figures double-count the encoder. Budget ~25 min at the ~2 Gb/s
this box gets from the hub.

VDN is **not** in that list on purpose: §1c–1e are finished and in RESULTS.md, and re-measuring them
costs 82 GB of download to reproduce a number to ±0.05 s. Add `vdn` only to re-baseline a new
driver or a new sglang.

> `hf`'s `--include` takes **one pattern per occurrence** as of huggingface_hub 1.x (the CLI is
> click-based now, not argparse). A second bare pattern is silently read as a *filename* and 404s as
> `resolve/main/tokenizer/%2A`. The script repeats the flag; don't "simplify" it back.

### 1b-venv. Install without Docker. **~10 min for the environment, then the same download.**

Same result, more moving parts; keep it for a box where Docker is unavailable, or to reproduce the
environment RESULTS.md was measured in exactly.

```bash
# what the two queued jobs need, and nothing else
WEIGHTS=t2va,ref2va setsid nohup bash /opt/dlami/nvme/vdn/sglang_bringup.sh \
    > /opt/dlami/nvme/sglang/bringup.log 2>&1 < /dev/null &
tail -f /opt/dlami/nvme/sglang/bringup.log        # ends with "=== [..] done"
```


The script asserts what matters instead of hoping: python is 3.12, `$VIRTUAL_ENV` is unset,
`ffmpeg`/`ffprobe` exist, and `import sglang.multimodal_gen.configs.pipeline_configs.minimax_h3_vdn`
succeeds. Five things it handles that are easy to get wrong on a fresh box:

* **sglang from git, not PyPI.** Checked here: release 0.5.19 has base MiniMax-H3 but zero
  occurrences of `vdn` or `hybrid_window_attn_h3`. On main VDN is a whole subsystem. Retry PyPI
  once 0.5.20 ships. The verified build is `0.5.6.post3.dev10594+g3f8eb35ea` (main at `3f8eb35e`).
* **`SGLANG_BUILD_RUST_EXTS=none`.** main's `setup.py` shells out to `cargo` just to *discover* the
  Rust router extensions, so with no toolchain the build dies in
  `get_requires_for_build_wheel` before any Python compiles. Nothing in the diffusion path uses
  them; installing rustup instead works and wastes ten minutes.
* **`ffmpeg` before the download, not after.** H3's pipeline raises at startup without it, on every
  rank, and the parent shows only an `EOFError` from the pipe — which reads like a crash.
* **The text encoder is not optional, for any arm.** The OpenVDN checkpoint has no `text_encoder/`
  or `processor/` — the Qwen3-VL conditioner was never on the old machine either, because the
  reference stack `torch.load`s offline prompt caches (see the "text encoding is not in any of these
  numbers" note in RESULTS.md). A server that takes a prompt over **HTTP** needs it, so every
  `WEIGHTS` entry except `vdn` pulls it.
* **`huggingface_hub[hf_transfer]` no longer exists.** hub 1.x prints "does not provide the extra
  'hf-transfer'", installs plain hub, and leaves you on the slow path; `HF_HUB_ENABLE_HF_TRANSFER=1`
  now only earns a `FutureWarning` pointing at `HF_XET_HIGH_PERFORMANCE`. The fast backend is
  `hf-xet`, a default dependency. The script imports each one and exports whichever knob matches,
  and prints which — check that line in the log before walking away from a 200 GiB download.

> **On a fresh DLAMI, apt is broken and the ffmpeg step is the first thing to hit it.** `/` is not
> instance store, so a stop/start keeps the fix — a **replaced** instance does not, and this comes
> back. `/etc/apt/sources.list.d/cuda-ubuntu2604-x86_64.list` serves a malformed `Packages` file
> ("Encountered a section with no Package: header"); clearing `/var/lib/apt/lists` does not help,
> because apt re-fetches the same bad file. Move that one source aside and re-run:
>
> ```bash
> sudo mv /etc/apt/sources.list.d/cuda-ubuntu2604-x86_64.list /root/cuda.list.disabled
> sudo apt-get update -qq && sudo apt-get install -y ffmpeg && ffprobe -version | head -1
> ```
>
> Nothing here installs CUDA from apt (each venv carries its own), so leaving it disabled is fine —
> but **know that it is disabled** before blaming apt for something else. Do this *before*
> `sglang_bringup.sh` if you want to be sure, or just let the script fail on the ffmpeg step, fix it,
> and re-run — every step in it is idempotent.

### 1c. Serve and measure. **~6 min startup, ~2 min per bench.**

```bash
cd /opt/dlami/nvme/vdn
setsid nohup bash sglang_arm.sh serve 480 > /dev/null 2>&1 < /dev/null &
tail -f /opt/dlami/nvme/sglang/logs/serve_480p.log     # wait for warmup to finish
bash sglang_arm.sh bench 480 10                        # 1 discarded warmup + 10 measured
```

Then the same with `768`. **Stop and launch must be separate ssh invocations** — see trap 8.

```bash
bash sglang_arm.sh stop
```

**Measured, 2026-09-16, sglang main `3f8eb35e`, 345 frames, ten requests each:**

| 345 f | E2E median | server `inference_time_s` | peak/GPU | reference stack |
|---|---:|---:|---:|---:|
| 480p (864×480) | **8.02 s** | 6.9 s | 54.7 GB | 11.44 s → **1.43×** |
| 768p (1344×768) | **19.04 s** | 17.0 s | 62.1 GB | 33.21 s → **1.74×** |

Both ratios are **floors**: SGLang encodes the text prompt inside the request, and every number in
RESULTS.md excludes text encoding entirely. The ~1 s / ~2 s between E2E and `inference_time_s` is
mux plus disk write, outside the model.

`sglang_arm.sh` carries five fixes, each commented with the symptom it produced. Two of them this
runbook did not predict and you cannot skip:

* **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is what makes it fit.** `--quantization fp8`
  on SM90 is *online* quantization: the loader reads the 65.71 GiB bf16 checkpoint onto the card
  and casts afterwards, so freed bf16 blocks leave holes fp8 weights cannot reuse. The first 480p
  attempt died with 51.89 GiB allocated and **18.70 GiB reserved-but-unallocated** — never 12 GiB
  short, fragmented by that much. Costs no latency.
* **`NCCL_NET_PLUGIN=none`, on AWS specifically.** The DLAMI puts the EFA/OFI plugin on the system
  loader path, `deep_ep`'s `check_nccl_so()` scans `/proc/self/maps`, sees it next to the venv's
  `libnccl.so.2` and asserts "Duplicate NCCL runtime" — fatal because sglang guards that import
  with `except ImportError`, which cannot catch `AssertionError`. Surfaces two layers away as
  `Model architectures ['MiniMaxH3Qwen3VLEncoder'] failed to be inspected`. **Drop this line before
  running anything across nodes.**

**The offload ladder was never needed.** 768p fits with the DiT, both VAEs and the Qwen3-VL
conditioner all resident at `--performance-mode speed`, peak 62.1 GB against 79.18 GiB visible.
If a future build stops fitting, trade in this order — each costs latency, so record which one was
used, because "SGLang is faster" and "SGLang is faster while offloading the DiT" are different
findings:

```bash
bash sglang_arm.sh serve 768 --performance-mode auto
bash sglang_arm.sh serve 768 --layerwise-offload-components text_encoder
bash sglang_arm.sh serve 768 --layerwise-offload-components dit,text_encoder \
                             --dit-layerwise-resident-layers 14
```

### 1d. fl2va, on the same server. **~5 min, no restart, no second checkpoint.**

Run **after** 1c: it cuts its keyframes out of an existing render at the target canvas, so
`/opt/dlami/nvme/vdn/outputs/` needs one 864×480 and one 1344×768 mp4 already there.

```bash
cd /opt/dlami/nvme/vdn
python3 sglang_cond.py 2>&1 | tee /opt/dlami/nvme/sglang/logs/cond.log
```

**Measured**, 3 requests each after a discarded warmup:

| 345 f | task | E2E median | server inference | peak/GPU | vs t2va |
|---|---|---:|---:|---:|---:|
| 480p | t2va | 7.85 s | 6.82 s | 54,682 MB | — |
| | fl2va first+last | 8.88 s | 7.85 s | 55,162 MB | **+13.1 %** |
| | fl2va first only | 9.07 s | 7.81 s | 54,804 MB | |
| 768p | t2va | 18.11 s | 16.33 s | 62,022 MB | — |
| | fl2va first+last | 20.96 s | 19.09 s | 62,364 MB | **+15.7 %** |
| | fl2va first only | 19.94 s | 18.09 s | 62,382 MB | |

The wire format, from `runtime/.../minimax_h3/{task_profiles,request_validation}.py`:

```json
{"task": "fl2va",
 "conditions": [{"role": "keyframe", "type": "image", "uri": "/abs/path.png", "frame_index": 0},
                {"role": "keyframe", "type": "image", "uri": "/abs/path.png", "frame_index": -1}]}
```

`MINIMAX_H3_FL2VA_KEYFRAME_SIGNATURES` is `((0,), (-1,), (0,-1))`: first-only, last-only and
first+last are all the one task name. Only +340–480 MB resident.

**`ref2va` is refused, and correctly** — `MINIMAX_H3_TASK_PARTITIONS` maps t2va and fl2va to the
`fl2va` partition and ref2va to a `ref2va` partition, and VDN shipped only the former:

> `VDN-H3 serves t2va and fl2va; ref2va was not trained (got task='ref2va'). Use
> MiniMaxAI/MiniMax-H3 --model-variant ref2va for that task.`

That is a training-time limit, not a runtime one. No flag, patch or framework gets around it; base
MiniMax-H3 has ref2va but no 8-step distill, i.e. a different and much slower model.

**Two gotchas in the API**, both of which cost a debugging cycle:

* **Do not send `seconds`.** `VideoGenerationsRequest` types it `Optional[int]` and 345f at 24 fps
  is 14.375, so every request 400s with pydantic's `int_from_float` — and `bench_serving` reports
  `0/10 in 0.01 s` rather than an error. `target.duration_seconds` takes the float.
* **`file_path` in the response is relative to the server's cwd**, `/opt/dlami/nvme/vdn`, not the
  caller's. The API is async: `POST /v1/videos` returns `queued`; poll `GET /v1/videos/{id}` for
  `completed`, which carries `inference_time_s` and `peak_memory_mb`.

### 1e. Same prompt, same seed as the reference stack. **~1 min, server already up.**

The quality question, asked at matched input. `prompts/example_2.pt` carries the prompt *text* next
to its embeddings, so the same 5,720 characters go over HTTP and SGLang re-encodes them with the
same conditioner the cache came from; `seed: 42` is `src/config/inference.py:167`.

```bash
cd /opt/dlami/nvme/vdn
/opt/dlami/nvme/sglang/.venv/bin/python -u sglang_parity.py
```

Measured: **480p 8.55 s, 768p 19.10 s** — a real production prompt costs ~0.5 s more than vbench's
short ones at 480p and nothing extra at 768p. Both stacks then render the prompt's eight shots at
the prompt's timestamps with the same subject; the pixels differ (15.35 / 14.42 dB PSNR, *inside*
the same-stack same-seed band — see trap 9). **Do not read PSNR as quality here.** Compare by
watching:

```bash
# [on your Mac], after --get'ing both
ffmpeg -i samples/n_480p_seg4.mp4 -i out/sglang/parity/parity_480p_345f_seed42.mp4 \
  -filter_complex "[0:v][1:v]hstack" -map 1:a -c:v libx264 -crf 20 sidebyside.mp4
```

`output_path` in the request is treated as a **directory**, not a filename — the mp4 lands inside it
under a UUID.

### 1f. Base MiniMax-H3, the denominator. **~8 min startup, ~5 min for four arms.**

"VDN is 1.43× / 1.74× faster than the reference stack" compares two *implementations of VDN*. This
compares two *models*, on the same eight cards, same prompt, same seed 42, same 345 frames. It needs
its own server, on port 30011 so the VDN server on 30010 can stay up — but 8 GPUs cannot hold both,
so in practice stop VDN first (`bash sglang_arm.sh stop`).

```bash
cd /opt/dlami/nvme/vdn
setsid nohup bash sglang_base_arm.sh serve 480 \
    > /opt/dlami/nvme/sglang/base_wrap.log 2>&1 < /dev/null &
tail -f /opt/dlami/nvme/sglang/logs/serve_base_480p.log     # wait for uvicorn on 30011
/opt/dlami/nvme/sglang/.venv/bin/python -u sglang_base_steps.py 480:50 480:8 768:50 768:8
```

The `serve 480` argument only sizes the startup warmup; both resolutions are then requested against
the one server. 50 is base H3's own asserted schedule
(`MiniMaxH3SamplingParams.num_inference_steps = 50`); the `8` arms are step-cost probes, not
shippable renders. Measured: **480p 45.13 s / 768p 187.50 s at 50 steps**, against VDN's 8.55 / 19.10
— **5.28× and 9.82×** — and at a matched 8 steps base costs 0.98 vs 0.95 s/step at 480p but 3.73 vs
2.17 s/step at 768p. Full reading in `RESULTS.md`, "Base MiniMax-H3 on the same eight cards".

**Pull the mp4s before you stop the box.** They were lost once already: `/opt/dlami/nvme` is instance
store and the four base renders never left it.

```bash
bash scripts/p5.sh 'cd /opt/dlami/nvme/vdn/pull/base && for d in base_*; do \
  mv "$d"/*.mp4 "$d".mp4 && rmdir "$d"; done && ls -l'
bash scripts/p5.sh --get /opt/dlami/nvme/vdn/pull/base out/sglang/base    # [on your Mac]
```

**The Turbo LoRA arm does not run at fp8, and knowing why saves an hour.** `--lora-path` together
with `--quantization fp8` aborts during warmup with `AttributeError:
'RowParallelLinearWithLoRA' object has no attribute 'quant_method'` — the dynamic-LoRA wrapper is not
quantization-aware. (This is *not* the failure the g7e project recorded; that one, an in-place add on
a transposed fp8 weight, is fixed upstream.) And the offline bf16 merge cannot be pointed at a t2va
tree: the adapter is named for the native layout, every t2va transformer on disk is named for the
diffusers layout, and `scripts/lora_merge_h3.py` matches 0/259 and refuses. What still works:

```bash
# bf16, so --lora-path runs with no key mapping. FOR PICTURES ONLY -- bf16 latency is not
# comparable to any fp8 arm here, and 62 GB of bf16 weights may need --use-fsdp-inference to fit.
cd /opt/dlami/nvme/vdn && QUANT= LOGTAG=turbo setsid nohup bash sglang_base_arm.sh serve 480 \
    --lora-path /opt/dlami/nvme/vdn/lora/minimax_h3_turbo_v4_step600_ema.safetensors \
    --lora-nickname turbo > /opt/dlami/nvme/sglang/turbo_wrap.log 2>&1 < /dev/null &
```

The fp8 Turbo *latency* needs no run at all: merging an adapter changes weight values, not shapes and
not the graph, so it is the 8-step row already measured — 9.02 s at 480p, 31.57 s at 768p.

### 1g. ref2va + lightx2v Turbo, the "melting" question. **~15 min startup, ~3 min for all seven arms.**

Read `REF2VA.md` first — it names the three causes each arm tests, and the arms are cheap enough
(every one under a minute of GPU) that running them in the wrong order is the only way to waste time.
**ref2va means base MiniMax-H3, not VDN**: upstream never trained ref2va for VDN and the server says
so. That is why this is its own server on its own port (30012), and why it starts slower — it wants
the `Ref2VA` partition, ~134 GiB, and it is bf16 because `--lora-path` does not survive fp8.

```bash
cd /opt/dlami/nvme/vdn && mkdir -p lora ref pull/ref2va
# The two ref2v LoRAs, 1.3 GB each. bf16 diffusers-named PEFT; both declare alpha 8, rank 128.
for n in minimax_h3_ref2v_turbo_8step_v1.0_768p_bf16 minimax_h3_ref2v_turbo_4step_v0.1_bf16; do
  .venv/bin/hf download lightx2v/Minimax-h3-Turbo "$n.safetensors" --local-dir lora
done
# One reference image, cut from an existing render so nothing measures a resize of unknown origin.
ffmpeg -y -v error -i pull/base/base_480p_8step.mp4 -vf 'select=eq(n\,0)' -vframes 1 ref/subject.png
```

Arm A first, because it is the control and every other number is unattributable without it:

```bash
cd /opt/dlami/nvme/vdn && LOGTAG=A setsid nohup bash sglang_ref2va_arm.sh serve 480 \
    > /opt/dlami/nvme/sglang/ref2va_A.log 2>&1 < /dev/null &
# wait for "Uvicorn running", then:
python3 sglang_ref2va.py ref=ref/subject.png tag=A 480:50
```

Then B and C — **the pair that answers the customer's question** — back to back, one seed apart from
nothing:

```bash
L=/opt/dlami/nvme/vdn/lora/minimax_h3_ref2v_turbo_8step_v1.0_768p_bf16.safetensors
bash sglang_ref2va_arm.sh stop
LORA=$L LOGTAG=B setsid nohup bash sglang_ref2va_arm.sh serve 480 \
    > /opt/dlami/nvme/sglang/ref2va_B.log 2>&1 < /dev/null &
python3 sglang_ref2va.py ref=ref/subject.png tag=B 480:8      # alpha 8 from the file = scale 0.0625

bash sglang_ref2va_arm.sh stop
LORA=$L LORA_ALPHA=128 LOGTAG=C setsid nohup bash sglang_ref2va_arm.sh serve 480 \
    > /opt/dlami/nvme/sglang/ref2va_C.log 2>&1 < /dev/null &
python3 sglang_ref2va.py ref=ref/subject.png tag=C 480:8      # 16x overdrive, on purpose
```

**Check the server log for the scale it resolved before trusting either arm.** Arm B must not show
alpha 128 anywhere: SGLang reads `alpha: 8` out of the safetensors metadata by itself, and
`--lora-alpha 128` — which is what lightx2v's own 768p README command line contains — overrides it
to 16x. That single flag is the leading explanation for "melting" and arm C exists to confirm it.

D, E, F, G are the same shape; F is the one code change:

```bash
# This arm REPRODUCES the customer's melting; do not carry it into production. See REF2VA.md,
# "Correction: what actually melts": the LoRA is clean at 2048 and melts at 1024.
bash sglang_ref2va_arm.sh refedge 1024      # then restart; `refedge restore` puts 2048 back
bash sglang_ref2va_arm.sh refedge restore
# in a container, where a patched module dies with the container that patched it:
REFEDGE=1024 LOGTAG=F bash docker/h3.sh serve ref2va 480
```

Score them, do not watch them first:

```bash
python3 melt_metrics.py --csv /opt/dlami/nvme/vdn/pull/ref2va/curves.csv \
    /opt/dlami/nvme/vdn/pull/ref2va/*.mp4
# `decay` < 1 is melting: detail present at the start of the clip and gone by the end.
# Compare `sat` and `clip%` between B and C -- a 16x scale shows up there before it shows up in sharp.
```

**Pull the mp4s and the csv before you stop the box.** `/opt/dlami/nvme` is instance store; this is
how the four base-H3 renders were lost.

```bash
cd /opt/dlami/nvme/vdn/pull/ref2va && for f in *.mp4 curves.csv; do echo "$f"; done
# [on your Mac]
bash scripts/p5.sh --get /opt/dlami/nvme/vdn/pull/ref2va/curves.csv out/ref2va/
```

For a *latency* number rather than a picture, merge and go back to fp8 — the merge needs no key
translation because `transformer_ref/` and lightx2v's LoRA are both diffusers-named:

```bash
SRC=$HF_HOME/hub/models--MiniMaxAI--MiniMax-H3/snapshots/*/transformer_ref \
LORA=$L DST=/opt/dlami/nvme/vdn/ref2v_turbo_bf16 \
  /opt/dlami/nvme/sglang/.venv/bin/python lora_merge_h3.py     # prints the resolved SCALE; check it
QUANT=fp8 MERGED=/opt/dlami/nvme/vdn/ref2v_turbo_bf16 bash sglang_ref2va_arm.sh serve 480
```

---

## 2. The reference stack — the control

Rebuild this only to **re-establish a baseline**: new box, new driver, new CUDA, or a claim that
needs a same-machine comparison. It is not a deployment path any more; SGLang beats it at both
canvases, keeps 768p resident where patch 11's residency died on the second request, and serves
fl2va without a restart.

### 2a. Check what is there

```bash
bash /opt/dlami/nvme/vdn/box_check.sh
```

```
repo        : HEAD 5f4406b, 12 commits past 2f740c9, 0 tracked files modified
configs     : 4/4 h100 yamls
weights     : 82G (expect 82G)
prompts     : 3 caches
drivers     : 3/3 in /opt/dlami/nvme/vdn
python env  : torch 2.13.0+cu129  torchvision 0.28.0+cu129  diffusers 0.40.0.dev0  gpus 8
```

**Both `+cu129` suffixes matter more than anything else on that list** — see trap 1. If `gpus busy`
is not 0, something is still running (`nvidia-smi`; `pkill -f 'infer_ulysses\.py'`).

### 2b. Rebuild from scratch. **~10 min, mostly download.**

**[on your Mac]**, stage the files the box cannot fetch:

```bash
cd /Users/henanwan/Documents/workspace/bytedance/minimax_h3_h100
bash scripts/p5.sh 'mkdir -p /opt/dlami/nvme/vdn/staging'
bash scripts/p5.sh --put scripts/h100_bringup.sh /opt/dlami/nvme/vdn/h100_bringup.sh
bash scripts/p5.sh --put scripts/box_check.sh    /opt/dlami/nvme/vdn/box_check.sh
for f in patches/*.patch configs/*.yaml scripts/{runone,h100_grid}.sh \
         scripts/{summarize,text_encoder_bench,mux_bench,reload_bench,clipinfo,vidcmp,vidscale,vidshift}.py; do
  bash scripts/p5.sh --put "$f" "/opt/dlami/nvme/vdn/staging/$(basename "$f")"
done
```

Then, on the box:

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

# --- 5. verify, then run the control below before trusting any new number
bash /opt/dlami/nvme/vdn/box_check.sh
```

`scripts/decode_parity.py`, `src/inference/utils/parallel_vae.py` and
`src/inference/utils/yuv.py` come from the patch series, not from step 4 — if they are missing,
`git am` did not run. `git am --3way` has been verified clean against a fresh 2f740c9 on this box,
and the tree it produces is byte-identical to the one the 11.44 s measurement came off.

### 2c. The two control runs

The pattern is always the same, from `/opt/dlami/nvme/vdn`. `runone.sh` sources the venv, sets
`OMP_NUM_THREADS=24` (trap 2) and `expandable_segments`, kills any previous `infer_ulysses.py`, and
passes `render.record=true`. Overrides are OmegaConf dotlist. Each run pays a ~220 s model build.

```bash
# 480p / 345f, ten requests. ~7 min. Expect 11.44 +- 0.05 s over requests 2-10.
setsid nohup bash runone.sh f1_480p_rep10 8nfe_480p_345f_ulysses_h100.yaml 8 \
    render.repeat=10 > f1_480p_rep10.log 2>&1 < /dev/null &

# 768p / 345f, ten requests. ~10 min. Expect 33.21 +- 0.85 s, and peak reserved 63.59 GiB
# IDENTICAL at request 2 and request 10 -- zero memory growth is as much the point as the mean.
setsid nohup bash runone.sh f2_768p_rep10 8nfe_768p_345f_ulysses_h100.yaml 8 \
    render.repeat=10 > f2_768p_rep10.log 2>&1 < /dev/null &

grep -E "^(steady|reserved after| req|  [0-9])" f1_480p_rep10.log
```

480p breakdown to expect: denoise 8.71, video VAE 1.56, decode+encode 1.17; request 1 ≈ 16.1 s.
**Outside ±2 % on the mean, stop and check the two `+cu129` suffixes** before reading anything into
it. The 480p arm has already reproduced the old 11.447 ± 0.040 to 0.06 % on this box.

### 2d. The other reference-stack arms, and why they are no longer queued

Kept for the record; run one only if a specific question needs it.

* **`vae_after_decode=keep` at 768p, expected to OOM.** The load-bearing negative result of the
  whole 768p design rests on one OOM from one run, and it has never been reproduced. Cheap
  (~6 min) and it re-measures the non-PyTorch floor on the current driver:
  `runone.sh f3_768p_keepkeep 8nfe_768p_345f_ulysses_h100.yaml 8 render.repeat=2 parallel.vae_after_decode=keep`.
  Expect request 2 to die on ranks **5, 6, 7** — exactly the three linear-branch ranks at
  `softmax_ranks: 5`.
* **fl2va in the reference stack** (`PROMPT=prompts/image/example_fl2va.pt`, +8.9 % denoise at
  768p) — **superseded by 1d**, which measures it end to end with the visual tokenizer inline.
  The two numbers are reconciled in RESULTS.md: the +8.9 % was denoise-only against pre-encoded
  latents.
* **Pricing a conditioner in the request path** (`text_encoder_bench.py --transfer 6.1`) —
  **moot**. The conditioner is resident in the winning configuration and its cost is already
  inside the 8.02 s.
* **cu130 in a second venv** — **effectively answered, and not by this arm.** SGLang resolves its
  own torch and the winning configuration runs **torch 2.13.0+cu130** with nvcc 13.4
  (`out/sglang/parity/sglang_env.txt`, a full freeze off the box). cu130 on sm90 is therefore not a
  blocker and not a win by itself; cu129 remains the reference stack's pin because that is what
  upstream VDN requires.

---

## 3. Still open

* **The four base-H3 renders, and the 8-step quality question.** §1f's latencies are measured but the
  mp4s died with the instance store, so nothing shows whether base H3 at 8 steps is as undercooked as
  the theory says. Re-render the two `8` arms: 41 s of GPU once a server is up. **Do this first
  tomorrow** — it is the cheapest open item on the list.
* **Base + Turbo LoRA, as a picture.** §1f has the working bf16 invocation. The fp8 latency needs no
  run. Open question if anyone wants it at fp8: invert the three diffusers↔native translations named
  in `scripts/lora_merge_h3.py`, or wait for SGLang to make its dynamic-LoRA wrapper
  quantization-aware, which is the smaller upstream fix and would make `--lora-path --quantization
  fp8` work directly.
* **ref2va melting: seven arms, all measured, none run.** `REF2VA.md` plus §1g. The pair that
  matters is **B vs C** — lightx2v's ref2v LoRA at the alpha its own file declares (8, scale 0.0625)
  against the alpha their README's command line passes (128, scale 1.0). If C melts and B does not,
  the customer's problem is one flag and the answer is "delete it". **Arm A, the 50-step control,
  runs first** or nothing else is attributable.
* **Whether bf16 ref2va fits at all.** Every LoRA arm above is bf16 because `--lora-path` dies at
  fp8, and bf16 is ~62 GiB of DiT per rank under Ulysses on top of a 9.7 GiB video VAE. 480p with
  one reference should fit; 768p may not. First fallback is `--tp-size 2 --ulysses-degree 4`, which
  shards the DiT weights instead of only the sequence. Unknown until it is tried.
* **The non-inference tail.** SGLang's E2E minus `inference_time_s` is ~1 s at 480p and ~2 s at
  768p: mux plus disk write. This repo's finding that libx264 does not parallelise itself and needs
  segmented encoding points straight at it — but that is upstream code now, so it is a PR, not a
  patch.
* **Why the reference stack's 768p steady denoise is *slower* than its own first request**
  (19.77 → 20.97 s) while 480p is flat. `parallel.profile=true` appended to the 768p control. A
  property of the reference stack that no SGLang number explains.
* **2K upscale is not open source.** MiniMax's own page says so. Nothing to run.
* **Territory.** The MiniMax H3 Community License's applicable territory excludes the United
  States, EU, UK and South Korea, and this box is in `us-east-2`. Flagged in README.md; a licensing
  question, not a technical one.

---

## 4. Results

**SGLang** writes its mp4s under `/opt/dlami/nvme/vdn/outputs/` (the server's cwd) and its logs
under `/opt/dlami/nvme/sglang/logs/`. Latency comes from the bench log and from each response's
`inference_time_s` / `peak_memory_mb`.

**The reference stack** writes an mp4 and, because `runone.sh` passes `render.record=true`, a JSON
record next to it holding the checkpoint identity, the resolved config, the actual kernel state,
per-NFE and per-request timings, and `sequence_splits`:

```bash
cd /opt/dlami/nvme/vdn/vdn-minimax-h3
.venv/bin/python scripts/clipinfo.py /opt/dlami/nvme/vdn/out/<tag>.mp4   # canvas, frames, audio
python /opt/dlami/nvme/vdn/summarize.py /opt/dlami/nvme/vdn/out          # the RESULTS.md tables
```

**Copy records off the box even for a run that failed** — they are the only durable record, and the
previous instance was terminated with an mp4 transfer half-finished.

**[on your Mac]**:

```bash
cd /Users/henanwan/Documents/workspace/bytedance/minimax_h3_h100
bash scripts/p5.sh --get '/opt/dlami/nvme/sglang/logs/*.log'            out/sglang/logs/
bash scripts/p5.sh --get '/opt/dlami/nvme/vdn/outputs/<name>.mp4'       out/sglang/
bash scripts/p5.sh --get '/opt/dlami/nvme/vdn/out/<tag>.mp4.inference.json' rescue/
```

`out/` is gitignored — mp4s stay local. What was rescued off this instance before it was shut down,
all under `out/sglang/`:

| local | what |
|---|---|
| `sglang_{480,768}p_345f.mp4` | the t2va renders the 8.02 / 19.04 s numbers came from |
| `parity/parity_{480,768}p_345f_seed42.mp4` | same prompt + seed 42 as the reference stack (§1e) |
| `parity/sidebyside_{480,768}p_seed42.mp4` | reference left, SGLang right, labelled |
| `parity/sheet_{480,768}p.png` | frames 0/110/220/344 matched, reference row above SGLang row |
| `parity/cond_{480,768}p_{t2va,fl2va_first_last,fl2va_first_only}.mp4` | the six conditioning arms (§1d) |
| `parity/{first,last}_{480,768}.png` | the keyframes those fl2va requests were given |
| `parity/sglang_env.txt` | python 3.12.14, sglang `0.5.6.post3.dev10594+g3f8eb35ea`, torch 2.13.0+cu130, 237-package freeze |
| `logs/`, `parity/{cond,parity,serve_480p}.log` | the bench, conditioning, parity and server logs |

---

## 5. Traps

1. **`torchvision` must be `+cu129`** *(reference stack only)*. `pyproject.toml` pins
   `torchvision==0.28.0` with no local version, so `uv pip install -e .` takes the PyPI default —
   **cu130** — and then every `import diffusers.loaders.peft` dies with `PyTorch has CUDA
   Version=12.9 and torchvision has CUDA Version=13.0`, surfacing unhelpfully as `Could not import
   module 'BloomPreTrainedModel'`. `h100_bringup.sh` names both wheels from the cu129 index up
   front, which is the only ordering with no repair step, because the obvious repair is a **no-op**
   (0.28.0 already satisfies 0.28.0, so `0.28.0+cu129` is never fetched). If already broken:

   ```bash
   uv pip install --reinstall-package torchvision torchvision==0.28.0 \
       --index-url https://download.pytorch.org/whl/cu129
   ```

2. **`torchrun` sets `OMP_NUM_THREADS=1`** and says so in its own output. Right for a GPU render,
   wrong for patch 2's host-side LoRA merge and fp8 quantise: at 1 thread all 8 ranks sit at 99 %
   of a single core and are still going after six minutes. `runone.sh` sets `192 / nproc_per_node`.

3. **Do not use the DLAMI's `/opt/pytorch`.** Python 3.13 + cu130. The reference stack requires
   `>=3.12,<3.13` and cu129. **SGLang fails there too, and the error names the wrong culprit:**
   `sglang[diffusion]` pins an `outlines_core` 0.1.x whose newest wheel is **cp312** (every 0.1.x
   stops at cp312; cp313 first appears in 0.2.9), so under 3.13 pip falls back to the sdist and
   dies on `error: can't find Rust compiler`. Installing Rust is the wrong fix — it makes that one
   package build and leaves you on 3.13 for the next gap. Both bringup scripts build a private 3.12.

4. **Everything on `/opt/dlami/nvme`.** `/` is 484 GB, the HF cache is 407 GB, the NVMe is 27 TB —
   but it is an instance store and a stop/start wipes it.

5. **The two venvs are not interchangeable and must stay separate.**
   `/opt/dlami/nvme/vdn/vdn-minimax-h3/.venv` is torch 2.13.0+cu129 and must keep reproducing
   11.44 s; `/opt/dlami/nvme/sglang/.venv` resolves its own torch. They share only `HF_HOME`.
   Installing sglang into the reference venv overwrites the number sglang is compared against.

6. **Host RAM is a real constraint for the reference stack.** Patch 2 assembles on the CPU, so 8
   ranks each hold ~78 GiB of bf16 weights at once, ~620 GB. Fine on this box's 2 TB.

7. **`git checkout .` / `git stash` / `git reset --hard` inside `vdn-minimax-h3`.** Harmless in the
   current state (the 12 patches are commits, `box_check.sh` says `tree clean`), destructive in the
   older one where they were file copies. `box_check.sh` tells you which state you are in.

8. **`pkill -f 'sglang.*serve'` kills the ssh session that issues it.** `ssh box 'pkill -f
   sglang.*serve; bash sglang_arm.sh serve 480'` matches its own command line, returns 255, and
   starts nothing. `sglang_arm.sh stop` uses the `[s]glang` bracket trick, but the real rule is
   that **stop and launch must be separate ssh invocations** — any command line genuinely
   containing both words matches.

9. **Multi-rank denoise is not bit-reproducible** — `index_add_` atomics plus `all_reduce` reorder
   floating-point work, so the same seed across two runs gives ~17 dB PSNR, not identity. Any
   parity claim about the decode has to be made in one process with fixed latents, which is what
   `scripts/decode_parity.py` does. Same reason 1d verifies keyframes by PSNR *against a control*
   (31.6 dB vs the right keyframe, 12.7 dB vs the wrong one) rather than by equality.
