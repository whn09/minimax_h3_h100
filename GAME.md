# Serving VDN-H3 to an interactive game: 480p, 345 frames, two H100 boxes

**Deployed and measured 2026-09-19 on P5-1 and P5-2** (2 × p5.48xlarge, 8 × H100 80GB, us-east-2).
One SGLang Diffusion server per box in Docker, `OpenVDN/vdn-minimax-h3`, 8-step, fp8, 864×480,
**345 frames = 14.375 s of video with audio**, HTTP on loopback, forwarded to `127.0.0.1:30010` and
`:30011` on the Mac. `scripts/game_client.py` hands you a finished mp4.

| what the game sees | measured |
|---|---:|
| server-side denoise + decode (`inference_time_s`) | **6.70 s** |
| clip finished **on the box** (POST → `completed`) | **8.34 s** |
| clip finished **on the Mac** (incl. download) | **11.32 s** |
| both boxes, sustained, clip on the box | **14.0 clips/min = 3.36× real-time** |
| both boxes, sustained, clip on the Mac | **10.6 clips/min = 2.53× real-time** |
| peak memory | **54,562 MB/GPU**, nothing offloaded |

That 6.70 s is slightly *better* than the 6.94–7.15 s in RESULTS.md, and 8.34 s matches its 8.02 s —
so this deployment is the measured configuration, not an approximation of it.

**It is not real-time, and no configuration of it is.** VDN-H3 has no streaming decode: the clip does
not begin to exist until all 8 steps and both VAEs are done. A game can use it as *generated content*
pipelined ahead of the player; it cannot call it inside a frame loop.
[What "interactive" can actually mean](#what-interactive-can-actually-mean) is the last section and the
part worth arguing with.

---

## Use it

Everything is already up. On the Mac:

```bash
cd /Users/henanwan/Documents/workspace/bytedance/minimax_h3_h100
bash scripts/game_tunnel.sh P5-1 P5-2          # if the tunnels dropped
bash scripts/game_tunnel.sh status
python3 scripts/game_client.py "a lantern-lit alley in the rain, footsteps and distant thunder"
python3 scripts/game_client.py --bench 6       # re-measure the table above
```

mp4s land in `out/game/`. Verified on the first clip: `864x480`, `nb_frames=345`, `24/1` fps,
`duration=14.375`, h264 + aac — exactly the requested shape, with audio.

From game code:

```python
from game_client import Renderer

r = Renderer()                 # reads the replica file game_tunnel.sh wrote
r.warm()                       # once, at startup -- see "the warmup flag lies", below
fut = r.submit("the gate grinds open, dust falls from the lintel")
path = fut.result()            # local mp4

# continue from where the last clip ended -- fl2va, keyframe pinned to frame 0.
# The path is on the SERVER: the worker resolves URIs, so a local path fails with
# "file not found" naming a file that plainly exists on your Mac.
fut = r.submit("she turns and runs", keyframe="/opt/dlami/nvme/vdn/outputs/prev_last.png")
```

One worker per replica pulls from one queue, so `submit()` never blocks and both boxes stay busy.
Two requests to the *same* replica do **not** interleave — the whole box is one model — so the replica
is the unit of concurrency and the pool is fixed at the replica count on purpose. **Two boxes buy
throughput, not latency:** 2 clips in flight, one clip still costs ~8 s.

**Failures raise, and are not retried.** A 400 means an off-lattice frame count or `ref2va` (which VDN
refuses: it ships only the `fl2va` partition — a training limit no flag changes); an OOM means the
memory arithmetic below. Retrying any of the three just makes a configuration error slower.

## Three things this deployment found that the runbook did not predict

**1. `--warmup-resolutions` does not warm the resolution.** Upstream's `_synthetic_warmup_target()`
(`configs/sample/minimax_h3.py:155`) uses that string *only* to pick the nearest aspect ratio;
`short_edge` is `MINIMAX_H3_RECOMMENDED_SHORT_EDGE = 768`, hardcoded. So a 480p server warms at 768p —
the log says `server warmup req (1344x768x345f)` even though the flag said `864x480`. `--warmup-num-frames`
*is* honoured (345 frames, confirmed). Cost: the first 480p request took **7.58 s** of server time
against a **6.70 s** steady state. Small, but it lands on the first clip the player waits for, so
`Renderer.warm()` / `game_client.py --warm` throws away one clip per replica at the real shape.

**2. Fetching the clip cost more than half the latency, and most of that was the ssh handshake.**
Cold scp of a 946 KB mp4 from us-east-2 to this Mac: **5.4–6.1 s**. Through a `ControlMaster` socket:
**2.65–3.4 s**. The tunnel already holds a connection to the same box, so `game_tunnel.sh` now opens it
as a master and `game_client.py` rides it — end-to-end **14.19 s → 11.32 s**, throughput 8.5 → 10.6
clips/min. What is left (~2.7 s) is the actual transfer at ~350 KB/s over the internet.
**If the game runs in-region, that whole 3 s disappears** and you are at the 8.34 s figure.

**3. `output_path` is a directory prefix, not a file prefix.** The server creates
`<output_path>/<uuid>.mp4` and returns the full path in `file_path` (and `url: null` — there is
genuinely no download endpoint). `outputs/` therefore accumulates one directory per clip; at ~950 KB
each on a 27 TB instance store this is not urgent, but a long-running game should sweep it.

## What it took to stand up (for the next box)

| step | time | note |
|---|---:|---|
| `game_push.sh P5-1 P5-2` | seconds | five files per box, by ssh alias |
| `h3.sh probe` | ~3 min | pulls `lmsysorg/sglang:dev`; all 17 imports `have`, no `MISSING` |
| `h3.sh weights vdn` | ~10 min | **78 GiB** (the VDN repo only, not the ~200 GiB of every arm) |
| `h3.sh serve game` → `Application startup complete` | 3 min 19 s | 65.65 GiB bf16 read, fp8 online |
| the server's own warmup render | 55 s | at 768p, see above |

```bash
bash scripts/game_push.sh P5-1 P5-2                    # on the Mac
# then on each box:
cd /opt/dlami/nvme/vdn/docker
bash h3.sh probe && bash h3.sh weights vdn
BASE=lmsysorg/sglang@sha256:d46a59f4b98658f728a1e006c003ad5ee0628e999fd8b2bef71ac1bb61b814da \
  FRAMES=345 bash h3.sh serve game
bash h3.sh logs game            # wait for "Application startup complete"
bash h3.sh stop                 # separate ssh invocation from the launch -- RUNBOOK trap 8
```

**Pin the digest.** `:dev` is rebuilt nightly, so two boxes set up an hour apart can differ. Both of
these run `sha256:d46a59f4b986…`, sglang `0.0.0.dev1+g20518d851`, bundled source
`20518d8518375f49be0d14ead7ea474dbc2721d0` (2026-09-17), torch 2.13.0+cu130, python 3.12.3.

**`/opt/dlami/nvme` is instance store: the 78 GiB is gone after a stop/start** and the weights step
runs again. That is the price of the fast disk, and the reason the image does not bake them in.

The servers bind **127.0.0.1 only**, deliberately: `ssh -L` reaches loopback anyway, and `0.0.0.0`
would put an unauthenticated video generator on the VPC for no gain.

## GPU count, if you ever run this on a smaller box

`game_serve.sh` counts cards itself, so it will start on anything — but the numbers above belong to 8.

**Ulysses shards the sequence, not the weights.** Every rank holds the whole DiT, both VAEs and the
Qwen3-VL conditioner; `--ulysses-degree` only decides how many ways the 43,759-row sequence is split.
Going 8 → 1 card does **not** reduce the resident footprint; it keeps it and multiplies each rank's
activation share by 8. The measured 8-rank peak is 54,562 MB on an 81,559 MiB card. By arithmetic off
that peak, a single card lands within a couple of GB of the limit — a coin-flip I have not flipped. If
a single-card warmup OOMs, that is why, and the ladder is:

```bash
FRAMES=345 bash h3.sh serve game --performance-mode auto
FRAMES=345 bash h3.sh serve game --layerwise-offload-components text_encoder
FRAMES=345 bash h3.sh serve game --layerwise-offload-components dit,text_encoder \
                                 --dit-layerwise-resident-layers 14
```

Each rung costs latency — record which one you used, because "it serves" and "it serves while
offloading the DiT" are different claims.

**I did not build one 16-GPU cross-node job.** It would need EFA (so the `NCCL_NET_PLUGIN=none` line in
`scripts/_env.sh` has to come out — it is there because `deep_ep` mistakes the AWS OFI plugin for a
duplicate NCCL runtime), the sequence-parallel all-to-all would then cross the network every layer,
and nothing in this repo has measured that. Two independent replicas need none of it and are what the
throughput numbers above come from.

## The frame count is a lattice

`align_num_frames(n, 17, 5)` snaps to the VAE's 17-frames-to-5-latents chunking. **Reachable lengths
are `5 + 17k` only:**

| frames | seconds @ 24 fps | |
|---:|---:|---|
| 226 | 9.417 | |
| **243** | **10.125** | the shorter beat; 0.72× the tokens, so roughly 0.72× the time |
| 260 | 10.833 | |
| 328 | 13.667 | |
| **345** | **14.375** | **deployed — the shape every number here and in RESULTS.md was measured at** |
| 362 | 15.083 | when the requirement is literally "15 seconds" |

Nothing exists between 345 and 362. **245 frames is not on the lattice** ((245−5)/17 = 14.1) and the
server does not reject it — it rounds silently, so you would get a length you did not ask for with no
log line saying so. `game_serve.sh` and `game_client.py` both refuse instead.

At 480p each latent frame is 405 tokens (54×30 latent, 2×2 patch → 27×15), so 345 f is 41,310 video
rows + 1299 text + 1150 audio = **43,759**. Shortening the clip is the one honest latency knob.

## Cost

At 8.34 s/clip on a p5.48xlarge, 480p is **$0.005261 per finished second of video** (README's
breakdown) — about **7.6 cents** per 14.375 s clip. Two boxes bill whether or not they are rendering,
so throughput is what decides affordability: at the measured 14.0 clips/min the pair produces roughly
$63/hour of video against roughly $70/hour of on-demand p5.48xlarge. Idle, it is $70/hour of nothing —
**stop the boxes when you are not rendering**, and remember the 78 GiB comes down again on restart.

## What "interactive" can actually mean

Three shapes work with ~8 s of latency and 14.4 s of output. One does not.

* **Render ahead.** The current clip plays for 14.4 s; you have 8 s to produce the next. Measured
  3.36× real-time across the pair means you can hold two replicas' worth of lookahead — e.g.
  speculatively render the two most likely player choices and discard one, and still keep ahead.
* **Scene beats.** Generate on a deliberate transition — entering a room, a dialogue turn — and cover
  the ~8 s with a fade or a still. **Verified on this deployment:** last frame of one clip → keyframe
  of the next costs **7.29 s server / 12.32 s E2E** (+8.8 % over t2va, better than the +13–16 % in
  RESULTS.md), and frame 0 of the continuation matches that keyframe at **33.16 dB PSNR** against
  **8.52 dB** for an unrelated frame. The seam is real, not aspirational.

  ```bash
  # cut the keyframe on the box (the worker resolves URIs, so it must live there)
  ssh P5-1 'docker exec h3-game bash -lc "ffmpeg -y -sseof -0.05 -i <clip>.mp4 -frames:v 1 \
      /opt/dlami/nvme/vdn/outputs/prev_last.png"'
  python3 scripts/game_client.py --keyframe /opt/dlami/nvme/vdn/outputs/prev_last.png "she turns and runs"
  ```
* **A visible wait.** The player types, waits ~11 s, gets a clip. Honest, and it is what the latency is.
* **Not this:** a clip per player input at interaction latency (<200 ms), or continuous streamed video.
  Nothing in VDN-H3 emits frames incrementally, and neither 243 frames nor more GPUs changes the kind
  of thing it is.

If the game needs sub-second reaction, the split is VDN-H3 for the pre-rendered/lookahead layer and
something else entirely for the frame-rate layer. I would rather say that now than after you have
built a loop around an 8 s call.

## Files

| file | runs on | what |
|---|---|---|
| `scripts/game_push.sh` | Mac | copies the five files each box needs, by ssh alias |
| `scripts/game_serve.sh` | box | one replica: GPU auto-detect, lattice check, loopback bind |
| `scripts/game_tunnel.sh` | Mac | one local port per replica, the ControlMaster socket, the replica file |
| `scripts/game_client.py` | Mac | submit/poll/scp, one worker per replica, `--warm`, `--bench N` |
| `docker/h3.sh` | box | the container wrapper; `serve game` is this arm |

`game_serve.sh` is deliberately separate from `scripts/sglang_arm.sh`. That file is the measurement
driver — its defaults are the published workload and every number in RESULTS.md and RUNBOOK.md is
quoted against it, so retuning it for a game would make those numbers refer to a command that no
longer exists. The serve flags are transcribed unchanged; only the shape, the GPU count and the
warmup differ. **If `sglang_arm.sh` gains a flag, `game_serve.sh` needs the same flag.**
