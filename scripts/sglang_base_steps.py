#!/usr/bin/env python3
"""Base MiniMax-H3 against VDN on the same eight cards, at the schedule each was trained for.

    python3 sglang_base_steps.py 480:50 480:8 768:50 768:8

Each argument is `<short_edge>:<num_inference_steps>`, run in the order given, one measured
request each with no warmup discard -- the server already warmed up at startup, and the box this
runs on is on a clock.

WHY THE STEP COUNT IS THE WHOLE STORY. Three members of one family, all with the same DiT shape
and the same VAEs, differing in how many denoise steps they were distilled for:

    MiniMaxAI/MiniMax-H3      50 steps   configs/sample/minimax_h3.py:41
    FastH3                     5 sigma points = 4 DiT forwards, t2va only (asserts)
    OpenVDN/vdn-minimax-h3     9 sigma points = 8 DiT forwards          (asserts)

So `480:50` is what "normal H3" costs, and `480:8` is NOT a model anyone would ship -- base
weights denoised on a schedule they were not trained for come out undercooked. It is here because
it isolates the two things the 50-vs-8 comparison confounds: how much of VDN's win is fewer steps,
and how much is a cheaper step (the hybrid linear/softmax branch VDN adds and base H3 does not
have). Subtract and you get both.

WHAT IT ANSWERED, and the answer flips with resolution, which is why both edges are in the ladder:
at 480p the two models cost the same per step (base 0.98 vs VDN 0.95 s/step, 3 %), so all of the
5.28x is the schedule; at 768p VDN's step is 1.72x cheaper (2.17 vs 3.73 s/step) and 6.25x x 1.72x
is the 9.82x measured. Per-step cost from 480p to 768p, against 2.49x the tokens: base 3.81x
(superlinear, dense softmax), VDN 2.28x (sublinear). The hybrid attention earns nothing until the
sequence is long. Full numbers in RESULTS.md.

The Turbo LoRA arm is not here, and not only because --lora-path is a *server* argument that needs
a restart: it does not run alongside --quantization fp8 at all on this build, and it cannot be
merged offline into a t2va tree without a key-layout translation. Both reasons, with the exact
failures, are in scripts/sglang_base_arm.sh and scripts/lora_merge_h3.py. The latency it would
have reported is the `8`-step row above unchanged, since merging an adapter moves weight values
and not shapes.
"""
import json
import sys
import time
import urllib.error
import urllib.request

HOST = "http://127.0.0.1:30011"     # 30011 = base H3; 30010 is the VDN server
FRAMES = 345
DURATION = FRAMES / 24
SEED = 42
OUTDIR = "/opt/dlami/nvme/vdn/pull/base"
PROMPT_PT = "/opt/dlami/nvme/vdn/vdn-minimax-h3/prompts/example_2.pt"


def prompt_text() -> str:
    import torch

    return torch.load(PROMPT_PT, weights_only=False, map_location="cpu")["prompt"]


def go(edge: int, steps: int, prompt: str, tag: str = "") -> None:
    body = {
        "prompt": prompt,
        "task": "t2va",
        "conditions": [],
        "target": {"short_edge": edge, "aspect_ratio": "16:9", "duration_seconds": DURATION},
        "num_inference_steps": steps,
        "seed": SEED,
        "output_path": f"{OUTDIR}/base_{edge}p_{steps}step{tag}",
    }
    t0 = time.time()
    req = urllib.request.Request(
        f"{HOST}/v1/videos",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            j = json.load(r)
    except urllib.error.HTTPError as e:
        print(f"{edge}p {steps} steps REJECTED: {e.read().decode()[:500]}", flush=True)
        return
    while True:
        with urllib.request.urlopen(f"{HOST}/v1/videos/{j['id']}", timeout=30) as r:
            d = json.load(r)
        if d["status"] in ("completed", "failed"):
            break
        time.sleep(0.5)
    dt = time.time() - t0
    if d["status"] != "completed":
        print(f"{edge}p {steps} steps FAILED: {str(d.get('error'))[:400]}", flush=True)
        return
    inf = d["inference_time_s"]
    print(f"{edge}p {steps:>2} steps{tag}: E2E {dt:7.2f} s  inference {inf:7.2f} s  "
          f"{inf / steps:5.2f} s/step  peak {d['peak_memory_mb']:.0f} MB  -> {d.get('file_path')}",
          flush=True)


if __name__ == "__main__":
    p = prompt_text()
    tag = ""
    args = []
    for a in sys.argv[1:]:
        if a.startswith("tag="):
            tag = "_" + a[4:]
        else:
            args.append(a)
    print(f"base H3, {FRAMES} f, seed {SEED}, prompt {len(p)} chars{tag}", flush=True)
    for a in args or ["480:50"]:
        edge, steps = a.split(":")
        go(int(edge), int(steps), p, tag)
