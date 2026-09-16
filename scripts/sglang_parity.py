#!/usr/bin/env python3
"""Same prompt, same seed, both stacks -- so the quality question is asked at matched input.

    python3 sglang_parity.py            # needs `sglang_arm.sh serve` already up

The reference stack's renders (samples/n_480p_seg4.mp4, samples/p_768p_free.mp4) are t2va from
`prompts/example_2.pt` at seed 42 (`src/config/inference.py:167`). That .pt carries the prompt
*text* next to its embeddings, so the same words can go over HTTP to sglang, which re-encodes
them with the same Qwen3-VL conditioner the cache was built from. Same words, same conditioner,
same canvas, same frame count, same seed.

WHAT THIS IS NOT: a pixel comparison. Three separate reasons, none of them fixable here.
  1. Seed 42 does not mean the same noise. The two stacks sample the initial latent with
     different generator placement and different per-rank shapes, so `seed` fixes the stream,
     not the tensor.
  2. fp8 differs. The reference stack assembles per-tensor fp8 on the host (patch 2); sglang
     quantizes per-channel online after loading bf16 onto the card.
  3. Multi-rank denoise is not bit-reproducible even against itself -- `index_add_` atomics plus
     `all_reduce` reorder float work, and RESULTS.md measures ~17 dB PSNR between two runs of the
     *same* stack at the same seed (RUNBOOK trap 9).
So the output is a same-prompt qualitative pair to look at, and any PSNR computed against the
reference render measures reordered arithmetic, not quality. Look at the videos.
"""
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

HOST = "http://127.0.0.1:30010"
FRAMES = 345
DURATION = FRAMES / 24
SEED = 42                      # src/config/inference.py:167, the reference stack's default
PROMPT_PT = Path("/opt/dlami/nvme/vdn/vdn-minimax-h3/prompts/example_2.pt")
OUTDIR = Path("/opt/dlami/nvme/vdn/pull")


def prompt_text() -> str:
    """The words the reference stack rendered, read out of its own prompt cache."""
    import torch  # only needed for this; the sglang venv has it

    d = torch.load(PROMPT_PT, weights_only=False, map_location="cpu")
    return d["prompt"]


def go(edge: int, prompt: str) -> None:
    out = OUTDIR / f"parity_{edge}p_345f_seed{SEED}.mp4"
    body = {
        "prompt": prompt,
        "task": "t2va",
        "conditions": [],
        "target": {"short_edge": edge, "aspect_ratio": "16:9", "duration_seconds": DURATION},
        "seed": SEED,
        "flow_shift": 12.0,
        "audio_flow_shift": 3.0,
        "output_path": str(out),
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
        print(f"{edge}p REJECTED: {e.read().decode()[:600]}", flush=True)
        return
    while True:
        with urllib.request.urlopen(f"{HOST}/v1/videos/{j['id']}", timeout=30) as r:
            d = json.load(r)
        if d["status"] in ("completed", "failed"):
            break
        time.sleep(0.25)
    if d["status"] != "completed":
        print(f"{edge}p FAILED: {str(d.get('error'))[:400]}", flush=True)
        return
    print(f"{edge}p seed {SEED}: E2E {time.time() - t0:5.2f} s  "
          f"inference {d['inference_time_s']:5.2f} s  peak {d['peak_memory_mb']:.0f} MB  "
          f"-> {d.get('file_path')}", flush=True)


if __name__ == "__main__":
    OUTDIR.mkdir(parents=True, exist_ok=True)
    p = prompt_text()
    print(f"prompt: {len(p)} chars from {PROMPT_PT.name}, seed {SEED}, {FRAMES} f", flush=True)
    for edge in (480, 768):
        go(edge, p)
