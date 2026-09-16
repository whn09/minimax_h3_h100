#!/usr/bin/env python3
"""What conditioning costs on the SGLang server: fl2va against t2va, both canvases.

    python3 sglang_cond.py            # needs `sglang_arm.sh serve` already up

The reference stack measured fl2va as +8.9% denoise over t2va at 768p (RESULTS.md,
"Conditioning: what fl2va costs over t2va"), from an offline prompt cache with the keyframes
already latent. Here the keyframes go in over HTTP as files and the visual tokenizer runs
inside the request (`visual_tokenizer_encode=True` on the fl2va condition rule), so this
measures the part that arithmetic could not: the encode of two images at the target canvas.

ref2va is probed once, not measured. MINIMAX_H3_TASK_PARTITIONS maps t2va and fl2va to the
`fl2va` partition and ref2va to a `ref2va` partition, and vdn-minimax-h3 ships only the
former -- upstream never trained ref2va for VDN. The probe is here so the refusal is on the
record as the server's own words rather than mine.
"""
import json
import statistics
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

HOST = "http://127.0.0.1:30010"
FRAMES = 345
DURATION = FRAMES / 24
KEYDIR = Path("/opt/dlami/nvme/vdn/keyframes")
OUTPUTS = Path("/opt/dlami/nvme/vdn/outputs")
CANVASES = {480: (864, 480), 768: (1344, 768)}
REPS = 3


def post(body: dict) -> dict:
    req = urllib.request.Request(
        f"{HOST}/v1/videos",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        return {"__error__": e.read().decode()[:600]}


def wait(vid: str, timeout: float = 300.0) -> dict:
    """Poll to completion. The API is async: POST returns `queued`."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        with urllib.request.urlopen(f"{HOST}/v1/videos/{vid}", timeout=30) as r:
            d = json.load(r)
        if d["status"] in ("completed", "failed"):
            return d
        time.sleep(0.2)
    raise TimeoutError(vid)


def keyframes(edge: int) -> tuple[Path, Path]:
    """First and last frame of an existing render at this canvas, as PNGs.

    Using the model's own output as the conditioning input keeps the keyframes exactly on the
    target canvas, so nothing here measures a resize that a real caller would not pay. It also
    means fl2va is asked to reproduce endpoints it is known to be able to draw.
    """
    w, h = CANVASES[edge]
    first, last = KEYDIR / f"first_{edge}.png", KEYDIR / f"last_{edge}.png"
    if first.exists() and last.exists():
        return first, last
    KEYDIR.mkdir(parents=True, exist_ok=True)
    src = next(
        (
            p
            for p in sorted(OUTPUTS.glob("*.mp4"), key=lambda p: -p.stat().st_mtime)
            if subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=width", "-of", "csv=p=0", str(p)],
                capture_output=True, text=True,
            ).stdout.strip() == str(w)
        ),
        None,
    )
    if src is None:
        raise SystemExit(f"no existing {w}x{h} mp4 under {OUTPUTS} to cut keyframes from")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(src),
                    "-vf", "select=eq(n\\,0)", "-vframes", "1", str(first)], check=True)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-sseof", "-0.2", "-i", str(src),
                    "-update", "1", "-q:v", "2", str(last)], check=True)
    return first, last


def body(edge: int, task: str, conds: list) -> dict:
    # `seconds` is deliberately absent: it is typed int and 345f at 24fps is 14.375s.
    return {
        "prompt": "a chef plates a dish in a bright kitchen, warm light, ambient clatter",
        "task": task,
        "conditions": conds,
        "target": {"short_edge": edge, "aspect_ratio": "16:9",
                   "duration_seconds": DURATION},
        "flow_shift": 12.0,
        "audio_flow_shift": 3.0,
    }


def run(edge: int, task: str, conds: list, reps: int) -> None:
    e2e, infer = [], []
    for i in range(reps + 1):  # +1 warmup, dropped: first request of a shape pays extra
        t0 = time.time()
        r = post(body(edge, task, conds))
        if "__error__" in r:
            print(f"  {task} {edge}p REJECTED: {r['__error__']}")
            return
        d = wait(r["id"])
        dt = time.time() - t0
        if d["status"] != "completed":
            print(f"  {task} {edge}p FAILED: {str(d.get('error'))[:400]}")
            return
        if i:
            e2e.append(dt)
            infer.append(d["inference_time_s"])
    print(f"  {task:6s} {edge}p  E2E median {statistics.median(e2e):6.2f} s "
          f"(n={len(e2e)}, {min(e2e):.2f}-{max(e2e):.2f})   "
          f"server inference median {statistics.median(infer):6.2f} s   "
          f"peak {d['peak_memory_mb']:.0f} MB")


if __name__ == "__main__":
    for edge in (480, 768):
        first, last = keyframes(edge)
        print(f"{edge}p, {FRAMES} f, {DURATION} s:")
        run(edge, "t2va", [], REPS)
        run(edge, "fl2va", [
            {"role": "keyframe", "type": "image", "uri": str(first), "frame_index": 0},
            {"role": "keyframe", "type": "image", "uri": str(last), "frame_index": -1},
        ], REPS)
        run(edge, "fl2va", [
            {"role": "keyframe", "type": "image", "uri": str(first), "frame_index": 0},
        ], REPS)
    print("ref2va probe (expected to be refused: VDN ships the fl2va partition only):")
    f480, _ = keyframes(480)
    run(480, "ref2va", [
        {"role": "reference", "type": "image", "uri": str(f480)},
    ], 1)
