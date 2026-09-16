#!/usr/bin/env python3
"""One ref2va render per arm, at fixed seed and fixed prompt, so quality is the only variable.

    python3 sglang_ref2va.py ref=/opt/dlami/nvme/vdn/ref/subject.png tag=B 480:8
    python3 sglang_ref2va.py ref=... tag=A 480:50            # the control
    python3 sglang_ref2va.py ref=... tag=F1024 480:8 shift=12

Positional arguments are `<short_edge>:<num_inference_steps>`. Server-side settings (which LoRA,
which alpha, which reference short edge) are NOT here -- they are `sglang_ref2va_arm.sh` flags and
need a restart. That split is the whole reason this script is small: one arm per server, one tag
per arm, and the mp4s land side by side for `melt_metrics.py`.

WHY THE PROMPT IS THIS PROMPT. The claim under test is "融化 / melting": a subject losing
structural integrity while it moves. Every community report of it names the same three triggers --
hands (#33: "this lora messes the hands, they become blurry"), fast motion (#13: "fight-scene
outputs suffer from severe artifacts, body distortion and blurring"), and fine moving detail
(#33: "hair flowing in the wind lacks details"). A calm talking-head prompt would not discriminate
between any of the arms in REF2VA.md, so the default prompt asks for all three plus a line of
dialogue, which also exposes the audio-shift language-blending report in #30. Override with
prompt=<file> if the customer supplies their own; keep it identical across arms either way.

Reference-image conditions carry no frame_index (`requires_frame_index=False` on the ref2va
reference rules), unlike fl2va keyframes. One reference by default: the row cost is quadratic in
the reference short edge and 12 references at 2048 would be 87,552 rows, more than the 480p video
itself -- so a multi-reference arm is a separate question from melting, and is not queued.
"""
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HOST = "http://127.0.0.1:30012"     # 30012 = ref2va; 30011 base t2va; 30010 VDN
FRAMES = 345
DURATION = FRAMES / 24
SEED = 42
OUTDIR = Path("/opt/dlami/nvme/vdn/pull/ref2va")

# Hands, fast motion, moving fine detail, and one spoken line. See the docstring.
PROMPT = (
    "A woman in a dark green jacket spins to face the camera in a windy courtyard at golden hour, "
    "her long hair whipping across her face, then raises both hands and counts to three on her "
    "fingers, holding each hand open and close to the lens. She says: \"Three. Two. One.\" "
    "Handheld camera, natural light, wind noise, her voice close and clear."
)


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
        return {"__error__": e.read().decode()[:800]}


def go(edge: int, steps: int, ref: str, tag: str, prompt: str, shift: float | None) -> None:
    body = {
        "prompt": prompt,
        "task": "ref2va",
        "conditions": [{"role": "reference", "type": "image", "uri": ref}],
        "target": {"short_edge": edge, "aspect_ratio": "16:9", "duration_seconds": DURATION},
        "num_inference_steps": steps,
        "seed": SEED,
        "output_path": str(OUTDIR / f"ref2va_{tag}_{edge}p_{steps}step"),
    }
    # Unset means the server's own ref2va default, which is already lightx2v's recommendation for
    # the ref2v LoRAs: video 12.0 / audio 3.0 (task_profiles.py, and lightx2v discussion #51).
    # Only the 768p *fl2v* models want 6. Pass shift= to test that, not to "fix" anything.
    if shift is not None:
        body["flow_shift"] = shift
        body["audio_flow_shift"] = 3.0
    t0 = time.time()
    r = post(body)
    if "__error__" in r:
        print(f"{tag} {edge}p {steps} steps REJECTED: {r['__error__']}", flush=True)
        return
    while True:
        with urllib.request.urlopen(f"{HOST}/v1/videos/{r['id']}", timeout=30) as g:
            d = json.load(g)
        if d["status"] in ("completed", "failed"):
            break
        time.sleep(0.5)
    dt = time.time() - t0
    if d["status"] != "completed":
        print(f"{tag} {edge}p {steps} steps FAILED: {str(d.get('error'))[:500]}", flush=True)
        return
    inf = d["inference_time_s"]
    print(f"{tag:>8s} {edge}p {steps:>2} steps: E2E {dt:7.2f} s  inference {inf:7.2f} s  "
          f"{inf / steps:5.2f} s/step  peak {d['peak_memory_mb']:.0f} MB  -> {d.get('file_path')}",
          flush=True)


if __name__ == "__main__":
    ref = tag = None
    prompt, shift, arms = PROMPT, None, []
    for a in sys.argv[1:]:
        if a.startswith("ref="):
            ref = a[4:]
        elif a.startswith("tag="):
            tag = a[4:]
        elif a.startswith("shift="):
            shift = float(a[6:])
        elif a.startswith("prompt="):
            prompt = Path(a[7:]).read_text().strip()
        else:
            arms.append(a)
    if not ref or not Path(ref).is_file():
        raise SystemExit(f"ref=<image> is required and must exist (got {ref!r}). Cut one from an "
                         "existing render: ffmpeg -i out.mp4 -vf 'select=eq(n\\,0)' -vframes 1 "
                         "subject.png")
    if not tag:
        raise SystemExit("tag=<arm name> is required; it is what tells two mp4s apart afterwards")
    OUTDIR.mkdir(parents=True, exist_ok=True)
    print(f"ref2va, {FRAMES} f, seed {SEED}, ref {ref}, prompt {len(prompt)} chars, "
          f"shift {'server default (12/3)' if shift is None else shift}", flush=True)
    for a in arms or ["480:8"]:
        edge, steps = a.split(":")
        go(int(edge), int(steps), ref, tag, prompt, shift)
