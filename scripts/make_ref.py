#!/usr/bin/env python3
"""Render the ONE reference image every ref2va arm shares, against the base t2va server.

    FRAMES=97 bash h3.sh serve t2va 768        # 97-frame warmup, not 345
    bash h3.sh exec make_ref.py                # -> /opt/dlami/nvme/vdn/ref/subject.png

WHY THIS EXISTS AT ALL. REF2VA.md's arms A-G differ in exactly one server flag each; the reference
image is the input they all hold fixed, and there was no way to produce one. The two mp4s already on
the box are useless as references -- s4_768p (cathedral square) has a ~30 px distant figure and
f_768p_fl2va (a sail) has no subject -- and "融化" is a claim about a *subject* losing structure, so a
reference without a subject cannot discriminate between any two arms.

WHY GENERATED, NOT A PHOTO. Three reasons, in order of how much they would cost to get wrong:
  * It is regenerable. Prompt, seed, steps and the frame-choice rule are all in this file, so the
    reference can be reproduced bit-for-bit on another box. A downloaded photo is an unversioned
    200 KB blob that has to travel with the results, and if it is ever replaced, every arm measured
    before the swap silently stops being comparable to every arm after it.
  * It is in-distribution. A reference the base model itself produced cannot be blamed for the
    melting: if arm A (base, 50 steps, no LoRA) holds the subject together on this reference and
    arm B does not, the difference is the LoRA. A stock photo introduces a second explanation --
    lighting, grain, compression, a crop the model has never seen -- that no metric can separate.
  * No likeness. Nobody's face ends up in a benchmark artifact.

WHY THE SUBJECT LOOKS LIKE THIS. It has to carry the three things every melting report names, or the
arms cannot differ: visible hands with separated fingers, long hair, and a face. Hands especially --
the reference is where the fingers' *identity* comes from, and lightx2v #33 is specifically that the
LoRA blurs them. A reference with hands in pockets would make arm B look fine.

WHY IT IS A NEAR-STILL PROMPT. This clip is a source of one sharp frame, not a video. Static camera,
minimal motion, no motion blur to inherit. Sharpness of the *reference* is a confound the whole
comparison would otherwise carry: a soft reference produces mush in every arm, which reads exactly
like melting and is not.

WHY 97 FRAMES. DiT cost is linear in latent rows and rows are linear in frames, so the clip should be
as short as the server allows: 97 frames is ~28% of the 345-frame arms, about 50 s against 187 at
768p/50 steps. It cannot go lower. The floor is a server-side validation, not a shape constraint --
`ValueError: target.duration_seconds must be in [4, 15]` -- and 4 s at 24 fps is 96 frames, which is
0 mod 8 where the model wants 1 mod 8 (345 = 8*43+1), so 97 it is. Serve the arm with FRAMES=97 too:
warmup then compiles the shape this request uses, and the render pays no recompile.

WHY THE SHARPEST FRAME, not frame 0. Frame 0 of these renders is routinely the softest -- the first
latent has the least temporal context -- and a reference is a still, so there is no reason to accept
it. Laplacian variance over every frame is the standard focus measure, it is deterministic given the
mp4, and it prints so the choice is auditable. This is the same measure melt_metrics.py uses per
frame, which is deliberate: the reference is scored by the same ruler as the arms.
"""
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HOST = "http://127.0.0.1:30011"        # 30011 = base t2va (sglang_base_arm.sh's default port)
FRAMES = 97                            # the shortest clip the server accepts; see below
SEED = 42                              # same seed as every arm, for no reason except consistency
STEPS = 50                             # base H3 is a 50-step model by design; this is not an arm
EDGE = 768                             # short edge of the *source* clip, not of the reference crop
OUTDIR = Path("/opt/dlami/nvme/vdn/ref")

PROMPT = (
    "Medium close-up of a woman in a dark green jacket standing in a courtyard at golden hour, "
    "facing the camera. She holds both hands up at chest height, palms toward the lens, fingers "
    "spread and clearly separated. Her long dark hair falls over her shoulders. She stands still "
    "and looks into the lens; the camera does not move. Sharp focus on her face and hands, soft "
    "background, natural warm light, quiet ambience."
)


def post(body: dict) -> dict:
    req = urllib.request.Request(f"{HOST}/v1/videos", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        return {"__error__": e.read().decode()[:800]}


def render(prompt: str, edge: int, frames: int, steps: int) -> str:
    body = {
        "prompt": prompt,
        "task": "t2va",
        "target": {"short_edge": edge, "aspect_ratio": "16:9", "duration_seconds": frames / 24},
        "num_inference_steps": steps,
        "seed": SEED,
        "output_path": str(OUTDIR / f"refsrc_{edge}p_{frames}f"),
    }
    t0 = time.time()
    r = post(body)
    if "__error__" in r:
        raise SystemExit(f"server rejected the request: {r['__error__']}")
    while True:
        with urllib.request.urlopen(f"{HOST}/v1/videos/{r['id']}", timeout=30) as g:
            d = json.load(g)
        if d["status"] in ("completed", "failed"):
            break
        time.sleep(0.5)
    if d["status"] != "completed":
        raise SystemExit(f"render failed: {str(d.get('error'))[:600]}")
    print(f"source clip: {edge}p {frames} f {steps} steps in {time.time() - t0:.1f} s "
          f"(inference {d['inference_time_s']:.1f} s) -> {d.get('file_path')}", flush=True)
    return d["file_path"]


def pick_sharpest(mp4: str, out: Path) -> dict:
    # cv2 ships in the image (opencv-python-headless, via sglang[diffusion]), so no ffmpeg pipe and
    # no intermediate PNGs: decode, score, keep the best frame in memory, write it once.
    import cv2

    cap = cv2.VideoCapture(mp4)
    best = (-1.0, -1, None)
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        v = float(cv2.Laplacian(g, cv2.CV_64F).var())
        if v > best[0]:
            best = (v, i, frame.copy())
        i += 1
    cap.release()
    if best[2] is None:
        raise SystemExit(f"no decodable frames in {mp4}")
    out.parent.mkdir(parents=True, exist_ok=True)
    # PNG, not JPEG: the reference is re-encoded and resized by the server anyway, and adding JPEG
    # ringing to the one fixed input of a quality benchmark is a free way to lose an argument.
    cv2.imwrite(str(out), best[2])
    h, w = best[2].shape[:2]
    print(f"reference: frame {best[1]}/{i} laplacian var {best[0]:.1f}  {w}x{h}  -> {out}",
          flush=True)
    return {"frame_index": best[1], "frames_scanned": i, "laplacian_var": round(best[0], 2),
            "width": w, "height": h}


if __name__ == "__main__":
    prompt, edge, frames, steps = PROMPT, EDGE, FRAMES, STEPS
    out = OUTDIR / "subject.png"
    src = None
    for a in sys.argv[1:]:
        if a.startswith("prompt="):
            prompt = Path(a[7:]).read_text().strip()
        elif a.startswith("out="):
            out = Path(a[4:])
        elif a.startswith("edge="):
            edge = int(a[5:])
        elif a.startswith("frames="):
            frames = int(a[7:])
        elif a.startswith("steps="):
            steps = int(a[6:])
        elif a.startswith("from="):
            src = a[5:]          # skip the render, cut from an mp4 that already exists
    mp4 = src or render(prompt, edge, frames, steps)
    meta = pick_sharpest(mp4, out)
    # The sidecar is what makes the reference citable in RESULTS.md: it says which prompt, which
    # seed and which frame produced the image every arm was conditioned on.
    meta.update(source_mp4=mp4, prompt=prompt, seed=SEED,
                steps=steps if src is None else None, short_edge=edge if src is None else None)
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {out.with_suffix('.json')}", flush=True)
