#!/usr/bin/env python3
"""Run a case against MiniMax's own hosted H3 (the CN platform v2 API), not our SGLang box.

    export MINIMAX_API_KEY=...            # never put the key in a file; it is read from the env
    python3 scripts/minimax_api.py ir  --case case/demo_raw.txt --task ref2va
    python3 scripts/minimax_api.py gen --case case/demo_raw.txt --task ref2va --resolution 2K
    python3 scripts/minimax_api.py gen --prompt-file out/api/ir_ref2va.txt --image ref/demo_ref.jpg

WHY THIS EXISTS. Everything else in this repo drives the open-source release through SGLang, where
`prompt -> model` is the whole pipeline: presentation.py:109 `minimax_h3_text_only_ids()` is
"verbatim prompt, no special tokens", there is no chat template under model_specific_stages/, and
README.md:94 says H3-Context-IR is not in the open-source release. So the local renders are the
model *without* the layer MiniMax's own README calls "critical to the quality of the final output".
The CN API exposes both halves as separate endpoints, which makes the missing layer observable:

    POST /v2/h3_context_ir           -> a task whose result is `content.prompt`, an enriched prompt.
                                        Docs, verbatim: "H3-Context-IR 是一个复杂系统，暂不提供开源
                                        实现；本 API 既可用于验证 Full 2K-Workflow 的官方效果".
    POST /v2/video_generation        -> a task whose result is `content.url`, an mp4.
    GET  /v2/query/video_generation/{task_id}   -> both, discriminated by task_type.

So `ir` then `gen` is the official two-stage workflow, and it is also the cheap way round: the IR
call returns text, costs a fraction of a render, and the prompt it hands back can be replayed on the
g7 box for free. Run `ir` first, read the prompt, then `gen`.

TWO THINGS THE SCHEMA DOES *NOT* LET US DO, both load-bearing for this case:

* `first_frame`/`last_frame` and `reference_image`/`reference_video`/`reference_audio` are MUTUALLY
  EXCLUSIVE inside one `content` array ("图生视频与多模态参考生视频互斥"). The local fl2va arm
  (ref2va.jpg pinned as the last keyframe) and the local ref2va arm are therefore two separate
  requests here, never one.
* There is no seed, no step count and no `short_edge`. The only geometry knob is
  `resolution: 768P | 2K` plus `ratio`, and for a reference request `ratio` defaults to `adaptive`.
  So an API render is NOT matched-config against a local one and must not be quoted as an ablation
  of it; it answers "what does the vendor's own stack produce", which is a different question.

`resolution: "2K"` is worth asking for even though every local arm is 768p. It is not available
locally at all -- resolved_plan.py:42 MINIMAX_H3_MAX_PIXELS = 768 * 1344 = 1,032,192 and 1344x768 is
that cap exactly, so a larger short_edge resolves straight back to it (resolved_plan.py:159-165) --
and PROMPT_IR.md §3.4 pins the handwriting failure on precisely that grid: a chalk stroke is ~8-10 px
= ~0.5 of one 16x latent cell. 2K is the one configuration in which that arithmetic changes.

THE IMAGE GOES IN AS A data: URI. Limits are >= 256 px on both sides, <= 5760 px, aspect (w/h) in
[0.4, 2.5], <= 30 MB per file, <= 9 reference images, and <= 64 MB for the whole request body; the
docs say to prefer public URLs for *large* files, and ref/demo_ref.jpg is 131 kB / 2752x1536, which is
inside every one of those and needs no hosting. Note that the create call validates the *shape*
only: a 1x1 px probe image was accepted and returned a task_id, so a bad image surfaces later as a
failed task, not as a 400.
"""
import argparse
import base64
import json
import mimetypes
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = "https://api.minimax.cn/v2"
# The item shape is nested, and this was measured rather than read: the docs collapse the child
# attributes of `content`, and a flat {"type":"image_url","url":...} comes back with
# `content[1].image_url.url is empty for type=image_url (2013)`.
ROLE_REFERENCE = "reference_image"
ROLE_LAST = "last_frame"
ROLE_FIRST = "first_frame"
POLL_SECONDS = 10
POLL_LIMIT = 180  # 30 min; a 2K render is minutes, and the loop prints every status change


def key() -> str:
    k = os.environ.get("MINIMAX_API_KEY")
    if not k:
        raise SystemExit("MINIMAX_API_KEY is not set. Export it; do not write it into a file.")
    return k


def post(path: str, body: dict) -> dict:
    req = urllib.request.Request(f"{BASE}/{path}", data=json.dumps(body).encode(),
                                headers={"Authorization": f"Bearer {key()}",
                                         "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        # Params/auth/balance errors come back as a real 4xx with the MiniMax code in the body.
        raise SystemExit(f"POST {path} -> {e.code}: {e.read().decode()[:800]}")


def get(path: str) -> dict:
    req = urllib.request.Request(f"{BASE}/{path}",
                                headers={"Authorization": f"Bearer {key()}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def data_uri(p: Path) -> str:
    mime = mimetypes.guess_type(p.name)[0] or "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(p.read_bytes()).decode()


def wait(task_id: str) -> dict:
    """Poll until terminal. status: queued | running | succeeded | failed | cancelled."""
    last, t0 = None, time.time()
    for _ in range(POLL_LIMIT):
        t = get(f"query/video_generation/{task_id}")["task"]
        st = t.get("status")
        if st != last:
            print(f"  [{time.time() - t0:6.1f}s] {st}", flush=True)
            last = st
        if st in ("succeeded", "failed", "cancelled"):
            return t
        time.sleep(POLL_SECONDS)
    raise SystemExit(f"task {task_id} still {last} after {POLL_LIMIT * POLL_SECONDS}s")


def case_line(case: Path, task: str) -> tuple[str, str | None]:
    """(prompt, image filename or None) for the first `task:` line -- same format as sglang_case.py.

    Deliberately the same parser contract as the local driver, including the NBSP strip (the real
    case.txt has a U+00A0 after `t2va:`), so the API and the local box are fed byte-identical text.
    """
    for raw in case.read_text().splitlines():
        line = raw.replace("\u00a0", " ").strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        label, rest = (s.strip() for s in line.split(":", 1))
        if label.split("@")[0] != task:
            continue
        head, _, lastw = rest.rpartition(" ")
        if head and "." in lastw and "/" not in lastw and " " not in lastw:
            return head.strip().replace("\\n", "\n"), lastw
        return rest.replace("\\n", "\n"), None
    raise SystemExit(f"no `{task}:` line in {case}")


def build(model: str, prompt: str, image: Path | None, role: str, duration: int,
          ratio: str | None, resolution: str | None) -> dict:
    content: list[dict] = [{"type": "text", "text": prompt}]
    if image:
        content.append({"type": "image_url", "role": role,
                        "image_url": {"url": data_uri(image)}})
    body: dict = {"model": model, "content": content, "duration": duration}
    if ratio:
        body["ratio"] = ratio
    if resolution:
        body["resolution"] = resolution
    # `extra` is left out on purpose. The only declared member is prompt_expansion_mode (default
    # "balanced"), and the docs warn explicitly against sending "balance", an empty string, a boolean
    # or any undeclared field -- so the default is the only value this script will send.
    return body


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("ir", "gen", "query"))
    ap.add_argument("--case", type=Path)
    ap.add_argument("--task", default="ref2va", help="which line of the case file")
    ap.add_argument("--prompt-file", type=Path, help="use this text instead of the case line")
    ap.add_argument("--image", type=Path, help="override the case file's image")
    ap.add_argument("--role", default=ROLE_REFERENCE,
                    choices=(ROLE_REFERENCE, ROLE_FIRST, ROLE_LAST))
    ap.add_argument("--model", default="MiniMax-H3")
    ap.add_argument("--duration", type=int, default=5)
    ap.add_argument("--ratio", default="16:9")
    ap.add_argument("--resolution", default="2K")
    ap.add_argument("--outdir", type=Path, default=Path("out/api"))
    ap.add_argument("--tag", default="")
    ap.add_argument("--task-id", help="for `query`: poll an existing task")
    a = ap.parse_args()
    a.outdir.mkdir(parents=True, exist_ok=True)

    if a.mode == "query":
        t = wait(a.task_id)
        print(json.dumps(t, ensure_ascii=False, indent=2))
        return

    if a.prompt_file:
        prompt, img = a.prompt_file.read_text().strip(), a.image
    else:
        if not a.case:
            raise SystemExit("--case or --prompt-file is required")
        prompt, name = case_line(a.case, a.task)
        img = a.image or (a.case.parent / name if name else None)
    if img and not img.is_file():
        raise SystemExit(f"image {img} does not exist")

    tag = a.tag or a.task
    if a.mode == "ir":
        # No resolution on the IR endpoint: it only takes model/content/duration/ratio.
        body = build(a.model, prompt, img, a.role, a.duration, a.ratio, None)
        print(f"IR  {len(prompt)} chars, image {img}, role {a.role}, {a.duration}s {a.ratio}",
              flush=True)
        tid = post("h3_context_ir", body)["task_id"]
        print(f"  task_id {tid}", flush=True)
        t = wait(tid)
        if t["status"] != "succeeded":
            raise SystemExit(json.dumps(t, ensure_ascii=False)[:1200])
        out = a.outdir / f"ir_{tag}.txt"
        out.write_text(t["content"]["prompt"])
        print(f"  -> {out}  ({len(t['content']['prompt'])} chars)  usage {t.get('usage')}")
        return

    body = build(a.model, prompt, img, a.role, a.duration, a.ratio, a.resolution)
    print(f"GEN {len(prompt)} chars, image {img}, role {a.role}, "
          f"{a.duration}s {a.ratio} {a.resolution} on {a.model}", flush=True)
    tid = post("video_generation", body)["task_id"]
    print(f"  task_id {tid}", flush=True)
    t = wait(tid)
    if t["status"] != "succeeded":
        raise SystemExit(json.dumps(t, ensure_ascii=False)[:1200])
    url = t["content"]["url"]
    dst = a.outdir / f"{tag}_{a.resolution}_{a.duration}s.mp4"
    with urllib.request.urlopen(url, timeout=300) as r, open(dst, "wb") as f:
        f.write(r.read())
    print(f"  {t['content'].get('resolution')} {t['content'].get('duration')}s  "
          f"usage {t.get('usage')}\n  -> {dst} ({dst.stat().st_size} bytes)")
    (a.outdir / f"{tag}_task.json").write_text(json.dumps(t, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.exit(main())
