"""The real thing: MiniMax's own `POST /v2/h3_context_ir`, used as an oracle.

MiniMax ships H3-Context-IR as a paid endpoint. It does not create a video -- it returns only the
enhanced prompt, in exactly the field format this package validates. That makes it three things we
did not have before:

  1. **A ground truth for the validator.** Genuine IR output must never be rejected. One documented
     sample already lives in `golds/minimax_official_api.json` and is asserted in the test suite;
     `harvest` lets us grow that corpus and treat any rule that fires on official output as a bug.
  2. **A teacher.** (request, official IR) pairs are exactly the supervision a distilled local
     compiler needs. At the published $0.90/M in + $3.60/M out and ~9.1k tokens per call, a pair
     costs about $0.017, so a 10k-pair set is on the order of $170 -- cheap relative to one day of
     8xB300 time.
  3. **A ceiling to measure against.** Not a production path: the documented worked example took
     29 s wall clock (`updated_at - created_at`), which is over three times our Haiku-4.5 compile.

Deliberately stdlib-only (`urllib`) so harvesting works in any venv, and deliberately reads the key
from the environment: `MINIMAX_API_KEY` is never written to a file in this repo.

  export MINIMAX_API_KEY=...      # platform.minimax.io -> Account Management -> API Keys
  python -m h3ir.cli oracle --duration 5 --ratio 16:9 'A boy playing basketball by the sea'
  python -m h3ir.cli harvest --in requests.jsonl --out golds/harvest.jsonl
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field as _dc_field

from .validate import Report, validate

# MiniMax runs two independent platforms with separate accounts and NON-INTERCHANGEABLE keys, and
# the OpenAPI spec only documents the international host. A China-console key against api.minimax.io
# returns `invalid api key (2049)` with HTTP 401 -- indistinguishable from a bad key, which is the
# trap. Measured 2026-09-18: the same key that 401s on `.io` returns 200 on `api.minimaxi.com` and
# on the legacy `api.minimax.chat`. Pick with MINIMAX_REGION=cn|global, or override outright.
HOSTS = {
    "cn": "https://api.minimaxi.com",       # platform.minimaxi.com console; api.minimax.chat also works
    "global": "https://api.minimax.io",     # platform.minimax.io console (the documented one)
}
BASE_URL = os.environ.get(
    "MINIMAX_BASE_URL") or HOSTS.get(os.environ.get("MINIMAX_REGION", "cn").lower(), HOSTS["cn"])
CREATE_PATH = "/v2/h3_context_ir"
QUERY_PATH = "/v2/query/video_generation/{task_id}"

# Published pay-as-you-go rates for MiniMax-H3-Context-IR, USD per million tokens
# (platform.minimax.io/docs/guides/pricing-paygo#video, retrieved 2026-09-18).
PRICE_IN_PER_MTOK = 0.90
PRICE_OUT_PER_MTOK = 3.60

TERMINAL = {"succeeded", "failed", "cancelled"}

# `ratio` is required and may not be `adaptive` for text-only content; for first/last-frame input it
# is forced to `adaptive` by the server. See H3ContextIRReq in the OpenAPI spec.
CONCRETE_RATIOS = ("21:9", "16:9", "4:3", "1:1", "3:4", "9:16")


class OracleError(RuntimeError):
    pass


@dataclass
class OracleResult:
    request: dict
    prompt: str | None
    status: str
    task_id: str
    usage: dict = _dc_field(default_factory=dict)
    seconds: float = 0.0
    server_seconds: int = 0
    error: dict | None = None
    report: Report | None = None

    @property
    def cost_usd(self) -> float:
        u = self.usage or {}
        return (u.get("prompt_tokens", 0) * PRICE_IN_PER_MTOK
                + u.get("completion_tokens", 0) * PRICE_OUT_PER_MTOK) / 1e6

    def to_json(self) -> dict:
        out = {
            "request": self.request,
            "task_id": self.task_id,
            "status": self.status,
            "prompt": self.prompt,
            "usage": self.usage,
            "cost_usd": round(self.cost_usd, 6),
            "wall_seconds": round(self.seconds, 2),
            "server_seconds": self.server_seconds,
        }
        if self.error:
            out["error"] = self.error
        if self.report is not None:
            # The interesting column. Errors here are validator bugs, not prompt bugs.
            out["validator"] = {
                "ok": self.report.ok,
                "errors": [v.rule for v in self.report.errors],
                "warnings": [v.rule for v in self.report.warnings],
            }
        return out


def _api_key() -> str:
    key = os.environ.get("MINIMAX_API_KEY", "").strip()
    if not key:
        raise OracleError(
            "MINIMAX_API_KEY is not set. Get one at platform.minimax.io -> Account Management -> "
            "API Keys, enable the Pay-as-you-go API for video, and export it in your shell. Do not "
            "put it in a file in this repo.")
    return key


def _call(method: str, path: str, body: dict | None = None, timeout: int = 120) -> dict:
    req = urllib.request.Request(
        BASE_URL + path,
        method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Authorization": f"Bearer {_api_key()}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:800]
        hint = ""
        if e.code == 401:
            other = "global" if BASE_URL == HOSTS["cn"] else "cn"
            hint = (f"\n  A 401 here is as likely to be the wrong HOST as a bad key: CN and global "
                    f"keys are not interchangeable. This call went to {BASE_URL}; retry with "
                    f"MINIMAX_REGION={other} ({HOSTS[other]}).")
        raise OracleError(f"{method} {path} -> HTTP {e.code}: {detail}{hint}") from None


def build_content(text: str, first_frame: str | None = None, last_frame: str | None = None,
                  reference_images: list[str] | None = None,
                  reference_videos: list[str] | None = None,
                  reference_audios: list[str] | None = None) -> list[dict]:
    """Assemble the `content` array. One non-empty text item is mandatory, and the keyframe roles
    are mutually exclusive with the reference roles -- the server rejects a mix, so do not build one.
    """
    if not text.strip():
        raise OracleError("content must include one non-empty text item")
    keyframes = [x for x in (first_frame, last_frame) if x]
    refs = (reference_images or []) + (reference_videos or []) + (reference_audios or [])
    if keyframes and refs:
        raise OracleError("first_frame/last_frame and reference_* roles cannot be mixed in one "
                          "request (H3ContextIRReq: i2va and r2va are mutually exclusive)")
    content: list[dict] = [{"type": "text", "text": text.strip()}]
    for url, role in ((first_frame, "first_frame"), (last_frame, "last_frame")):
        if url:
            content.append({"type": "image_url", "image_url": {"url": url}, "role": role})
    for url in reference_images or []:
        content.append({"type": "image_url", "image_url": {"url": url},
                        "role": "reference_image"})
    for url in reference_videos or []:
        content.append({"type": "video_url", "video_url": {"url": url},
                        "role": "reference_video"})
    for url in reference_audios or []:
        content.append({"type": "audio_url", "audio_url": {"url": url},
                        "role": "reference_audio"})
    return content


def infer_task(content: list[dict]) -> str:
    """Which of this package's task names the returned IR should be validated as."""
    roles = {c.get("role") for c in content}
    if roles & {"reference_image", "reference_video", "reference_audio"}:
        return "ref2va"
    if {"first_frame", "last_frame"} <= roles:
        return "fl2va"
    if "last_frame" in roles:
        return "l2va"
    if "first_frame" in roles:
        return "i2va"
    return "t2va"


def context_ir(text: str, duration: int = 5, ratio: str = "16:9", *,
               poll_interval: float = 2.0, timeout_s: float = 300.0,
               validate_output: bool = True, **media) -> OracleResult:
    """One call to the closed H3-Context-IR compiler. Async: create, then poll until terminal."""
    if not 4 <= duration <= 15:
        raise OracleError(f"duration={duration} outside the API's 4-15 s range")
    content = build_content(text, **media)
    task = infer_task(content)
    if task == "t2va" and ratio not in CONCRETE_RATIOS:
        raise OracleError(f"text-only requests require a concrete ratio, one of {CONCRETE_RATIOS}")
    body = {"model": "MiniMax-H3", "content": content, "duration": duration, "ratio": ratio}

    t0 = time.time()
    created = _call("POST", CREATE_PATH, body)
    # The spec types the create response as VideoGenerationV2Resp, but the CN host actually returns a
    # flat `{"task_id": "..."}`. Accept both rather than trusting either.
    task_id = (created.get("task") or {}).get("id") or created.get("task_id") or ""
    if not task_id:
        raise OracleError(f"no task_id in create response: {json.dumps(created)[:400]}")

    while True:
        time.sleep(poll_interval)
        t = _call("GET", QUERY_PATH.format(task_id=task_id)).get("task", {})
        status = t.get("status", "")
        if status in TERMINAL:
            break
        if time.time() - t0 > timeout_s:
            raise OracleError(f"task {task_id} still {status!r} after {timeout_s:.0f}s")

    prompt = (t.get("content") or {}).get("prompt")
    res = OracleResult(
        request=body, prompt=prompt, status=status, task_id=task_id,
        usage=t.get("usage") or {}, seconds=time.time() - t0,
        server_seconds=max(0, (t.get("updated_at") or 0) - (t.get("created_at") or 0)),
        error=t.get("error"),
    )
    if prompt and validate_output:
        # `duration` is an integer here but the renderer's real clip length is frames/24, so the
        # timestamp bound is checked against the API's own number -- the looser of the two.
        res.report = validate(prompt, task, duration_s=float(duration))
    return res
