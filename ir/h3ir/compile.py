"""The compiler: Bedrock `converse` -> mechanical validator -> repair loop -> fail closed.

The loop is the whole point. An LLM asked for a rigid format gets it mostly right and then
silently breaks one rule; the validator names the rule, the model fixes exactly that, and a prompt
that still does not validate after `max_repairs` is NOT returned. Three seconds of retry against a
105-180 s render is a trade with no downside.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field as _dc_field

from . import guides
from .validate import Report, Violation, duration_for_frames, validate

# Bedrock rejects the bare model ids for these ("on-demand throughput isn't supported"); the
# us.-prefixed cross-region inference profiles are what works from us-east-1/us-west-2.
MODELS = {
    "opus5": "us.anthropic.claude-opus-5",
    "gpt6": "us.openai.gpt-6-astra",
    "sonnet5": "us.anthropic.claude-sonnet-5",
    "sonnet46": "us.anthropic.claude-sonnet-4-6",
    "haiku45": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "fable51": "us.anthropic.claude-fable-5-1",
}

# `us.anthropic.claude-opus-5` rejects `temperature` outright ("deprecated for this model"), so it
# is omitted for every model rather than special-cased -- format compliance does not want sampling
# entropy anyway.
MAX_TOKENS = 4096

# MiniMax-H3's SGLang task profiles want frames 1 mod 8 with duration in [4, 15] s at 24 fps, so
# these three are the real choices (../PROMPT_IR.md sec. 4.4). NVlabs' diffusers path used 243
# frames for its 10 s showcase clips, which is not 1 mod 8 -- a different runtime, not a conflict.
LEGAL_FRAMES = {121: 5.04, 241: 10.04, 345: 14.38}


@dataclass
class Attempt:
    text: str
    report: Report
    seconds: float
    stop_reason: str = ""
    usage: dict = _dc_field(default_factory=dict)


@dataclass
class CompileResult:
    task: str
    duration_s: float
    model: str
    attempts: list[Attempt] = _dc_field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.attempts) and self.attempts[-1].report.ok

    @property
    def prompt(self) -> str | None:
        """The compiled prompt, or None. Fail closed: an invalid prompt is never returned."""
        return self.attempts[-1].text if self.ok else None

    @property
    def repairs(self) -> int:
        return max(0, len(self.attempts) - 1)

    def summary(self) -> str:
        head = (f"{self.task} / {self.duration_s:.2f}s / {self.model}: "
                f"{'CLEAN' if self.ok else 'REJECTED'} after {len(self.attempts)} call(s), "
                f"{sum(a.seconds for a in self.attempts):.1f}s")
        rows = []
        for i, a in enumerate(self.attempts, 1):
            rules = ", ".join(v.rule for v in a.report.errors) or "clean"
            warns = ", ".join(v.rule for v in a.report.warnings)
            rows.append(f"  call {i}: {rules}" + (f"  (warn: {warns})" if warns else ""))
        return "\n".join([head, *rows])

    def to_json(self) -> dict:
        return {
            "task": self.task,
            "duration_seconds": self.duration_s,
            "model": self.model,
            "ok": self.ok,
            "calls": len(self.attempts),
            "repairs": self.repairs,
            "seconds": round(sum(a.seconds for a in self.attempts), 3),
            "prompt": self.prompt,
            "attempts": [
                {
                    "seconds": round(a.seconds, 3),
                    "stop_reason": a.stop_reason,
                    "usage": a.usage,
                    "errors": [{"rule": v.rule, "message": v.message, "excerpt": v.excerpt}
                               for v in a.report.errors],
                    "warnings": [{"rule": v.rule, "message": v.message}
                                 for v in a.report.warnings],
                    "text": a.text,
                }
                for a in self.attempts
            ],
        }


def bedrock_client(read_timeout: int = 600):
    """botocore's default read timeout is 60 s with 3 attempts, which is wrong for this workload:
    a reasoning model composing a 2000-character prompt took 176 s here, so the default turns one
    slow call into two silent retries and then a ReadTimeoutError. One long attempt instead."""
    import boto3
    from botocore.config import Config
    return boto3.client("bedrock-runtime", config=Config(
        read_timeout=read_timeout, connect_timeout=20,
        retries={"max_attempts": 1, "mode": "standard"}))


def _strip_fences(text: str) -> str:
    """Models sometimes wrap the answer despite the contract. Unwrap rather than fail on it."""
    text = text.strip()
    m = re.match(r"^```[a-zA-Z]*\n(.*?)\n?```$", text, re.S)
    return m.group(1).strip() if m else text


def _user_message(request: str, task: str, duration_s: float, frames: int,
                  assets: list[str], script: str | None) -> str:
    lines = [
        "# Request",
        "",
        request.strip(),
        "",
        "# Parameters",
        f"task: {task}",
        f"duration_seconds: {duration_s:.2f}",
        f"num_frames: {frames} at 24 fps",
    ]
    if assets:
        lines.append("reference assets, in label order: " + ", ".join(assets))
    if script:
        lines += [
            "",
            "# Spoken content, supplied by the caller -- use it VERBATIM inside `<d>` and do not "
            "rewrite, translate, shorten or paraphrase it, except that any digit-letter token or "
            "operator must be respelled the way it is said out loud (measured rule 1):",
            script.strip(),
        ]
    else:
        lines += [
            "",
            "# The caller supplied no spoken content.",
            "If the request implies someone speaks, write the line yourself, keep it inside the "
            "syllable budget for this duration, and put it in `<d>`. Never leave a speaking "
            "subject without a `<d>` block.",
        ]
    lines += ["", "Emit the prompt now, and nothing else."]
    return "\n".join(lines)


def _repair_message(report: Report) -> str:
    bullets = "\n".join(
        f"  - {v.rule}: {v.message}" + (f"\n      at: {v.excerpt}" if v.excerpt else "")
        for v in report.errors)
    return (
        "The prompt you emitted fails the format validator on the following rules. These are "
        "mechanical checks against MiniMax's format and against rules measured on the model, not "
        "opinions.\n\n"
        f"{bullets}\n\n"
        "Fix exactly these violations and re-emit the complete prompt. Change nothing else. "
        "Output only the prompt."
    )


def compile_prompt(
    request: str,
    task: str = "t2va",
    frames: int = 121,
    model: str = "opus5",
    assets: list[str] | None = None,
    script: str | None = None,
    max_repairs: int = 3,
    cache_system: bool = True,
    client=None,
) -> CompileResult:
    """Compile one request into validated MiniMax-H3 IR.

    `frames` must be a legal MiniMax-H3 frame count (1 mod 8, 4-15 s at 24 fps).
    `script` is the caller's verbatim spoken line, if they have one -- see PROMPT_IR.md sec. 4.2.
    """
    if frames not in LEGAL_FRAMES:
        raise ValueError(f"frames={frames} is not one of {sorted(LEGAL_FRAMES)} "
                         f"(must be 1 mod 8, duration 4-15 s at 24 fps)")
    duration_s = duration_for_frames(frames)
    model_id = MODELS.get(model, model)

    if client is None:
        client = bedrock_client()

    # The system prompt is ~29 KB (~8k tokens) of guides + golds and is byte-identical for every
    # request of a given task, so it is the textbook cache target: a cachePoint after it makes the
    # repair call and every later request reuse it. Note this is a COST lever much more than a
    # latency one -- the compile is output-bound (~900 tokens out), not input-bound.
    system = [{"text": guides.system_prompt(task)}]
    if cache_system:
        system.append({"cachePoint": {"type": "default"}})
    messages = [{"role": "user", "content": [
        {"text": _user_message(request, task, duration_s, frames, assets or [], script)}]}]

    result = CompileResult(task=task, duration_s=duration_s, model=model_id)

    for _ in range(max_repairs + 1):
        t0 = time.time()
        resp = client.converse(
            modelId=model_id,
            system=system,
            messages=messages,
            inferenceConfig={"maxTokens": MAX_TOKENS},
        )
        elapsed = time.time() - t0
        text = _strip_fences("".join(
            b.get("text", "") for b in resp["output"]["message"]["content"]))
        report = validate(text, task, duration_s=duration_s)
        result.attempts.append(Attempt(text=text, report=report, seconds=elapsed,
                                       stop_reason=resp.get("stopReason", ""),
                                       usage=resp.get("usage", {})))
        if report.ok:
            break
        messages.append({"role": "assistant", "content": [{"text": text}]})
        messages.append({"role": "user", "content": [{"text": _repair_message(report)}]})

    return result


def route_task(has_first_frame: bool, has_last_frame: bool, has_reference: bool) -> str:
    """PROMPT_IR.md sec. 3.3 / 4.5 step 3. The distinction that matters: an image of the FINISHED
    state of the requested action is a last keyframe (`l2va`/`fl2va`), not a reference (`ref2va`).
    `ref2va` means "keep this subject and scene", not "arrive at this frame"."""
    if has_first_frame and has_last_frame:
        return "fl2va"
    if has_last_frame:
        return "l2va"
    if has_first_frame:
        return "i2va"
    if has_reference:
        return "ref2va"
    return "t2va"
