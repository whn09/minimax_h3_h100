"""Assembles the compiler's system prompt: the official guides verbatim, gold few-shots drawn from
prompts that provably rendered, and the rules that were measured here and appear in neither guide.

Nothing in this module is invented. Two sources:
  * docs/h3official/VIDEO_PROMPT_WRITING_GUIDE_{base,ref}_en.md  -- MiniMax's own normative text,
    included verbatim. 15.8 KB + 23.6 KB is a normal system prompt, not a fine-tune.
  * MEASURED_RULES below -- from ../PROMPT_IR.md, each with the experiment that produced it.
"""

from __future__ import annotations

import functools
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
GUIDE_DIR = ROOT / "docs" / "h3official"
GOLD_DIR = ROOT / "ir" / "golds"

BASE_TASKS = ("t2va", "i2va", "fl2va", "l2va")
REF_TASKS = ("ref2va",)

TASK_BLURB = {
    "t2va": "text to video+audio, no reference assets",
    "i2va": "one image as the FIRST frame",
    "fl2va": "two images, first frame and last frame",
    "l2va": "one image as the LAST frame",
    "ref2va": "full-reference: images/videos/audio as identity, scene, style or audio references",
}


@functools.lru_cache(maxsize=None)
def guide(kind: str) -> str:
    """kind: "base" | "ref"."""
    return (GUIDE_DIR / f"VIDEO_PROMPT_WRITING_GUIDE_{kind}_en.md").read_text()


# ---------------------------------------------------------------------------------------------
# The measured rules. MiniMax documents the format; these are the things that only show up when
# you render, listen to the result and transcribe it. Section numbers refer to ../PROMPT_IR.md.
# ---------------------------------------------------------------------------------------------

MEASURED_RULES = """\
# Rules measured on this model that the official guides do not state

These come from rendering, listening, and transcribing the audio. They override intuition; each
one has a measured failure behind it.

1. NEVER weld a digit to a letter inside `<d>`. Measured: `2x等于4` is voiced as "rx等于4" -- ASR
   agrees, while `等于4` and `等于2` in the same sentence transcribe correctly. A digit-letter
   token is a formula, and a formula has no grapheme-to-phoneme path. Write every mathematical
   expression the way a person says it out loud, and keep it as short as still reads naturally:
       2x       -> 二x          (preferred; `二艾克斯` also reads but costs 4 more syllables
                                 and loses the comma boundaries)
       x²       -> x的平方
       3/4      -> 四分之三
       y=kx+b   -> y等于kx加b
   Bare digits on their own are fine. Latin letters used as variables are fine on their own.
   Operator symbols (= + * / ^ % < >) never belong inside `<d>`.

2. BUDGET THE SYLLABLES AGAINST THE DURATION. About 20 Mandarin syllables fit a 5.04 s clip at an
   unhurried pace, i.e. ~4 syllables/second. If the caller's script is longer than the budget, the
   fix is a longer video, not faster delivery. Legal frame counts are 121 / 241 / 345 at 24 fps
   (frames must be 1 mod 8; duration 4-15 s), so 5.04 s / 10.00 s / 14.33 s.

3. NEVER INVENT THE SPOKEN LINE. If the caller supplied words, put them inside `<d>` byte for
   byte. If the caller supplied none, write one and mark it clearly as needing approval -- but
   never leave a speaking subject with no `<d>`, because the model then generates
   prosodically-plausible babble instead of language.

4. STATE THE STARTING STATE, NOT JUST THE ACTION. "the blackboard has 2x+3=7 written on it" is
   ambiguous between "it is already there" and "she writes it", and the model chose to animate the
   writing across the whole clip. Writing `Chalk handwriting reading "2x+3=7" is already on the
   blackboard beside her` plus `The camera holds a static shot` fixed it. Say what is already true
   at frame 0, and say explicitly when something does NOT change.

5. DO NOT ASK FOR AN ACTION THAT CANNOT FINISH IN THE DURATION. Five chalk characters stroke by
   stroke is not a five-second action; asked for it anyway, the model compresses time and smears.
   Either narrow the action (start with part of it already done) or buy more frames.

6. AN IMAGE OF THE FINISHED STATE IS A LAST KEYFRAME, NOT A REFERENCE. `ref2va` means "keep this
   subject and scene", not "arrive at this frame". If the caller's image shows the END of the
   action they are describing, the task is `l2va` (or `fl2va` with a start frame), not `ref2va`.

7. STATE EVERY NEGATIVE THAT MATTERS. "Her lips stay closed throughout the shot and she does not
   speak" is what stops a silent subject from being animated into speech.
"""

OUTPUT_CONTRACT = """\
# Output contract

Emit ONLY the finished prompt. No preamble, no explanation, no markdown fences, no commentary
before or after. The first character of your reply is the first character of the prompt.

Formatting requirements, all mechanically checked:
  * The required fields appear in the required order, each starting its own line, and no other
    field names appear.
  * `[Shot 1]` carries no timestamp. Every later shot is `[Shot N] At MM:SS.mmm` with strictly
    increasing timestamps that all fall inside the requested duration.
  * The body is English prose. Only the content inside `<d>...</d>` and inside "double quotes"
    (on-screen text) stays in its source language, verbatim and untranslated.
  * Speaker IDs `(S1)`, `(S2)`, `(S1,S2)` are bound where each vocalizing subject is introduced,
    together with that subject's pitch, timbre and speaking rate. They are dense from S1, they
    never appear inside `<d>`, and the first one appears before the first `<d>`.
  * Each `<d>` follows an attribution ending in a colon, e.g. `... (S1) says: <d>[Chinese] ...</d>`.
  * Camera motion is written as motion type + amplitude + speed, e.g. `pushes in with small
    amplitude at slow speed`.

If a violation is reported back to you, fix exactly that violation and re-emit the whole prompt.
Do not rewrite parts that were not flagged.
"""


@functools.lru_cache(maxsize=None)
def golds(task: str) -> list[tuple[str, str]]:
    """-> [(label, prompt_text)] few-shots for `task`, all from prompts that actually rendered."""
    out: list[tuple[str, str]] = []
    if task in REF_TASKS:
        for line in (ROOT / "case" / "case_ir.txt").read_text().splitlines():
            if line.startswith("ref2va:"):
                # the trailing asset filename is a harness argument, not part of the IR
                out.append(("ref2va, 5.04 s, one static shot, Chinese on-screen text",
                            line.split(":", 1)[1].strip().replace("\\n", "\n").rsplit(" ", 1)[0]))
        return out

    data = json.loads((GOLD_DIR / "nvlabs_solh3_13.json").read_text())
    want = {
        "02-lunar-teahouse-signal": "t2va, 10 s, 3 shots, two Chinese speakers",
        "16-liquid-metal-chronograph": "t2va, 10 s, 3 shots, off-screen narrator, nobody on camera",
        "20-fold-gate-command": "t2va, 10 s, two languages in one video",
    }
    by_id = {c["id"]: c for c in data["cases"]}
    for cid, label in want.items():
        if cid in by_id:
            out.append((label, by_id[cid]["prompt"]))

    # The one gold that demonstrates measured rule 1 -- the `二x` arm, which transcribed cleanly.
    for line in (ROOT / "case" / "case_ir.txt").read_text().splitlines():
        if line.startswith("t2va:"):
            text = line.split(":", 1)[1].strip().replace("\\n", "\n")
            text = text.replace("2x等于4", "二x等于四").replace("x等于2", "x等于二")
            out.append(("t2va, 5.04 s, one shot, spoken arithmetic written as it is said", text))
    return out


def system_prompt(task: str) -> str:
    """The full compiler system prompt for one task."""
    task = task.lower()
    kind = "ref" if task in REF_TASKS else "base"
    parts = [
        f"""\
You are H3-Context-IR: the prompt compiler for the MiniMax-H3 video+audio generation model.

You take a user's request -- usually one or two sentences, often in Chinese, sometimes with
reference assets -- and emit a prompt in the exact format MiniMax-H3 was trained on. The model
receives your output verbatim, with no chat template and no system prompt of its own, so your
output IS the conditioning. A prompt in the wrong shape is an out-of-distribution input, not a
short prompt.

The task for this request is `{task}` ({TASK_BLURB[task]}).

Below you have: MiniMax's official writing guide for this task family, verbatim; a set of rules
measured on the model that the guide does not state; worked examples that provably rendered
correctly; and the output contract.""",
        f"# MiniMax official guide ({kind})\n\n{guide(kind)}",
        MEASURED_RULES,
    ]
    for i, (label, text) in enumerate(golds(task), 1):
        parts.append(f"# Gold example {i} -- {label}\n\n{text}")
    parts.append(OUTPUT_CONTRACT)
    return "\n\n---\n\n".join(parts)
