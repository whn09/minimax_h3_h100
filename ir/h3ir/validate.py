"""Mechanical validator for MiniMax-H3 prompt IR.

This is the piece that makes the compiler a compiler instead of a wish. Every rule here is
either normative text from docs/h3official/ or a rule measured in ../PROMPT_IR.md that the
official guides do not state.

Two severities:

  ERROR  structural / measured-defect rules. A prompt with any ERROR is not sent (fail closed).
  WARN   soft budgets (word counts, syllable rate). Reported, never blocking -- the calibration
         corpus in ../golds/ contains prompts that provably rendered well and still trip some
         of them, so making them hard would reject known-good output.

Calibration (see ir/tests/test_validate.py):
  * NVlabs' 13 validated Sol-H3 showcase prompts separate fields with a SINGLE newline, not the
    blank line the base guide's section 2.2 shows. A blank-line rule would reject all 13. So the
    separator check accepts either and only rejects fields run together on one line.
  * `non_diegetic_music: N/A` is legal (the guide says so, and the prompts this repo measured used
    it and rendered correctly) even though none of the 13 use it. ../case/demo_ir.txt keeps that
    shape.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field as _dc_field

# ---------------------------------------------------------------------------- field schemas

BASE_FIELDS = ("integrated_multimodal_description", "overall_soundscape", "non_diegetic_music")

REF_FIELDS = (
    "subject_definitions",
    "summary",
    "retention_analysis",
    "detailed_description",
    "overall_soundscape",
    "non_diegetic_music",
)

BASE_TASKS = ("t2va", "i2va", "fl2va", "l2va")
REF_TASKS = ("ref2va",)
ALL_TASKS = BASE_TASKS + REF_TASKS

ALL_FIELD_NAMES = tuple(sorted(set(BASE_FIELDS) | set(REF_FIELDS)))

# guide base_en.md section 2.1. Reproduced literally, including the guide's own inconsistency:
# FL2VA is written with bare `Picture 1 (from Shot 1)` while I2VA/L2VA use `<Picture 1>` (from
# [Shot 1]). Do not normalize it -- the instruction is what the closed rewriter emits.
INSTRUCTION_PATTERNS = {
    "i2va": re.compile(
        r"^For the target video, at 0\.00 seconds into the target video, "
        r"<Picture 1> \(from \[Shot 1\]\) is fully referenced\.$"
    ),
    "fl2va": re.compile(
        r"^How the reference pictures align with the target video — "
        r"Picture 1 \(from Shot 1\) aligns with the 0\.00-second mark of the target video; "
        r"Picture 2 \(from Shot \d+\) aligns with the \d+\.\d{2}-second mark of the target video\.$"
    ),
    "l2va": re.compile(
        r"^How the reference pictures align with the target video — "
        r"<Picture 1> \(from \[Shot \d+\]\) aligns with the "
        r"\d+\.\d{2}-second mark of the target video\.$"
    ),
}

# guide ref_en.md section 3.2 / 3.3
VISIBLE_MARKERS = ("fully_preserved", "partially_preserved", "attribute_transfer", "weak_reference")
AUDIO_MARKERS = ("fully_copy", "partially_copy", "reference", "weak_reference")
TASK_TYPES = (
    "keyframe completion",
    "reference generation",
    "video editing",
    "video continuation",
    "audio reuse",
    "audio reference",
)

# PROMPT_IR.md section 3.1.1: ~20 unhurried Mandarin syllables fit a 5.04 s clip.
SYLLABLES_PER_SECOND = 4.0

# PROMPT_IR.md section 4.4 / task_profiles.py: frames must be 1 mod 8, duration in [4, 15] s.
LEGAL_FRAME_COUNTS = (121, 241, 345)


@dataclass
class Violation:
    rule: str
    severity: str  # "error" | "warn"
    message: str
    excerpt: str = ""

    def __str__(self) -> str:
        tail = f"  |  {self.excerpt}" if self.excerpt else ""
        return f"[{self.severity.upper():5}] {self.rule}: {self.message}{tail}"


@dataclass
class Report:
    task: str
    violations: list[Violation] = _dc_field(default_factory=list)
    fields: dict[str, str] = _dc_field(default_factory=dict)
    instruction: str = ""

    @property
    def errors(self) -> list[Violation]:
        return [v for v in self.violations if v.severity == "error"]

    @property
    def warnings(self) -> list[Violation]:
        return [v for v in self.violations if v.severity == "warn"]

    @property
    def ok(self) -> bool:
        """Fail closed: only an error-free prompt is sent to the model."""
        return not self.errors

    def render(self) -> str:
        if not self.violations:
            return "clean"
        return "\n".join(str(v) for v in self.violations)


# ---------------------------------------------------------------------------- helpers

_FIELD_RE = re.compile(r"^(" + "|".join(ALL_FIELD_NAMES) + r"):", re.M)
_D_RE = re.compile(r"<d>(.*?)</d>", re.S)
_SHOT_RE = re.compile(r"\[Shot (\d+)\](?:\s+At (\d{2}):(\d{2})\.(\d{3}))?")
_SPEAKER_RE = re.compile(r"\((S\d+(?:\s*,\s*S\d+)*)\)")
_LABEL_RE = re.compile(r"<(Subject|Picture|Video|Audio) (\d+)>")
_QUOTED_RE = re.compile(r'"[^"]*"')

# Scripts that must not appear in the English body. Latin + digits + punctuation only outside
# <d> and outside "..." (base guide 4.5, ref guide 4; every worked example in both guides).
# Codepoint ranges, not unicodedata names -- `unicodedata.name('一')` is "CJK UNIFIED
# IDEOGRAPH-4E00", which no "HAN" prefix test will ever match.
_SCRIPT_RANGES = (
    ("HAN", 0x3400, 0x4DBF), ("HAN", 0x4E00, 0x9FFF), ("HAN", 0xF900, 0xFAFF),
    ("HAN", 0x20000, 0x2FA1F),
    ("CJK-PUNCT", 0x3000, 0x303F), ("CJK-PUNCT", 0xFF01, 0xFF65),
    ("HIRAGANA", 0x3040, 0x309F), ("KATAKANA", 0x30A0, 0x30FF),
    ("HANGUL", 0x1100, 0x11FF), ("HANGUL", 0x3130, 0x318F), ("HANGUL", 0xAC00, 0xD7AF),
    ("ARABIC", 0x0600, 0x06FF), ("ARABIC", 0x0750, 0x077F), ("ARABIC", 0xFB50, 0xFDFF),
    ("ARABIC", 0xFE70, 0xFEFF),
    ("HEBREW", 0x0590, 0x05FF), ("CYRILLIC", 0x0400, 0x04FF), ("GREEK", 0x0370, 0x03FF),
    ("THAI", 0x0E00, 0x0E7F), ("DEVANAGARI", 0x0900, 0x097F),
)

# A `<d>` block is an attributed quotation: the guides write `... (S1) says: <d>` or
# `... (S1,S2) shout together, <d>`, and all 38 dialogue blocks in golds/nvlabs_solh3_13.json
# end their attribution with a colon. A `<d>` dropped into a sentence with no attribution is the
# failure mode this catches.
_ATTRIBUTION_RE = re.compile(r"[:,]\s*$")


def _script_of(ch: str) -> str:
    cp = ord(ch)
    for name, lo, hi in _SCRIPT_RANGES:
        if lo <= cp <= hi:
            return name
    return ""


def split_fields(text: str) -> tuple[str, dict[str, str], list[str]]:
    """-> (instruction_block, {field: value}, field_order).

    The instruction block is whatever precedes the first field name (empty for t2va/ref2va).
    """
    marks = list(_FIELD_RE.finditer(text))
    if not marks:
        return text.strip(), {}, []
    instruction = text[: marks[0].start()].strip()
    fields: dict[str, str] = {}
    order: list[str] = []
    for i, m in enumerate(marks):
        name = m.group(1)
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        order.append(name)
        fields[name] = text[m.end():end].strip()
    return instruction, fields, order


def strip_dialogue_and_quotes(text: str) -> str:
    return _QUOTED_RE.sub(" ", _D_RE.sub(" ", text))


def count_syllables(dialogue: str) -> float:
    """Rough syllable count. Han/Hangul/Kana = 1 each; Latin = vowel groups."""
    han = sum(1 for c in dialogue if _script_of(c) in ("HAN", "HANGUL", "HIRAGANA", "KATAKANA"))
    latin = len(re.findall(r"[aeiouyAEIOUY]+", re.sub(r"[^A-Za-z ]", " ", dialogue)))
    return han + latin


# ---------------------------------------------------------------------------- rule groups

def _check_structure(task: str, instruction: str, fields: dict, order: list, text: str,
                     out: list[Violation]) -> None:
    expected = list(REF_FIELDS if task in REF_TASKS else BASE_FIELDS)

    for name in expected:
        if name not in fields:
            out.append(Violation("E001-MISSING-FIELD", "error", f"required field `{name}:` absent"))
    extra = [n for n in order if n not in expected]
    for name in extra:
        out.append(Violation("E003-UNKNOWN-FIELD", "error",
                             f"`{name}:` is not a field of {task}; expected only {expected}"))
    present = [n for n in order if n in expected]
    if present != [n for n in expected if n in fields]:
        out.append(Violation("E002-FIELD-ORDER", "error",
                             f"fields out of order: {present}, expected {expected}"))

    # Separator: the guide shows a blank line, NVlabs' 13 validated prompts use a single "\n".
    # Both accepted. What is rejected is two fields sharing a line, which the model does emit
    # when it treats the format as prose.
    for m in re.finditer(r"(\S[ \t]*)(" + "|".join(expected) + r"):", text):
        out.append(Violation("E004-FIELD-NOT-AT-LINE-START", "error",
                             f"`{m.group(2)}:` appears mid-line; each field starts its own line",
                             text[max(0, m.start() - 40):m.end() + 20]))

    # Instruction line (base guide 2.1)
    pat = INSTRUCTION_PATTERNS.get(task)
    if pat is not None:
        first = instruction.splitlines()[0].strip() if instruction else ""
        if not pat.match(first):
            out.append(Violation("E005-INSTRUCTION-LINE", "error",
                                 f"{task} requires its fixed alignment instruction as the first "
                                 f"line, verbatim per base guide 2.1",
                                 first[:160] or "(no instruction line)"))
    elif instruction:
        out.append(Violation("E006-UNEXPECTED-INSTRUCTION", "error",
                             f"{task} has no alignment instruction; the prompt must begin with "
                             f"`{expected[0]}:`", instruction[:120]))


def _check_shots(body: str, duration_s: float | None, out: list[Violation]) -> None:
    shots = list(_SHOT_RE.finditer(body))
    if not shots:
        out.append(Violation("E010-NO-SHOT-MARKER", "error",
                             "the description must open with `[Shot 1]` (base guide 4.2)"))
        return

    numbers = [int(m.group(1)) for m in shots]
    if numbers[0] != 1:
        out.append(Violation("E011-SHOT-NUMBERING", "error",
                             f"first shot marker is [Shot {numbers[0]}], must be [Shot 1]"))
    # Markers may repeat (retention_analysis references shots), so check the set is dense.
    uniq = sorted(set(numbers))
    if uniq != list(range(1, len(uniq) + 1)):
        out.append(Violation("E011-SHOT-NUMBERING", "error",
                             f"shot indices {uniq} are not dense from 1"))

    seen_ts: list[tuple[int, float]] = []
    for m in shots:
        n = int(m.group(1))
        has_ts = m.group(2) is not None
        if n == 1 and has_ts:
            out.append(Violation("E012-SHOT1-TIMESTAMPED", "error",
                                 "[Shot 1] carries no timestamp; it starts at 0", m.group(0)))
        if n > 1 and not has_ts:
            out.append(Violation("E013-SHOT-NO-TIMESTAMP", "error",
                                 f"[Shot {n}] needs `At MM:SS.mmm` (base guide 4.2)", m.group(0)))
        if has_ts:
            t = int(m.group(2)) * 60 + int(m.group(3)) + int(m.group(4)) / 1000.0
            seen_ts.append((n, t))

    for (n1, t1), (n2, t2) in zip(seen_ts, seen_ts[1:]):
        if n2 > n1 and t2 <= t1:
            out.append(Violation("E014-TIMESTAMP-NOT-INCREASING", "error",
                                 f"[Shot {n2}] at {t2:.3f}s does not follow [Shot {n1}] at "
                                 f"{t1:.3f}s"))
    if duration_s is not None:
        for n, t in seen_ts:
            if t >= duration_s:
                out.append(Violation("E015-TIMESTAMP-PAST-DURATION", "error",
                                     f"[Shot {n}] starts at {t:.3f}s, outside a "
                                     f"{duration_s:.2f}s video"))


def _check_dialogue(body: str, duration_s: float | None, out: list[Violation]) -> None:
    if body.count("<d>") != body.count("</d>"):
        out.append(Violation("E020-D-UNBALANCED", "error",
                             f"{body.count('<d>')} `<d>` vs {body.count('</d>')} `</d>`"))

    dialogues = list(_D_RE.finditer(body))

    # Speaker IDs are bound where the speaker is INTRODUCED, together with the voice description
    # ("...with a warm gravelly Mandarin voice (S1), pours tea"), and later lines are attributed by
    # name. All 13 validated prompts in golds/ work that way, so requiring an `(SN)` adjacent to
    # every `<d>` -- which is how the guide's short examples read, and what PROMPT_IR.md sec. 4.1
    # assumed -- would reject every one of them. What is enforceable is: the IDs exist, they are
    # dense from S1, they are declared outside `<d>`, and the first one precedes the first `<d>`.
    ids_outside = _SPEAKER_RE.findall(_D_RE.sub(" ", body))
    if dialogues:
        if not ids_outside:
            out.append(Violation("E022-NO-SPEAKER-ID", "error",
                                 "the description has dialogue but declares no `(SN)` speaker ID; "
                                 "bind an ID to each vocalizing subject where it is introduced, "
                                 "with its pitch/timbre/rate (base guide 4.4)"))
        first_id = _SPEAKER_RE.search(body)
        if first_id and first_id.start() > dialogues[0].start():
            out.append(Violation("E025-SPEAKER-ID-AFTER-FIRST-LINE", "error",
                                 "the first `(SN)` is declared after the first `<d>`; a speaker "
                                 "must be established before it speaks"))
    if _SPEAKER_RE.search(" ".join(m.group(1) for m in dialogues)):
        out.append(Violation("E026-SPEAKER-ID-INSIDE-D", "error",
                             "`(SN)` appears inside `<d>`; only the language tag and the verbatim "
                             "spoken content go inside (base guide 4.4)"))

    total_syll = 0.0
    speakers: list[str] = [s for group in ids_outside for s in re.findall(r"S\d+", group)]
    for m in dialogues:
        inner = m.group(1)
        before = body[:m.start()]

        if not re.match(r"\s*\[[A-Z][A-Za-z ]*\]", inner):
            out.append(Violation("E021-D-NO-LANGUAGE-TAG", "error",
                                 "`<d>` must open with a `[Language]` tag (base guide 4.4)",
                                 inner[:60]))
        line = re.sub(r"^\s*\[[A-Za-z ]*\]\s*", "", inner)
        # `<scenetrans>` and `<cutoff>` are legal, and sometimes mandatory, INSIDE `<d>`: a line that
        # crosses a cut carries the marker at both connecting points. Their angle brackets are not
        # math. Strip them before the operator scan -- without this, E031 fires four times on a
        # correct two-shot voiceover, which is how MiniMax's own API output exposed the bug
        # (logs/harvest_cn.jsonl case 4).
        spoken = re.sub(r"<(?:scenetrans|cutoff)>", "", line)

        if not _ATTRIBUTION_RE.search(before):
            out.append(Violation("E027-D-UNATTRIBUTED", "error",
                                 "`<d>` must follow an attribution ending in `:` or `,` "
                                 "(`... (S1) says: <d>`)", before[-60:]))

        # Measured in PROMPT_IR.md 3.1.1 and in NEITHER guide -- and, as of 2026-09-18, not
        # implemented by MiniMax's own H3-Context-IR either: asked to have a subject explain an
        # equation out loud with no script supplied, the official API invented a digit-letter token
        # inside `<d>` unprompted. `2x` renders as "rx". This rule is the clearest thing this package
        # adds over buying the API. The harvest record of that probe is not in the repo (it used a
        # customer's shot) -- golds/minimax_official_harvest_cn.json's `note` says how to re-harvest it.
        for bad in re.finditer(r"\d[A-Za-z]|[A-Za-z]\d", spoken):
            out.append(Violation("E030-D-MIXED-SCRIPT-TOKEN", "error",
                                 f"`{bad.group(0)}` welds a digit to a letter: no grapheme-to-"
                                 f"phoneme path, voiced as noise. Write it as it is spoken "
                                 f"(`2x` -> `二x`).", spoken[max(0, bad.start() - 12):bad.end() + 12]))
        for bad in re.finditer(r"[=+*/^<>%²³½¾]|(?<=\d)-(?=\d)", spoken):
            out.append(Violation("E031-D-MATH-OPERATOR", "error",
                                 f"`{bad.group(0)}` is an operator, not a word; spell the "
                                 f"expression out loud",
                                 spoken[max(0, bad.start() - 12):bad.end() + 12]))

        # base guide 4.4 / PROMPT_IR.md 4.4: an off-screen voiceover must say the lips stay shut,
        # or the model animates a talking mouth on a subject who is not on camera.
        if re.search(r"off-screen voiceover|voice-over|voiceover", before[-160:], re.I):
            # The real obligation is "assert somewhere adjacent that no visible mouth is speaking".
            # Do NOT pin it to one sentence: the guide gives only the "lips remain closed" form,
            # golds case 16-liquid-metal-chronograph renders fine with "no speaking face appears",
            # and MiniMax's own API writes a third ("with her unseen mouth remaining completely
            # still throughout" -- logs/harvest_cn.jsonl case 4). Three independent phrasings for one
            # requirement is the signal that the requirement, not the wording, is what to check.
            window = before[-400:] + inner + body[m.end():m.end() + 300]
            if not re.search(
                    r"lips (remain|stay|are kept|remaining|staying)[a-z, ]*(closed|shut|still)"
                    r"|no (speaking|visible|talking) (face|mouth|lips)"
                    r"|(unseen|unshown|off-?screen|no visible) (mouth|face|lips)[a-z, ]*"
                    r"(remain|remains|remaining|stay|stays|staying|is|are)?[a-z, ]*"
                    r"(still|closed|shut|unmoving|motionless)"
                    r"|(mouth|face|lips) (remain|remains|remaining|stay|stays|staying|is|are)"
                    r"[a-z, ]*(still|closed|shut|unmoving|motionless)", window, re.I):
                out.append(Violation("E024-VOICEOVER-NO-LIPS-CLAUSE", "error",
                                     "a voiceover `<d>` must assert nearby that no visible mouth is "
                                     "speaking -- \"lips remain closed\", \"no speaking face appears "
                                     "on screen\", or \"her unseen mouth remains still\"", inner[:60]))

        total_syll += count_syllables(line)

    # base guide 4.4: a line crossing a cut needs `<scenetrans>` at BOTH connecting points.
    n_trans = body.count("<scenetrans>")
    if n_trans == 1:
        out.append(Violation("E028-SCENETRANS-UNPAIRED", "error",
                             "`<scenetrans>` appears once; it marks both connecting points of the "
                             "cut a line crosses, so it comes in pairs"))
    if body.count("<cutoff>") and duration_s is not None:
        tail = body[body.index("<cutoff>"):]
        if _SHOT_RE.search(tail):
            out.append(Violation("E029-CUTOFF-NOT-AT-END", "error",
                                 "`<cutoff>` marks speech truncated by the END of the video, but "
                                 "another shot follows it"))

    if speakers:
        idx = sorted({int(s[1:]) for s in speakers})
        if idx != list(range(1, len(idx) + 1)):
            out.append(Violation("E023-SPEAKER-IDS-NOT-DENSE", "error",
                                 f"speaker IDs {idx} are not dense from S1"))

    if duration_s is not None and total_syll > SYLLABLES_PER_SECOND * duration_s:
        out.append(Violation("W032-SYLLABLE-BUDGET", "warn",
                             f"{total_syll:.0f} syllables of dialogue in {duration_s:.2f}s is "
                             f"{total_syll / duration_s:.1f}/s; ~{SYLLABLES_PER_SECOND:.0f}/s is "
                             f"unhurried. Raise the duration, do not talk faster "
                             f"(PROMPT_IR.md 3.1.1)"))


def _check_body_language(fields: dict, out: list[Violation]) -> None:
    """The body is English; only dialogue and on-screen text keep the source script."""
    for name, value in fields.items():
        stripped = strip_dialogue_and_quotes(value)
        bad = {}
        for ch in stripped:
            s = _script_of(ch)
            if s:
                bad.setdefault(s, ch)
        if bad:
            out.append(Violation("E040-NON-ENGLISH-BODY", "error",
                                 f"`{name}` contains {sorted(bad)} outside `<d>` and outside "
                                 f'double quotes; the body is English prose with foreign content '
                                 f"quoted", "".join(bad.values())))


def _check_reference_sections(fields: dict, out: list[Violation]) -> None:
    defs = fields.get("subject_definitions", "")
    retention = fields.get("retention_analysis", "")
    summary = fields.get("summary", "")

    defined: set[str] = set()
    for line in defs.splitlines():
        m = _LABEL_RE.match(line.strip())
        if m:
            defined.add(m.group(0))
    if not defined:
        out.append(Violation("E050-NO-LABELS-DEFINED", "error",
                             "`subject_definitions` defines no `<Subject N>` / `<Picture N>` / "
                             "`<Video N>` / `<Audio N>` label (ref guide 2)"))

    used = {m.group(0) for name, v in fields.items() if name != "subject_definitions"
            for m in _LABEL_RE.finditer(v)}
    for label in sorted(used - defined):
        out.append(Violation("E051-LABEL-UNDEFINED", "error",
                             f"{label} is referenced but never defined in `subject_definitions`"))

    for label in sorted(defined):
        lines = [ln for ln in retention.splitlines() if ln.strip().startswith(label)]
        if len(lines) != 1:
            out.append(Violation("E052-RETENTION-CARDINALITY", "error",
                                 f"{label} has {len(lines)} lines in `retention_analysis`; "
                                 f"exactly one is required"))
            continue
        legal = AUDIO_MARKERS if label.startswith("<Audio") else VISIBLE_MARKERS
        if not any(re.search(rf"\b{m}\b", lines[0]) for m in legal):
            out.append(Violation("E053-BAD-MARKER", "error",
                                 f"{label}'s retention line carries no legal marker; one of "
                                 f"{list(legal)}", lines[0][:120]))

    # ref guide 4.3: (Sx) speaker IDs must not appear in retention_analysis.
    if _SPEAKER_RE.search(retention):
        out.append(Violation("E054-SPEAKER-ID-IN-RETENTION", "error",
                             "`retention_analysis` must not carry `(SN)` speaker IDs; they "
                             "belong in `detailed_description`",
                             _SPEAKER_RE.search(retention).group(0)))

    m = re.match(r"\[([^\]]+)\]", summary.strip())
    if not m:
        out.append(Violation("E055-SUMMARY-NO-TASK-TYPE", "error",
                             "`summary` must open with a bracketed task type, e.g. "
                             "`[reference generation]`", summary[:80]))
    else:
        for t in m.group(1).split(" + "):
            if t not in TASK_TYPES:
                out.append(Violation("E056-BAD-TASK-TYPE", "error",
                                     f"`{t}` is not one of the six task types {list(TASK_TYPES)}"))

    dd = fields.get("detailed_description", "")
    words = len(re.findall(r"[A-Za-z][A-Za-z'-]*", strip_dialogue_and_quotes(dd)))
    if dd and not 350 <= words <= 500:
        out.append(Violation("W060-DETAILED-DESCRIPTION-LENGTH", "warn",
                             f"`detailed_description` is {words} English words; the guide asks "
                             f"for 350-500"))


def _check_soft_lengths(fields: dict, out: list[Violation]) -> None:
    def sentences(s: str) -> int:
        return len([x for x in re.split(r"(?<=[.!?])\s+", s.strip()) if x])

    sound = fields.get("overall_soundscape", "")
    if sound and sentences(sound) > 4:
        out.append(Violation("W061-SOUNDSCAPE-LENGTH", "warn",
                             f"`overall_soundscape` is {sentences(sound)} sentences; 1-4 is the "
                             f"shape the guides use"))
    music = fields.get("non_diegetic_music", "")
    if music and music.strip() != "N/A" and sentences(music) > 3:
        out.append(Violation("W062-MUSIC-LENGTH", "warn",
                             f"`non_diegetic_music` is {sentences(music)} sentences; 1-3, or "
                             f"`N/A`"))


# ---------------------------------------------------------------------------- entry point

def validate(text: str, task: str, duration_s: float | None = None) -> Report:
    """Validate one IR prompt. `duration_s` enables the timing and syllable-budget rules."""
    task = task.lower()
    if task not in ALL_TASKS:
        raise ValueError(f"unknown task {task!r}; one of {ALL_TASKS}")

    out: list[Violation] = []
    instruction, fields, order = split_fields(text)
    _check_structure(task, instruction, fields, order, text, out)

    body_field = "detailed_description" if task in REF_TASKS else BASE_FIELDS[0]
    body = fields.get(body_field, "")
    if body:
        _check_shots(body, duration_s, out)
        _check_dialogue(body, duration_s, out)
    _check_body_language(fields, out)
    if task in REF_TASKS:
        _check_reference_sections(fields, out)
    _check_soft_lengths(fields, out)

    return Report(task=task, violations=out, fields=fields, instruction=instruction)


def duration_for_frames(frames: int) -> float:
    """Clip duration of `frames` at 24 fps. `frames / 24`, not `(frames - 1) / 24`: the latter is
    the timestamp of the last frame, and the repo's convention throughout (../PROMPT_IR.md) is that
    121 frames is a 5.04 s clip."""
    return frames / 24.0
