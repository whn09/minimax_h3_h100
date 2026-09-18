"""Calibration tests. Run: ir/../../../tmp/irvenv/bin/python -m pytest ir/tests -q

The point of these tests is not that the validator is self-consistent -- it is that the validator
accepts prompts that *provably rendered well* and rejects the specific defects that were measured.
Two corpora:

  golds/nvlabs_solh3_13.json     NVlabs' 13 validated Sol-H3 showcase prompts (t2va, 10 s, 243 f).
  golds/minimax_official_api.json MiniMax's own documented POST /v2/h3_context_ir response -- the
                                  only sample of genuine H3-Context-IR output in existence publicly,
                                  and therefore the single most load-bearing test in this file.
  ../case/demo_ir.txt            a synthetic pair in the same format as this repo's render grid
                                 (PROMPT_IR.md sec. 2). Synthetic because the measured prompts were a
                                 customer's and are not in this repo; the rules they taught us are.

demo_ir.txt's t2va arm is deliberately expected to FAIL, on exactly one rule: it welds a digit to a
letter inside `<d>`, which is the defect PROMPT_IR.md sec. 3.1.1 measured (such a token is voiced as
noise). It stands in for a known-bad-audio render, so a validator that passed it would be useless --
and `h3ir.guides.golds()` feeds the model the repaired form of this same line as the t2va few-shot.
"""

from __future__ import annotations

import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ir"))

from h3ir.validate import Report, duration_for_frames, validate  # noqa: E402

GOLDS = ROOT / "ir" / "golds"


def _nvlabs() -> list[dict]:
    return json.loads((GOLDS / "nvlabs_solh3_13.json").read_text())["cases"]


NVLABS_DURATION = json.loads(
    (GOLDS / "nvlabs_solh3_13.json").read_text())["protocol"]["duration_seconds"]


def _case_ir() -> dict[str, str]:
    out = {}
    for line in (ROOT / "case" / "demo_ir.txt").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        task, body = line.split(":", 1)
        out[task.strip()] = body.strip().replace("\\n", "\n")
    return out


# ---------------------------------------------------------------- the validated corpus passes

@pytest.mark.parametrize("case", _nvlabs(), ids=lambda c: c.get("id", "?"))
def test_nvlabs_validated_prompts_pass(case):
    """All 13 rendered NVlabs showcase clips. Any error here is a validator bug, not a prompt bug."""
    rep = validate(case["prompt"], "t2va", duration_s=NVLABS_DURATION)
    assert rep.ok, f"{case['id']}:\n{rep.render()}"


def test_minimax_official_api_output_passes():
    """The oracle test. This is real H3-Context-IR output, produced by the closed system itself for
    the request in `["request"]`. A rule that fires here is a wrong rule, full stop: no amount of
    reading the writing guide outweighs one sample of what the actual IR compiler emits."""
    gold = json.loads((GOLDS / "minimax_official_api.json").read_text())
    rep = validate(gold["prompt"], gold["task"], duration_s=gold["duration_seconds"])
    assert rep.ok, rep.render()


def _harvest() -> list[dict]:
    return json.loads((GOLDS / "minimax_official_harvest_cn.json").read_text())["cases"]


@pytest.mark.parametrize("case", _harvest(), ids=lambda c: c["id"])
def test_live_official_ir_matches_expectations(case):
    """Live output from the closed compiler, harvested 2026-09-18 for $0.05. Ground truth, with any
    deliberate exception recorded per case in `expect_errors`.

    These found real bugs in this validator: `E031` was counting the angle brackets of a legal
    in-`<d>` `<scenetrans>` marker as math operators, and `E024` knew only two of the three phrasings
    the ecosystem uses for "no visible mouth is speaking". Both are fixed; this test is what keeps
    them fixed.

    TWO OF THE FOUR HARVESTED CASES ARE NO LONGER HERE, including the only one that expected an
    error, so the strongest evidence for `E030` is now cited rather than executed -- see the `note`
    field in the fixture for what it showed and how to re-harvest an equivalent. The probe used a
    customer's shot, so the record went with the rest of that material. The rule itself is unaffected:
    it was measured from a render, not from this fixture.
    """
    rep = validate(case["prompt"], case["task"], duration_s=case["duration_seconds"])
    assert [v.rule for v in rep.errors] == case["expect_errors"], f"{case['id']}:\n{rep.render()}"


def test_scenetrans_inside_dialogue_is_not_a_math_operator():
    """The E031 regression, minimised. `<scenetrans>` is mandatory at both connecting points of a
    line that crosses a cut, so its brackets must never be read as operators."""
    text = BASE_OK.replace("<d>[English] We are out of time.</d>",
                           "<d>[English] We are out of<scenetrans></d>")
    text = text.replace("\noverall_soundscape:",
                        "\n[Shot 3] At 00:04.000, he (S1) continues: "
                        "<d>[English] <scenetrans>time.</d>\noverall_soundscape:")
    assert "E031-D-MATH-OPERATOR" not in err_rules(text)


def test_case_ir_ref2va_passes():
    ref = _case_ir()["ref2va"].rsplit(" ", 1)[0]  # trailing asset filename is not part of the IR
    rep = validate(ref, "ref2va", duration_s=duration_for_frames(121))
    assert rep.ok, rep.render()


# ---------------------------------------------------------------- the measured defects are caught

def test_case_ir_t2va_is_rejected_for_the_measured_mixed_script_defect():
    rep = validate(_case_ir()["t2va"], "t2va", duration_s=duration_for_frames(121))
    rules = {v.rule for v in rep.errors}
    assert rules == {"E030-D-MIXED-SCRIPT-TOKEN"}, rep.render()


def test_the_cn_arm_that_rendered_cleanly_passes():
    """PROMPT_IR.md sec. 3.1.1, the `cn` arm: respelling the digit-letter token is the ONE change,
    and it is enough -- everything else about the prompt is held byte-identical. This is the same
    repair `h3ir.guides.golds()` applies before using the line as a few-shot; if the two ever drift
    the compiler starts teaching the model a prompt the validator rejects."""
    fixed = _case_ir()["t2va"].replace("2L", "两升")
    rep = validate(fixed, "t2va", duration_s=duration_for_frames(121))
    assert rep.ok, rep.render()


# ---------------------------------------------------------------- structural rules, one by one

BASE_OK = (
    "integrated_multimodal_description: [Shot 1] Live-action, a static medium shot frames a man.\n"
    "[Shot 2] At 00:02.500, the camera cuts to a wide shot and pushes in with small amplitude at "
    "slow speed as the man (S1) says: <d>[English] We are out of time.</d>\n"
    "overall_soundscape: Street room tone throughout.\n"
    "non_diegetic_music: N/A"
)


def err_rules(text: str, task: str = "t2va", duration_s: float | None = 10.0) -> set[str]:
    return {v.rule for v in validate(text, task, duration_s).errors}


def test_baseline_fixture_is_clean():
    assert err_rules(BASE_OK) == set()


def test_missing_field():
    text = BASE_OK.replace("non_diegetic_music: N/A", "")
    assert "E001-MISSING-FIELD" in err_rules(text)


def test_field_order():
    lines = BASE_OK.split("\n")
    swapped = "\n".join(lines[:2] + [lines[3], lines[2]])
    assert "E002-FIELD-ORDER" in err_rules(swapped)


def test_fields_run_together_on_one_line():
    text = BASE_OK.replace("\noverall_soundscape:", " overall_soundscape:")
    assert "E004-FIELD-NOT-AT-LINE-START" in err_rules(text)


def test_blank_line_separator_is_also_accepted():
    assert err_rules(BASE_OK.replace("\noverall", "\n\noverall")) == set()


def test_i2va_requires_its_instruction_line_verbatim():
    assert "E005-INSTRUCTION-LINE" in err_rules(BASE_OK, task="i2va")
    good = ("For the target video, at 0.00 seconds into the target video, <Picture 1> "
            "(from [Shot 1]) is fully referenced.\n\n" + BASE_OK)
    assert err_rules(good, task="i2va") == set()


def test_fl2va_instruction_line_keeps_the_guides_bare_labels():
    good = ("How the reference pictures align with the target video — Picture 1 (from Shot 1) "
            "aligns with the 0.00-second mark of the target video; Picture 2 (from Shot 2) "
            "aligns with the 8.00-second mark of the target video.\n\n" + BASE_OK)
    assert err_rules(good, task="fl2va") == set()
    # angle brackets are the L2VA/I2VA spelling; FL2VA's line is written without them
    assert "E005-INSTRUCTION-LINE" in err_rules(
        good.replace("Picture 1 (from Shot 1)", "<Picture 1> (from [Shot 1])"), task="fl2va")


def test_t2va_must_not_carry_an_instruction_line():
    assert "E006-UNEXPECTED-INSTRUCTION" in err_rules("A leading sentence.\n\n" + BASE_OK)


def test_shot1_may_not_be_timestamped():
    assert "E012-SHOT1-TIMESTAMPED" in err_rules(BASE_OK.replace("[Shot 1]", "[Shot 1] At 00:00.000"))


def test_later_shot_needs_a_timestamp():
    assert "E013-SHOT-NO-TIMESTAMP" in err_rules(BASE_OK.replace("[Shot 2] At 00:02.500", "[Shot 2]"))


def test_timestamps_must_increase_and_fit_the_duration():
    three = BASE_OK.replace("out of time.</d>",
                            "out of time.</d>\n[Shot 3] At 00:01.000, a door closes.")
    assert "E014-TIMESTAMP-NOT-INCREASING" in err_rules(three)
    assert "E015-TIMESTAMP-PAST-DURATION" in err_rules(
        BASE_OK.replace("00:02.500", "00:12.500"), duration_s=5.04)


def test_shot_numbering_must_be_dense():
    assert "E011-SHOT-NUMBERING" in err_rules(BASE_OK.replace("[Shot 2]", "[Shot 4]"))


def test_dialogue_tag_rules():
    assert "E020-D-UNBALANCED" in err_rules(BASE_OK.replace("</d>", ""))
    assert "E021-D-NO-LANGUAGE-TAG" in err_rules(BASE_OK.replace("[English] ", ""))
    assert "E022-NO-SPEAKER-ID" in err_rules(BASE_OK.replace("(S1) ", ""))
    assert "E027-D-UNATTRIBUTED" in err_rules(BASE_OK.replace("says: <d>", "says <d>"))
    assert "E026-SPEAKER-ID-INSIDE-D" in err_rules(
        BASE_OK.replace("[English] We are", "[English] (S1) We are"))
    assert "E023-SPEAKER-IDS-NOT-DENSE" in err_rules(BASE_OK.replace("(S1)", "(S2)"))


def test_voiceover_needs_the_lips_clause():
    vo = BASE_OK.replace("(S1) says:", "(S1) says in an off-screen voiceover:")
    assert "E024-VOICEOVER-NO-LIPS-CLAUSE" in err_rules(vo)
    assert err_rules(vo.replace("</d>", "</d> Her lips remain closed throughout.")) == set()


def test_math_operators_inside_dialogue():
    assert "E031-D-MATH-OPERATOR" in err_rules(
        BASE_OK.replace("We are out of time.", "The answer is x = seven."))


def test_body_must_be_english_but_quotes_and_dialogue_may_not_be():
    assert "E040-NON-ENGLISH-BODY" in err_rules(BASE_OK.replace("a man.", "一个男人。"))
    # on-screen text stays in the source language, in double quotes, untranslated
    assert err_rules(BASE_OK.replace("a man.", 'a neon sign reading "营业中".')) == set()
    # ... and so does the dialogue itself
    assert err_rules(BASE_OK.replace("[English] We are out of time.",
                                     "[Chinese] 我们没有时间了。")) == set()


def test_syllable_budget_is_a_warning_not_an_error():
    long_line = "[Chinese] " + "今天的豆子是云南的中度烘焙" * 4
    text = BASE_OK.replace("[English] We are out of time.", long_line)
    rep = validate(text, "t2va", duration_s=5.04)
    assert rep.ok
    assert "W032-SYLLABLE-BUDGET" in {v.rule for v in rep.warnings}


# ---------------------------------------------------------------- ref2va-only rules

REF_OK = _case_ir()["ref2va"].rsplit(" ", 1)[0]


def ref_err(text: str) -> set[str]:
    return {v.rule for v in validate(text, "ref2va", 5.04).errors}


def test_ref2va_summary_needs_a_legal_bracketed_task_type():
    assert "E055-SUMMARY-NO-TASK-TYPE" in ref_err(
        REF_OK.replace("[reference generation] ", ""))
    assert "E056-BAD-TASK-TYPE" in ref_err(
        REF_OK.replace("[reference generation]", "[style transfer]"))
    assert ref_err(REF_OK.replace("[reference generation]",
                                  "[reference generation + audio reference]")) == set()


def test_ref2va_every_label_needs_exactly_one_retention_line_with_a_legal_marker():
    dropped = "\n".join(ln for ln in REF_OK.splitlines() if not ln.startswith("<Subject 2> (appears"))
    assert "E052-RETENTION-CARDINALITY" in ref_err(dropped)
    assert "E053-BAD-MARKER" in ref_err(REF_OK.replace("fully_preserved", "kept the same", 1))
    # audio labels take the other marker table
    assert "E053-BAD-MARKER" in ref_err(
        REF_OK.replace("<Subject 2>", "<Audio 1>").replace("fully_preserved -", "fully_copy -", 1))


def test_ref2va_undefined_label_is_caught():
    # the mutation has to land in detailed_description, not in subject_definitions -- an undefined
    # label is one the body USES and the definitions never introduce.
    mutated = REF_OK.replace("counter of <Subject 2> with the espresso",
                             "counter of <Subject 7> with the espresso")
    assert mutated != REF_OK, "fixture drifted: mutation target no longer present"
    assert "E051-LABEL-UNDEFINED" in ref_err(mutated)


def test_ref2va_retention_analysis_may_not_carry_speaker_ids():
    assert "E054-SPEAKER-ID-IN-RETENTION" in ref_err(
        REF_OK.replace("<Subject 1> (appears in [Shot 1]):", "<Subject 1> (S1) (appears in [Shot 1]):"))


def test_ref2va_fields_are_not_base_fields():
    assert "E003-UNKNOWN-FIELD" in ref_err(
        REF_OK + "\nintegrated_multimodal_description: [Shot 1] something.")


def test_scenetrans_comes_in_pairs_and_cutoff_ends_the_video():
    once = BASE_OK.replace("out of time.</d>", "out of time.<scenetrans></d>")
    assert "E028-SCENETRANS-UNPAIRED" in err_rules(once)
    paired = once.replace("[Shot 2] At 00:02.500, the camera cuts",
                          "[Shot 2] At 00:02.500, <scenetrans> the camera cuts")
    assert "E028-SCENETRANS-UNPAIRED" not in err_rules(paired)
    early = BASE_OK.replace("frames a man.", "frames a man who trails off <cutoff>")
    assert "E029-CUTOFF-NOT-AT-END" in err_rules(early)


# ---------------------------------------------------------------- the oracle's request assembly
# No network here. These are the two constraints in MiniMax's own H3ContextIRReq schema that are
# easy to violate and expensive to discover from a 400: the roles decide the task, and the keyframe
# roles may not be mixed with the reference roles.

def test_oracle_infers_the_task_from_the_content_roles():
    from h3ir.oracle import build_content, infer_task
    assert infer_task(build_content("x")) == "t2va"
    assert infer_task(build_content("x", first_frame="u")) == "i2va"
    assert infer_task(build_content("x", last_frame="u")) == "l2va"
    assert infer_task(build_content("x", first_frame="u", last_frame="v")) == "fl2va"
    assert infer_task(build_content("x", reference_audios=["u"])) == "ref2va"


def test_oracle_refuses_to_mix_keyframe_and_reference_roles():
    from h3ir.oracle import OracleError, build_content
    with pytest.raises(OracleError):
        build_content("x", first_frame="u", reference_images=["v"])
    with pytest.raises(OracleError):
        build_content("   ")
