"""CLI for the H3 prompt compiler.

  compile  one request -> a validated prompt (or a non-zero exit and the violations)
  check    validate an existing prompt file, no model call
  bench    the same requests through several models, reporting first-pass clean rate and repairs
  oracle   one call to MiniMax's own closed H3-Context-IR API, and our verdict on what it returns
  harvest  many such calls into a JSONL corpus: validator ground truth and distillation data

Examples:
  python -m h3ir.cli check --task t2va --frames 121 prompt.txt
  python -m h3ir.cli compile --frames 241 --script '今天的豆子是云南的，牛奶用两升装的。' \\
      '固定镜头，中景，一位年轻男咖啡师站在吧台后面，指着身后写好的小黑板介绍今天的豆子'
  python -m h3ir.cli bench --models haiku45,opus5 --out logs/bench.json
  MINIMAX_API_KEY=... python -m h3ir.cli oracle --duration 5 'A boy playing basketball by the sea'
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

from .compile import LEGAL_FRAMES, MODELS, compile_prompt
from .validate import duration_for_frames, validate

# The bench set. Each request exercises something the validator can fail on, so the numbers mean
# something: a model that scores well here is producing prompts that are actually renderable.
BENCH = [
    # A supplied script containing a digit welded to a letter (`2L`). The compiler must respell it
    # inside `<d>` and must not touch the on-screen text in double quotes -- measured rule 1, the one
    # defect neither official guide states.
    dict(name="spoken-unit-t2va", task="t2va", frames=121,
         request="固定镜头，中景，一位穿深灰色T恤、戴棕色围裙的年轻男咖啡师站在咖啡馆吧台后面，"
                 "他身后墙上的小黑板已经写好\"TODAY: YUNNAN\"，他指着黑板介绍今天的豆子。",
         script="今天的豆子是云南的，牛奶用2L装的。"),
    dict(name="three-shot-market", task="t2va", frames=241,
         request="10秒，三个镜头：清晨的菜市场，摊主大声吆喝；切到买菜的老太太挑西红柿；"
                 "最后切到远景，整个市场在晨光里。要有环境音和轻快的背景音乐。",
         script=None),
    dict(name="voiceover-product", task="t2va", frames=241,
         request="高端耳机产品广告，全程没有人出现在画面里，只有画外女声旁白，"
                 "最后一个镜头产品悬浮旋转，屏幕上出现 \"静界 Pro\" 字样。",
         script="真正的安静，是听见你想听的。"),
    # ref2va: both a person and a room to preserve, an action that must FINISH inside the duration,
    # and no speech -- so the compiler has to write the six reference sections, label both subjects,
    # give each a retention marker, and still say the lips stay closed.
    dict(name="latte-art-ref", task="ref2va", frames=241,
         request="参考图里的咖啡师站在同一个吧台后面，把奶泡从钢杯倒进面前那杯浓缩咖啡里，"
                 "最后在表面拉出一片叶子的图案。要保持这个人和这个店，固定镜头，他全程不说话。",
         script=None, assets=["<Picture 1> = the reference photo of the barista at the counter"]),
]


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--task", default="t2va",
                   choices=["t2va", "i2va", "fl2va", "l2va", "ref2va"])
    p.add_argument("--frames", type=int, default=121, choices=sorted(LEGAL_FRAMES),
                   help="121 = 5.04 s, 241 = 10.04 s, 345 = 14.38 s at 24 fps")


def cmd_check(args: argparse.Namespace) -> int:
    text = pathlib.Path(args.file).read_text() if args.file != "-" else sys.stdin.read()
    rep = validate(text.strip(), args.task, duration_s=duration_for_frames(args.frames))
    print(rep.render())
    print(f"\n{len(rep.errors)} error(s), {len(rep.warnings)} warning(s) -- "
          f"{'PASS' if rep.ok else 'REJECTED'}")
    return 0 if rep.ok else 1


def cmd_compile(args: argparse.Namespace) -> int:
    res = compile_prompt(
        request=args.request,
        task=args.task,
        frames=args.frames,
        model=args.model,
        assets=args.asset,
        script=args.script,
        max_repairs=args.max_repairs,
        cache_system=not args.no_cache,
    )
    print(res.summary(), file=sys.stderr)
    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(res.to_json(), ensure_ascii=False, indent=2))
    if not res.ok:
        print("\nfail closed: no prompt emitted. Last attempt's violations:", file=sys.stderr)
        print(res.attempts[-1].report.render(), file=sys.stderr)
        return 1
    print(res.prompt)
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    out: list[dict] = []

    def flush() -> None:
        if args.out:
            p = pathlib.Path(args.out)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(out, ensure_ascii=False, indent=2))

    for model in args.models.split(","):
        for case in BENCH:
            row = dict(model=model, case=case["name"])
            try:
                res = compile_prompt(model=model, max_repairs=args.max_repairs,
                                     cache_system=not args.no_cache,
                                     **{k: v for k, v in case.items() if k != "name"})
            except Exception as exc:  # a transport failure is a bench result, not a crash
                row.update(ok=False, error=f"{type(exc).__name__}: {exc}")
                print(f"{model:28} {case['name']:22} ERROR {type(exc).__name__}", file=sys.stderr)
                out.append(row)
                flush()
                continue
            first_clean = res.attempts[0].report.ok
            row.update(ok=res.ok, first_pass_clean=first_clean, repairs=res.repairs,
                       seconds=round(sum(a.seconds for a in res.attempts), 2),
                       first_pass_errors=[v.rule for v in res.attempts[0].report.errors],
                       warnings=[v.rule for v in res.attempts[-1].report.warnings],
                       prompt=res.prompt)
            print(f"{model:28} {case['name']:22} "
                  f"first-pass {'clean' if first_clean else 'dirty'}  "
                  f"repairs {res.repairs}  {'OK' if res.ok else 'REJECTED'}  "
                  f"{row['seconds']}s", file=sys.stderr)
            out.append(row)
            flush()

    for model in args.models.split(","):
        rows = [r for r in out if r["model"] == model]
        done = [r for r in rows if "error" not in r]
        clean = sum(r["first_pass_clean"] for r in done)
        print(f"\n{model}: first-pass clean {clean}/{len(done)}, "
              f"validated after repair {sum(r['ok'] for r in done)}/{len(done)}, "
              f"mean repairs {sum(r['repairs'] for r in done) / max(1, len(done)):.2f}, "
              f"transport errors {len(rows) - len(done)}", file=sys.stderr)
    return 0 if all(r["ok"] for r in out) else 1


def _oracle_media(args: argparse.Namespace) -> dict:
    return dict(first_frame=args.first_frame, last_frame=args.last_frame,
                reference_images=args.reference_image, reference_videos=args.reference_video,
                reference_audios=args.reference_audio)


def cmd_oracle(args: argparse.Namespace) -> int:
    """One call to MiniMax's own closed H3-Context-IR. Costs real money (~$0.017 a call)."""
    from .oracle import context_ir

    res = context_ir(args.request, duration=args.duration, ratio=args.ratio, **_oracle_media(args))
    print(f"task {res.task_id}  {res.status}  {res.seconds:.1f}s wall / "
          f"{res.server_seconds}s server  ${res.cost_usd:.4f}  {res.usage}", file=sys.stderr)
    if res.report is not None:
        verdict = "accepts it" if res.report.ok else "REJECTS IT -- our rules are wrong"
        print(f"our validator {verdict}", file=sys.stderr)
        if not res.report.ok:
            print(res.report.render(), file=sys.stderr)
    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(res.to_json(), ensure_ascii=False, indent=2))
    if not res.prompt:
        print(f"no prompt returned: {res.error}", file=sys.stderr)
        return 1
    print(res.prompt)
    return 0


def cmd_harvest(args: argparse.Namespace) -> int:
    """Build a (request -> official IR) corpus: validator ground truth, and distillation data.

    Appends one JSON object per line and flushes as it goes, so an interrupted or rate-limited run
    keeps everything it already paid for.
    """
    from .oracle import OracleError, context_ir

    if args.infile:
        reqs = [json.loads(ln) for ln in pathlib.Path(args.infile).read_text().splitlines()
                if ln.strip() and not ln.lstrip().startswith("#")]
    else:  # no corpus supplied: the bench set is at least a real one
        reqs = [{"request": c["request"], "duration": 5 if c["frames"] == 121 else 10}
                for c in BENCH]

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists() and not args.overwrite:
        for ln in out.read_text().splitlines():
            if ln.strip():
                done.add(json.loads(ln)["request"]["content"][0]["text"])

    spent, rejected, n = 0.0, 0, 0
    with out.open("a" if not args.overwrite else "w") as fh:
        for i, r in enumerate(reqs, 1):
            text = r.get("request") or r.get("text") or ""
            if text.strip() in done:
                print(f"[{i}/{len(reqs)}] already harvested, skipping", file=sys.stderr)
                continue
            try:
                res = context_ir(text, duration=int(r.get("duration", 5)),
                                 ratio=r.get("ratio", "16:9"),
                                 first_frame=r.get("first_frame"), last_frame=r.get("last_frame"),
                                 reference_images=r.get("reference_images"),
                                 reference_videos=r.get("reference_videos"),
                                 reference_audios=r.get("reference_audios"))
            except OracleError as exc:
                print(f"[{i}/{len(reqs)}] FAILED {exc}", file=sys.stderr)
                continue
            fh.write(json.dumps(res.to_json(), ensure_ascii=False) + "\n")
            fh.flush()
            n += 1
            spent += res.cost_usd
            bad = res.report is not None and not res.report.ok
            rejected += bad
            print(f"[{i}/{len(reqs)}] {res.status}  {res.server_seconds}s  ${res.cost_usd:.4f}  "
                  + ("VALIDATOR DISAGREES: " + ",".join(v.rule for v in res.report.errors)
                     if bad else "validator clean"), file=sys.stderr)

    print(f"\nharvested {n} pair(s) into {out} for ${spent:.2f}. "
          f"{rejected} official output(s) failed our validator"
          + (" -- fix those rules before trusting the compiler." if rejected else "."),
          file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="h3ir", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("check", help="validate an existing prompt, no model call")
    _add_common(c)
    c.add_argument("file", help="prompt file, or - for stdin")
    c.set_defaults(fn=cmd_check)

    c = sub.add_parser("compile", help="compile a request into a validated prompt")
    _add_common(c)
    c.add_argument("request")
    # haiku45 is the default on measurement: 4/4 first-pass validator-clean at an 8.1 s mean,
    # against opus5's 4/4 at 19.5 s. The validator is what makes the cheap model safe here.
    c.add_argument("--model", default="haiku45", help=f"one of {list(MODELS)} or a full Bedrock id")
    c.add_argument("--script", help="the caller's verbatim spoken line")
    c.add_argument("--asset", action="append", help="a labelled reference asset; repeatable")
    c.add_argument("--max-repairs", type=int, default=3)
    c.add_argument("--no-cache", action="store_true",
                   help="do not put a Bedrock cachePoint after the ~8k-token system prompt "
                        "(caching is a cost lever: the compile is output-bound, not input-bound)")
    c.add_argument("--json", help="write the full attempt trace here")
    c.set_defaults(fn=cmd_compile)

    c = sub.add_parser("bench", help="run the bench set through one or more models")
    c.add_argument("--models", default="haiku45,opus5")
    c.add_argument("--max-repairs", type=int, default=3)
    c.add_argument("--no-cache", action="store_true")
    c.add_argument("--out")
    c.set_defaults(fn=cmd_bench)

    def _media(p: argparse.ArgumentParser) -> None:
        p.add_argument("--first-frame", help="public URL of the first-frame image")
        p.add_argument("--last-frame", help="public URL of the last-frame image")
        p.add_argument("--reference-image", action="append", help="repeatable, max 9")
        p.add_argument("--reference-video", action="append", help="repeatable, max 3")
        p.add_argument("--reference-audio", action="append", help="repeatable, max 3")

    c = sub.add_parser("oracle", help="call MiniMax's own closed H3-Context-IR (needs "
                                      "MINIMAX_API_KEY; costs about $0.017 per call)")
    c.add_argument("request")
    c.add_argument("--duration", type=int, default=5, help="4-15 s, integer (the API's own units)")
    c.add_argument("--ratio", default="16:9",
                   help="required and non-adaptive for text-only requests")
    c.add_argument("--json", help="write the full result, usage and validator verdict here")
    _media(c)
    c.set_defaults(fn=cmd_oracle)

    c = sub.add_parser("harvest", help="build a (request -> official IR) corpus from the real API")
    c.add_argument("--in", dest="infile",
                   help="JSONL of requests; each line {request, duration, ratio, ...}. "
                        "Defaults to this file's bench set.")
    c.add_argument("--out", default="golds/harvest.jsonl")
    c.add_argument("--overwrite", action="store_true",
                   help="rewrite the output instead of appending and skipping what is already there")
    c.set_defaults(fn=cmd_harvest)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
