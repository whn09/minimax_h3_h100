#!/usr/bin/env python3
"""Render a customer case file verbatim -- the prompt is the input under test, not a knob.

    python3 sglang_case.py case=/opt/dlami/nvme/vdn/case.txt task=ref2va tag=FP8 768:25:121
    python3 sglang_case.py case=... task=t2va   tag=BF16 768:25:121

WHY NOT sglang_ref2va.py / sglang_base_steps.py. Both of those own their prompt: the ref2va one
hard-codes a spin-and-count-fingers English prompt chosen to *provoke* melting, and the base one
loads VDN's example_2.pt. That is right for attribution arms -- one prompt across every cell or the
comparison means nothing -- and wrong here, where the prompt IS what the customer asked about and
must not be paraphrased, re-punctuated or translated. So this script reads the case file and sends
what is in it, byte for byte after stripping the leading `task:` label.

THE CASE FILE FORMAT, as the customer wrote it:

    <task>: <prompt text> [<reference image filename>]

one case per non-empty line. The task label picks the server port and the request shape, and for
ref2va the last whitespace-separated token is a filename relative to refdir= (the file itself is
not on the server's PATH-like search; H3 resolves conditioning URIs in the *worker*, so it has to
be an absolute path that exists inside the container). t2va lines carry no image and get
`conditions: []`. NBSP is stripped along with ASCII space: the real file has a U+00A0 after
`t2va:`, presumably from a paste out of a doc, and `.strip()` alone leaves it in the prompt.

WHY THE TASK PICKS THE PORT AND NOT A FLAG. `--model-variant` is a server argument: t2va and ref2va
are different weight partitions, so they cannot share a process. 30011 is the t2va server and 30012
the ref2va one (30010 is VDN), matching sglang_base_arm.sh and sglang_ref2va_arm.sh. Running both
at once needs two GPUs' worth of DiT, so on one card they run in sequence and only one port answers
at a time -- hence the explicit refusal below rather than a connection error 30 s in.

FRAME COUNTS ARE NOT FREE-FORM: the server validates duration in [4, 15] s and the model wants
frames = 1 mod 8. "5 seconds" is therefore 121 f (5.04 s), not 120.
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PORTS = {"t2va": 30011, "ref2va": 30012}
SEED = 42                                  # same seed as every other arm in this repo
FRAMES = 121                               # 5.04 s; see the docstring
# Both overridable by environment, because the p5/H100 route and the g7 route disagree about where
# the big local disk is mounted: /opt/dlami/nvme on a DLAMI instance store, /data on the g7 pods
# (hostPath to the node's NVMe RAID0). Getting this wrong on g7 is not a cosmetic error -- the pod's
# own writable layer is capped by ephemeral-storage and kubelet evicts the pod for exceeding it, so
# the mp4 has to land on the hostPath mount.
OUTDIR = Path(os.environ.get("OUTDIR") or "/opt/dlami/nvme/vdn/pull/case")
REFDIR = Path(os.environ.get("REFDIR") or "/opt/dlami/nvme/vdn/ref")


def parse(path: Path) -> list[tuple[str, str, str, str | None]]:
    """-> [(task, label, prompt, image or None)] in file order.

    `task@label:` gives the case its own output directory. Needed the moment a case file holds
    several variants of ONE prompt, which is what prompt A/B work looks like: output_path is
    f"{task}_{tag}_{edge}p..." so four t2va lines under one tag would land four mp4s in one
    directory under four uuids and nothing afterwards could say which was which. The label is
    appended to the tag rather than replacing it, so `tag=aud` plus `t2va@aike:` reads as
    `aud_aike` and both the serving arm and the prompt variant stay legible in the path.
    """
    out = []
    for raw in path.read_text().splitlines():
        line = raw.replace(" ", " ").strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            raise SystemExit(f"case line has no `task:` label: {line[:60]!r}")
        task, rest = line.split(":", 1)
        task, rest = task.strip(), rest.strip()
        label = ""
        if "@" in task:
            task, label = (p.strip() for p in task.split("@", 1))
        if task not in PORTS:
            raise SystemExit(f"unknown task {task!r}; known: {', '.join(PORTS)}")
        img = None
        if task == "ref2va":
            # The reference is the trailing token. Checked as a filename rather than assumed, so a
            # case file that forgot the image fails here instead of sending a prompt that mentions
            # a reference to a server that was not given one.
            head, _, last = rest.rpartition(" ")
            if not head or "." not in last:
                raise SystemExit(f"ref2va case has no trailing image filename: {rest[-40:]!r}")
            rest, img = head.strip(), last
        # `\n` -> newline, AFTER the image token has been split off. H3's official prompt format is
        # three blank-line-separated fields (integrated_multimodal_description / overall_soundscape /
        # non_diegetic_music, see docs/h3official/), and the case file is one case per LINE, so an IR
        # prompt cannot be written literally. Escapes are decoded here rather than the file switching
        # to a multi-line format, so case.txt (the customer's own text, single line) and case_ir.txt
        # (the rewritten form) stay the same format and the same parser.
        out.append((task, label, rest.replace("\\n", "\n"), img))
    return out


def request(port: int, task: str, prompt: str, ref: str | None, edge: int, steps: int,
            frames: int, tag: str, quality: str | None = None) -> None:
    body = {
        "prompt": prompt,
        "task": task,
        "conditions": ([{"role": "reference", "type": "image", "uri": ref}] if ref else []),
        "target": {"short_edge": edge, "aspect_ratio": "16:9", "duration_seconds": frames / 24},
        "num_inference_steps": steps,
        "seed": SEED,
        "output_path": str(OUTDIR / f"{task}_{tag}_{edge}p_{steps}step_{frames}f"),
    }
    # quality=high is a PER-REQUEST field, not a server flag, and it is the only Cache-DiT switch
    # that reaches H3. --cache-dit-config is read by diffusers_pipeline.py:592 and H3 runs the native
    # pipeline, so that flag is silently ignored (measured: byte-identical mp4 and 105.40 s against
    # the plain arm's 105.37 s). The native path asks MiniMaxH3DenoisingStage._cache_dit_requested(),
    # which is true only for sampling_params.quality == "high" or the SGLANG_CACHE_DIT_ENABLED env.
    # "high" then selects an AUDITED preset rather than whatever knobs an operator guessed --
    # constants.py:64 MINIMAX_H3_HIGH_QUALITY_CACHE_DIT_CONFIG = (4, 0.04, 1), i.e. warmup 4 steps,
    # residual-diff threshold 0.04, at most 1 consecutive cached step, Fn_compute_blocks 1, with
    # "Measured SSIM 0.931 / PSNR 28.16 dB against quality=lossless" recorded next to it. It is sent
    # top-level because VideoGenerationsRequest is ConfigDict(extra="allow") and does NOT declare
    # `quality`, so it lands in model_extra where request_extra_value() looks; a declared field would
    # have swallowed it and left the request silently lossless.
    if quality:
        body["quality"] = quality
    host = f"http://127.0.0.1:{port}"
    t0 = time.time()
    req = urllib.request.Request(f"{host}/v1/videos", data=json.dumps(body).encode(),
                                headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            j = json.load(r)
    except urllib.error.HTTPError as e:
        print(f"{tag} {task} REJECTED: {e.read().decode()[:600]}", flush=True)
        return
    except OSError as e:
        # The common operator error is "the other precision's server is still up", and a bare
        # ConnectionRefused with no port in it costs a minute of confusion.
        print(f"{tag} {task}: no server on {host} ({e}); is the {task} arm serving?", flush=True)
        return
    while True:
        with urllib.request.urlopen(f"{host}/v1/videos/{j['id']}", timeout=30) as g:
            d = json.load(g)
        if d["status"] in ("completed", "failed"):
            break
        time.sleep(0.5)
    dt = time.time() - t0
    if d["status"] != "completed":
        print(f"{tag} {task} FAILED: {str(d.get('error'))[:600]}", flush=True)
        return
    inf = d["inference_time_s"]
    print(f"{tag:>6s} {task:>6s} {edge}p {steps:>3} steps {frames:>4} f ({frames / 24:5.2f} s): "
          f"E2E {dt:7.2f} s  inference {inf:7.2f} s  {inf / steps:5.2f} s/step  "
          f"{inf / (frames / 24):6.2f} s per video-second  peak {d['peak_memory_mb']:.0f} MB\n"
          f"       -> {d.get('file_path')}", flush=True)


def main(argv: list[str]) -> None:
    case = task = tag = None
    quality = os.environ.get("QUALITY") or None
    refdir, arms = REFDIR, []
    for a in argv:
        if a.startswith("case="):
            case = Path(a[5:])
        elif a.startswith("task="):
            task = a[5:]
        elif a.startswith("tag="):
            tag = a[4:]
        elif a.startswith("quality="):
            quality = a[8:]
        elif a.startswith("refdir="):
            refdir = Path(a[7:])
        else:
            arms.append(a)
    if not case or not case.is_file():
        raise SystemExit(f"case=<file> is required and must exist (got {case!r})")
    if not tag:
        raise SystemExit("tag=<name> is required; it is what tells two mp4s apart afterwards")
    cases = parse(case)
    if task:
        cases = [c for c in cases if c[0] == task]
        if not cases:
            raise SystemExit(f"no {task} case in {case}")
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for t, label, prompt, img in cases:
        ref = None
        if img:
            ref = str(refdir / img)
            if not Path(ref).is_file():
                raise SystemExit(f"reference {ref} does not exist; upload it next to the case file")
        ctag = f"{tag}_{label}" if label else tag
        print(f"{t}{'@' + label if label else ''}: {len(prompt)} chars, "
              f"ref {ref or 'none'}, seed {SEED}", flush=True)
        for a in arms or [f"768:25:{FRAMES}"]:
            edge, steps, *rest = a.split(":")
            frames = int(rest[0]) if rest else FRAMES
            if frames % 8 != 1:
                raise SystemExit(f"{frames} frames is {frames % 8} mod 8; the model wants 1 mod 8")
            request(PORTS[t], t, prompt, ref, int(edge), int(steps), frames, ctag, quality)


if __name__ == "__main__":
    main(sys.argv[1:])
