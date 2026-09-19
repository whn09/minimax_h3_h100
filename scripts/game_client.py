#!/usr/bin/env python3
"""Client for the game: submit a clip, get a local mp4 back, across N tunnelled replicas.

    bash scripts/game_tunnel.sh H100-A H100-B     # once
    python3 scripts/game_client.py --bench 6      # measure what the game will actually feel
    python3 scripts/game_client.py "a lantern-lit alley in the rain, footsteps and distant thunder"

    from game_client import Renderer
    r = Renderer()                                # reads the replica file game_tunnel.sh wrote
    fut = r.submit("the gate grinds open, dust falls")
    path = fut.result()                           # local mp4

WHY A POOL AND NOT A ROUND-ROBIN COUNTER. The server takes one request at a time per replica in any
configuration that matters here -- the whole box is one model, Ulysses-sharded across its cards, so a
second concurrent request does not interleave, it queues behind the first and both get slower. So the
useful unit of concurrency is the REPLICA, and the right structure is one worker per replica pulling
from a shared queue. Two machines therefore buy throughput (2 clips in flight) and not latency (one
clip still costs what one clip costs).

WHY THE MP4 COMES BACK OVER SCP. The API is submit-then-poll and the response carries `file_path` --
a path on the SERVER's filesystem, because H3 takes and returns URIs the worker resolves. There is no
/v1/videos/{id}/content in anything this repo has exercised. So the clip is copied back over the same
ssh alias the tunnel uses. If a future build does expose a content endpoint, `--no-fetch` plus that
endpoint is the faster path and this whole scp step goes away.

WHAT IT DOES NOT DO. It does not retry a failed render. A failure here is a 400 (a shape off the
frame lattice, or ref2va, which VDN refuses) or an OOM, and all three are configuration errors that a
retry converts into a slow configuration error. They are raised.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import Future
from pathlib import Path

STATE = Path(os.environ.get("STATE", Path(tempfile.gettempdir()) / "h3game"))
REPLICAS = STATE / "replicas"
REMOTE_OUT = os.environ.get("REMOTE_OUT", "/opt/dlami/nvme/vdn/outputs")

# 345 f at 24 fps = 14.375 s. The lattice is 5 + 17k; see game_serve.sh. duration_seconds is a float
# and `seconds` is NOT sent -- VideoGenerationsRequest types that one as int, so 14.375 becomes a 400.
FRAMES = int(os.environ.get("FRAMES", 345))
SHORT_EDGE = int(os.environ.get("SHORT_EDGE", 480))
FLOW_SHIFT = 12.0
AUDIO_FLOW_SHIFT = 3.0


def legal_frames(n: int) -> bool:
    return n >= 22 and (n - 5) % 17 == 0


class Replica:
    def __init__(self, alias: str, endpoint: str) -> None:
        self.alias, self.endpoint = alias, endpoint

    def __repr__(self) -> str:
        return f"{self.alias}@{self.endpoint}"

    def post(self, body: dict) -> dict:
        req = urllib.request.Request(f"http://{self.endpoint}/v1/videos",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"{self} rejected the request: {e.read().decode()[:600]}") from None

    def wait(self, vid: str, timeout: float) -> dict:
        # 0.2 s, the same interval sglang_cond.py polls at. Finer does not help: the measured gap
        # between server inference_time_s and client E2E is ~1 s of mux and file write, so polling
        # granularity is not what bounds this.
        deadline = time.time() + timeout
        while time.time() < deadline:
            with urllib.request.urlopen(f"http://{self.endpoint}/v1/videos/{vid}", timeout=30) as r:
                d = json.load(r)
            if d["status"] in ("completed", "failed"):
                return d
            time.sleep(0.2)
        raise TimeoutError(f"{self} did not finish {vid} in {timeout:.0f}s")

    def fetch(self, remote: str, outdir: Path) -> Path:
        outdir.mkdir(parents=True, exist_ok=True)
        local = outdir / f"{self.alias}_{Path(remote).name}"
        # Ride game_tunnel.sh's ControlMaster socket if it is there: measured 946 KB from us-east-2,
        # 5.4-6.1 s cold against 2.65-3.4 s multiplexed. A ControlPath that does not exist is not an
        # error -- ssh just dials a fresh connection -- so this stays correct with --replicas too.
        cm = STATE / f"cm-{self.alias}"
        opts = ["-o", "StrictHostKeyChecking=accept-new"]
        if cm.exists():
            opts += ["-o", f"ControlPath={cm}"]
        subprocess.run(["scp", "-q", *opts, f"{self.alias}:{remote}", str(local)], check=True)
        return local


def read_replicas(spec: str | None) -> list[Replica]:
    """`--replicas alias=host:port,alias=host:port`, else the file game_tunnel.sh wrote."""
    if spec:
        out = []
        for part in spec.split(","):
            alias, _, endpoint = part.partition("=")
            out.append(Replica(alias, endpoint or alias))
        return out
    if not REPLICAS.exists():
        raise SystemExit(f"no replicas: run `bash scripts/game_tunnel.sh <alias> [<alias>]` "
                         f"or pass --replicas (looked in {REPLICAS})")
    out = []
    for line in REPLICAS.read_text().splitlines():
        if line.strip():
            alias, endpoint = line.split()
            out.append(Replica(alias, endpoint))
    return out


class Renderer:
    """One worker per replica, one shared queue. Close it or use it as a context manager."""

    def __init__(self, replicas: list[Replica] | None = None, outdir: Path | None = None,
                 frames: int = FRAMES, short_edge: int = SHORT_EDGE, fetch: bool = True,
                 timeout: float = 600.0) -> None:
        if not legal_frames(frames):
            raise ValueError(f"{frames} frames is off the 5 + 17k lattice; see game_serve.sh")
        self.replicas = replicas if replicas is not None else read_replicas(None)
        self.outdir = outdir or Path("out/game")
        self.frames, self.short_edge, self.fetch, self.timeout = frames, short_edge, fetch, timeout
        self.q: queue.Queue = queue.Queue()
        self._threads = [threading.Thread(target=self._worker, args=(r,), daemon=True)
                         for r in self.replicas]
        for t in self._threads:
            t.start()

    def submit(self, prompt: str, *, seed: int | None = None, keyframe: str | None = None,
               steps: int | None = None) -> Future:
        """keyframe is a path ON THE SERVER (fl2va, pinned to frame 0) -- the worker resolves URIs,
        so a local path would fail with a file-not-found naming a file that plainly exists here."""
        fut: Future = Future()
        self.q.put((prompt, seed, keyframe, steps, fut))
        return fut

    def render(self, prompt: str, **kw) -> Path | str:
        return self.submit(prompt, **kw).result()

    def warm(self, prompt: str = "a wide shot of a quiet harbour at dawn") -> None:
        """One throwaway clip per replica, at the shape this Renderer will actually send.

        NEEDED because `--warmup-resolutions` does not do what its name says on H3: the server's
        _synthetic_warmup_target() uses it ONLY to pick the nearest aspect ratio, and the short edge
        is MINIMAX_H3_RECOMMENDED_SHORT_EDGE = 768, hardcoded. So a 480p server warms at 768p, and
        the first 480p request pays for the shape change -- measured at ~0.9 s of extra server time
        on a freshly started replica (7.58 s against a 6.70 s steady state). Small, but it lands on
        the first clip the player waits for, which is the worst place for it.

        Loops until every replica has served one, rather than trusting N tasks to land on N workers.
        """
        seen: set[str] = set()
        for _ in range(3):
            pending = [self.submit(prompt) for _ in range(len(self.replicas) - len(seen))]
            for f in pending:
                f.result()
                seen.add(f.replica)  # type: ignore[attr-defined]
            if len(seen) >= len(self.replicas):
                return

    def close(self) -> None:
        for _ in self._threads:
            self.q.put(None)

    def __enter__(self) -> "Renderer":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _body(self, prompt: str, seed: int | None, keyframe: str | None,
              steps: int | None) -> dict:
        conditions = []
        if keyframe:
            # frame_index is REQUIRED for a keyframe role and rejected for a reference role. 0 pins
            # the opening frame, which is the one a game wants: continue from where the last clip
            # ended. VDN serves t2va and fl2va and refuses ref2va -- that is a training limit.
            conditions = [{"role": "keyframe", "type": "image", "uri": keyframe, "frame_index": 0}]
        body = {
            "prompt": prompt,
            "task": "fl2va" if keyframe else "t2va",
            "conditions": conditions,
            "target": {"short_edge": self.short_edge, "aspect_ratio": "16:9",
                       "duration_seconds": self.frames / 24},
            "flow_shift": FLOW_SHIFT,
            "audio_flow_shift": AUDIO_FLOW_SHIFT,
            "output_path": f"{REMOTE_OUT}/game_{int(time.time() * 1000)}",
        }
        if seed is not None:
            body["seed"] = seed
        if steps is not None:
            # The checkpoint is the 8-step Stage-DMD distill. Asking for fewer is not a supported
            # speed knob, it is off-schedule; asking for more costs time for nothing.
            body["num_inference_steps"] = steps
        return body

    def _worker(self, rep: Replica) -> None:
        while True:
            item = self.q.get()
            if item is None:
                return
            prompt, seed, keyframe, steps, fut = item
            if not fut.set_running_or_notify_cancel():
                continue
            try:
                t0 = time.time()
                r = rep.post(self._body(prompt, seed, keyframe, steps))
                d = rep.wait(r["id"], self.timeout)
                if d["status"] != "completed":
                    raise RuntimeError(f"{rep} failed: {str(d.get('error'))[:400]}")
                remote = d.get("file_path")
                out = rep.fetch(remote, self.outdir) if (self.fetch and remote) else remote
                fut.e2e = time.time() - t0              # type: ignore[attr-defined]
                fut.inference = d.get("inference_time_s")  # type: ignore[attr-defined]
                fut.replica = rep.alias                 # type: ignore[attr-defined]
                fut.set_result(out)
            except Exception as exc:  # noqa: BLE001 -- the caller decides what a failure means
                fut.set_exception(exc)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prompt", nargs="?", help="what to render; omit with --bench")
    ap.add_argument("--replicas", help="alias=host:port,... (default: the game_tunnel.sh file)")
    ap.add_argument("--frames", type=int, default=FRAMES, help="5 + 17k only; 345 = 14.375 s")
    ap.add_argument("--short-edge", type=int, default=SHORT_EDGE, choices=[480, 768])
    ap.add_argument("--keyframe", help="server-side image path; makes this fl2va, pinned to frame 0")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--outdir", type=Path, default=Path("out/game"))
    ap.add_argument("--no-fetch", action="store_true", help="leave the mp4 on the box, print its path")
    ap.add_argument("--warm", action="store_true",
                    help="throw away one clip per replica first; the server cannot warm 480p itself")
    ap.add_argument("--bench", type=int, metavar="N",
                    help="N clips through the pool: per-clip latency AND clips/minute")
    args = ap.parse_args()

    reps = read_replicas(args.replicas)
    print(f"{len(reps)} replica(s): {', '.join(map(str, reps))}", file=sys.stderr)

    with Renderer(reps, outdir=args.outdir, frames=args.frames, short_edge=args.short_edge,
                  fetch=not args.no_fetch) as r:
        if args.warm and not args.bench:
            r.warm()
            print("warmed every replica at this exact shape", file=sys.stderr)

        if args.bench:
            # One discarded clip per replica first: the server cannot warm 480p itself (see
            # Renderer.warm), and a first-of-shape request is systematically slower in the decode and
            # the tail -- the mistake that put a 13.20 s figure in this repo's own README once.
            r.warm()
            t0 = time.time()
            futs = [r.submit(f"a wide shot of a quiet harbour at dawn, boat {i}")
                    for i in range(args.bench)]
            e2e, infer = [], []
            for f in futs:
                f.result()
                e2e.append(f.e2e)              # type: ignore[attr-defined]
                if f.inference is not None:    # type: ignore[attr-defined]
                    infer.append(f.inference)  # type: ignore[attr-defined]
            wall = time.time() - t0
            print(f"\n{args.bench} clips of {args.frames} f / {args.frames/24:.3f} s "
                  f"at {args.short_edge}p over {len(reps)} replica(s)")
            print(f"  per-clip E2E     median {statistics.median(e2e):6.2f} s  "
                  f"({min(e2e):.2f}-{max(e2e):.2f})")
            if infer:
                print(f"  server inference median {statistics.median(infer):6.2f} s  "
                      f"-- the rest is mux, scp and polling")
            print(f"  throughput       {args.bench / wall * 60:6.2f} clips/min  "
                  f"({wall:.1f} s wall)")
            print(f"  video seconds per wall second: "
                  f"{args.bench * args.frames / 24 / wall:.2f}x")
            return 0

        if not args.prompt:
            ap.error("give a prompt, or --bench N")
        fut = r.submit(args.prompt, seed=args.seed, keyframe=args.keyframe)
        out = fut.result()
        print(f"{out}   ({fut.e2e:.2f} s E2E, "        # type: ignore[attr-defined]
              f"{fut.inference} s server, on {fut.replica})")  # type: ignore[attr-defined]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
