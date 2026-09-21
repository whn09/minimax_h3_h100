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

HOW THE MP4 COMES BACK. Over HTTP, through the tunnel that is already open: `GET /v1/videos/{id}/
content`. That endpoint IS there -- it is in the server's openapi.json -- and this repo spent a while
believing it was not, because the completed job object's `url` field is null and the response's
`file_path` is a path on the SERVER's filesystem. Measured on a 1.06 MB clip: 1.14-1.48 s over HTTP
against 2.65-3.4 s for a multiplexed scp and 5.4-6.1 s cold. scp is kept only as a fallback for a
build without the endpoint (`FETCH=scp`), and it is the one thing here that needs the ssh alias to be
a real alias rather than just a forwarded port.

WHAT IT DOES NOT DO. It does not retry a failed render. A failure here is a 400 (a shape off the
frame lattice, or ref2va, which VDN refuses) or an OOM, and all three are configuration errors that a
retry converts into a slow configuration error. They are raised.
"""
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import queue
import shutil
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
KEYFRAME_DIR = os.environ.get("KEYFRAME_DIR", "/opt/dlami/nvme/vdn/inputs/keyframes")

# 345 f at 24 fps = 14.375 s. The lattice is 5 + 17k; see game_serve.sh. duration_seconds is a float
# and `seconds` is NOT sent -- VideoGenerationsRequest types that one as int, so 14.375 becomes a 400.
FRAMES = int(os.environ.get("FRAMES", 345))
SHORT_EDGE = int(os.environ.get("SHORT_EDGE", 480))
FLOW_SHIFT = 12.0
AUDIO_FLOW_SHIFT = 3.0

# How the clip comes back and how a local keyframe goes out. Both default to HTTP so the only thing
# ssh is needed for is the tunnel itself. "scp" on either side is the fallback for a build without
# GET /v1/videos/{id}/content, or for a keyframe too big to want in a request body.
FETCH = os.environ.get("FETCH", "http")          # http | scp
KEYFRAME_MODE = os.environ.get("KEYFRAME_MODE", "inline")   # inline | scp


def legal_frames(n: int) -> bool:
    return n >= 22 and (n - 5) % 17 == 0


def data_uri(path: Path) -> str:
    """Inline an image as a `data:` URI, which H3's material resolver accepts.

    minimax_h3_localize_material_uri takes local paths, file://, http(s)://, data:/base64: and
    tar+offset. The data: form is what makes a keyframe work without touching the serving box's
    filesystem at all -- which matters because each replica resolves URIs in its OWN worker, so a
    keyframe staged on one box is simply absent on the other. Verified end to end: a 381 KB png
    became a 509 KB URI, rendered, and frame 0 came back at 32.88 dB PSNR against the keyframe --
    the same seam quality as the file-path route's 33.16 dB.
    """
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


class Replica:
    def __init__(self, alias: str, endpoint: str) -> None:
        self.alias, self.endpoint = alias, endpoint
        self._staged = False

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

    def _ssh_opts(self) -> list[str]:
        # Ride game_tunnel.sh's ControlMaster socket if it is there: measured 946 KB from us-east-2,
        # 5.4-6.1 s cold against 2.65-3.4 s multiplexed. A ControlPath that does not exist is not an
        # error -- ssh just dials a fresh connection -- so this stays correct with --replicas too.
        opts = ["-o", "StrictHostKeyChecking=accept-new"]
        cm = STATE / f"cm-{self.alias}"
        if cm.exists():
            opts += ["-o", f"ControlPath={cm}"]
        return opts

    def put_keyframe(self, local: Path) -> str:
        """Upload a keyframe and return ITS PATH ON THIS REPLICA.

        Each box has its own filesystem, and the pool picks a replica per request -- so a keyframe
        cut on one box is simply absent on the other, and the request dies with
        `FileNotFoundError: MiniMax H3 material source does not exist or is not a file: <path>`
        surfacing as a 500. Uploading per replica is the only thing that is correct without shared
        storage. ~390 KB of png, one scp on the multiplexed connection.
        """
        if not self._staged:
            subprocess.run(["ssh", *self._ssh_opts(), self.alias, f"mkdir -p {KEYFRAME_DIR}"],
                           check=True)
            self._staged = True
        remote = f"{KEYFRAME_DIR}/{local.stem}_{int(time.time() * 1000)}{local.suffix}"
        subprocess.run(["scp", "-q", *self._ssh_opts(), str(local), f"{self.alias}:{remote}"],
                       check=True)
        return remote

    def fetch_http(self, vid: str, outdir: Path) -> Path:
        """GET /v1/videos/{id}/content -- the fast path, and the reason scp is now a fallback.

        This endpoint exists (it is in the server's openapi.json) even though the response object's
        `url` field is null, which is what misled this repo into believing there was no way to
        download a clip over HTTP. Measured 1.06 MB: 1.14-1.48 s here against 2.65-3.4 s for a
        multiplexed scp and 5.4-6.1 s for a cold one -- it rides the ssh forward that is already
        open and warm, instead of standing up a second connection per clip.
        """
        outdir.mkdir(parents=True, exist_ok=True)
        local = outdir / f"{self.alias}_{vid}.mp4"
        with urllib.request.urlopen(f"http://{self.endpoint}/v1/videos/{vid}/content",
                                    timeout=300) as r, open(local, "wb") as f:
            shutil.copyfileobj(r, f)
        return local

    def fetch_scp(self, remote: str, outdir: Path) -> Path:
        """Fallback for a build without the content endpoint. Needs the ssh alias to be real."""
        outdir.mkdir(parents=True, exist_ok=True)
        local = outdir / f"{self.alias}_{Path(remote).name}"
        subprocess.run(["scp", "-q", *self._ssh_opts(), f"{self.alias}:{remote}", str(local)],
                       check=True)
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
                 timeout: float = 600.0, fetch_mode: str = FETCH,
                 keyframe_mode: str = KEYFRAME_MODE) -> None:
        if not legal_frames(frames):
            raise ValueError(f"{frames} frames is off the 5 + 17k lattice; see game_serve.sh")
        self.replicas = replicas if replicas is not None else read_replicas(None)
        self.outdir = outdir or Path("out/game")
        self.frames, self.short_edge, self.fetch, self.timeout = frames, short_edge, fetch, timeout
        self.fetch_mode, self.keyframe_mode = fetch_mode, keyframe_mode
        self.q: queue.Queue = queue.Queue()
        self._threads = [threading.Thread(target=self._worker, args=(r,), daemon=True)
                         for r in self.replicas]
        for t in self._threads:
            t.start()

    def submit(self, prompt: str, *, seed: int | None = None, keyframe: str | None = None,
               steps: int | None = None) -> Future:
        """fl2va when `keyframe` is given, pinned to frame 0; t2va otherwise.

        `keyframe` may be either a path on THIS machine -- in which case it is inlined as a `data:`
        URI (or scp'd, with keyframe_mode="scp") to whichever replica ends up serving the request --
        or a path/URL that already resolves on the boxes, which is passed through as-is. The
        distinction is made by `Path.is_file()` locally, so a server-side path that happens to also
        exist here resolves as local; name the staging dirs differently if that matters.

        The reason a local path cannot just be forwarded: H3 resolves condition URIs in the WORKER,
        on the serving box, so the file has to exist THERE -- and each replica has its own
        filesystem, so "there" is a different place per request.
        """
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
                # Resolve the keyframe HERE, not in submit(): which replica serves a request is not
                # known until a worker picks it up, and the file has to exist on THAT box.
                kf = keyframe
                if kf is not None and Path(kf).is_file():
                    kf = (data_uri(Path(kf)) if self.keyframe_mode == "inline"
                          else rep.put_keyframe(Path(kf)))
                r = rep.post(self._body(prompt, seed, kf, steps))
                d = rep.wait(r["id"], self.timeout)
                if d["status"] != "completed":
                    raise RuntimeError(f"{rep} failed: {str(d.get('error'))[:400]}")
                remote = d.get("file_path")
                if not self.fetch:
                    out = remote
                elif self.fetch_mode == "http":
                    out = rep.fetch_http(r["id"], self.outdir)
                else:
                    out = rep.fetch_scp(remote, self.outdir) if remote else remote
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
    ap.add_argument("--keyframe", help="image path (local or server-side): fl2va, pinned to frame 0")
    ap.add_argument("--keyframe-mode", choices=["inline", "scp"], default=KEYFRAME_MODE,
                    help="how a LOCAL keyframe reaches the box: data: URI (default) or scp")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--outdir", type=Path, default=Path("out/game"))
    ap.add_argument("--fetch-mode", choices=["http", "scp"], default=FETCH,
                    help="http = GET /v1/videos/{id}/content (default, ~2.3x faster than scp)")
    ap.add_argument("--no-fetch", action="store_true", help="leave the mp4 on the box, print its path")
    ap.add_argument("--warm", action="store_true",
                    help="throw away one clip per replica first; the server cannot warm 480p itself")
    ap.add_argument("--bench", type=int, metavar="N",
                    help="N clips through the pool: per-clip latency AND clips/minute")
    args = ap.parse_args()

    reps = read_replicas(args.replicas)
    print(f"{len(reps)} replica(s): {', '.join(map(str, reps))}", file=sys.stderr)

    with Renderer(reps, outdir=args.outdir, frames=args.frames, short_edge=args.short_edge,
                  fetch=not args.no_fetch, fetch_mode=args.fetch_mode,
                  keyframe_mode=args.keyframe_mode) as r:
        if args.warm and not args.bench:
            r.warm()
            print("warmed every replica at this exact shape", file=sys.stderr)
            # `--warm` with no prompt is a complete, useful invocation -- it is the startup step the
            # README's quick start tells you to run once, and it used to warm the replicas and THEN
            # exit 2 on "give a prompt".
            if not args.prompt:
                return 0

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
                      f"-- the rest is mux, download and polling")
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
