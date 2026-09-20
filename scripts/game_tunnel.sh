#!/usr/bin/env bash
# Bring each replica's loopback HTTP port to a distinct port on THIS laptop, and keep it there.
#
#   bash game_tunnel.sh H100-A H100-B     # -> 127.0.0.1:30010 and 127.0.0.1:30011
#   bash game_tunnel.sh down
#   bash game_tunnel.sh status
#
# The arguments are ssh host aliases (~/.ssh/config), not hostnames, for the same reason the rest of
# this repo takes them that way: the key path, the user and the long ec2 DNS name belong in one place.
#
# WHY A FILE OF REPLICAS AND NOT JUST PORTS. game_client.py fetches the finished mp4 over HTTP
# (`GET /v1/videos/{id}/content`, ~1.2 s for 1 MB), so the port alone would nearly do -- but the
# fallback paths need the ssh ALIAS: `--fetch-mode scp` for a build without that endpoint, and
# `--keyframe-mode scp` for an fl2va keyframe too large to inline in the request body. The mapping
# alias -> local port is written here and read there.
#
# WHY -N AND NOT A LOGIN SHELL. A tunnel that also allocates a pty dies differently: the remote shell
# can exit, or a `logout` can take the forward with it, and the failure looks like an HTTP timeout
# three layers away. -N forwards and nothing else.
set -uo pipefail

BASE_PORT=${BASE_PORT:-30010}
REMOTE_PORT=${REMOTE_PORT:-30010}
STATE=${STATE:-${TMPDIR:-/tmp}/h3game}
REPLICAS=$STATE/replicas

mkdir -p "$STATE"

case "${1:-}" in
  down)
    if [ -d "$STATE" ]; then
      for p in "$STATE"/*.pid; do
        [ -e "$p" ] || continue
        pid=$(cat "$p")
        kill "$pid" 2>/dev/null && echo "closed tunnel pid $pid ($(basename "$p" .pid))"
        rm -f "$p"
      done
      rm -f "$STATE"/cm-*     # the control sockets die with their master; don't leave stale paths
    fi
    rm -f "$REPLICAS"
    exit 0
    ;;
  status)
    [ -f "$REPLICAS" ] || { echo "no tunnels (no $REPLICAS)"; exit 0; }
    while read -r alias endpoint; do
      # GET /health, and nothing else. The first version of this POSTed an empty body to /v1/videos
      # on the theory that any answer proves the forward is alive -- but the server ACCEPTS `{}`,
      # fills in its defaults ("Adjusting number of frames from 1 to 29 based on number of GPUs") and
      # renders a video. So a liveness check cost ~8 s of eight H100s per replica and left a junk
      # clip in outputs/. Never probe a generation endpoint with a write verb.
      if curl -s -o /dev/null -m 3 -f "http://$endpoint/health"; then
        echo "up    $alias  ->  $endpoint"
      else
        echo "DOWN  $alias  ->  $endpoint"
      fi
    done < "$REPLICAS"
    exit 0
    ;;
esac

[ $# -ge 1 ] || { echo "usage: game_tunnel.sh <ssh-alias> [<ssh-alias> ...] | down | status"; exit 2; }

: > "$REPLICAS"
port=$BASE_PORT
for alias in "$@"; do
  # -o ExitOnForwardFailure=yes: without it ssh connects, fails to bind the local port (because a
  # previous tunnel still holds it), and sits there looking healthy while every request goes to the
  # OLD replica. That is the one failure mode of this script worth spending a flag on.
  # ServerAlive*: an idle NAT or an ELB will drop a forward that carries no traffic between clips,
  # and a game that renders one beat a minute is exactly that shape.
  # ControlMaster, for the scp fallbacks: a COLD scp is expensive -- measured us-east-2 -> this Mac,
  # 946 KB, 5.4-6.1 s cold vs 2.65-3.4 s through this socket. The handshake was half the cost of
  # fetching a clip back when scp was the only route, and the tunnel already holds a connection to
  # the same box, so game_client.py rides it instead of opening its own.
  ssh -N -f \
    -o ExitOnForwardFailure=yes \
    -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
    -o StrictHostKeyChecking=accept-new \
    -o ControlMaster=yes -o "ControlPath=$STATE/cm-$alias" \
    -L "127.0.0.1:$port:127.0.0.1:$REMOTE_PORT" \
    "$alias" || { echo "FAILED to forward $alias -> 127.0.0.1:$port"; exit 1; }
  # -f backgrounds ssh itself, so $! is not its pid. Find it by the forward spec, which is unique.
  pid=$(pgrep -f "L 127.0.0.1:$port:127.0.0.1:$REMOTE_PORT" | head -1)
  echo "$pid" > "$STATE/$alias.pid"
  echo "$alias 127.0.0.1:$port" >> "$REPLICAS"
  echo "forwarded $alias  ->  127.0.0.1:$port  (pid $pid)"
  port=$((port + 1))
done

echo
echo "replicas written to $REPLICAS"
echo "next: python3 scripts/game_client.py --bench 5"
