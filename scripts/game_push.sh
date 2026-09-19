#!/usr/bin/env bash
# Put the five files a game replica needs onto each box, by ssh alias.
#
#   bash game_push.sh H100-A H100-B
#
# WHY NOT scripts/p5.sh. That helper takes HOST=ubuntu@<ip> and hardcodes one region's .pem,
# because it was written for the single box RESULTS.md was measured on. Everything in the game path
# (game_tunnel.sh, game_client.py's scp) addresses machines by ssh ALIAS instead, so the key, the
# user and the DNS name live in ~/.ssh/config and there is exactly one place to fix when a box is
# replaced. This is the same idea for the upload direction.
#
# WHY SO FEW FILES. The container route needs no source tree on the box: `h3.sh` pulls upstream's
# nightly, which already carries sglang + the diffusion extras, and the scripts arrive through the
# bind mount rather than a COPY. So this is the arm script, the environment it sources, the weights
# fetcher, and the wrapper.
set -uo pipefail

FILES_VDN=(scripts/_env.sh scripts/fetch_weights.sh scripts/game_serve.sh)
FILES_DOCKER=(docker/h3.sh docker/Dockerfile)
DEST=${DEST:-/opt/dlami/nvme/vdn}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

[ $# -ge 1 ] || { echo "usage: game_push.sh <ssh-alias> [<ssh-alias> ...]"; exit 2; }

SSHOPTS=(-o StrictHostKeyChecking=accept-new -o ConnectTimeout=10)

for alias in "$@"; do
  echo "--- $alias"
  # /opt/dlami/nvme is instance store: it exists on a p5 and it is EMPTY after every stop/start, so
  # the mkdir is part of the normal path, not error handling.
  ssh "${SSHOPTS[@]}" "$alias" "mkdir -p $DEST/docker $DEST/outputs /opt/dlami/nvme/sglang/logs" \
    || { echo "cannot reach $alias -- check ~/.ssh/config"; exit 1; }
  for f in "${FILES_VDN[@]}"; do
    scp -q "${SSHOPTS[@]}" "$HERE/$f" "$alias:$DEST/$(basename "$f")" || exit 1
    echo "  -> $DEST/$(basename "$f")"
  done
  for f in "${FILES_DOCKER[@]}"; do
    scp -q "${SSHOPTS[@]}" "$HERE/$f" "$alias:$DEST/docker/$(basename "$f")" || exit 1
    echo "  -> $DEST/docker/$(basename "$f")"
  done
  # docker group membership, checked here rather than discovered three commands later when
  # `h3.sh probe` fails with a permission error on the socket. The DLAMI puts ubuntu in it already.
  ssh "${SSHOPTS[@]}" "$alias" 'docker info >/dev/null 2>&1 \
      && nvidia-smi -L | sed "s/^/  /" \
      || echo "  WARNING: docker not usable as this user"'
done

echo
echo "next, on each box:  cd $DEST/docker && bash h3.sh probe && bash h3.sh weights vdn"
echo "then:               FRAMES=345 bash h3.sh serve game && bash h3.sh logs game"
