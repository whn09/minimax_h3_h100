#!/usr/bin/env bash
# Push the patched sources, the 480p/768p configs and the driver onto the box.
#
# The box's working tree is not a git branch -- patches 1-3 were applied there by hand and
# `git am` would refuse on top of them. So the unit of sync is the FILE SET the series
# touches, copied from /tmp/vdnh3 (where the series lives as real commits). That makes the
# box byte-identical to the patch workspace rather than "probably up to date", which is the
# only property worth having when the numbers in RESULTS.md come off it.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORK=${WORK:-/tmp/vdnh3}
REPO=/opt/dlami/nvme/vdn/vdn-minimax-h3

for f in $(cd "$WORK" && git diff --name-only 2f740c9 HEAD); do
  echo "  src  $f"
  bash "$HERE/p5.sh" --put "$WORK/$f" "$REPO/$f"
done
for f in "$HERE"/../configs/*.yaml; do
  echo "  cfg  $(basename "$f")"
  bash "$HERE/p5.sh" --put "$f" "$REPO/configs/inference/$(basename "$f")"
done
for f in h100_grid.sh runone.sh summarize.py; do
  [ -f "$HERE/$f" ] || continue
  echo "  drv  $f"
  bash "$HERE/p5.sh" --put "$HERE/$f" "/opt/dlami/nvme/vdn/$f"
done
# These go INSIDE the repo, not next to the driver: each does
# `sys.path.insert(0, dirname(dirname(__file__)))` so it can import `src.*`, which only
# resolves when the file sits at <repo>/scripts/. decode_parity.py is not listed here
# because it is a tracked file of the patch series and the loop above already sent it.
bash "$HERE/p5.sh" "mkdir -p $REPO/scripts"
for f in text_encoder_bench.py mux_bench.py vidcmp.py vidscale.py vidshift.py; do
  echo "  drv  scripts/$f"
  bash "$HERE/p5.sh" --put "$HERE/$f" "$REPO/scripts/$f"
done
echo "sync done"
