#!/usr/bin/env bash
# Is this box in the state the numbers in RESULTS.md were measured in?  Run it on the box:
#
#   bash /opt/dlami/nvme/vdn/box_check.sh
#
# Every line is something that has silently been wrong at least once. The two `+cu129`
# suffixes are the important ones -- a bare `0.28.0` torchvision means RUNBOOK trap 1, which
# does not fail at install time and does not fail at import of torch, only at import of
# diffusers.loaders.peft, and reports itself as a missing Bloom model.
set -uo pipefail
ROOT=${ROOT:-/opt/dlami/nvme/vdn}
REPO=$ROOT/vdn-minimax-h3
BASE=2f740c9

cd "$REPO" 2>/dev/null || { echo "NOT SET UP: $REPO missing -- see RUNBOOK section 2"; exit 1; }

head=$(git rev-parse --short HEAD)
commits=$(git log --oneline "$BASE"..HEAD 2>/dev/null | wc -l | tr -d ' ')
dirty=$(git status --porcelain -uno | wc -l | tr -d ' ')
echo "repo        : HEAD $head, $commits commits past $BASE, $dirty tracked files modified"
# Two states are both correct, and they are not interchangeable if you ever run a git command
# that touches the working tree:
#   12 commits /  0 modified  -- the series applied with `git am` (section 2). git-clean.
#    0 commits /  6 modified  -- the series copied in as files (scripts/sync_box.sh from a
#                                Mac). `git checkout .` DESTROYS it.
if   [ "$commits" = 12 ] && [ "$dirty" = 0 ]; then echo "              -> series as commits, tree clean"
elif [ "$commits" = 0 ]  && [ "$dirty" = 6 ]; then echo "              -> series as file copies; do NOT run git checkout/stash/reset here"
else echo "              -> UNEXPECTED. Expected 12/0 (git am) or 0/6 (file copy)"; fi

echo "configs     : $(ls configs/inference/ 2>/dev/null | grep -c h100)/4 h100 yamls"
echo "weights     : $(du -sh --dereference ckpts 2>/dev/null | cut -f1) (expect 82G)"
echo "prompts     : $(ls prompts/*.pt 2>/dev/null | wc -l | tr -d ' ') caches"
echo "drivers     : $(ls "$ROOT"/runone.sh "$ROOT"/h100_grid.sh "$ROOT"/summarize.py 2>/dev/null | wc -l | tr -d ' ')/3 in $ROOT"
echo "free on nvme: $(df -h --output=avail "$ROOT" | tail -1 | tr -d ' ')"
echo "gpus busy   : $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l | tr -d ' ') processes"

.venv/bin/python - <<'PY' 2>/dev/null
import torch, torchvision, diffusers
import diffusers.loaders.peft            # the import trap 1 breaks
ok = torch.__version__.endswith("+cu129") and torchvision.__version__.endswith("+cu129")
print(f"python env  : torch {torch.__version__}  torchvision {torchvision.__version__}  "
      f"diffusers {diffusers.__version__}  gpus {torch.cuda.device_count()}")
print("              " + ("OK" if ok else "BAD -- both must end in +cu129; see RUNBOOK trap 1"))
PY
[ ${PIPESTATUS[0]:-0} -eq 0 ] || echo "python env  : IMPORT FAILED -- run the block in RUNBOOK trap 1"
