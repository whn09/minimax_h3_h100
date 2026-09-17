#!/usr/bin/env bash
# One wrapper for the container route. Run it on the box, from anywhere.
#
#   bash h3.sh probe            # what does the published image already have? ~3 min, pulls it
#   bash h3.sh build            # bake the diffusion extra + the pinned revision.   ~6 min
#   bash h3.sh weights t2va,ref2va                                                 # ~25 min
#   bash h3.sh serve ref2va 768        # runs sglang_ref2va_arm.sh inside, detached
#   LORA=$L/ref2v_8step.safetensors REFEDGE=1024 bash h3.sh serve ref2va 768   # env passes through
#   bash h3.sh logs                    # follow it
#   bash h3.sh exec sglang_ref2va.py 768:8 ref=/opt/dlami/nvme/vdn/keyframes/ref.png
#   bash h3.sh sh                      # interactive shell, same mounts
#   bash h3.sh stop
#
# THE ONE DESIGN DECISION: /opt/dlami/nvme is bind-mounted at the SAME PATH inside the container.
# Not /workspace, not /data. Every path in this repo -- the HF cache, the fused overlay cache, the
# reference images, the mp4s the server writes, the LoRA files, RUNBOOK's copy-paste commands -- is
# an absolute /opt/dlami/nvme/... path, and H3 takes its conditioning inputs as URIs the *worker*
# resolves. Mounting elsewhere would mean translating every one of them and would make a request
# that works outside the container fail inside it with "file not found" pointing at a path that
# plainly exists. Same path in both worlds, no translation, and scripts/_env.sh takes care of the
# rest (it activates the venv only if there is one).
#
# WHAT THIS DOES NOT DO: bake in weights. A ~200 GiB image is slower to move than the weights are
# to download, and /opt/dlami/nvme is instance store, so a fresh box pays the download either way.
# The image is worth ~10 minutes and four traps, not the 25-minute download -- see the Dockerfile.
set -euo pipefail

IMAGE=${IMAGE:-minimax-h3:local}
BASE=${BASE:-lmsysorg/sglang:dev}
NVME=${NVME:-/opt/dlami/nvme}
NAME=${NAME:-h3}
DOCKER=${DOCKER:-docker}
# -t only when there is a terminal: `docker run -it` fails with "the input device is not a TTY"
# under nohup, in a pipeline, or over `ssh box 'bash h3.sh exec ...'`.
TTY=(-i); [ -t 1 ] && TTY=(-it)
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# --shm-size 32g and --ipc=host are upstream's own H3 numbers (docs/src/snippets/_deployment.jsx):
#   eight ranks exchange tensors through shared memory, and the 64 MB docker default is where a
#   multi-rank diffusion worker dies with a bare "Bus error" or a NCCL timeout.
# --network host rather than -p: three servers on 30010/30011/30012 and a client on the same box.
# --ulimit memlock=-1: pinned host memory for the frames-to-host copy and for NCCL registration.
RUNFLAGS=(--gpus all --ipc=host --shm-size 32g --network host
          --ulimit memlock=-1:-1 --ulimit stack=67108864:67108864
          -v "$NVME:$NVME" -w "$NVME/vdn"
          -e "HF_HOME=$NVME/vdn/hf" -e "SGLANG_DIFFUSION_CACHE_ROOT=$NVME/sglang/cache")

mkdir -p "$NVME/vdn/hf" "$NVME/sglang/cache" "$NVME/sglang/logs"

# Pass the arm scripts' knobs through automatically. Every one of these is read as an environment
# variable by sglang_*_arm.sh or by the client scripts, and `LORA=... bash h3.sh serve ref2va` looks
# like it should work -- so make it work, rather than have the flag silently not reach the server
# and produce a clean-looking run of the wrong arm. Only variables actually set are forwarded, so
# the scripts' own defaults still apply.
# Seeded rather than empty, for two reasons: `"${arr[@]}"` on an empty array is an *unbound
# variable* under `set -u` before bash 4.4 (this dies on macOS's bash 3.2, and the box's bash 5 only
# hides it), and IN_H3_CONTAINER gives anything running inside a way to know which route it is on.
ENVFLAGS=(-e IN_H3_CONTAINER=1)
for v in QUANT LORA LORA_ALPHA MERGED REFEDGE GPUS FRAMES SEED PORT LOGTAG MODEL \
         WEIGHTS OUTDIR PROMPT SGLANG_DISABLE_COSMOS3_GUARDRAILS; do
  # `if`, not `[ ] && ...`: a for loop's status is its last command's, so a final unset variable
  # would make the loop return 1 and `set -e` would exit the script right here.
  if [ -n "${!v-}" ]; then ENVFLAGS+=(-e "$v=${!v}"); fi
done

cmd=${1:?probe|build|weights|serve|exec|sh|logs|stop}
shift || true

case "$cmd" in

probe)
  # Ask the published image what it has before building anything on top of it. If VDN and the
  # diffusion pipeline both import here, the overlay is unnecessary and `serve` can run $BASE
  # directly (IMAGE=$BASE bash h3.sh serve ...). Expect them NOT to: the image installs the LLM
  # extras only and bundles a source tree whose revision is as old as the tag.
  img=${1:-$BASE}
  $DOCKER pull "$img"
  $DOCKER run --rm --entrypoint bash "$img" -lc '
    python -c "import sglang, torch; print(\"sglang\", sglang.__version__, \"torch\", torch.__version__)"
    python -c "import sys; print(\"python\", sys.version.split()[0])"
    for m in sglang.multimodal_gen \
             sglang.multimodal_gen.configs.pipeline_configs.minimax_h3 \
             sglang.multimodal_gen.configs.pipeline_configs.minimax_h3_vdn; do
      python -c "import $m" 2>/dev/null && echo "have  $m" || echo "MISSING $m"
    done
    for b in ffmpeg ffprobe nvcc cargo git; do
      command -v $b >/dev/null && echo "have  $b" || echo "MISSING $b"
    done
    echo "CUDA_HOME=${CUDA_HOME:-unset}; /usr/local/cuda $( [ -d /usr/local/cuda ] && echo present || echo absent)"
    ls -d /sgl-workspace/sglang 2>/dev/null && git -C /sgl-workspace/sglang log -1 --format="bundled source: %H %cd"
  '
  ;;

build)
  # The build context is just docker/, because the Dockerfile has no COPY at all: the scripts
  # arrive through the bind mount, so editing one does not invalidate a layer or need a rebuild.
  $DOCKER build --build-arg "BASE=$BASE" ${SGLANG_REV:+--build-arg "SGLANG_REV=$SGLANG_REV"} \
    -t "$IMAGE" -f "$HERE/Dockerfile" "$HERE"
  echo "built $IMAGE"
  ;;

weights)
  # Inside the image, so the host needs no python, no uv and no venv -- only docker. Same
  # fetch_weights.sh the venv route calls.
  WEIGHTS=${1:?vdn,t2va,ref2va,fl2va,transformer_ref,all,none}
  $DOCKER run --rm -e "WEIGHTS=$WEIGHTS" "${RUNFLAGS[@]}" "${ENVFLAGS[@]}" --entrypoint bash \
    "$IMAGE" "$NVME/vdn/fetch_weights.sh"
  ;;

serve)
  # `serve <arm> [edge] [extra sglang flags...]` -- arm is vdn|t2va|ref2va, i.e. which of the three
  # arm scripts to run. Detached, with the server log going to the same $NVME/sglang/logs the venv
  # route writes to, so RUNBOOK's tail commands are unchanged.
  arm=${1:?vdn|t2va|ref2va}; shift || true
  case "$arm" in
    vdn)    script=sglang_arm.sh        ;;
    t2va)   script=sglang_base_arm.sh   ;;
    ref2va) script=sglang_ref2va_arm.sh ;;
    *) echo "arm must be vdn, t2va or ref2va"; exit 2 ;;
  esac
  $DOCKER rm -f "$NAME-$arm" >/dev/null 2>&1 || true
  $DOCKER run -d --name "$NAME-$arm" "${RUNFLAGS[@]}" "${ENVFLAGS[@]}" --entrypoint bash "$IMAGE" \
    "$NVME/vdn/$script" serve "$@"
  echo "started $NAME-$arm; follow it with: bash h3.sh logs $arm"
  ;;

exec)
  # Anything else, in a *new* container beside the server (the client scripts talk over
  # 127.0.0.1, which --network host makes the same loopback). `exec <script.py|cmd> [args...]`
  target=${1:?a script under /opt/dlami/nvme/vdn, or a command}
  shift || true
  if [ -f "$NVME/vdn/$target" ]; then
    case "$target" in
      *.py) set -- python "$NVME/vdn/$target" "$@" ;;
      *)    set -- bash   "$NVME/vdn/$target" "$@" ;;
    esac
  else
    set -- "$target" "$@"
  fi
  $DOCKER run --rm "${TTY[@]}" "${RUNFLAGS[@]}" "${ENVFLAGS[@]}" --entrypoint "$1" "$IMAGE" "${@:2}"
  ;;

sh)
  $DOCKER run --rm "${TTY[@]}" "${RUNFLAGS[@]}" "${ENVFLAGS[@]}" --entrypoint bash "$IMAGE"
  ;;

logs)
  arm=${1:-ref2va}
  $DOCKER logs -f "$NAME-$arm"
  ;;

stop)
  # Remove the containers. NOT `sglang_*_arm.sh stop`: that is a pkill, and without --pid=host a
  # second container has its own PID namespace, so the pkill matches nothing and reports success
  # while the server keeps holding all eight cards.
  for a in vdn t2va ref2va; do $DOCKER rm -f "$NAME-$a" >/dev/null 2>&1 && echo "removed $NAME-$a"; done
  exit 0
  ;;

*) echo "unknown command '$cmd': probe|build|weights|serve|exec|sh|logs|stop"; exit 2 ;;
esac
