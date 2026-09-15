#!/usr/bin/env bash
# ssh/scp helper for the p5.48xlarge box.  `p5.sh <cmd...>` runs remotely,
# `p5.sh --put <local> [remote]` / `--get <remote> [local]` copy.
KEY=${KEY:-/Users/henanwan/Documents/account/579019700964/henanwan/henanwan-us-east-2.pem}
HOST=${HOST:-ubuntu@ec2-18-189-225-222.us-east-2.compute.amazonaws.com}
OPTS=(-i "$KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR)
case ${1:-} in
  --put) shift; scp -q "${OPTS[@]}" "$1" "$HOST:${2:-/opt/dlami/nvme/vdn/}" ;;
  --get) shift; scp -q "${OPTS[@]}" "$HOST:$1" "${2:-.}" ;;
  *)     ssh "${OPTS[@]}" "$HOST" "$@" ;;
esac
