#!/usr/bin/env bash
# Live view of cluster membership. Run alongside run_local.sh in a second pane;
# this is the pane you watch while killing and restarting workers.
#
#   scripts/watch_cluster.sh          follow until Ctrl-C
#   scripts/watch_cluster.sh --once   print one snapshot and exit
set -uo pipefail
cd "$(dirname "$0")/.."

CONTROLLER="${CONTROLLER_URL:-http://127.0.0.1:8000}"
PYTHON="${PYTHON:-.venv/bin/python}"

render() {
  # Heartbeat age is computed against the controller's own as_of, so the numbers
  # do not drift if this machine's clock disagrees with the controller's.
  "$PYTHON" -c '
import json, sys

body = json.load(sys.stdin)
workers = body["workers"]
if not workers:
    print("  (no workers registered)")
for w in workers:
    print("  %-12s %-9s gen %-3d %-20s age %5.1fs  active %d" % (
        w["worker_id"], w["status"], w["generation"], w["address"],
        body["as_of"] - w["last_heartbeat"], w["active_requests"],
    ))
'
}

snapshot() {
  echo "cluster @ $CONTROLLER   $(date +%H:%M:%S)"
  curl -s --max-time 2 "$CONTROLLER/cluster/workers" | render 2>/dev/null \
    || echo "  controller is not answering"
}

if [[ "${1:-}" == "--once" ]]; then
  snapshot
  exit 0
fi

while true; do
  out="$(snapshot)"
  clear
  printf '%s\n' "$out"
  sleep 1
done
