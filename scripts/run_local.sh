#!/usr/bin/env bash
# Bring up a controller and two workers locally. Ctrl-C tears all three down.
set -euo pipefail
cd "$(dirname "$0")/.."

source .venv/bin/activate

pids=()
cleanup() { kill "${pids[@]}" 2>/dev/null || true; wait 2>/dev/null || true; }
trap cleanup EXIT INT TERM

CONTROLLER_PORT=8000 python -m controller.main & pids+=($!)
WORKER_WORKER_ID=worker-a WORKER_PORT=8001 python -m worker.main & pids+=($!)
WORKER_WORKER_ID=worker-b WORKER_PORT=8002 python -m worker.main & pids+=($!)

echo "controller :8000  workers :8001 :8002   (Ctrl-C to stop)"
wait
