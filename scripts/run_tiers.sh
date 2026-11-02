#!/usr/bin/env bash
# Bring up a master and one real-model worker per tier, all backed by a local
# Ollama. Ctrl-C tears all four down; Ollama itself is left running.
set -euo pipefail
cd "$(dirname "$0")/.."

source .venv/bin/activate

OLLAMA_URL="${WORKER_OLLAMA_URL:-http://127.0.0.1:11434}"

# Checked up front because a worker cannot tell a missing model from a broken
# one until the first request fails, which is a confusing place to find out.
if ! curl -sf "$OLLAMA_URL/api/version" >/dev/null; then
  echo "ollama is not answering at $OLLAMA_URL - start it with: brew services start ollama" >&2
  exit 1
fi

tiers=("small 8001 qwen2.5:0.5b" "medium 8002 qwen2.5:1.5b" "large 8003 qwen2.5:3b")

for tier in "${tiers[@]}"; do
  read -r _ _ model <<<"$tier"
  if ! ollama list | awk 'NR > 1 { print $1 }' | grep -qx "$model"; then
    echo "model $model is not pulled - run: ollama pull $model" >&2
    exit 1
  fi
done

pids=()
cleanup() { kill "${pids[@]}" 2>/dev/null || true; wait 2>/dev/null || true; }
trap cleanup EXIT INT TERM

MASTER_PORT=8000 python -m master.main & pids+=($!)

# One request at a time per worker: on one laptop the three share the same
# memory and cores, and a real model cannot run eight generations at once.
for tier in "${tiers[@]}"; do
  read -r name port model <<<"$tier"
  WORKER_WORKER_ID="worker-$name" WORKER_PORT="$port" WORKER_BACKEND=ollama \
    WORKER_MODEL="$model" WORKER_MAX_CONCURRENCY=1 WORKER_OLLAMA_URL="$OLLAMA_URL" \
    python -m worker.main & pids+=($!)
done

echo "master :8000  small :8001 (qwen2.5:0.5b)  medium :8002 (qwen2.5:1.5b)  large :8003 (qwen2.5:3b)"
echo "pick a tier by sending its model name, e.g. \"model\": \"qwen2.5:3b\"   (Ctrl-C to stop)"
wait
