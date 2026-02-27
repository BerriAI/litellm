#!/bin/bash
#
# Memory leak reproduction script for LiteLLM proxy.
#
# Starts a fake OpenAI backend, the LiteLLM proxy with multiple workers,
# a worker monitor, and a load generator. Ctrl-C to stop everything.
#
# Usage: bash run_repro.sh [NUM_WORKERS] [DURATION_SECS] [CONCURRENCY]
#
set -e

NUM_WORKERS=${1:-3}
DURATION=${2:-120}
CONCURRENCY=${3:-50}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROXY_PORT=4000
FAKE_PORT=18080
export PATH="$HOME/.local/bin:$PATH"

echo "=============================================="
echo " LiteLLM Memory Leak Reproduction"
echo "=============================================="
echo " Workers:     $NUM_WORKERS"
echo " Duration:    ${DURATION}s"
echo " Concurrency: $CONCURRENCY"
echo " Proxy port:  $PROXY_PORT"
echo " Fake backend: $FAKE_PORT"
echo "=============================================="
echo ""

cleanup() {
    echo ""
    echo "Cleaning up..."
    kill $FAKE_PID $MONITOR_PID $LOAD_PID 2>/dev/null || true
    # Kill the proxy (and its workers)
    if [ -n "$PROXY_PID" ]; then
        kill -- -$PROXY_PID 2>/dev/null || kill $PROXY_PID 2>/dev/null || true
    fi
    # Kill any leftover litellm/uvicorn processes we spawned
    pkill -f "fake_openai_server" 2>/dev/null || true
    pkill -f "litellm --config.*repro_memory_leak" 2>/dev/null || true
    echo "Done."
}
trap cleanup EXIT INT TERM

# 1) Start fake OpenAI backend
echo "[1/4] Starting fake OpenAI backend on port $FAKE_PORT..."
poetry run python "$SCRIPT_DIR/fake_openai_server.py" &
FAKE_PID=$!
sleep 1
if ! kill -0 $FAKE_PID 2>/dev/null; then
    echo "ERROR: Fake server failed to start"
    exit 1
fi
echo "  -> PID $FAKE_PID"

# 2) Start LiteLLM proxy
echo "[2/4] Starting LiteLLM proxy with $NUM_WORKERS workers on port $PROXY_PORT..."
NUM_WORKERS=$NUM_WORKERS poetry run litellm \
    --config "$SCRIPT_DIR/config.yaml" \
    --port $PROXY_PORT \
    --num_workers $NUM_WORKERS \
    --detailed_debug \
    2>&1 | tee /tmp/litellm_proxy.log &
PROXY_PID=$!
echo "  -> PID $PROXY_PID"

# Wait for proxy to be ready
echo "  Waiting for proxy to start..."
for i in $(seq 1 60); do
    if curl -s http://127.0.0.1:$PROXY_PORT/health > /dev/null 2>&1; then
        echo "  -> Proxy ready after ${i}s"
        break
    fi
    sleep 1
done

# Quick sanity check
echo ""
echo "Sanity check - sending a test request..."
RESP=$(curl -s -w "\n%{http_code}" \
    -X POST http://127.0.0.1:$PROXY_PORT/v1/chat/completions \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer sk-repro-test-1234" \
    -d '{"model":"fake-model","messages":[{"role":"user","content":"hello"}]}')
HTTP_CODE=$(echo "$RESP" | tail -1)
BODY=$(echo "$RESP" | head -n -1)
echo "  HTTP $HTTP_CODE"
if [ "$HTTP_CODE" != "200" ]; then
    echo "  WARNING: Non-200 response. Body: $BODY"
    echo "  (Continuing anyway - errors also exercise the code paths)"
fi
echo ""

# 3) Start worker monitor
echo "[3/4] Starting worker monitor..."
poetry run python "$SCRIPT_DIR/worker_monitor.py" --interval 10 &
MONITOR_PID=$!
echo "  -> PID $MONITOR_PID"
echo ""

# 4) Start load generator
echo "[4/4] Starting load generator ($CONCURRENCY concurrency, ${DURATION}s)..."
echo ""
poetry run python "$SCRIPT_DIR/load_generator.py" \
    --url "http://127.0.0.1:$PROXY_PORT" \
    --api-key "sk-repro-test-1234" \
    --concurrency $CONCURRENCY \
    --duration $DURATION
LOAD_PID=$!

echo ""
echo "=============================================="
echo " Reproduction complete"
echo "=============================================="
echo ""
echo "Check /tmp/litellm_proxy.log for proxy logs"
