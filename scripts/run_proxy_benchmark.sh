#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE="$(dirname "$SCRIPT_DIR")"
CONFIG="$WORKSPACE/benchmark_config.yaml"
RESULTS_DIR="$WORKSPACE/benchmark_results"
REQUESTS=2000
CONCURRENT=100
RUNS=3
WORKERS=4

mkdir -p "$RESULTS_DIR"

echo "============================================================"
echo "  LiteLLM Proxy Benchmark: Standard vs fast-litellm"
echo "============================================================"
echo "  Config:       $CONFIG"
echo "  Requests:     $REQUESTS"
echo "  Concurrency:  $CONCURRENT"
echo "  Runs:         $RUNS"
echo "  Workers:      $WORKERS"
echo "  Mode:         network_mock (pure proxy overhead)"
echo ""

wait_for_health() {
    local port=$1
    local timeout=120
    local elapsed=0
    while [ $elapsed -lt $timeout ]; do
        if curl -s "http://localhost:$port/health" > /dev/null 2>&1; then
            return 0
        fi
        sleep 1
        elapsed=$((elapsed + 1))
    done
    return 1
}

# ============================================================
# BENCHMARK 1: Standard LiteLLM Proxy
# ============================================================
echo "############################################################"
echo "  PHASE 1: Standard LiteLLM Proxy"
echo "############################################################"

LITELLM_LOG=ERROR poetry run litellm \
    --config "$CONFIG" \
    --port 4000 \
    --num_workers $WORKERS \
    > "$RESULTS_DIR/standard_proxy.log" 2>&1 &
STANDARD_PID=$!
echo "  Started standard proxy (PID: $STANDARD_PID)"
echo "  Waiting for proxy to be ready..."

if ! wait_for_health 4000; then
    echo "  ERROR: Standard proxy did not start"
    cat "$RESULTS_DIR/standard_proxy.log" | tail -30
    kill $STANDARD_PID 2>/dev/null
    exit 1
fi

echo "  Standard proxy is ready!"
echo "  Running benchmark..."

poetry run python "$SCRIPT_DIR/benchmark_mock.py" \
    --url "http://localhost:4000/chat/completions" \
    --requests $REQUESTS \
    --max-concurrent $CONCURRENT \
    --runs $RUNS \
    2>&1 | tee "$RESULTS_DIR/standard_results.txt"

echo "  Stopping standard proxy..."
kill $STANDARD_PID 2>/dev/null
wait $STANDARD_PID 2>/dev/null || true
sleep 3
echo "  Standard proxy stopped."

# ============================================================
# BENCHMARK 2: fast-litellm Accelerated Proxy
# ============================================================
echo ""
echo "############################################################"
echo "  PHASE 2: fast-litellm Accelerated Proxy"
echo "############################################################"

LITELLM_LOG=ERROR poetry run python -c "
import fast_litellm
import sys
sys.argv = ['litellm', '--config', '$CONFIG', '--port', '4001', '--num_workers', '$WORKERS']
from litellm.proxy.proxy_cli import run_server
run_server()
" > "$RESULTS_DIR/fast_proxy.log" 2>&1 &
FAST_PID=$!
echo "  Started fast-litellm proxy (PID: $FAST_PID)"
echo "  Waiting for proxy to be ready..."

if ! wait_for_health 4001; then
    echo "  ERROR: fast-litellm proxy did not start"
    cat "$RESULTS_DIR/fast_proxy.log" | tail -30
    kill $FAST_PID 2>/dev/null
    exit 1
fi

echo "  fast-litellm proxy is ready!"
echo "  Running benchmark..."

poetry run python "$SCRIPT_DIR/benchmark_mock.py" \
    --url "http://localhost:4001/chat/completions" \
    --requests $REQUESTS \
    --max-concurrent $CONCURRENT \
    --runs $RUNS \
    2>&1 | tee "$RESULTS_DIR/fast_results.txt"

echo "  Stopping fast-litellm proxy..."
kill $FAST_PID 2>/dev/null
wait $FAST_PID 2>/dev/null || true
echo "  fast-litellm proxy stopped."

echo ""
echo "============================================================"
echo "  Benchmark complete! Results saved to: $RESULTS_DIR/"
echo "============================================================"
