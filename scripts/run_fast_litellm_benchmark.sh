#!/usr/bin/env bash
#
# Benchmark: LiteLLM Proxy vs fast-litellm Proxy
#
# This script runs three scenarios and collects results:
#   1. Baseline: uvicorn, 1 worker, no fast-litellm
#   2. Standard: gunicorn, 4 workers, no fast-litellm
#   3. fast-litellm: gunicorn, 4 workers, all Rust features enabled
#
# All scenarios use network_mock: true to measure pure proxy overhead.

set -euo pipefail

RESULTS_DIR="/workspace/benchmark_results"
mkdir -p "$RESULTS_DIR"

BENCHMARK_CMD="poetry run python scripts/benchmark_mock.py --requests 2000 --max-concurrent 100 --runs 3"
HEALTH_URL="http://localhost:4000/health"
CONFIG="benchmark_config.yaml"
MAX_WAIT=120  # seconds to wait for proxy to start

wait_for_proxy() {
    echo "Waiting for proxy to become healthy..."
    local elapsed=0
    while ! curl -sf "$HEALTH_URL" > /dev/null 2>&1; do
        sleep 2
        elapsed=$((elapsed + 2))
        if [ "$elapsed" -ge "$MAX_WAIT" ]; then
            echo "ERROR: Proxy did not become healthy within ${MAX_WAIT}s"
            return 1
        fi
    done
    echo "Proxy is healthy (took ~${elapsed}s)"
    # Extra settle time
    sleep 3
}

kill_proxy() {
    echo "Stopping proxy..."
    # Kill any litellm/gunicorn/uvicorn on port 4000
    local pids
    pids=$(lsof -ti :4000 2>/dev/null || true)
    if [ -n "$pids" ]; then
        echo "  Killing PIDs: $pids"
        echo "$pids" | xargs kill -TERM 2>/dev/null || true
        sleep 3
        # Force kill if still running
        pids=$(lsof -ti :4000 2>/dev/null || true)
        if [ -n "$pids" ]; then
            echo "  Force killing PIDs: $pids"
            echo "$pids" | xargs kill -9 2>/dev/null || true
            sleep 2
        fi
    fi
    echo "Proxy stopped."
}

echo "=========================================="
echo "fast-litellm Proxy Benchmark"
echo "=========================================="
echo "Date: $(date)"
echo "CPUs: $(nproc)"
echo "Memory: $(free -h | awk '/Mem:/{print $2}')"
echo ""

# Ensure no proxy is running
kill_proxy

##############################################
# Scenario 1: Baseline (uvicorn, 1 worker)
##############################################
echo ""
echo "=========================================="
echo "SCENARIO 1: Baseline (uvicorn, 1 worker)"
echo "=========================================="

poetry run litellm --config "$CONFIG" --port 4000 &
PROXY_PID=$!

if wait_for_proxy; then
    echo "Running benchmark..."
    $BENCHMARK_CMD 2>&1 | tee "$RESULTS_DIR/scenario1_baseline.txt"
else
    echo "FAILED: Proxy did not start" | tee "$RESULTS_DIR/scenario1_baseline.txt"
fi

kill_proxy

##############################################
# Scenario 2: Standard (gunicorn, 4 workers)
##############################################
echo ""
echo "=========================================="
echo "SCENARIO 2: Standard (gunicorn, 4 workers)"
echo "=========================================="

poetry run litellm --config "$CONFIG" --port 4000 --run_gunicorn --num_workers 4 &
PROXY_PID=$!

if wait_for_proxy; then
    echo "Running benchmark..."
    $BENCHMARK_CMD 2>&1 | tee "$RESULTS_DIR/scenario2_standard_multiworker.txt"
else
    echo "FAILED: Proxy did not start" | tee "$RESULTS_DIR/scenario2_standard_multiworker.txt"
fi

kill_proxy

##############################################
# Scenario 3: fast-litellm (gunicorn, 4 workers)
##############################################
echo ""
echo "=========================================="
echo "SCENARIO 3: fast-litellm (gunicorn, 4 workers, all Rust features)"
echo "=========================================="

export LITELLM_RUST_RUST_ROUTING=enabled
export LITELLM_RUST_RUST_RATE_LIMITING=enabled
export LITELLM_RUST_RUST_CONNECTION_POOLING=enabled
export LITELLM_RUST_RUST_TOKEN_COUNTING=enabled
export LITELLM_CONFIG_FILE_PATH="$CONFIG"

poetry run gunicorn fast_app:app \
    --preload \
    -w 4 \
    -k uvicorn.workers.UvicornWorker \
    -b 0.0.0.0:4000 &
PROXY_PID=$!

if wait_for_proxy; then
    echo "Running benchmark..."
    $BENCHMARK_CMD 2>&1 | tee "$RESULTS_DIR/scenario3_fast_litellm.txt"
else
    echo "FAILED: Proxy did not start" | tee "$RESULTS_DIR/scenario3_fast_litellm.txt"
fi

kill_proxy

# Unset env vars
unset LITELLM_RUST_RUST_ROUTING
unset LITELLM_RUST_RUST_RATE_LIMITING
unset LITELLM_RUST_RUST_CONNECTION_POOLING
unset LITELLM_RUST_RUST_TOKEN_COUNTING
unset LITELLM_CONFIG_FILE_PATH

echo ""
echo "=========================================="
echo "ALL BENCHMARKS COMPLETE"
echo "=========================================="
echo "Results saved in: $RESULTS_DIR/"
ls -la "$RESULTS_DIR/"
