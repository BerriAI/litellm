#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"
mkdir -p "$RESULTS_DIR"

USERS=${1:-200}
SPAWN_RATE=${2:-50}
DURATION=${3:-60s}

echo "=== LiteLLM Sidecar Load Test ==="
echo "Users: $USERS, Spawn Rate: $SPAWN_RATE, Duration: $DURATION"
echo ""

# Function to wait for a service
wait_for_service() {
    local url=$1
    local name=$2
    local max_attempts=${3:-30}
    echo "Waiting for $name at $url..."
    for i in $(seq 1 $max_attempts); do
        if curl -s "$url" > /dev/null 2>&1; then
            echo "$name is ready!"
            return 0
        fi
        sleep 1
    done
    echo "ERROR: $name failed to start"
    return 1
}

# Step 1: Start mock server (if not running)
if ! curl -s http://127.0.0.1:18888/health > /dev/null 2>&1; then
    echo "Starting mock OpenAI server on port 18888..."
    cd "$WORKSPACE" && poetry run python tests/load_tests/mock_openai_server.py &
    MOCK_PID=$!
    wait_for_service "http://127.0.0.1:18888/health" "Mock Server"
else
    echo "Mock server already running on port 18888"
fi

# Step 2: Run baseline test
echo ""
echo "=== BASELINE TEST (Python-only) ==="
echo "Starting proxy on port 4000..."

# Kill any existing proxy on port 4000
lsof -ti:4000 | xargs kill -9 2>/dev/null || true
sleep 1

cd "$WORKSPACE" && poetry run litellm --config tests/load_tests/loadtest_config.yaml --port 4000 &
PROXY_PID=$!
wait_for_service "http://localhost:4000/health/liveliness" "LiteLLM Proxy" 30

echo "Running baseline locust test..."
cd "$WORKSPACE" && poetry run locust -f tests/load_tests/locustfile.py \
    --headless -u "$USERS" -r "$SPAWN_RATE" --run-time "$DURATION" \
    --host http://localhost:4000 \
    --csv "$RESULTS_DIR/baseline" \
    --only-summary 2>&1 | tee "$RESULTS_DIR/baseline_output.txt"

# Kill baseline proxy
kill $PROXY_PID 2>/dev/null || true
sleep 2

# Step 3: Run sidecar test
echo ""
echo "=== SIDECAR TEST (Rust forwarding) ==="

# Start sidecar
SIDECAR_BIN="$WORKSPACE/litellm-sidecar/target/release/litellm-sidecar"
if [ ! -f "$SIDECAR_BIN" ]; then
    echo "Building sidecar..."
    cd "$WORKSPACE/litellm-sidecar" && cargo build --release
fi

echo "Starting sidecar on port 8787..."
SIDECAR_PORT=8787 "$SIDECAR_BIN" &
SIDECAR_PID=$!
wait_for_service "http://127.0.0.1:8787/health" "Sidecar"

echo "Starting proxy on port 4000 (sidecar mode)..."
cd "$WORKSPACE" && USE_SIDECAR=true SIDECAR_PORT=8787 \
    poetry run litellm --config tests/load_tests/loadtest_config_sidecar.yaml --port 4000 &
PROXY_PID=$!
wait_for_service "http://localhost:4000/health/liveliness" "LiteLLM Proxy (Sidecar)" 30

echo "Running sidecar locust test..."
cd "$WORKSPACE" && poetry run locust -f tests/load_tests/locustfile.py \
    --headless -u "$USERS" -r "$SPAWN_RATE" --run-time "$DURATION" \
    --host http://localhost:4000 \
    --csv "$RESULTS_DIR/sidecar" \
    --only-summary 2>&1 | tee "$RESULTS_DIR/sidecar_output.txt"

# Cleanup
kill $PROXY_PID 2>/dev/null || true
kill $SIDECAR_PID 2>/dev/null || true

# Step 4: Compare results
echo ""
echo "=== COMPARISON ==="
echo ""
echo "--- Baseline ---"
tail -5 "$RESULTS_DIR/baseline_output.txt"
echo ""
echo "--- Sidecar ---"
tail -5 "$RESULTS_DIR/sidecar_output.txt"
echo ""
echo "Results saved to $RESULTS_DIR/"
