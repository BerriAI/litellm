#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"
mkdir -p "$RESULTS_DIR"

USERS=${1:-5000}
SPAWN_RATE=${2:-500}
DURATION=${3:-60s}

echo "================================================================"
echo "  LiteLLM Performance Comparison: Baseline vs Optimized"
echo "  Users: $USERS | Spawn Rate: $SPAWN_RATE | Duration: $DURATION"
echo "================================================================"

wait_for_service() {
    local url=$1
    local name=$2
    local max_attempts=${3:-60}
    echo "  Waiting for $name at $url..."
    for i in $(seq 1 $max_attempts); do
        if curl -s "$url" > /dev/null 2>&1; then
            echo "  $name is ready!"
            return 0
        fi
        sleep 1
    done
    echo "  ERROR: $name failed to start after $max_attempts seconds"
    return 1
}

kill_port() {
    local port=$1
    local pids=$(lsof -ti:$port 2>/dev/null || true)
    if [ -n "$pids" ]; then
        echo "$pids" | xargs kill -9 2>/dev/null || true
        sleep 1
    fi
}

# Start mock server
if ! curl -s http://127.0.0.1:18888/health > /dev/null 2>&1; then
    echo ""
    echo "Starting mock OpenAI server on port 18888..."
    cd "$WORKSPACE" && poetry run python tests/load_tests/mock_openai_server.py &
    MOCK_PID=$!
    wait_for_service "http://127.0.0.1:18888/health" "Mock Server" 15
else
    echo "Mock server already running on port 18888"
fi

# ---- PHASE 1: BASELINE (stash current changes) ----
echo ""
echo "================================================================"
echo "  PHASE 1: BASELINE (pre-optimization code)"
echo "================================================================"

cd "$WORKSPACE"
CURRENT_COMMIT=$(git rev-parse HEAD)
PARENT_COMMIT=$(git rev-parse HEAD~1)

echo "  Current commit (optimized): ${CURRENT_COMMIT:0:12}"
echo "  Parent commit (baseline):   ${PARENT_COMMIT:0:12}"
echo "  Checking out baseline..."
git checkout "$PARENT_COMMIT" -- \
    litellm/integrations/prometheus.py \
    litellm/litellm_core_utils/litellm_logging.py \
    litellm/litellm_core_utils/llm_cost_calc/tool_call_cost_tracking.py \
    litellm/litellm_core_utils/safe_json_dumps.py \
    litellm/proxy/litellm_pre_call_utils.py \
    litellm/proxy/spend_tracking/spend_tracking_utils.py \
    litellm/router.py \
    litellm/router_strategy/simple_shuffle.py \
    litellm/types/integrations/prometheus.py \
    2>/dev/null

kill_port 4000

echo "  Starting proxy on port 4000 (baseline)..."
cd "$WORKSPACE" && poetry run litellm --config tests/load_tests/loadtest_config_perf.yaml --port 4000 > "$RESULTS_DIR/baseline_proxy.log" 2>&1 &
PROXY_PID=$!
wait_for_service "http://localhost:4000/health/liveliness" "LiteLLM Proxy (baseline)" 60

echo "  Running baseline locust test..."
cd "$WORKSPACE" && poetry run locust -f tests/load_tests/locustfile.py \
    --headless -u "$USERS" -r "$SPAWN_RATE" --run-time "$DURATION" \
    --host http://localhost:4000 \
    --csv "$RESULTS_DIR/baseline" \
    --only-summary 2>&1 | tee "$RESULTS_DIR/baseline_output.txt"

kill $PROXY_PID 2>/dev/null || true
sleep 2
kill_port 4000

# ---- PHASE 2: OPTIMIZED (restore current changes) ----
echo ""
echo "================================================================"
echo "  PHASE 2: OPTIMIZED (with performance fixes)"
echo "================================================================"

cd "$WORKSPACE"
echo "  Restoring optimized code..."
git checkout "$CURRENT_COMMIT" -- \
    litellm/integrations/prometheus.py \
    litellm/litellm_core_utils/litellm_logging.py \
    litellm/litellm_core_utils/llm_cost_calc/tool_call_cost_tracking.py \
    litellm/litellm_core_utils/safe_json_dumps.py \
    litellm/proxy/litellm_pre_call_utils.py \
    litellm/proxy/spend_tracking/spend_tracking_utils.py \
    litellm/router.py \
    litellm/router_strategy/simple_shuffle.py \
    litellm/types/integrations/prometheus.py \
    2>/dev/null

kill_port 4000

echo "  Starting proxy on port 4000 (optimized)..."
cd "$WORKSPACE" && poetry run litellm --config tests/load_tests/loadtest_config_perf.yaml --port 4000 > "$RESULTS_DIR/optimized_proxy.log" 2>&1 &
PROXY_PID=$!
wait_for_service "http://localhost:4000/health/liveliness" "LiteLLM Proxy (optimized)" 60

echo "  Running optimized locust test..."
cd "$WORKSPACE" && poetry run locust -f tests/load_tests/locustfile.py \
    --headless -u "$USERS" -r "$SPAWN_RATE" --run-time "$DURATION" \
    --host http://localhost:4000 \
    --csv "$RESULTS_DIR/optimized" \
    --only-summary 2>&1 | tee "$RESULTS_DIR/optimized_output.txt"

kill $PROXY_PID 2>/dev/null || true

# ---- PHASE 3: COMPARE ----
echo ""
echo "================================================================"
echo "  RESULTS COMPARISON"
echo "================================================================"
cd "$WORKSPACE" && poetry run python tests/load_tests/compare_perf_results.py "$RESULTS_DIR"
