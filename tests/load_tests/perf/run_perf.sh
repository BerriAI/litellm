#!/usr/bin/env bash
set -euo pipefail

# Local RPS / latency / profiling harness for the LiteLLM proxy.
#
# Usage:
#   ./run_perf.sh single                # one request + exit
#   ./run_perf.sh batch                 # 100 sequential requests + exit
#   ./run_perf.sh rps [concurrency]     # max-RPS for 30s + exit (default concurrency=50)
#
# Every request carries a unique user-message content (counter + random suffix),
# so no two requests are identical — nothing is served from any request-keyed
# cache path on the proxy side.
#
# Outputs land in tests/load_tests/perf/results/<mode>-<timestamp>/:
#   driver.txt         client-side latency + RPS
#   top_functions.txt  functions ranked by TOTAL / SELF CPU time (text table)
#   profile.ystat      yappi native profile (rank_profile.py consumes this)
#   proxy.log          proxy stdout/stderr
#
# Prereqs:
#   pip install yappi
#   Proxy runs this branch's source via PYTHONPATH=$REPO_ROOT (no install needed).
#   httpx must be importable in the Python used by `python3`.
#
# Override the port with: LITELLM_PERF_PORT=4123 ./run_perf.sh single
#
# Why yappi (not py-spy / cProfile):
#   - py-spy requires sudo on macOS even for subprocess mode.
#   - cProfile profiles only the calling thread AND measures wall time, which
#     misattributes time to async functions while they're suspended.
#   - yappi with clock_type="cpu" is async-aware (suspended coroutines don't
#     accumulate time) and captures every thread. No sudo needed.
#
# Profiling window:
#   The proxy is started under [profile_proxy.py]. Profiling is OFF during
#   startup and toggled on/off via SIGUSR1/SIGUSR2 so import/JIT overhead
#   isn't in the profile.

PERF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$PERF_DIR/../../.." && pwd)"
CONFIG="$PERF_DIR/mock_config.yaml"
DRIVER="$PERF_DIR/driver.py"
PROFILE_PROXY="$PERF_DIR/profile_proxy.py"
RANKER="$PERF_DIR/rank_profile.py"
PORT="${LITELLM_PERF_PORT:-4001}"
WORKERS="${LITELLM_PERF_WORKERS:-1}"
RUNNER="${LITELLM_PERF_RUNNER:-uvicorn}"    # uvicorn | granian | uvicorn+fastpath
URL="http://localhost:$PORT/v1/chat/completions"

MODE="${1:-}"
CONCURRENCY="${2:-50}"

if [[ -z "$MODE" ]]; then
  echo "usage: $0 {single|batch|rps [concurrency]}" >&2
  exit 1
fi

for bin in python3 litellm lsof curl; do
  if ! command -v "$bin" >/dev/null 2>&1; then
    echo "error: '$bin' not found on PATH." >&2
    exit 1
  fi
done
if ! python3 -c "import yappi" >/dev/null 2>&1; then
  echo "error: yappi not installed. Run: pip install yappi" >&2
  exit 1
fi

for f in "$CONFIG" "$DRIVER" "$PROFILE_PROXY" "$RANKER"; do
  [[ -f "$f" ]] || { echo "error: $f missing" >&2; exit 1; }
done

if lsof -nP -iTCP:$PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
  echo "error: port $PORT already in use. Stop the other process and retry." >&2
  exit 1
fi

TS="$(date +%Y%m%d-%H%M%S)"
RESULTS="$PERF_DIR/results/${MODE}-${TS}"
mkdir -p "$RESULTS"

PROXY_PID=""

cleanup() {
  set +e
  if [[ -n "$PROXY_PID" ]] && kill -0 "$PROXY_PID" 2>/dev/null; then
    kill -TERM "$PROXY_PID" 2>/dev/null
    for _ in $(seq 1 20); do
      kill -0 "$PROXY_PID" 2>/dev/null || break
      sleep 0.5
    done
    kill -KILL "$PROXY_PID" 2>/dev/null
  fi
  local stray
  stray="$(lsof -nP -iTCP:$PORT -sTCP:LISTEN -t 2>/dev/null || true)"
  if [[ -n "$stray" ]]; then
    kill -TERM $stray 2>/dev/null
  fi
}
trap cleanup EXIT INT TERM

case "$MODE" in
  single)   DRIVER_ARGS=(single) ;;
  batch)    DRIVER_ARGS=(batch 100) ;;
  rps)      DRIVER_ARGS=(rps "$CONCURRENCY" 30) ;;
  *)
    echo "error: unknown mode '$MODE'. Use single|batch|rps." >&2
    exit 1
    ;;
esac

PROFILE="$RESULTS/profile.ystat"
PROFILING_ENABLED=1

if [[ "$RUNNER" == "uvicorn+fastpath" ]]; then
  # LiteLLM FastAPI app wrapped in the Rust fast-path middleware. Mock-eligible
  # requests (model in model_list with mock_response) are handled entirely in
  # Rust; everything else passes through to the normal LiteLLM pipeline.
  PROFILING_ENABLED=0
  echo ">>> starting proxy: $WORKERS uvicorn workers + Rust fast-path middleware, port=$PORT"
  (
    cd "$REPO_ROOT"
    export PYTHONPATH="$PERF_DIR:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
    export CONFIG_FILE_PATH="$CONFIG"
    exec python3 -m uvicorn fast_path_app:app \
      --host 127.0.0.1 --port "$PORT" --workers "$WORKERS" --loop uvloop
  ) >"$RESULTS/proxy.log" 2>&1 &
  PROXY_PID=$!
elif [[ "$RUNNER" == "granian" ]]; then
  # Granian is a tokio+hyper ASGI server written in Rust. Replaces uvicorn
  # entirely. Cannot be combined with yappi profile_proxy.py (which wraps the
  # litellm CLI to run uvicorn).
  PROFILING_ENABLED=0
  echo ">>> starting proxy: $WORKERS granian worker(s), port=$PORT (profiling DISABLED)"
  (
    cd "$REPO_ROOT"
    export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
    export CONFIG_FILE_PATH="$CONFIG"
    GRANIAN_EXTRA_ARGS=(${LITELLM_PERF_GRANIAN_ARGS:-})
    exec granian --interface asgi --host 127.0.0.1 --port "$PORT" \
      --workers "$WORKERS" --no-ws \
      "${GRANIAN_EXTRA_ARGS[@]}" \
      litellm.proxy.proxy_server:app
  ) >"$RESULTS/proxy.log" 2>&1 &
  PROXY_PID=$!
elif [[ "$WORKERS" -gt 1 ]]; then
  # uvicorn forks fresh Python processes for each worker. yappi hook only
  # lives in the supervisor, which doesn't serve traffic — skip profiling.
  PROFILING_ENABLED=0
  echo ">>> starting proxy: $WORKERS uvicorn workers, port=$PORT (profiling DISABLED)"
  (
    cd "$REPO_ROOT"
    export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
    exec litellm --config "$CONFIG" --port "$PORT" --num_workers "$WORKERS"
  ) >"$RESULTS/proxy.log" 2>&1 &
  PROXY_PID=$!
else
  echo ">>> starting proxy under yappi (port=$PORT, profile=$PROFILE)"
  (
    cd "$REPO_ROOT"
    export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
    exec python3 "$PROFILE_PROXY" "$PROFILE" \
      --config "$CONFIG" --port "$PORT" --num_workers 1
  ) >"$RESULTS/proxy.log" 2>&1 &
  PROXY_PID=$!
fi

echo ">>> waiting for /health/liveness ..."
READY=0
for i in $(seq 1 60); do
  if curl -fsS "http://localhost:$PORT/health/liveness" >/dev/null 2>&1; then
    echo "    ready after ~$(( i * 500 ))ms wall time"
    READY=1
    break
  fi
  if ! kill -0 "$PROXY_PID" 2>/dev/null; then
    echo "error: proxy died before becoming healthy. Tail of proxy.log:" >&2
    tail -n 80 "$RESULTS/proxy.log" >&2
    exit 1
  fi
  sleep 0.5
done
if [[ "$READY" -ne 1 ]]; then
  echo "error: proxy did not become healthy within 30s" >&2
  exit 1
fi

# Warm up (batch / rps only) — happens BEFORE profiling starts, so import /
# JIT / first-hit allocation aren't in the profile.
if [[ "$MODE" != "single" ]]; then
  echo ">>> warmup: 20 varied requests (NOT profiled)"
  (cd "$REPO_ROOT" && LITELLM_PERF_URL="$URL" python3 "$DRIVER" batch 20) >/dev/null
fi

if [[ "$PROFILING_ENABLED" -eq 1 ]]; then
  echo ">>> SIGUSR1 -> yappi.start()"
  kill -USR1 "$PROXY_PID"
  sleep 0.2
fi

echo ">>> running driver: ${DRIVER_ARGS[*]} against $URL"
(cd "$REPO_ROOT" && LITELLM_PERF_URL="$URL" python3 "$DRIVER" "${DRIVER_ARGS[@]}") | tee "$RESULTS/driver.txt"

if [[ "$PROFILING_ENABLED" -eq 1 ]]; then
  echo ">>> SIGUSR2 -> yappi.stop() + dump profile"
  kill -USR2 "$PROXY_PID"
  for _ in $(seq 1 20); do
    [[ -s "$PROFILE" ]] && break
    sleep 0.25
  done
fi

echo ">>> shutting down proxy"
kill -TERM "$PROXY_PID" 2>/dev/null || true
wait "$PROXY_PID" 2>/dev/null || true
PROXY_PID=""

if [[ "$PROFILING_ENABLED" -eq 1 ]]; then
  if [[ -s "$PROFILE" ]]; then
    echo ">>> ranking functions -> $RESULTS/top_functions.txt"
    python3 "$RANKER" "$PROFILE" --top 40 >"$RESULTS/top_functions.txt" 2>&1 || \
      echo "warning: ranker failed; see $RESULTS/top_functions.txt" >&2
  else
    echo "warning: profile $PROFILE is empty or missing — nothing to rank" >&2
  fi
fi

echo ""
echo "=========================================="
echo "results: $RESULTS"
echo "  driver.txt         client-side latency / RPS"
echo "  top_functions.txt  ranked TOTAL / SELF function times"
echo "  profile.ystat      raw yappi profile"
echo "  proxy.log          proxy stdout/stderr"
echo "=========================================="
