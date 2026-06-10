#!/usr/bin/env bash
#
# Start the vendored example_openai_endpoint mock server.
#
# Usage:
#   ./tests/mock_endpoints/start_mock_server.sh                 # foreground
#   ./tests/mock_endpoints/start_mock_server.sh --background    # background, prints PID
#
# Environment:
#   PORT                 - port to bind (default 8090)
#   MOCK_SERVER_LOG_FILE - log file when running in background (default /tmp/mock_openai_endpoint.log)
#   MOCK_SERVER_PID_FILE - PID file when running in background (default /tmp/mock_openai_endpoint.pid)
#   MOCK_SERVER_TIMEOUT  - seconds to wait for /chat/completions to respond in --background (default 30)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="${SCRIPT_DIR}/example_openai_endpoint"

PORT="${PORT:-8090}"
LOG_FILE="${MOCK_SERVER_LOG_FILE:-/tmp/mock_openai_endpoint.log}"
PID_FILE="${MOCK_SERVER_PID_FILE:-/tmp/mock_openai_endpoint.pid}"
TIMEOUT="${MOCK_SERVER_TIMEOUT:-30}"

# Prefer the project's venv if it has the deps; otherwise fall back to system python3.
# Callers can override with PYTHON_BIN=/path/to/python.
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
if [[ -z "${PYTHON_BIN:-}" ]]; then
    if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
        PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
    else
        PYTHON_BIN="python3"
    fi
fi

if [[ "${1:-}" == "--background" ]]; then
    PORT="${PORT}" nohup "${PYTHON_BIN}" "${APP_DIR}/main.py" >"${LOG_FILE}" 2>&1 &
    PID=$!
    echo "${PID}" >"${PID_FILE}"
    echo "Started mock server: pid=${PID} port=${PORT} log=${LOG_FILE}"

    # Wait until the server responds (or until we exhaust the timeout).
    for _ in $(seq 1 "${TIMEOUT}"); do
        if curl -fsS -o /dev/null \
            -X POST "http://127.0.0.1:${PORT}/chat/completions" \
            -H 'Authorization: Bearer sk-test' \
            -H 'Content-Type: application/json' \
            -d '{"model":"gpt-4o-mini","messages":[{"role":"user","content":"hi"}]}'; then
            echo "Mock server is ready on http://127.0.0.1:${PORT}"
            exit 0
        fi
        sleep 1
    done

    echo "Mock server failed to become ready within ${TIMEOUT}s. Logs:" >&2
    tail -n 100 "${LOG_FILE}" >&2 || true
    kill "${PID}" 2>/dev/null || true
    exit 1
fi

exec env PORT="${PORT}" "${PYTHON_BIN}" "${APP_DIR}/main.py"
