#!/usr/bin/env bash
#
# End-to-end manual test for per-MCP RPM rate limiting.
#
# What it does, with no other setup required:
#   1. Writes a throwaway proxy config with one stdio MCP server (the
#      `uvx mcp-server-fetch` server, aliased "fetch_mcp").
#   2. Boots the proxy in the background and waits until it is ready.
#   3. Generates two keys, both with full access to fetch_mcp:
#        - "limited" key:  mcp_rpm_limit caps "fetch_mcp" at 2 req/min.
#        - "control" key:  mcp_rpm_limit caps a DIFFERENT server name
#                          ("other_mcp") at 2 req/min, so calls to fetch_mcp
#                          are uncapped.
#   4. Fires 4 fetch_mcp calls with each key. The limited key must trip at the
#      3rd call (429); the control key must never be rate limited. This proves
#      the limit is keyed per MCP server name, not globally per key.
#
# Using one physical server with two keys (rather than two servers) keeps the
# test deterministic: a single server's access resolution is exercised, and the
# only variable between the two runs is which server name the key's limit
# targets.
#
# Usage:
#   ./scripts/test_mcp_rpm_limit.sh
#
# Requirements: jq, curl, uvx (for the stdio fetch MCP server), and a reachable
# DATABASE_URL (read from .env).

set -uo pipefail

# --- locate repo root (this script lives in <repo>/scripts) -------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

RPM_LIMIT=2
SERVER="fetch_mcp"
OTHER_SERVER="other_mcp"
WORKDIR="$(mktemp -d)"
CONFIG="${WORKDIR}/mcp_rpm_test_config.yaml"
PROXY_LOG="${WORKDIR}/proxy.log"
PROXY_PID=""

# --- load secrets (DATABASE_URL, provider keys, master key) -------------------
# Parse .env line-by-line and export each KEY=VALUE verbatim. We avoid
# `source`-ing it because some values contain characters (e.g. '#') that the
# shell would try to execute.
if [[ -f .env ]]; then
  while IFS= read -r line; do
    [[ "${line}" =~ ^[A-Za-z_][A-Za-z0-9_]*= ]] || continue
    key="${line%%=*}"
    val="${line#*=}"
    val="${val%\"}"; val="${val#\"}"   # strip surrounding double quotes
    val="${val%\'}"; val="${val#\'}"   # strip surrounding single quotes
    export "${key}=${val}"
  done < .env
fi
MASTER_KEY="${LITELLM_MASTER_KEY:-sk-1234}"

# --- pick a free TCP port (start at 4000) so we never collide with a proxy
# already running from a prior session -----------------------------------------
PORT=""
for candidate in $(seq 4000 4050); do
  if ! lsof -iTCP:"${candidate}" -sTCP:LISTEN -n -P >/dev/null 2>&1; then
    PORT="${candidate}"
    break
  fi
done
if [[ -z "${PORT}" ]]; then
  echo "ERROR: no free port found in 4000-4050"
  exit 1
fi
BASE="http://localhost:${PORT}"
echo ">> using port ${PORT}"

cleanup() {
  if [[ -n "${PROXY_PID}" ]] && kill -0 "${PROXY_PID}" 2>/dev/null; then
    echo ">> stopping proxy (pid ${PROXY_PID})"
    kill "${PROXY_PID}" 2>/dev/null
    # kill the whole process group in case uvicorn spawned children
    pkill -P "${PROXY_PID}" 2>/dev/null
    wait "${PROXY_PID}" 2>/dev/null
  fi
  echo ">> logs kept at: ${PROXY_LOG}"
}
trap cleanup EXIT

require() { command -v "$1" >/dev/null 2>&1 || { echo "ERROR: '$1' is required but not installed"; exit 1; }; }
require jq
require curl

# --- 1. write throwaway config ------------------------------------------------
cat > "${CONFIG}" <<YAML
model_list: []

mcp_servers:
  ${SERVER}:
    transport: "stdio"
    command: "uvx"
    args: ["mcp-server-fetch", "--ignore-robots-txt"]
    alias: "${SERVER}"
    allow_all_keys: true

general_settings:
  master_key: ${MASTER_KEY}
  store_model_in_db: false
YAML

echo ">> config written to ${CONFIG}"

# --- 2. start proxy -----------------------------------------------------------
echo ">> starting proxy on :${PORT} (log: ${PROXY_LOG})"
# Put the repo root first on PYTHONPATH so the local litellm source shadows any
# stale `litellm` installed in site-packages (running the cli as a script puts
# litellm/proxy/ on sys.path instead of the repo root).
PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" python litellm/proxy/proxy_cli.py \
  --config "${CONFIG}" \
  --port "${PORT}" \
  --detailed_debug \
  --use_v2_migration_resolver > "${PROXY_LOG}" 2>&1 &
PROXY_PID=$!

echo -n ">> waiting for readiness"
ready=false
for _ in $(seq 1 90); do
  if curl -sf "${BASE}/health/readiness" >/dev/null 2>&1; then
    ready=true
    break
  fi
  if ! kill -0 "${PROXY_PID}" 2>/dev/null; then
    echo ""
    echo "ERROR: proxy process died during startup. Tail of log:"
    tail -n 40 "${PROXY_LOG}"
    exit 1
  fi
  echo -n "."
  sleep 1
done
echo ""
if [[ "${ready}" != "true" ]]; then
  echo "ERROR: proxy did not become ready in time. Tail of log:"
  tail -n 40 "${PROXY_LOG}"
  exit 1
fi
echo ">> proxy is ready"

# --- 3. generate the two keys -------------------------------------------------
# Both keys get explicit access to fetch_mcp. They differ only in which server
# name their mcp_rpm_limit targets.
generate_key() {
  local rpm_target="$1"
  curl -sf -X POST "${BASE}/key/generate" \
    -H "Authorization: Bearer ${MASTER_KEY}" \
    -H "Content-Type: application/json" \
    -d "{\"mcp_rpm_limit\": {\"${rpm_target}\": ${RPM_LIMIT}}, \"object_permission\": {\"mcp_servers\": [\"${SERVER}\"]}}" \
    | jq -r '.key'
}

echo ">> generating limited key  (mcp_rpm_limit {\"${SERVER}\": ${RPM_LIMIT}})"
LIMITED_KEY="$(generate_key "${SERVER}")"
echo ">> generating control key  (mcp_rpm_limit {\"${OTHER_SERVER}\": ${RPM_LIMIT}})"
CONTROL_KEY="$(generate_key "${OTHER_SERVER}")"
for k in "${LIMITED_KEY}" "${CONTROL_KEY}"; do
  if [[ -z "${k}" || "${k}" == "null" ]]; then
    echo "ERROR: /key/generate failed. Is DATABASE_URL set and reachable?"
    tail -n 40 "${PROXY_LOG}"
    exit 1
  fi
done
echo ">> limited key: ${LIMITED_KEY:0:12}...  control key: ${CONTROL_KEY:0:12}..."

# --- discover a real tool name on the server ----------------------------------
TOOL_NAME="$(curl -sf "${BASE}/mcp-rest/tools/list?server_id=${SERVER}" \
  -H "Authorization: Bearer ${LIMITED_KEY}" 2>/dev/null \
  | jq -r '.tools[0].name // empty')"
if [[ -z "${TOOL_NAME}" ]]; then
  echo ">> could not auto-discover a tool name; falling back to 'fetch'"
  TOOL_NAME="fetch"
fi
echo ">> using tool: ${TOOL_NAME}"

# tool_call: the server alias is accepted directly as server_id. Point the fetch
# tool at the proxy's own health endpoint so the call is fast and always
# reachable; that way a non-429 response unambiguously means "the rate limiter
# let this through" rather than "the upstream fetch flaked".
FETCH_URL="${BASE}/health/readiness"
call_mcp() {
  local key="$1"
  curl -s -o /dev/null -w "%{http_code}" \
    -X POST "${BASE}/mcp-rest/tools/call" \
    -H "Authorization: Bearer ${key}" \
    -H "Content-Type: application/json" \
    -d "{\"server_id\": \"${SERVER}\", \"name\": \"${TOOL_NAME}\", \"arguments\": {\"url\": \"${FETCH_URL}\", \"max_length\": 100}}"
}

# --- 4a. limited key: expect 429 once the cap is exceeded ---------------------
echo ""
echo "=== limited key (caps ${SERVER} at ${RPM_LIMIT}/min) ==="
limited_codes=()
for i in 1 2 3 4; do
  code="$(call_mcp "${LIMITED_KEY}")"
  limited_codes+=("${code}")
  echo "   call ${i} -> HTTP ${code}"
done

# --- 4b. control key: caps a different server name, so fetch_mcp is uncapped --
echo ""
echo "=== control key (caps ${OTHER_SERVER}, so ${SERVER} is uncapped) ==="
control_codes=()
for i in 1 2 3 4; do
  code="$(call_mcp "${CONTROL_KEY}")"
  control_codes+=("${code}")
  echo "   call ${i} -> HTTP ${code}"
done

# --- evaluate -----------------------------------------------------------------
echo ""
echo "=== result ==="
pass=true

# limited: first two must NOT be 429, last two MUST be 429
[[ "${limited_codes[0]}" != "429" ]] || { echo "FAIL: limited call 1 was rate limited"; pass=false; }
[[ "${limited_codes[1]}" != "429" ]] || { echo "FAIL: limited call 2 was rate limited"; pass=false; }
[[ "${limited_codes[2]}" == "429" ]] || { echo "FAIL: limited call 3 was NOT rate limited (got ${limited_codes[2]})"; pass=false; }
[[ "${limited_codes[3]}" == "429" ]] || { echo "FAIL: limited call 4 was NOT rate limited (got ${limited_codes[3]})"; pass=false; }

# control: none may be 429
for c in "${control_codes[@]}"; do
  [[ "${c}" != "429" ]] || { echo "FAIL: control key was rate limited on ${SERVER} (got ${c})"; pass=false; }
done

if [[ "${pass}" == "true" ]]; then
  echo "PASS: ${SERVER} tripped at call 3 (429) under the limited key; the control key (which caps ${OTHER_SERVER}) was never rate limited on ${SERVER}."
  exit 0
else
  echo "See proxy log for detail: ${PROXY_LOG}"
  exit 1
fi
