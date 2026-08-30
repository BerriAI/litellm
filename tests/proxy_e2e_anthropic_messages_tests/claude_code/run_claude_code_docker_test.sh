#!/usr/bin/env bash
set -euo pipefail

# Dockerized Claude Code + LiteLLM integration test (Anthropic focus).
#
# Required:
#   ANTHROPIC_API_KEY
#
# Optional:
#   MODEL_NAME=<litellm model alias>
#   LITELLM_IMAGE=litellm-claude-code-e2e:local
#   CLAUDE_CODE_IMAGE=litellm-claude-code-client:local
#   LITELLM_SKIP_BUILD=true
#   CLAUDE_CODE_SKIP_BUILD=true
#   PROXY_PORT=4000

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="${SCRIPT_DIR}/docker-compose.yaml"
ENV_FILE="${SCRIPT_DIR}/.env"
MODEL_NAME="${MODEL_NAME:-claude-sonnet-4-6}"
LITELLM_MASTER_KEY="${LITELLM_MASTER_KEY:-sk-1234}"
PROXY_PORT="${PROXY_PORT:-4000}"
PROXY_URL="http://127.0.0.1:${PROXY_PORT}"
STARTUP_TIMEOUT_S="${STARTUP_TIMEOUT_S:-300}"
COMPOSE_PROJECT_NAME="${COMPOSE_PROJECT_NAME:-litellm-claude-code-e2e}"
TMP_DIR="$(mktemp -d)"
CLAUDE_CODE_LITELLM_CONFIG="${TMP_DIR}/config.yaml"
HEADERS_FILE="${TMP_DIR}/headers.txt"
BODY_FILE="${TMP_DIR}/body.json"

cleanup() {
  if [[ "${KEEP_CONTAINERS:-}" == "1" ]]; then
    echo "KEEP_CONTAINERS=1 set; leaving docker compose project ${COMPOSE_PROJECT_NAME} running."
    return
  fi
  docker compose -p "${COMPOSE_PROJECT_NAME}" -f "${COMPOSE_FILE}" down -v >/dev/null 2>&1 || true
  rm -rf "${TMP_DIR}"
}
trap cleanup EXIT

if [[ -f "${ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
fi

: "${ANTHROPIC_API_KEY:?ANTHROPIC_API_KEY is required}"

# Create LiteLLM config
cat >"${CLAUDE_CODE_LITELLM_CONFIG}" <<EOF
model_list:
  - model_name: ${MODEL_NAME}
    litellm_params:
      model: anthropic/${MODEL_NAME}
      api_key: os.environ/ANTHROPIC_API_KEY

general_settings:
  forward_client_headers_to_llm_api: true
  master_key: "${LITELLM_MASTER_KEY}"

litellm_settings:
  drop_params: true
  modify_params: true
EOF

export ANTHROPIC_API_KEY
export CLAUDE_CODE_LITELLM_CONFIG
export LITELLM_IMAGE="${LITELLM_IMAGE:-litellm-claude-code-e2e:local}"
export CLAUDE_CODE_IMAGE="${CLAUDE_CODE_IMAGE:-litellm-claude-code-client:local}"
export LITELLM_MASTER_KEY
export MODEL_NAME
export PROXY_PORT

echo "[0/5] Preparing Docker images..."
if [[ "${LITELLM_SKIP_BUILD:-}" == "true" && "${CLAUDE_CODE_SKIP_BUILD:-}" == "true" ]]; then
  echo "Skipping image builds."
else
  docker compose -p "${COMPOSE_PROJECT_NAME}" -f "${COMPOSE_FILE}" build litellm claude-code
fi

echo "[1/5] Starting Postgres and LiteLLM proxy..."
docker compose -p "${COMPOSE_PROJECT_NAME}" -f "${COMPOSE_FILE}" up -d postgres litellm

echo "[2/5] Waiting for LiteLLM readiness at ${PROXY_URL}..."
READY=0
for _ in $(seq 1 "${STARTUP_TIMEOUT_S}"); do
  if curl -fsS --connect-timeout 5 --max-time 15 \
      -H "Authorization: Bearer ${LITELLM_MASTER_KEY}" "${PROXY_URL}/v1/models" >/dev/null 2>&1; then
    READY=1
    break
  fi
  sleep 1
done

if [[ "${READY}" != "1" ]]; then
  echo "LiteLLM failed to become ready."
  docker compose -p "${COMPOSE_PROJECT_NAME}" -f "${COMPOSE_FILE}" logs --tail=200 litellm || true
  exit 1
fi

echo "[3/5] Checking configured model..."
curl -fsS --connect-timeout 5 --max-time 20 \
  -H "Authorization: Bearer ${LITELLM_MASTER_KEY}" "${PROXY_URL}/v1/models" | grep -q "\"${MODEL_NAME}\""

echo "[4/5] Running back-to-back Claude Code requests (V0 Verification)..."
if ! docker compose -p "${COMPOSE_PROJECT_NAME}" -f "${COMPOSE_FILE}" run --rm -T claude-code; then
  echo "Claude Code integration test suite failed."
  docker compose -p "${COMPOSE_PROJECT_NAME}" -f "${COMPOSE_FILE}" logs --tail=200 litellm || true
  exit 1
fi

echo "[5/5] Verifying LiteLLM request headers on Anthropic messages endpoint..."
REQUEST_BODY="$(python3 - "${MODEL_NAME}" <<'PY'
import json
import sys
print(json.dumps({"model": sys.argv[1], "max_tokens": 16, "messages": [{"role": "user", "content": "Respond with the word ok."}]}))
PY
)"

curl -fsS --connect-timeout 5 --max-time 30 -D "${HEADERS_FILE}" -o "${BODY_FILE}" \
  -X POST "${PROXY_URL}/v1/messages" \
  -H "Authorization: Bearer ${LITELLM_MASTER_KEY}" \
  -H "Content-Type: application/json" \
  -H "anthropic-version: 2023-06-01" \
  --data "${REQUEST_BODY}"

grep -qi "^x-litellm-call-id:" "${HEADERS_FILE}" || {
  echo "Missing x-litellm-call-id header."
  exit 1
}

echo "Success: Claude Code (Anthropic only) verified with back-to-back requests and secure isolation."
