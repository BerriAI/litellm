#!/usr/bin/env bash
set -euo pipefail

# Dockerized Claude Code + LiteLLM integration test.
#
# Required:
#   ANTHROPIC_API_KEY or tests/proxy_e2e_anthropic_messages_tests/claude_code/.env
#
# Optional:
#   MODEL_NAME=claude-sonnet-4-6
#   LITELLM_UPSTREAM_MODEL=anthropic/${MODEL_NAME}
#   LITELLM_IMAGE=litellm-claude-code-e2e:local
#   CLAUDE_CODE_IMAGE=litellm-claude-code-client:local
#   LITELLM_SKIP_BUILD=true
#   CLAUDE_CODE_SKIP_BUILD=true
#   CLAUDE_CODE_VERSION=latest
#   PROXY_PORT=4000

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="${SCRIPT_DIR}/docker-compose.yaml"
ENV_FILE="${SCRIPT_DIR}/.env"
MODEL_NAME="${MODEL_NAME:-claude-sonnet-4-6}"
LITELLM_UPSTREAM_MODEL="${LITELLM_UPSTREAM_MODEL:-anthropic/${MODEL_NAME}}"
LITELLM_MASTER_KEY="${LITELLM_MASTER_KEY:-sk-darcy-sf-onsite-interview}"
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
    echo "Generated config left at ${CLAUDE_CODE_LITELLM_CONFIG}"
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

if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
  echo "ANTHROPIC_API_KEY is required in the environment or ${ENV_FILE}"
  exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required"
  exit 1
fi

if ! docker compose version >/dev/null 2>&1; then
  echo "docker compose v2 is required"
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required"
  exit 1
fi

is_port_available() {
  python3 - "$1" <<'PY'
import socket
import sys

port = int(sys.argv[1])
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    sock.bind(("127.0.0.1", port))
    print("yes")
except OSError:
    print("no")
finally:
    sock.close()
PY
}

if [[ "$(is_port_available "${PROXY_PORT}")" != "yes" ]]; then
  echo "Port ${PROXY_PORT} is in use. Searching for a free port..."
  FOUND_PORT=""
  for candidate in $(seq 4001 4050); do
    if [[ "$(is_port_available "${candidate}")" == "yes" ]]; then
      FOUND_PORT="${candidate}"
      break
    fi
  done
  if [[ -z "${FOUND_PORT}" ]]; then
    echo "No free local port found between 4001-4050."
    exit 1
  fi
  PROXY_PORT="${FOUND_PORT}"
  PROXY_URL="http://127.0.0.1:${PROXY_PORT}"
  echo "Using fallback PROXY_PORT=${PROXY_PORT}"
fi

cat >"${CLAUDE_CODE_LITELLM_CONFIG}" <<EOF
model_list:
  - model_name: ${MODEL_NAME}
    litellm_params:
      model: ${LITELLM_UPSTREAM_MODEL}
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
  echo "Skipping image builds (LITELLM_SKIP_BUILD=true and CLAUDE_CODE_SKIP_BUILD=true)."
elif [[ "${LITELLM_SKIP_BUILD:-}" == "true" ]]; then
  docker compose -p "${COMPOSE_PROJECT_NAME}" -f "${COMPOSE_FILE}" build claude-code
elif [[ "${CLAUDE_CODE_SKIP_BUILD:-}" == "true" ]]; then
  docker compose -p "${COMPOSE_PROJECT_NAME}" -f "${COMPOSE_FILE}" build litellm
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
  echo "LiteLLM failed to become ready within ${STARTUP_TIMEOUT_S}s"
  docker compose -p "${COMPOSE_PROJECT_NAME}" -f "${COMPOSE_FILE}" logs --tail=200 litellm || true
  exit 1
fi

echo "[3/5] Checking configured model..."
curl -fsS --connect-timeout 5 --max-time 20 \
  -H "Authorization: Bearer ${LITELLM_MASTER_KEY}" "${PROXY_URL}/v1/models" | grep -q "\"${MODEL_NAME}\""

echo "[4/5] Running back-to-back Claude Code requests in an isolated non-root container..."
FIRST_RESPONSE="$(
docker compose -p "${COMPOSE_PROJECT_NAME}" -f "${COMPOSE_FILE}" run --rm \
  -e CLAUDE_CODE_PROMPT="Respond with exactly this text and nothing else: Hello from LiteLLM Claude Code request one." \
  -e CLAUDE_CODE_OUTPUT_FILE="/tmp/claude-output-1.txt" \
  claude-code
)"

SECOND_RESPONSE="$(
docker compose -p "${COMPOSE_PROJECT_NAME}" -f "${COMPOSE_FILE}" run --rm \
  -e CLAUDE_CODE_PROMPT="Respond with exactly this text and nothing else: Hello from LiteLLM Claude Code request two." \
  -e CLAUDE_CODE_OUTPUT_FILE="/tmp/claude-output-2.txt" \
  claude-code
)"

[[ -n "${FIRST_RESPONSE}" ]] || {
  echo "First Claude Code request produced no response output."
  exit 1
}
[[ -n "${SECOND_RESPONSE}" ]] || {
  echo "Second Claude Code request produced no response output."
  exit 1
}

printf '%s\n' "${FIRST_RESPONSE}" | grep -qi "request one" || {
  echo "First Claude Code response did not contain expected content."
  printf '%s\n' "${FIRST_RESPONSE}" | sed -n '1,40p'
  exit 1
}
printf '%s\n' "${SECOND_RESPONSE}" | grep -qi "request two" || {
  echo "Second Claude Code response did not contain expected content."
  printf '%s\n' "${SECOND_RESPONSE}" | sed -n '1,40p'
  exit 1
}

echo "[5/5] Verifying LiteLLM request headers on Anthropic messages endpoint..."
REQUEST_BODY="$(python3 - "${MODEL_NAME}" <<'PY'
import json
import sys

print(
    json.dumps(
        {
            "model": sys.argv[1],
            "max_tokens": 16,
            "messages": [
                {
                    "role": "user",
                    "content": "Respond with the word ok.",
                }
            ],
        }
    )
)
PY
)"

curl -fsS --connect-timeout 5 --max-time 30 -D "${HEADERS_FILE}" -o "${BODY_FILE}" \
  -X POST "${PROXY_URL}/v1/messages" \
  -H "Authorization: Bearer ${LITELLM_MASTER_KEY}" \
  -H "Content-Type: application/json" \
  -H "anthropic-version: 2023-06-01" \
  --data "${REQUEST_BODY}"

grep -qi "^x-litellm-call-id:" "${HEADERS_FILE}" || {
  echo "Missing x-litellm-call-id header; proxy record signal not found."
  sed -n '1,80p' "${HEADERS_FILE}"
  sed -n '1,80p' "${BODY_FILE}"
  exit 1
}

if ! grep -qi "^x-litellm-response-cost:" "${HEADERS_FILE}" && \
   ! grep -qi "^x-litellm-response-cost-original:" "${HEADERS_FILE}"; then
  echo "Missing LiteLLM response cost headers; proxy usage signal not found."
  echo "Expected one of: x-litellm-response-cost or x-litellm-response-cost-original"
  sed -n '1,80p' "${HEADERS_FILE}"
  sed -n '1,80p' "${BODY_FILE}"
  exit 1
fi

echo "Success: Claude Code ran in its own non-root container through LiteLLM, with Postgres-backed proxy headers present."
