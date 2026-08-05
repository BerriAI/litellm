#!/usr/bin/env bash
set -euo pipefail

export AWS_REGION_NAME="${AWS_REGION_NAME:-us-west-2}"
export BEDROCK_MODEL="${BEDROCK_MODEL:-us.anthropic.claude-sonnet-4-5-20250929-v1:0}"
export LITELLM_MASTER_KEY="${LITELLM_MASTER_KEY:-harness-master-key}"
export PORT="${PORT:-4001}"

if [[ -z "${AWS_BEARER_TOKEN_BEDROCK:-}" ]]; then
  echo "AWS_BEARER_TOKEN_BEDROCK is required" >&2
  exit 1
fi

if ! command -v claude >/dev/null 2>&1; then
  echo "claude CLI not found on PATH" >&2
  exit 1
fi

workspace="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
log=/tmp/litellm-bedrock-claude-code-gateway.log
ready_url="http://127.0.0.1:${PORT}/health/readiness"

if curl -sf "$ready_url" >/dev/null 2>&1; then
  echo "reusing the gateway already listening on ${PORT} (its model_list is whatever it was started with, not this script's BEDROCK_MODEL)"
else
  cargo run --manifest-path "${workspace}/Cargo.toml" -p litellm-ai-gateway --features server >"$log" 2>&1 &
  gateway_pid=$!
  trap 'kill "$gateway_pid" 2>/dev/null || true' EXIT

  for _ in {1..180}; do
    if ! kill -0 "$gateway_pid" 2>/dev/null; then
      echo "gateway exited before becoming ready; see $log" >&2
      tail -20 "$log" >&2
      exit 1
    fi
    curl -sf "$ready_url" >/dev/null && break
    sleep 1
  done

  if ! curl -sf "$ready_url" >/dev/null; then
    echo "gateway never became ready; see $log" >&2
    tail -20 "$log" >&2
    exit 1
  fi
fi

echo "smoke"
smoke_body=/tmp/litellm-bedrock-claude-code-smoke.json
smoke_code="$(curl -sS -o "$smoke_body" -w '%{http_code}' \
  -H "authorization: Bearer ${LITELLM_MASTER_KEY}" -H "content-type: application/json" \
  "http://127.0.0.1:${PORT}/v1/messages" \
  -d "{\"model\":\"${BEDROCK_MODEL}\",\"max_tokens\":16,\"messages\":[{\"role\":\"user\",\"content\":\"Reply with one word: hello\"}]}")"
if [[ "$smoke_code" != "200" ]]; then
  echo "smoke request failed with HTTP ${smoke_code}; not starting a claude session" >&2
  cat "$smoke_body" >&2
  echo >&2
  exit 1
fi
jq '{type,model,usage}' "$smoke_body"

unset ANTHROPIC_API_KEY
export ANTHROPIC_BASE_URL="http://127.0.0.1:${PORT}"
export ANTHROPIC_AUTH_TOKEN="${LITELLM_MASTER_KEY}"
export ANTHROPIC_MODEL="${BEDROCK_MODEL}"
# The gateway's env-built router holds exactly one deployment ($BEDROCK_MODEL),
# so every model slot Claude Code can reach for has to point at that same id.
export ANTHROPIC_SMALL_FAST_MODEL="${BEDROCK_MODEL}"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="${BEDROCK_MODEL}"
export ANTHROPIC_DEFAULT_SONNET_MODEL="${BEDROCK_MODEL}"
export ANTHROPIC_DEFAULT_OPUS_MODEL="${BEDROCK_MODEL}"
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1

echo "claude session against ${ANTHROPIC_BASE_URL} (model ${ANTHROPIC_MODEL}); gateway log: $log"
claude "$@"
