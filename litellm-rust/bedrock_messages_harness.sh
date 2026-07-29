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

cargo run -p litellm-ai-gateway --features server >/tmp/litellm-bedrock-gateway.log 2>&1 &
gateway_pid=$!
trap 'kill "$gateway_pid" 2>/dev/null || true' EXIT
for _ in {1..60}; do
  curl -sf "http://127.0.0.1:${PORT}/health/readiness" >/dev/null && break
  sleep 1
done

headers=(-H "authorization: Bearer ${LITELLM_MASTER_KEY}" -H "content-type: application/json")
url="http://127.0.0.1:${PORT}/v1/messages"

echo "simple"
curl -sS "${headers[@]}" "$url" -d "{\"model\":\"${BEDROCK_MODEL}\",\"max_tokens\":32,\"messages\":[{\"role\":\"user\",\"content\":\"Reply with one word: hello\"}]}" | jq '{type,id,model,usage}'

echo "streaming"
curl -sS "${headers[@]}" "$url" -d "{\"model\":\"${BEDROCK_MODEL}\",\"stream\":true,\"max_tokens\":32,\"messages\":[{\"role\":\"user\",\"content\":\"Reply with one word: hello\"}]}" | grep -E '^(event:|data:)' | head -20

echo "tool_use"
curl -sS "${headers[@]}" "$url" -d "{\"model\":\"${BEDROCK_MODEL}\",\"max_tokens\":64,\"tools\":[{\"name\":\"get_weather\",\"description\":\"Get weather\",\"input_schema\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}},\"required\":[\"city\"]}}],\"tool_choice\":{\"type\":\"tool\",\"name\":\"get_weather\"},\"messages\":[{\"role\":\"user\",\"content\":\"What is the weather in Paris?\"}]}" | jq '{type,model,content}'

echo "bad-model"
curl -sS -o /tmp/litellm-bedrock-bad-model.json -w 'HTTP %{http_code}\n' "${headers[@]}" "$url" -d '{"model":"us.anthropic.invalid-v1:0","max_tokens":8,"messages":[{"role":"user","content":"hello"}]}'
