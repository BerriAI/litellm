#!/usr/bin/env bash
# E2E check: start LiteLLM proxy with the aws_secret_key content filter, send one message.
# Usage (repo root, litellm installed from this checkout): ./e2e_aws_secret_key.sh "<message text>"
PORT=4123
MODEL="ollama/gemma4:31b"
API_BASE="http://localhost:11434"
MSG="${1:-Please review /usr/local/Documents/CodingProjects/PyCharmProjects/README.md}"

TMP=$(mktemp -d)
trap 'kill $PID 2>/dev/null; rm -rf "$TMP"' EXIT

cat > "$TMP/config.yaml" <<EOF
model_list:
  - model_name: test-model
    litellm_params:
      model: $MODEL
      api_base: $API_BASE
general_settings:
  dangerously_permit_weak_or_unset_master_key: true
guardrails:
  - guardrail_name: aws-secret-key-filter
    litellm_params:
      guardrail: litellm_content_filter
      mode: pre_call
      default_on: true
      patterns:
        - pattern_type: prebuilt
          pattern_name: aws_secret_key
          action: BLOCK
EOF

uv run litellm --config "$TMP/config.yaml" --port $PORT > "$TMP/proxy.log" 2>&1 &
PID=$!

for _ in $(seq 90); do
  curl -sf "http://127.0.0.1:$PORT/health/liveliness" > /dev/null && break
  kill -0 $PID 2>/dev/null || { echo "Proxy failed to start:"; tail -20 "$TMP/proxy.log"; exit 1; }
  sleep 1
done

# escape backslashes (Windows paths) and quotes for JSON
J=${MSG//\\/\\\\}; J=${J//\"/\\\"}
CODE=$(curl -s -o "$TMP/out.json" -w '%{http_code}' "http://127.0.0.1:$PORT/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"test-model\",\"messages\":[{\"role\":\"user\",\"content\":\"$J\"}]}")

echo "commit : $(git rev-parse --short HEAD)"
echo "message: $MSG"
echo "status : $CODE"
[ "$CODE" -ge 400 ] && echo "result : BLOCKED by guardrail" || echo "result : ALLOWED (model replied)"
echo "body   : $(head -c 300 "$TMP/out.json")"