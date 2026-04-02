#!/bin/bash
set -e

BASE_URL="${LITELLM_BASE_URL:-http://localhost:4000}"
MASTER_KEY="${LITELLM_MASTER_KEY:?LITELLM_MASTER_KEY is not set}"

echo "Testing LiteLLM at $BASE_URL"

# Health check
echo -n "Health check... "
curl -sf "$BASE_URL/health/liveliness" > /dev/null
echo "OK"

# Chat completion
echo -n "Chat completion... "
RESPONSE=$(curl -sf "$BASE_URL/v1/chat/completions" \
  -H "Authorization: Bearer $MASTER_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini-2.5-flash",
    "messages": [{"role": "user", "content": "Say hello in one word."}],
    "max_tokens": 10
  }')
echo "$RESPONSE" | python3 -c "import sys,json; print('OK -', json.load(sys.stdin)['choices'][0]['message']['content'])"