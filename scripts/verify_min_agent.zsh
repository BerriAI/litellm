#!/usr/bin/env zsh
set -euo pipefail; setopt NO_NOMATCH PIPE_FAIL
H="${API_HOST:-127.0.0.1}"; P="${API_PORT:-8788}"; B="${OLLAMA_URL:-http://ollama:11434}"; M="${MODEL:-ollama/qwen3:8b}"
REQ='{"messages":[{"role":"user","content":"hi"}],"model":"'"$M"'","api_base":"'"$B"'","base_url":"'"$B"'","tool_backend":"local","use_tools":false}'
OUT=$(curl -sS -H 'content-type: application/json' --data "$REQ" "http://$H:$P/agent/run" || true)
echo "$OUT" | grep -q '"ok":true' && echo "✅ Agent API OK" || { echo "$OUT"; echo "❌ Agent API fail"; exit 1; }
