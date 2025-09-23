# STATUS NOW — Agent API (Green)
- Endpoint: http://127.0.0.1:8788/agent/run
- Model: ollama/qwen3:8b
- Provider base: http://ollama:11434 (Docker network `llmnet`)

Verify:
curl -sS -H 'content-type: application/json' \
  --data '{"messages":[{"role":"user","content":"hi"}],
           "model":"ollama/qwen3:8b",
           "api_base":"http://ollama:11434",
           "base_url":"http://ollama:11434",
           "tool_backend":"local","use_tools":false}' \
  http://127.0.0.1:8788/agent/run
