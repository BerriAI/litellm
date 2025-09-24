# 0) Assumes: Ollama is running on host at 127.0.0.1:11434
#    If not, either start Ollama or switch to your OpenAI-compatible vars in compose.

# 1) Validate the compose file renders
docker compose -f local/docker/compose.exec.yml config

# 2) Rebuild images
docker compose -f local/docker/compose.exec.yml build agent-api exec-rpc

# 3) Bring up both services
API_HOST=127.0.0.1 API_PORT=8788 API_CONTAINER_PORT=8788 \
EXEC_HOST=127.0.0.1 EXEC_PORT=8792 EXEC_CONTAINER_PORT=8790 \
docker compose -f local/docker/compose.exec.yml up -d --remove-orphans agent-api exec-rpc

# 4) Watch status/ports
docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}' | grep -E 'agent-api|exec-rpc' || true

# 5) Probe Agent API (uses a REAL model; matches compose env)
curl -sS -X POST -H 'content-type: application/json' \
  --data '{"messages":[{"role":"user","content":"hi"}],
           "model":"ollama/qwen3:8b","tool_backend":"local","use_tools":false}' \
  http://127.0.0.1:8788/agent/run

# 6) Probe Exec RPC
curl -sS -H 'content-type: application/json' \
  --data '{"language":"python","code":"print(1)","timeout_sec":1.0}' \
  http://127.0.0.1:8792/exec

# 7) If either probe fails, tail logs quickly
docker logs docker-agent-api-1 --tail=120
docker logs docker-exec-rpc-1 --tail=120
