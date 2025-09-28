#!/usr/bin/env bash
set -euo pipefail

# Try to determine a host-accessible base URL for an Ollama Docker container.
# Prints a URL suitable for LITELLM/OLLAMA API bases.

PORT_LINE=$(docker ps --format '{{.Names}} {{.Ports}}' | grep -E 'ollama' | head -n 1 || true)
if [ -z "$PORT_LINE" ]; then
  # Fallback to internal DNS on llmnet (inside compose network)
  echo "http://ollama:11434"
  exit 0
fi

# Example PORTS column: 0.0.0.0:11434->11434/tcp, [::]:11434->11434/tcp
HOST_PORT=$(echo "$PORT_LINE" | sed -E 's/.*:([0-9]+)->11434.*/\1/' || true)
if [[ "$HOST_PORT" =~ ^[0-9]+$ ]]; then
  echo "http://127.0.0.1:${HOST_PORT}"
else
  echo "http://127.0.0.1:11434"
fi

