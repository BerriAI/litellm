#!/usr/bin/env bash
# Apply local litellm/ patch: restart proxy to reload bind-mounted source.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

echo "Restarting LiteLLM proxy (loading litellm/ from $(cd ../.. && pwd)/litellm)..."
docker compose restart litellm

echo -n "Waiting for health"
for _ in $(seq 1 60); do
  if curl -sf http://localhost:4000/health/liveliness >/dev/null 2>&1; then
    echo " OK"
    docker compose exec -T litellm python3 -c "import litellm; print('  litellm:', litellm.__file__)"
    exit 0
  fi
  echo -n "."
  sleep 2
done
echo " TIMEOUT" >&2
exit 1
