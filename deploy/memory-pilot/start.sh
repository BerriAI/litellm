#!/usr/bin/env bash
set -euo pipefail
export PATH="$PWD/.memory-pilot-venv/bin:$PATH"
prisma migrate deploy --schema litellm-proxy-extras/litellm_proxy_extras/schema.prisma
export WORKER_CONFIG="$PWD/deploy/memory-pilot/proxy_config.yaml"
export PYTHONPATH="$PWD/deploy/memory-pilot${PYTHONPATH:+:$PYTHONPATH}"
exec uvicorn pilot:create_app --factory --host 0.0.0.0 --port "${PORT:-4000}"
