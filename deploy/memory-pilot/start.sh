#!/usr/bin/env bash
set -euo pipefail
export PATH="$PWD/.memory-pilot-venv/bin:$PATH"
export PRISMA_BINARY_CACHE_DIR="$PWD/.memory-pilot-prisma"
export PRISMA_CLI_PATH="$PRISMA_BINARY_CACHE_DIR/node_modules/.bin/prisma"
prisma migrate deploy --schema litellm-proxy-extras/litellm_proxy_extras/schema.prisma
export WORKER_CONFIG="$PWD/deploy/memory-pilot/proxy_config.yaml"
export PYTHONPATH="$PWD/deploy/memory-pilot${PYTHONPATH:+:$PYTHONPATH}"
exec uvicorn pilot:create_app --factory --host 0.0.0.0 --port "${PORT:-4000}"
