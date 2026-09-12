#!/usr/bin/env bash
set -euo pipefail
python -m pip install uv==0.11.7
export UV_PROJECT_ENVIRONMENT="$PWD/.memory-pilot-venv"
uv sync --frozen --extra proxy --no-default-groups
export PATH="$UV_PROJECT_ENVIRONMENT/bin:$PATH"
prisma generate --schema schema.prisma
(
  cd ui/litellm-dashboard
  npm ci
  npm run build
)
rm -rf litellm/proxy/_experimental/out
cp -R ui/litellm-dashboard/out litellm/proxy/_experimental/out
