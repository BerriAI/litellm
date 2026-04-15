#!/usr/bin/env bash
set -e

echo "[post-create] Installing uv"
curl -LsSf https://astral.sh/uv/0.10.9/install.sh | env UV_NO_MODIFY_PATH=1 sh
export PATH="$HOME/.local/bin:$PATH"

echo "[post-create] Installing Python dependencies (uv)"
uv sync --frozen --group proxy-dev --extra proxy

echo "[post-create] Generating Prisma client"
uv run --no-sync prisma generate

echo "[post-create] Installing npm dependencies"
cd ui/litellm-dashboard && npm ci

echo "[post-create] Done"
