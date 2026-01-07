#!/usr/bin/env bash
set -e

echo "[post-create] Installing poetry via pip"
python -m pip install --upgrade pip
python -m pip install poetry

echo "[post-create] Installing Python dependencies (poetry)"
poetry install --with dev --extras proxy

echo "[post-create] Generating Prisma client"
poetry run prisma generate

echo "[post-create] Installing npm dependencies"
cd ui/litellm-dashboard && npm install --no-audit --no-fund

echo "[post-create] Done"