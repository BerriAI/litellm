#!/usr/bin/env bash
set -euo pipefail
# Render's preinstalled Rust toolchain directory is read-only. Maturin needs
# writable homes to install the repository's pinned toolchain during uv sync.
export RUSTUP_HOME="$PWD/.memory-pilot-rustup"
export CARGO_HOME="$PWD/.memory-pilot-cargo"
python -m pip install uv==0.11.7
export UV_PROJECT_ENVIRONMENT="$PWD/.memory-pilot-venv"
uv sync --frozen --extra proxy --extra extra_proxy --no-default-groups
export PATH="$UV_PROJECT_ENVIRONMENT/bin:$PATH"
prisma generate --schema schema.prisma
(
  cd ui/litellm-dashboard
  npm ci
  npm run build
)
rm -rf litellm/proxy/_experimental/out
cp -R ui/litellm-dashboard/out litellm/proxy/_experimental/out
