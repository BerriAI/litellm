#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

# Add venv scripts to PATH so subprocesses can find prisma, litellm, etc.
export PATH="$PWD/.venv/Scripts:$PATH"

# Load environment variables from .env if it exists
if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

# Force UTF-8 to avoid encoding issues on Windows
export PYTHONUTF8=1

# Workaround: keep prisma-client-py accessible via a path without spaces
export PATH="/c/litellm-bin:$PATH"

.venv/Scripts/litellm --config litellm_config.yaml --port 4000 --detailed_debug
