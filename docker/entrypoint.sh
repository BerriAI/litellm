#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PYTHON="$REPO_ROOT/.venv/bin/python"
MIGRATION_SCRIPT="$REPO_ROOT/litellm/proxy/prisma_migration.py"

if [ -x "$VENV_PYTHON" ]; then
    "$VENV_PYTHON" "$MIGRATION_SCRIPT"
elif command -v uv >/dev/null 2>&1; then
    (cd "$REPO_ROOT" && uv run --no-sync python "$MIGRATION_SCRIPT")
else
    python3 "$MIGRATION_SCRIPT"
fi

echo "Migration script ran successfully!"
