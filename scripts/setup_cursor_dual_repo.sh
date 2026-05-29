#!/usr/bin/env bash
# Set up litellm + litellm-docs for Cursor multi-root workspace (local Mac/Linux).
set -euo pipefail

LITELLM_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PARENT_DIR="$(dirname "$LITELLM_ROOT")"
DOCS_DIR="${LITELLM_DOCS_DIR:-$PARENT_DIR/litellm-docs}"
DOCS_REPO="${LITELLM_DOCS_REPO:-https://github.com/BerriAI/litellm-docs.git}"
WORKSPACE_FILE="$LITELLM_ROOT/litellm-full.local.code-workspace"
AGENTS_TEMPLATE="$LITELLM_ROOT/contrib/litellm-docs-AGENTS.md"

echo "LiteLLM root:     $LITELLM_ROOT"
echo "litellm-docs dir: $DOCS_DIR"

if [[ ! -d "$LITELLM_ROOT/.git" ]]; then
  echo "error: $LITELLM_ROOT does not look like the litellm git repo" >&2
  exit 1
fi

if [[ ! -f "$LITELLM_ROOT/litellm-full.code-workspace" ]]; then
  echo "error: litellm-full.code-workspace missing — pull latest litellm (PR #28497 or main)" >&2
  exit 1
fi

if [[ ! -d "$DOCS_DIR/.git" ]]; then
  echo "Cloning litellm-docs..."
  git clone "$DOCS_REPO" "$DOCS_DIR"
else
  echo "litellm-docs already cloned; pulling latest..."
  git -C "$DOCS_DIR" pull --ff-only || true
fi

if [[ -f "$AGENTS_TEMPLATE" && ! -f "$DOCS_DIR/AGENTS.md" ]]; then
  echo "Installing AGENTS.md in litellm-docs..."
  # Drop the "copy to..." header from the template file.
  awk 'BEGIN{p=0} /^---$/ && p==0 {p=1; next} p' "$AGENTS_TEMPLATE" > "$DOCS_DIR/AGENTS.md"
elif [[ -f "$DOCS_DIR/AGENTS.md" ]]; then
  echo "litellm-docs AGENTS.md already exists; skipping."
fi

# Machine-specific workspace (absolute paths) — gitignored.
cat > "$WORKSPACE_FILE" <<EOF
{
  "folders": [
    {
      "name": "litellm",
      "path": "$LITELLM_ROOT"
    },
    {
      "name": "litellm-docs",
      "path": "$DOCS_DIR"
    }
  ],
  "settings": {
    "search.exclude": {
      "**/node_modules": true,
      "**/.venv": true,
      "**/dist": true,
      "**/build": true,
      "**/.next": true
    }
  }
}
EOF

echo ""
echo "Created: $WORKSPACE_FILE"
echo ""
echo "Next steps in Cursor (manual — cannot be automated from cloud):"
echo "  1. File → Open Workspace from File…"
echo "     → $WORKSPACE_FILE"
echo "  2. Cursor Settings → Docs → Add https://docs.litellm.ai"
echo "  3. Wait for indexing (Cmd+Shift+P → Reindex if needed)"
echo ""

if command -v cursor >/dev/null 2>&1; then
  echo "Opening workspace with Cursor CLI..."
  exec cursor "$WORKSPACE_FILE"
elif [[ "$(uname -s)" == "Darwin" ]] && [[ -d "/Applications/Cursor.app" ]]; then
  echo "Opening workspace with Cursor.app..."
  open -a Cursor "$WORKSPACE_FILE"
else
  echo "Install Cursor CLI (Shell Command: Install 'cursor' command) or open the workspace file above manually."
fi
