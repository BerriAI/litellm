#!/usr/bin/env bash
# Verify local Cursor dual-repo setup. Exit 0 = ready, non-zero = fix items printed.
set -euo pipefail

LITELLM_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PARENT_DIR="$(dirname "$LITELLM_ROOT")"
DOCS_DIR="${LITELLM_DOCS_DIR:-$PARENT_DIR/litellm-docs}"
LOCAL_WS="$LITELLM_ROOT/litellm-full.local.code-workspace"
SHARED_WS="$LITELLM_ROOT/litellm-full.code-workspace"
RULES="$LITELLM_ROOT/.cursor/rules/docs-repo.mdc"

ok=0
warn=0

check() {
  local name="$1"
  local status="$2" # ok | warn | fail
  local detail="$3"
  case "$status" in
    ok)   echo "[OK]   $name — $detail" ;;
    warn) echo "[WARN] $name — $detail"; warn=$((warn + 1)) ;;
    fail) echo "[FAIL] $name — $detail"; ok=1 ;;
  esac
}

echo "LiteLLM:     $LITELLM_ROOT"
echo "litellm-docs: $DOCS_DIR"
echo ""

if [[ -d "$LITELLM_ROOT/.git" ]]; then
  branch="$(git -C "$LITELLM_ROOT" branch --show-current 2>/dev/null || echo unknown)"
  check "litellm git repo" ok "on branch $branch"
else
  check "litellm git repo" fail "not a git checkout"
fi

if [[ -f "$RULES" ]]; then
  check "Cursor rule" ok ".cursor/rules/docs-repo.mdc present"
else
  check "Cursor rule" fail "missing $RULES — pull branch with PR #28497"
fi

if [[ -d "$DOCS_DIR/.git" ]]; then
  check "litellm-docs clone" ok "$DOCS_DIR"
else
  check "litellm-docs clone" fail "missing $DOCS_DIR — run ./scripts/setup_cursor_dual_repo.sh"
fi

if [[ -f "$DOCS_DIR/AGENTS.md" ]]; then
  check "litellm-docs AGENTS.md" ok "present"
else
  check "litellm-docs AGENTS.md" warn "missing (optional; re-run setup script)"
fi

if [[ -f "$LOCAL_WS" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    python3 -c "import json; json.load(open('$LOCAL_WS'))" 2>/dev/null && \
      check "local workspace JSON" ok "$LOCAL_WS" || \
      check "local workspace JSON" fail "invalid JSON in $LOCAL_WS"
  else
    check "local workspace" ok "$LOCAL_WS (not validated)"
  fi
  if [[ -f "$DOCS_DIR/docs/proxy/health.md" ]]; then
    check "docs health page" ok "docs/proxy/health.md"
  else
    check "docs health page" warn "docs/proxy/health.md not found"
  fi
elif [[ -f "$SHARED_WS" && -d "$DOCS_DIR" ]]; then
  check "workspace file" warn "use setup script to generate $LOCAL_WS, or open $SHARED_WS with sibling ../litellm-docs"
else
  check "workspace file" fail "missing $LOCAL_WS — run ./scripts/setup_cursor_dual_repo.sh"
fi

if [[ -f "$LITELLM_ROOT/litellm/proxy/health_endpoints/_health_endpoints.py" ]]; then
  check "code health endpoint" ok "litellm/proxy/health_endpoints/"
else
  check "code health endpoint" fail "proxy health module missing"
fi

echo ""
if [[ $ok -ne 0 ]]; then
  echo "Not ready — fix [FAIL] items above."
  exit 1
fi
if [[ $warn -gt 0 ]]; then
  echo "Mostly ready — optional [WARN] items above."
  echo ""
  echo "In Cursor:"
  echo "  • Sidebar shows two roots: litellm + litellm-docs"
  echo "  • Settings → Docs includes https://docs.litellm.ai (manual)"
  exit 0
fi
echo "Repo setup looks ready."
echo ""
echo "Confirm in Cursor UI:"
echo "  1. Two workspace folders in the sidebar"
echo "  2. Opened workspace: litellm-full.local.code-workspace (or multi-root from setup)"
echo "  3. Settings → Docs → https://docs.litellm.ai added (optional @Docs)"
echo "  4. Ask: 'Where is proxy /health documented and implemented?' — should cite both repos"
exit 0
