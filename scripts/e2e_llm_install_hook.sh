#!/usr/bin/env bash
# e2e_llm_install_hook.sh — install (or remove) a local git pre-commit
# hook that blocks commits touching litellm/ source while the e2e LLM
# refactor loop is active. The hook lives at .git/hooks/pre-commit and
# is local-only (not pushed). It's belt-and-suspenders alongside
# scripts/e2e_llm_lock.sh because chmod doesn't stop root.
#
# Usage:
#   scripts/e2e_llm_install_hook.sh install
#   scripts/e2e_llm_install_hook.sh uninstall
#
# The hook checks for the sentinel file .e2e-llm-loop-active at repo
# root. While the sentinel exists, any staged change under litellm/
# (except documentation-only) is rejected. Remove the sentinel
# (`rm .e2e-llm-loop-active`) to exit the locked mode.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HOOK="${ROOT}/.git/hooks/pre-commit"

cmd="${1:-install}"

case "${cmd}" in
  install)
    if [[ -e "${HOOK}" ]] && ! grep -q "e2e-llm-loop-active" "${HOOK}"; then
      backup="${HOOK}.bak"
      if [[ -e "${backup}" ]]; then
        echo "Refusing to overwrite existing pre-commit hook: ${HOOK}" >&2
        echo "A backup already exists at ${backup}. Resolve manually before re-running." >&2
        exit 2
      fi
      mv "${HOOK}" "${backup}"
      echo "Backed up existing pre-commit hook to ${backup}"
    fi
    cat > "${HOOK}" <<'HOOKBODY'
#!/usr/bin/env bash
# Installed by scripts/e2e_llm_install_hook.sh. Blocks commits to
# litellm/ source while the e2e LLM translation loop is active.
set -euo pipefail
repo_root="$(git rev-parse --show-toplevel)"
sentinel="${repo_root}/.e2e-llm-loop-active"
if [[ ! -f "${sentinel}" ]]; then
  exit 0
fi
blocked=$(git diff --cached --name-only --diff-filter=ACMR \
  | grep -E '^litellm/' \
  | grep -v -E '\.(md|rst|txt)$' || true)
if [[ -n "${blocked}" ]]; then
  echo "E2E LLM translation loop is active (.e2e-llm-loop-active present)." >&2
  echo "Commits to litellm/ source are blocked. Staged files:" >&2
  echo "${blocked}" | sed 's/^/  /' >&2
  echo "" >&2
  echo "To proceed:" >&2
  echo "  1. Unstage litellm/ changes: git reset HEAD -- litellm/" >&2
  echo "  2. Or exit the loop:        rm ${sentinel}" >&2
  exit 1
fi
exit 0
HOOKBODY
    chmod +x "${HOOK}"
    echo "Installed pre-commit hook at ${HOOK}"
    echo "Touch .e2e-llm-loop-active to activate the block."
    ;;
  uninstall)
    if [[ -f "${HOOK}" ]] && grep -q "e2e-llm-loop-active" "${HOOK}"; then
      rm "${HOOK}"
      echo "Removed pre-commit hook."
      backup="${HOOK}.bak"
      if [[ -e "${backup}" ]]; then
        mv "${backup}" "${HOOK}"
        echo "Restored previous pre-commit hook from ${backup}"
      fi
    else
      echo "No matching pre-commit hook to remove."
    fi
    ;;
  *)
    echo "Usage: $0 {install|uninstall}" >&2
    exit 2
    ;;
esac
