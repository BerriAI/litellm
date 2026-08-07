#!/usr/bin/env bash
# e2e_llm_lock.sh — make non-test code read-only so a long-running agent
# can only modify tests. Used by the e2e LLM translation refactor loop.
#
# Usage:
#   scripts/e2e_llm_lock.sh lock    # set non-test code to read-only
#   scripts/e2e_llm_lock.sh unlock  # restore writable mode
#   scripts/e2e_llm_lock.sh status  # report current mode
#
# Locked directories: everything under litellm/ EXCEPT tests/ stays
# writable. The .git tree and tests/ are always writable so commits and
# new tests can land. chmod is per-user (u-w / u+w) so it doesn't affect
# other accounts.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TARGET_DIR="${ROOT}/litellm"

cmd="${1:-status}"

case "$cmd" in
  lock)
    echo "Locking ${TARGET_DIR} (read-only for current user)..."
    find "${TARGET_DIR}" -type f -exec chmod u-w {} +
    find "${TARGET_DIR}" -type d -exec chmod u-w {} +
    echo "Locked. Run 'scripts/e2e_llm_lock.sh unlock' to reverse."
    ;;
  unlock)
    echo "Unlocking ${TARGET_DIR}..."
    find "${TARGET_DIR}" -type d -exec chmod u+w {} +
    find "${TARGET_DIR}" -type f -exec chmod u+w {} +
    echo "Unlocked."
    ;;
  status)
    # Sample a known file. If it's writable, we're unlocked.
    sample="${TARGET_DIR}/__init__.py"
    if [[ -w "${sample}" ]]; then
      echo "unlocked (${sample} is writable)"
    else
      echo "locked (${sample} is read-only)"
    fi
    ;;
  *)
    echo "Usage: $0 {lock|unlock|status}" >&2
    exit 2
    ;;
esac
