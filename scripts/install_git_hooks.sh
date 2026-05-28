#!/usr/bin/env bash
# scripts/install_git_hooks.sh — opt-in installer for the Conventional Commits
# and Conventional Branches git hooks shipped in .githooks/.
#
# Sets `core.hooksPath` for the current repo to `.githooks` so that the hooks
# tracked in this repository are the ones git invokes locally. This is opt-in;
# `make install-dev` does NOT call this script.
#
# Bypass for a single commit/push: pass `--no-verify`.
set -euo pipefail

repo_root=$(git rev-parse --show-toplevel 2>/dev/null || true)
if [[ -z "$repo_root" ]]; then
  echo "install_git_hooks.sh: must be run inside a git repository" >&2
  exit 1
fi

cd "$repo_root"

if [[ ! -d .githooks ]]; then
  echo "install_git_hooks.sh: .githooks/ directory not found in $repo_root" >&2
  exit 1
fi

# Make sure the hooks are executable (mode bits may be lost on some checkouts).
for hook in .githooks/*; do
  [[ -f "$hook" ]] || continue
  chmod +x "$hook"
done

git config core.hooksPath .githooks

echo "Installed git hooks: core.hooksPath -> .githooks"
echo "  commit-msg : Conventional Commits 1.0.0"
echo "  pre-push   : Conventional Branches"
echo ""
echo "Bypass a single commit/push with --no-verify."
