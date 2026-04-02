#!/usr/bin/env bash
#
# harden_old_branches.sh — Sync CI/CD and supply-chain-sensitive files from
# main to all remote branches, reducing attack surface on stale branches.
#
# What it does:
#   1. SYNCS files from main   → overwrites the branch copy with main's version
#   2. DELETES files            → removes files that should no longer exist
#   3. Commits & pushes         → one commit per branch, only if there are changes
#
# Usage:
#   ./scripts/harden_old_branches.sh [OPTIONS]
#
# Options:
#   --dry-run           Show what would be changed without pushing (default)
#   --execute           Actually push changes to remote branches
#   --since YYYY-MM-DD  Only process branches with commits after this date (default: all)
#   --branch PATTERN    Only process branches matching this grep pattern
#   --exclude PATTERN   Skip branches matching this grep pattern
#   --max N             Process at most N branches (useful for testing)
#   --log FILE          Append per-branch results to this file
#   --resume-from NAME  Skip branches until this one is found, then continue
#
# Prerequisites:
#   - Run from repo root
#   - git fetch origin must have been run recently
#   - All CI/CD runs on non-main branches should be DISABLED before running with --execute
#
# Safety:
#   - Uses a detached worktree so your working directory is never touched
#   - Default mode is --dry-run
#   - Skips branches where the commit would be empty (already up to date)
#   - Skips protected branches (main, master, release/*)
#
set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────

MAIN_REF="origin/main"

# Files to sync from main (overwrite on branch with main's version)
SYNC_FILES=(
  ".circleci/config.yml"
  ".circleci/requirements.txt"
  ".github/actions/helm-oci-chart-releaser/action.yml"
  ".github/dependabot.yaml"
  "docker/build_admin_ui.sh"
  "ui/litellm-dashboard/build_ui.sh"
  "ui/litellm-dashboard/build_ui_custom_path.sh"
  "scripts/install.sh"
)

# Directories to sync from main (entire directory replaced with main's version)
SYNC_DIRS=(
  ".github/workflows"
)

# Files to delete if they exist on the branch
DELETE_FILES=(
  "ci_cd/publish-proxy-extras.sh"
  ".pre-commit-config.yaml"
  "ci_cd/security_scans.sh"
  "ci_cd/.grype.yaml"
  ".trivyignore"
  "ui/litellm-dashboard/.trivyignore"
  "docs/my-website/.trivyignore"
)

COMMIT_MSG="[Infra] Harden branch: sync CI/CD files from main, remove attack surface

Automated supply-chain hardening:
- Sync .circleci/, .github/, build scripts from main
- Remove dead/dangerous files (security_scans.sh, .pre-commit-config.yaml, etc.)
- SHA256-verified NVM installs, pip --only-binary"

# Protected branch patterns (never touch these)
PROTECTED_PATTERNS="^(main|master|release/.*)$"

# ── CLI Parsing ───────────────────────────────────────────────────────────

DRY_RUN=true
SINCE=""
BRANCH_FILTER=""
EXCLUDE_FILTER=""
MAX_BRANCHES=0
LOG_FILE=""
RESUME_FROM=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)    DRY_RUN=true; shift ;;
    --execute)    DRY_RUN=false; shift ;;
    --since)      SINCE="$2"; shift 2 ;;
    --branch)     BRANCH_FILTER="$2"; shift 2 ;;
    --exclude)    EXCLUDE_FILTER="$2"; shift 2 ;;
    --max)        MAX_BRANCHES="$2"; shift 2 ;;
    --log)        LOG_FILE="$2"; shift 2 ;;
    --resume-from) RESUME_FROM="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,/^$/p' "$0" | sed 's/^# \?//'
      exit 0
      ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

# ── Preflight ─────────────────────────────────────────────────────────────

if [[ ! -d .git && ! -f .git ]]; then
  echo "ERROR: Run this from the repo root." >&2
  exit 1
fi

if $DRY_RUN; then
  echo "=== DRY RUN MODE (use --execute to push) ==="
  echo ""
fi

# Fetch latest main
echo "Fetching latest main..."
git fetch origin main --quiet

# ── Build branch list ─────────────────────────────────────────────────────

echo "Building branch list..."

if [[ -n "$SINCE" ]]; then
  # Only branches with recent commits
  BRANCHES=$(git for-each-ref --sort=-committerdate \
    --format='%(committerdate:short) %(refname:short)' refs/remotes/origin/ \
    | awk -v since="$SINCE" '$1 >= since { print $2 }')
else
  BRANCHES=$(git for-each-ref --format='%(refname:short)' refs/remotes/origin/)
fi

# Strip origin/ prefix, filter, dedupe
BRANCHES=$(echo "$BRANCHES" | sed 's|^origin/||' | sort -u)

# Remove protected branches
BRANCHES=$(echo "$BRANCHES" | grep -vE "$PROTECTED_PATTERNS" || true)

# Remove HEAD
BRANCHES=$(echo "$BRANCHES" | grep -v '^HEAD$' || true)

# Apply user filters
if [[ -n "$BRANCH_FILTER" ]]; then
  BRANCHES=$(echo "$BRANCHES" | grep "$BRANCH_FILTER" || true)
fi
if [[ -n "$EXCLUDE_FILTER" ]]; then
  BRANCHES=$(echo "$BRANCHES" | grep -v "$EXCLUDE_FILTER" || true)
fi

# Resume support
if [[ -n "$RESUME_FROM" ]]; then
  BRANCHES=$(echo "$BRANCHES" | sed -n "/$RESUME_FROM/,\$p")
fi

# Cap count
TOTAL=$(echo "$BRANCHES" | grep -c . || true)
if [[ "$MAX_BRANCHES" -gt 0 && "$TOTAL" -gt "$MAX_BRANCHES" ]]; then
  BRANCHES=$(echo "$BRANCHES" | head -n "$MAX_BRANCHES")
  TOTAL=$MAX_BRANCHES
fi

echo "Found $TOTAL branches to process."
echo ""

if [[ "$TOTAL" -eq 0 ]]; then
  echo "Nothing to do."
  exit 0
fi

# ── Setup worktree ────────────────────────────────────────────────────────

WORKTREE_DIR=$(mktemp -d "${TMPDIR:-/tmp}/harden-branches.XXXXXX")
trap 'echo "Cleaning up worktree..."; git worktree remove --force "$WORKTREE_DIR" 2>/dev/null || rm -rf "$WORKTREE_DIR"' EXIT

# Create a detached worktree from main initially
git worktree add --detach "$WORKTREE_DIR" "$MAIN_REF" --quiet

# ── Helper: log result ────────────────────────────────────────────────────

log_result() {
  local branch="$1" status="$2" detail="$3"
  local line="$(date -u +%Y-%m-%dT%H:%M:%SZ) | $status | $branch | $detail"
  echo "  [$status] $detail"
  if [[ -n "$LOG_FILE" ]]; then
    echo "$line" >> "$LOG_FILE"
  fi
}

# ── Process each branch ──────────────────────────────────────────────────

COUNT=0
CHANGED=0
SKIPPED=0
FAILED=0

for BRANCH in $BRANCHES; do
  COUNT=$((COUNT + 1))
  echo "[$COUNT/$TOTAL] $BRANCH"

  # Try to check out the branch in the worktree
  if ! git -C "$WORKTREE_DIR" checkout "origin/$BRANCH" --detach --quiet 2>/dev/null; then
    log_result "$BRANCH" "SKIP" "could not checkout"
    SKIPPED=$((SKIPPED + 1))
    continue
  fi

  CHANGES_MADE=false

  # ── Sync individual files from main ──
  for FILE in "${SYNC_FILES[@]}"; do
    # Check if file exists on main
    if ! git show "${MAIN_REF}:${FILE}" > /dev/null 2>&1; then
      continue
    fi

    # Check if file exists on branch AND is already identical to main
    MAIN_HASH=$(git rev-parse "${MAIN_REF}:${FILE}" 2>/dev/null || true)
    BRANCH_HASH=$(git rev-parse "origin/${BRANCH}:${FILE}" 2>/dev/null || true)

    if [[ "$MAIN_HASH" == "$BRANCH_HASH" ]]; then
      continue  # Already in sync
    fi

    # File differs or doesn't exist on branch — sync it
    mkdir -p "$WORKTREE_DIR/$(dirname "$FILE")"
    git show "${MAIN_REF}:${FILE}" > "$WORKTREE_DIR/$FILE"
    git -C "$WORKTREE_DIR" add "$FILE"
    CHANGES_MADE=true
  done

  # ── Sync directories from main ──
  for DIR in "${SYNC_DIRS[@]}"; do
    # Remove existing directory on branch (if any) so we get an exact copy
    if [[ -d "$WORKTREE_DIR/$DIR" ]]; then
      rm -rf "$WORKTREE_DIR/$DIR"
    fi
    mkdir -p "$WORKTREE_DIR/$DIR"

    # Extract all files from main for this directory
    MAIN_DIR_FILES=$(git ls-tree -r --name-only "${MAIN_REF}" "$DIR" 2>/dev/null || true)
    if [[ -z "$MAIN_DIR_FILES" ]]; then
      continue
    fi

    for FILE in $MAIN_DIR_FILES; do
      mkdir -p "$WORKTREE_DIR/$(dirname "$FILE")"
      git show "${MAIN_REF}:${FILE}" > "$WORKTREE_DIR/$FILE"
    done
    git -C "$WORKTREE_DIR" add "$DIR"

    # Check if anything actually changed
    if ! git -C "$WORKTREE_DIR" diff --cached --quiet -- "$DIR"; then
      CHANGES_MADE=true
    fi
  done

  # ── Delete files ──
  for FILE in "${DELETE_FILES[@]}"; do
    if [[ -f "$WORKTREE_DIR/$FILE" ]]; then
      rm -f "$WORKTREE_DIR/$FILE"
      git -C "$WORKTREE_DIR" add "$FILE"
      CHANGES_MADE=true
    fi
  done

  # ── Commit & push ──
  if ! $CHANGES_MADE; then
    # Double-check with git
    if git -C "$WORKTREE_DIR" diff --cached --quiet 2>/dev/null; then
      log_result "$BRANCH" "SKIP" "already up to date"
      SKIPPED=$((SKIPPED + 1))
      continue
    fi
  fi

  # Check if there are actually staged changes
  if git -C "$WORKTREE_DIR" diff --cached --quiet 2>/dev/null; then
    log_result "$BRANCH" "SKIP" "already up to date"
    SKIPPED=$((SKIPPED + 1))
    continue
  fi

  if $DRY_RUN; then
    DIFF_STAT=$(git -C "$WORKTREE_DIR" diff --cached --stat | tail -1)
    log_result "$BRANCH" "DRY-RUN" "would change: $DIFF_STAT"
    # Reset for next branch
    git -C "$WORKTREE_DIR" reset --quiet HEAD 2>/dev/null || true
    CHANGED=$((CHANGED + 1))
  else
    # Commit
    git -C "$WORKTREE_DIR" commit --quiet -m "$COMMIT_MSG" --author="litellm-bot <bot@litellm.ai>"

    # Push the detached HEAD to the remote branch
    COMMIT_SHA=$(git -C "$WORKTREE_DIR" rev-parse HEAD)
    if git push origin "$COMMIT_SHA:refs/heads/$BRANCH" --quiet 2>/dev/null; then
      log_result "$BRANCH" "OK" "pushed"
      CHANGED=$((CHANGED + 1))
    else
      log_result "$BRANCH" "FAIL" "push rejected (branch may be protected)"
      FAILED=$((FAILED + 1))
    fi
  fi
done

# ── Summary ───────────────────────────────────────────────────────────────

echo ""
echo "=== Summary ==="
echo "  Processed: $COUNT"
echo "  Changed:   $CHANGED"
echo "  Skipped:   $SKIPPED"
echo "  Failed:    $FAILED"

if $DRY_RUN && [[ "$CHANGED" -gt 0 ]]; then
  echo ""
  echo "Run with --execute to apply these changes."
fi
