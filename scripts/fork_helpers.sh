#!/usr/bin/env bash
# Fork branch / sync helpers for Bitovi litellm.
# Usage via Makefile: make fork-branch NAME=... | make fork-sync-status | make upstream-branch NAME=...

set -euo pipefail

STAGING_BRANCH="litellm_internal_staging"
UPSTREAM_URL="https://github.com/BerriAI/litellm.git"

ensure_upstream() {
  if ! git remote get-url upstream >/dev/null 2>&1; then
    git remote add upstream "$UPSTREAM_URL"
    echo "Added upstream remote: $UPSTREAM_URL"
  fi
}

validate_name() {
  local name="$1"
  if [[ -z "$name" ]]; then
    echo "NAME is required (e.g. NAME=litellm_my_change)" >&2
    exit 1
  fi
  if [[ "$name" == */* ]]; then
    echo "Branch name must not contain '/': got '$name'" >&2
    exit 1
  fi
}

latest_stable_tag() {
  git tag -l 'v*' | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$' | sort -V | tail -1
}

cmd_fork_branch() {
  local name="${NAME:-}"
  validate_name "$name"
  ensure_upstream
  git fetch origin "$STAGING_BRANCH"
  git checkout "$STAGING_BRANCH"
  git pull --ff-only origin "$STAGING_BRANCH"
  if git show-ref --verify --quiet "refs/heads/$name"; then
    echo "Branch '$name' already exists; checking it out"
    git checkout "$name"
  else
    git checkout -b "$name"
    echo "Created branch '$name' from $STAGING_BRANCH"
  fi
  echo "PR base should be: $STAGING_BRANCH"
}

cmd_upstream_branch() {
  local name="${NAME:-}"
  validate_name "$name"
  ensure_upstream
  git fetch upstream main
  if git show-ref --verify --quiet "refs/heads/$name"; then
    echo "Branch '$name' already exists; checking it out"
    git checkout "$name"
  else
    git checkout -b "$name" upstream/main
    echo "Created branch '$name' from upstream/main"
  fi
  echo "This branch is for a PR to BerriAI/litellm only (not bitovi staging)"
}

cmd_fork_sync_status() {
  ensure_upstream
  git fetch upstream --tags
  git fetch origin "$STAGING_BRANCH"
  local tag
  tag="$(latest_stable_tag)"
  if [[ -z "$tag" ]]; then
    echo "No stable tags found matching vX.Y.Z"
    exit 1
  fi
  local tag_sha staging_sha behind ahead
  tag_sha="$(git rev-list -n 1 "$tag")"
  staging_sha="$(git rev-parse "origin/$STAGING_BRANCH")"
  behind="$(git rev-list --count "origin/$STAGING_BRANCH..$tag" 2>/dev/null || echo 0)"
  ahead="$(git rev-list --count "$tag..origin/$STAGING_BRANCH" 2>/dev/null || echo 0)"
  echo "Latest upstream stable: $tag ($tag_sha)"
  echo "origin/$STAGING_BRANCH: $staging_sha"
  echo "Commits on tag not in staging: $behind"
  echo "Commits on staging not in tag: $ahead"
  if git merge-base --is-ancestor "$tag_sha" "origin/$STAGING_BRANCH"; then
    echo "Status: staging already contains $tag (no sync needed)"
  else
    echo "Status: sync needed — merge $tag into $STAGING_BRANCH (see FORK.md)"
  fi
}

main() {
  local cmd="${1:-}"
  case "$cmd" in
    fork-branch) cmd_fork_branch ;;
    upstream-branch) cmd_upstream_branch ;;
    fork-sync-status) cmd_fork_sync_status ;;
    *)
      echo "Usage: $0 {fork-branch|upstream-branch|fork-sync-status}" >&2
      exit 1
      ;;
  esac
}

main "$@"
