#!/usr/bin/env bash
set -uo pipefail

category="${1:?usage: path_filter.sh <backend|client>}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

run_full() {
  echo "path-filter[$category]: running job ($1)"
  exit 0
}

[ -n "${CIRCLE_PULL_REQUEST:-}" ] || run_full "not a pull request"

candidate_bases="main litellm_internal_staging litellm_oss_staging"
merge_base=""
for base in $candidate_bases; do
  git fetch --quiet origin "$base" 2>/dev/null || continue
  candidate="$(git merge-base HEAD FETCH_HEAD 2>/dev/null)" || continue
  [ -n "$candidate" ] || continue
  if [ -z "$merge_base" ] || git merge-base --is-ancestor "$merge_base" "$candidate" 2>/dev/null; then
    merge_base="$candidate"
  fi
done

[ -n "$merge_base" ] || run_full "could not resolve a merge base against $candidate_bases"

changed="$(git diff --name-only "$merge_base" HEAD 2>/dev/null)" || run_full "git diff failed"
[ -n "$changed" ] || run_full "no files changed vs $merge_base"

echo "path-filter[$category]: changed files vs ${merge_base}:"
printf '%s\n' "$changed" | sed 's/^/  /' || true

decision="$(printf '%s\n' "$changed" | bash "$here/classify_changes.sh" "$category")" || run_full "classify_changes.sh failed"

if [ "$decision" = run ]; then
  run_full "$category-relevant changes detected"
fi

echo "path-filter[$category]: only unrelated (docs/client) changes detected; halting job as successful"
circleci-agent step halt
