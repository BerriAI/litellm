#!/usr/bin/env bash
set -uo pipefail

category="${1:?usage: classify_changes.sh <backend|client|ui>}"

has_client=false
has_backend=false
has_ci=false
while IFS= read -r file || [ -n "$file" ]; do
  [ -n "$file" ] || continue
  case "$file" in
    ui/* | tests/e2e/ui/*) has_client=true ;;
    docs/* | *.md | *.mdx) : ;;
    .github/* | .circleci/*) has_ci=true; has_backend=true ;;
    *) has_backend=true ;;
  esac
done

case "$category" in
  backend)
    [ "$has_backend" = true ] && echo run || echo skip
    ;;
  client)
    { [ "$has_client" = true ] || [ "$has_backend" = true ]; } && echo run || echo skip
    ;;
  ui)
    { [ "$has_client" = true ] || [ "$has_ci" = true ]; } && echo run || echo skip
    ;;
  *)
    echo run
    ;;
esac
