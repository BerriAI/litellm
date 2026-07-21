#!/usr/bin/env bash
set -uo pipefail

category="${1:?usage: classify_changes.sh <backend|client>}"

has_client=false
has_backend=false
while IFS= read -r file || [ -n "$file" ]; do
  [ -n "$file" ] || continue
  case "$file" in
    ui/* | tests/e2e/ui/*) has_client=true ;;
    docs/* | *.md | *.mdx) : ;;
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
  *)
    echo run
    ;;
esac
