#!/bin/sh
# Moved to litellm/scripts/quickstart.sh.
forward_quickstart() {
  tmp=$(mktemp) || { echo 'LiteLLM: could not create a temporary file.' >&2; exit 1; }
  trap 'rm -f "$tmp"' 0
  trap 'exit 1' HUP INT TERM
  if ! curl -fsSL https://raw.githubusercontent.com/BerriAI/litellm/main/litellm/scripts/quickstart.sh -o "$tmp"; then
    echo 'LiteLLM: could not download the quickstart script.' >&2
    exit 1
  fi
  sh "$tmp" "$@"
  exit "$?"
}
forward_quickstart "$@"
