#!/usr/bin/env bash
set -uo pipefail

has_file=false
has_file_outside_src=false
while IFS= read -r file || [ -n "$file" ]; do
  [ -n "$file" ] || continue
  has_file=true
  case "$file" in
    src/*) ;;
    *) has_file_outside_src=true ;;
  esac
done

{ [ "$has_file" = true ] && [ "$has_file_outside_src" = false ]; } && echo related || echo full
