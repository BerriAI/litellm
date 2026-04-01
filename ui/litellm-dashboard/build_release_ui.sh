#!/bin/bash
set -e

destination_dir="../../litellm/proxy/_experimental/out"

chmod +x ./build_ui.sh
./build_ui.sh

commit_message="chore: update Next.js build artifacts ($(date -u +"%Y-%m-%d %H:%M UTC"), node $(node -v))"

if git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
  git add -f "$destination_dir"/

  if ! git diff --cached --quiet; then
    git commit -m "$commit_message"
    echo "Git commit created."
  else
    echo "No changes to commit."
  fi
else
  echo "Not a git repository. Skipping commit."
fi
