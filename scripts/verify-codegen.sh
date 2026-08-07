#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${REPO_ROOT}"

GENERATED_PATHS=(
  "openapi.json"
  "ui/litellm-dashboard/src/lib/http/schema.d.ts"
)

"${REPO_ROOT}/scripts/update-codegen.sh"

if [[ -z "$(git status --porcelain -- "${GENERATED_PATHS[@]}")" ]]; then
  echo "Generated files are in sync with the proxy."
  exit 0
fi

if [[ -n "${GITHUB_ACTIONS:-}" ]]; then
  echo "::error file=openapi.json::Generated files are out of sync with the proxy. Run scripts/update-codegen.sh and commit the result."
fi

{
  echo ""
  echo "Generated files are out of sync with the proxy."
  echo "A backend route or response model changed without regenerating them."
  echo ""
  git status --short -- "${GENERATED_PATHS[@]}"
  git diff --stat -- "${GENERATED_PATHS[@]}"
  echo ""
  echo "The regenerated files are already in your working tree. Review and commit them,"
  echo "or run scripts/update-codegen.sh yourself to reproduce this."
} >&2

exit 1
