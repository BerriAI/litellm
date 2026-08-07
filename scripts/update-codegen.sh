#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DASHBOARD_DIR="${REPO_ROOT}/ui/litellm-dashboard"

if [[ -z "${LITELLM_PYTHON:-}" && -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  export LITELLM_PYTHON="${REPO_ROOT}/.venv/bin/python"
fi

if [[ ! -x "${DASHBOARD_DIR}/node_modules/.bin/openapi-typescript" ]]; then
  echo "Installing dashboard dependencies..." >&2
  npm ci --prefix "${DASHBOARD_DIR}"
fi

cd "${DASHBOARD_DIR}"
npm run gen:api
