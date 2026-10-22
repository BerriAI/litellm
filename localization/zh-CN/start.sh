#!/bin/bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  printf '用法：%s /绝对路径/litellm\n' "$0" >&2
  exit 2
fi

patch_root="$(cd "$(dirname "$0")" && pwd)"
repo_root="$(cd "$1" && pwd)"
exec docker compose --project-name litellm-local --env-file "$repo_root/.env" \
  -f "$repo_root/local-dev/compose.yaml" -f "$patch_root/compose.zh-cn.yaml" up -d
