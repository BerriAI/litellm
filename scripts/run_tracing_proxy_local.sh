#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "run_tracing_proxy_local.sh is deprecated; use bash scripts/lens_dev.sh" >&2
exec "$repo_root/scripts/lens_dev.sh" "$@"
