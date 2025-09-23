#!/usr/bin/env bash
set -euo pipefail
host="${1:-127.0.0.1}"; port="${2:-8790}"
echo "# /health"
curl -fsS "http://${host}:${port}/health" || true
echo
echo "# /exec (must include t_ms)"
resp=$(curl -fsS -H 'content-type: application/json' \
  -d '{"language":"python","code":"print(1)","timeout_sec":1.0}' \
  "http://${host}:${port}/exec" || true)
echo "$resp"
if ! echo "$resp" | grep -q '"t_ms"'; then
  echo "ERROR: /exec response missing t_ms" >&2
  exit 2
fi