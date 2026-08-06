#!/usr/bin/env bash
set -euo pipefail

max_attempts="${UV_SYNC_MAX_ATTEMPTS:-5}"
delay_seconds="${UV_SYNC_RETRY_DELAY_SECONDS:-15}"

export CARGO_HTTP_MULTIPLEXING="${CARGO_HTTP_MULTIPLEXING:-false}"
export CARGO_NET_RETRY="${CARGO_NET_RETRY:-5}"

if [[ "$#" -eq 0 ]]; then
  echo "usage: $0 <uv sync args...>" >&2
  exit 2
fi

for attempt in $(seq 1 "${max_attempts}"); do
  echo "uv sync attempt ${attempt}/${max_attempts}"
  status=0
  if uv sync "$@"; then
    exit 0
  else
    status=$?
  fi

  if [[ "${attempt}" -eq "${max_attempts}" ]]; then
    echo "uv sync failed after ${max_attempts} attempts" >&2
    exit "${status}"
  fi

  echo "uv sync failed; retrying in ${delay_seconds}s..."
  sleep "${delay_seconds}"
done
