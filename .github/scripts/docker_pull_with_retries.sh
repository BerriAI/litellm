#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 || -z "${1:-}" || "${1:-}" == -* ]]; then
  echo "usage: $0 <image>" >&2
  exit 2
fi

readonly image="$1"
log="$(mktemp)"
readonly log
trap 'rm -f "$log"' EXIT

for attempt in 1 2 3; do
  echo "docker pull attempt ${attempt}/3 for ${image}"
  if timeout --kill-after=5s 45s docker pull "$image" > "$log" 2>&1; then
    cat "$log"
    exit 0
  else
    status=$?
    cat "$log" >&2
  fi

  if [[ "$status" -ne 124 ]] && ! grep -Eiq \
    'HTTP status: (429|5[0-9]{2})|toomanyrequests|i/o timeout|TLS handshake timeout|connection reset by peer' "$log"; then
    exit "$status"
  fi
  if [[ "$attempt" -eq 3 ]]; then
    echo "docker pull failed after ${attempt} attempts" >&2
    exit "$status"
  fi
  echo "transient image pull failure, retrying in $((attempt * 2))s" >&2
  sleep "$((attempt * 2))"
done
