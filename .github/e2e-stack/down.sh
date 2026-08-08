#!/usr/bin/env bash
set -uo pipefail

STACK_DIR="${E2E_STACK_DIR:-${RUNNER_TEMP:-/tmp}/litellm-e2e-stack}"

for pid_file in "${STACK_DIR}"/pids/*.pid; do
  [[ -f "${pid_file}" ]] || continue
  pkill -TERM -P "$(cat "${pid_file}")" 2>/dev/null
  kill -TERM "$(cat "${pid_file}")" 2>/dev/null
  rm -f "${pid_file}"
done

for container in e2e-nginx e2e-valkey e2e-jaeger e2e-postgres; do
  docker rm -f "${container}" >/dev/null 2>&1
done

exit 0
