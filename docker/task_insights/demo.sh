#!/usr/bin/env bash
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
env_file="$here/.env"
port="${TASK_INSIGHTS_PORT:-4000}"
project="${TASK_INSIGHTS_PROJECT:-litellm-task-insights-demo}"
launcher="bash docker/task_insights/demo.sh"

fail() {
  printf 'demo.sh: %s\n' "$*" >&2
  exit 1
}

compose() {
  TASK_INSIGHTS_PORT="$port" docker compose -p "$project" \
    -f "$here/docker-compose.yml" -f "$here/docker-compose.light.yml" "$@"
}

require_docker() {
  command -v docker >/dev/null || fail "Docker is not installed. Install Docker Desktop or Docker Engine with the Compose plugin."
  docker compose version >/dev/null 2>&1 || fail "Docker Compose v2 is missing. Install the docker compose plugin."
  docker info >/dev/null 2>&1 || fail "Docker is not running. Start it and try again."
}

random_key() {
  printf 'sk-%s' "$(od -An -N32 -tx1 /dev/urandom | tr -d ' \n')"
}

ensure_env() {
  [ -e "$env_file" ] || (umask 077 && : >"$env_file")
  if [ -s "$env_file" ] && [ -n "$(tail -c1 "$env_file")" ]; then
    printf '\n' >>"$env_file"
  fi
  for name in LITELLM_MASTER_KEY LITELLM_SALT_KEY; do
    grep -q "^$name=." "$env_file" || printf '%s=%s\n' "$name" "$(random_key)" >>"$env_file"
  done
}

port_busy() {
  (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null
}

warn_low_memory() {
  local memory
  memory="$(docker info --format '{{.MemTotal}}' 2>/dev/null || echo 0)"
  if [ "$memory" -lt 6000000000 ]; then
    printf 'Warning: Docker has %s GiB of memory. The dashboard build needs about 6 GiB, so raise it in Docker Desktop settings if the build fails.\n' \
      "$((memory / 1073741824))" >&2
  fi
}

start() {
  [[ "$port" =~ ^[0-9]+$ ]] || fail "TASK_INSIGHTS_PORT must be a port number, got '$port'."
  ensure_env
  if [ -z "$(compose ps -q --status running litellm 2>/dev/null)" ] && port_busy; then
    fail "Port $port is already in use on this machine. Pick a free one, e.g. TASK_INSIGHTS_PORT=4100 $launcher"
  fi
  warn_low_memory
  printf 'Building the preview image. The first run downloads about 1 GB of base images and builds the dashboard, which takes a few minutes.\n'
  compose build litellm
  compose up -d --wait litellm
  compose run --rm task-insights-seed
  cat <<EOF

Task Insights seeded preview is running.
  URL:       http://localhost:$port/ui
  Username:  admin
  Password:  run '$launcher password' to print it in your terminal
Sign in, open Usage, then pick the Task Insights tab.
Stop it with '$launcher stop'. The sample database is kept until '$launcher reset'.
EOF
}

case "${1:-start}" in
  start)
    require_docker
    start
    ;;
  stop)
    require_docker
    compose down
    ;;
  reset)
    require_docker
    compose down --volumes
    printf 'Deleted the preview database. Credentials in %s are kept.\n' "$env_file"
    ;;
  password)
    [ -s "$env_file" ] || fail "No credentials yet. Run '$launcher' first."
    sed -n 's/^LITELLM_MASTER_KEY=//p' "$env_file" | tail -n 1
    ;;
  logs)
    require_docker
    compose logs --tail 200 litellm
    ;;
  *)
    fail "Usage: $launcher [start|stop|reset|password|logs]"
    ;;
esac
