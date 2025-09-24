#!/usr/bin/env zsh
set -euo pipefail
setopt NO_NOMATCH PIPE_FAIL

# Config: adjust path if your compose file lives elsewhere
COMPOSE_FILE="local/docker/compose.exec.yml"

# Args (optional)
TAIL_LINES="${1:-200}"     # how many lines to show
FOLLOW="${2:-}"            # pass "follow" to stream

need() { command -v "$1" >/dev/null 2>&1 || { print -r -- "missing: $1" >&2; exit 1; } }
need docker
need sed

cid() {
  # returns container ID for a compose service
  docker compose -f "$COMPOSE_FILE" ps -q "$1" 2>/dev/null || true
}

print_header() {
  print -r -- "\n========== $1 ==========\n"
}

print_status() {
  print_header "docker compose services"
  docker compose -f "$COMPOSE_FILE" ps || true

  print_header "ports"
  docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}' | sed -n '1,200p' || true
}

logs_for() {
  local svc="$1"
  local id; id="$(cid "$svc")"
  if [[ -z "$id" ]]; then
    print -r -- "no container for service: $svc" >&2
    return
  fi

  # health + name
  local name health
  name="$(docker inspect -f '{{.Name}}' "$id" 2>/dev/null | sed 's#^/##')"
  health="$(docker inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{else}}n/a{{end}}' "$id" 2>/dev/null || true)"
  print_header "$svc  name=$name  health=$health"

  # show recent logs
  if [[ "$FOLLOW" == "follow" ]]; then
    docker logs -f --tail="$TAIL_LINES" "$id"
  else
    docker logs --tail="$TAIL_LINES" "$id"
  fi
}

print_status
logs_for "agent-api"
logs_for "exec-rpc"
