#!/usr/bin/env bash
# One-command Lens local dev loop: proxy + Lens worker + hot-reload dashboard.
#
#   LENS_DEV_PROXY_PORT   proxy port (default 4000)
#   LENS_DEV_UI_PORT      next dev port (default 3000)
#   LENS_DEV_MASTER_KEY   master key, also the admin UI password
#                         (default: random, generated once into .lens-dev/master_key)
#   LENS_DEV_CONFIG       proxy config to use instead of the generated one
#   LENS_DEV_DATABASE_URL Postgres URL (default: the tracing stack's litellm DB on :15432)
#   LENS_DEV_SEED         trace seed profile (default|large), same as --seed
#   LENS_DEV_SEED_LOGS    request-log seed profile (default|large), same as --seed-logs
#
# State (master key, worker token, generated config, logs) lives in .lens-dev/ (gitignored).
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_release_tag="sha-$(git -C "$repo_root" rev-parse HEAD)"
proxy_port="${LENS_DEV_PROXY_PORT:-4000}"
ui_port="${LENS_DEV_UI_PORT:-3000}"
lens_port="${LENS_DEV_SERVICE_PORT:-4318}"
state_dir="${LENS_DEV_STATE_DIR:-$repo_root/.lens-dev}"
log_dir="$state_dir/logs"
token_file="$state_dir/worker_token"
key_file="$state_dir/master_key"
service_key_file="$state_dir/service_key"
proxy_url="http://localhost:$proxy_port"
py="${LENS_DEV_PYTHON:-$repo_root/.venv/bin/python}"
database_url="${LENS_DEV_DATABASE_URL:-postgresql://litellm:litellm@127.0.0.1:15432/litellm}"
clickhouse_url=http://default:local-tracing@127.0.0.1:18123
master_key=""
startup_timeout="${LENS_DEV_STARTUP_TIMEOUT_SECONDS:-300}"
readiness_request_timeout="${LENS_DEV_READINESS_REQUEST_TIMEOUT_SECONDS:-5}"
pids=()

die() { echo "lens-dev: $*" >&2; exit 1; }
listening() { lsof -nP -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1; }

# A fixed key would let anyone who can reach the proxy sign in as admin, so default to a
# random key generated once per checkout and kept next to the worker token.
load_master_key() {
  if [ -n "${LENS_DEV_MASTER_KEY:-}" ]; then
    master_key="$LENS_DEV_MASTER_KEY"
    return
  fi
  if [ ! -s "$key_file" ]; then
    (umask 077 && printf 'sk-%s\n' "$(openssl rand -hex 24)" > "$key_file")
  fi
  master_key="$(cat "$key_file")"
}

# Only reuse a listener on 15432/18123 if it accepts the tracing stack's credentials;
# start the compose service when nothing is listening; fail if something else is.
# A LENS_DEV_DATABASE_URL is left to the proxy, which may use Prisma-only URL params.
ensure_services() {
  local services=()
  if [ -z "${LENS_DEV_DATABASE_URL:-}" ] && listening 15432; then
    "$py" -c 'import sys, psycopg; psycopg.connect(sys.argv[1], connect_timeout=5).close()' "$database_url" 2>/dev/null \
      || die "port 15432 is taken by something that isn't the tracing Postgres (litellm/litellm)"
  elif [ -z "${LENS_DEV_DATABASE_URL:-}" ]; then
    services+=(db)
  fi
  if listening 18123; then
    [ "$(curl -fsS --max-time 5 "$clickhouse_url/?query=SELECT%201" 2>/dev/null)" = 1 ] \
      || die "port 18123 is taken by something that isn't the tracing ClickHouse (default/local-tracing)"
  else
    services+=(clickhouse)
  fi
  if [ "${#services[@]}" -gt 0 ]; then
    LITELLM_MASTER_KEY="$master_key" LITELLM_LENS_SERVICE_TOKEN="${service_key:-}" \
      LITELLM_RELEASE_TAG="$source_release_tag" \
      docker compose -f docker/docker-compose.tracing.yml up -d --wait "${services[@]}"
  else
    echo "lens-dev: reusing running Postgres and ClickHouse"
  fi
}

write_default_config() {
  cat > "$1" <<'EOF'
model_list:
  - model_name: gpt-6.1-sol
    litellm_params:
      model: openai/gpt-6.1-sol
      api_key: os.environ/OPENAI_API_KEY
general_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
  store_prompts_in_spend_logs: true
  tracing:
    store:
      type: lens
EOF
}

# litellm's implicit load_dotenv() walks up from a worktree into the parent checkout's
# .env and picks up REDIS_* / UI_* from there. LITELLM_MODE=PRODUCTION turns that off;
# this prints export lines for the same .env minus those vars, so provider keys still load.
dotenv_exports() {
  "$py" - <<'PY'
import os, re, shlex
from dotenv import dotenv_values, find_dotenv

path = find_dotenv(usecwd=True)
skip = re.compile(r"REDIS_.*|UI_USERNAME|UI_PASSWORD|LITELLM_MODE|ANTHROPIC_BASE_URL|ANTHROPIC_AUTH_TOKEN|ANTHROPIC_CUSTOM_HEADERS|OPENAI_BASE_URL|OPENAI_API_BASE")
for key, value in (dotenv_values(path) if path else {}).items():
    if value is not None and key not in os.environ and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) and not skip.fullmatch(key):
        print(f"export {key}={shlex.quote(value)}")
PY
}

# Run in the proxy's subshell: drop inherited settings that would point it at someone
# else's services, then set the local stack's.
proxy_env() {
  local var
  # Claude Code and similar tools export these; provider calls would go to them.
  unset ANTHROPIC_BASE_URL ANTHROPIC_AUTH_TOKEN ANTHROPIC_CUSTOM_HEADERS OPENAI_BASE_URL OPENAI_API_BASE
  for var in $(compgen -e | grep '^REDIS_' || true); do unset "$var"; done
  eval "$1"
  export LITELLM_RELEASE_TAG="$source_release_tag"
  export LENS_WORKER_IMAGE=litellm-lens-worker:local
  export LITELLM_MODE=PRODUCTION
  export LITELLM_MASTER_KEY="$master_key"
  export LITELLM_SALT_KEY=sk-local-tracing-salt-key
  export DATABASE_URL="$database_url"
  export STORE_MODEL_IN_DB=True
  unset CLICKHOUSE_URL CLICKHOUSE_DATABASE
  export LITELLM_LENS_URL="http://127.0.0.1:$lens_port"
  export LITELLM_LENS_PUBLIC_URL="http://localhost:$lens_port"
  export LITELLM_LENS_SERVICE_TOKEN="${service_key:-}"
  export LITELLM_LOCAL_MODEL_COST_MAP=True
  export PROXY_BASE_URL="$proxy_url"
  export LITELLM_UI_PATH="$repo_root/ui/litellm-dashboard/out"
  export UI_USERNAME=admin
  export UI_PASSWORD="$master_key"
}

# POST JSON as the admin and print one field of the response; dies with the body on failure.
admin_post() {
  local body
  body="$(curl -sS --fail-with-body "$proxy_url$1" -H "Authorization: Bearer $master_key" \
    -H "Content-Type: application/json" -d "$2")" || die "POST $1 failed: $body"
  "$py" -c 'import json, sys; print(json.loads(sys.argv[1])[sys.argv[2]])' "$body" "$3"
}

register_worker() {
  local key_hash worker_token
  key_hash="$(admin_post /key/generate "{\"key_alias\": \"lens-dev-$(date +%s)\"}" token)"
  worker_token="$(admin_post /lens/workers/register "{\"name\": \"lens-dev\", \"analysis_key_id\": \"$key_hash\"}" token)"
  (umask 077 && printf '%s\n' "$worker_token" > "$token_file")
  echo "lens-dev: registered a new Lens worker (token in $token_file)"
}

# Auth runs before the handler, so protocol_version=1 answers 401 for a bad token and
# 409 for a good one without claiming a job.
ensure_worker_token() {
  local status
  if [ ! -s "$token_file" ]; then
    register_worker
    return
  fi
  status="$(curl -sS -o /dev/null -w '%{http_code}' -X POST "$proxy_url/lens/worker/claim?protocol_version=1" \
    -H "Authorization: Bearer $(cat "$token_file")")"
  case "$status" in
    409) echo "lens-dev: reusing worker token from $token_file" ;;
    401) echo "lens-dev: stored worker token was rejected"; register_worker ;;
    *) die "unexpected HTTP $status checking the worker token" ;;
  esac
}

wait_for_proxy() {
  local proxy_pid="$1"
  echo "lens-dev: waiting for the proxy (log: $log_dir/proxy.log)"
  for _ in $(seq 1 "$startup_timeout"); do
    kill -0 "$proxy_pid" 2>/dev/null || die "proxy exited; see $log_dir/proxy.log"
    curl -fsS --max-time "$readiness_request_timeout" "$proxy_url/health/readiness" -H "Authorization: Bearer $master_key" >/dev/null 2>&1 && return
    sleep 1
  done
  die "proxy not ready after ${startup_timeout}s; see $log_dir/proxy.log"
}

wait_for_lens() {
  local lens_pid="$1"
  for _ in $(seq 1 "$startup_timeout"); do
    kill -0 "$lens_pid" 2>/dev/null || die "Lens exited; see $log_dir/worker.log"
    curl -fsS --max-time "$readiness_request_timeout" "http://127.0.0.1:$lens_port/health/ready" >/dev/null 2>&1 && return
    sleep 1
  done
  die "Lens not ready after ${startup_timeout}s; see $log_dir/worker.log"
}

wait_for_ui() {
  local ui_pid="$1"
  echo "lens-dev: waiting for the UI (log: $log_dir/ui.log)"
  for _ in $(seq 1 "$startup_timeout"); do
    kill -0 "$ui_pid" 2>/dev/null || die "UI exited; see $log_dir/ui.log"
    if curl -fsS --max-time "$readiness_request_timeout" "http://localhost:$ui_port/ui/login/" >/dev/null 2>&1; then
      kill -0 "$ui_pid" 2>/dev/null || die "UI exited; see $log_dir/ui.log"
      return
    fi
    sleep 1
  done
  die "UI not ready after ${startup_timeout}s; see $log_dir/ui.log"
}

# Children run in their own process groups (set -m), so killing -pid takes their trees too.
cleanup() {
  local alive pid
  trap - EXIT INT TERM
  [ "${#pids[@]}" -gt 0 ] || return 0
  echo "lens-dev: stopping"
  for pid in "${pids[@]}"; do kill -TERM -- "-$pid" 2>/dev/null || true; done
  for _ in $(seq 1 20); do
    alive=0
    for pid in "${pids[@]}"; do kill -0 "$pid" 2>/dev/null && alive=1; done
    [ "$alive" = 0 ] && break
    sleep 0.5
  done
  for pid in "${pids[@]}"; do kill -KILL -- "-$pid" 2>/dev/null || true; done
}

build_dashboard() {
  local dashboard_dir="$repo_root/ui/litellm-dashboard"
  case "${LENS_DEV_BUILD_UI:-0}" in
    0) return ;;
    1) ;;
    *) die "LENS_DEV_BUILD_UI must be 0 or 1" ;;
  esac
  echo "lens-dev: building the proxy dashboard (log: $log_dir/ui-build.log)"
  (
    cd "$dashboard_dir"
    NEXT_PUBLIC_BASE_URL="" LENS_DEV_PROXY_URL="" "$repo_root/scripts/with_dashboard_node.sh" npm run build
  ) > "$log_dir/ui-build.log" 2>&1 || die "UI build failed; see $log_dir/ui-build.log"
}

# Trace fixtures feed Lens (ClickHouse + spend rows); request logs feed the Logs page
# (Postgres only) with rows sized to stress the log detail drawer.
seed_data() {
  (
    proxy_env ""
    export CLICKHOUSE_URL="$clickhouse_url"
    export CLICKHOUSE_DATABASE=litellm
    export LENS_DEV_UI_URL="http://localhost:$ui_port"
    if [ -n "$seed_profile" ]; then
      "$py" -m scripts.seed_tracing_fixtures --profile "$seed_profile" ${seed_options[@]+"${seed_options[@]}"}
    fi
    if [ -n "$seed_logs_profile" ]; then
      "$py" -m scripts.seed_request_logs --profile "$seed_logs_profile"
    fi
  )
}

# --seed and --seed-logs take an optional profile; a bare flag means default.
seed_profile_arg() {
  if [ "${1:-}" = default ] || [ "${1:-}" = large ]; then echo "$1"; else echo default; fi
}

parse_args() {
  seed_profile="${LENS_DEV_SEED:-}"
  seed_logs_profile="${LENS_DEV_SEED_LOGS:-}"
  seed_only=0
  seed_options=()
  while [ "$#" -gt 0 ]; do
    case "$1" in
      --seed)
        seed_profile="$(seed_profile_arg "${2:-}")"
        [ "$seed_profile" = "${2:-}" ] && shift
        ;;
      --seed-logs)
        seed_logs_profile="$(seed_profile_arg "${2:-}")"
        [ "$seed_logs_profile" = "${2:-}" ] && shift
        ;;
      --copies)
        [ "$#" -ge 2 ] && [[ "$2" =~ ^[1-9][0-9]*$ ]] || die "--copies requires a positive integer"
        seed_options=(--copies "$2"); shift ;;
      --seed-only) seed_only=1 ;;
      --help)
        echo "Usage: $0 [--seed [default|large]] [--seed-logs [default|large]] [--copies N] [--seed-only]"
        exit 0 ;;
      *) die "unknown argument: $1 (use --help)" ;;
    esac
    shift
  done
  if [ "$seed_only" = 1 ] && [ -z "$seed_profile" ] && [ -z "$seed_logs_profile" ]; then seed_profile=default; fi
  case "$seed_profile" in ""|default|large) ;; *) die "seed profile must be default or large" ;; esac
  case "$seed_logs_profile" in ""|default|large) ;; *) die "seed-logs profile must be default or large" ;; esac
  [ "${#seed_options[@]}" = 0 ] || [ -n "$seed_profile" ] || die "--copies requires --seed"
}

main() {
  local config_file exports proxy_pid ui_pid lens_pid pid key_hint
  parse_args "$@"
  if [ -n "${LENS_DEV_CONFIG:-}" ]; then
    [ -f "$LENS_DEV_CONFIG" ] || die "LENS_DEV_CONFIG not found: $LENS_DEV_CONFIG"
    config_file="$(cd "$(dirname "$LENS_DEV_CONFIG")" && pwd)/$(basename "$LENS_DEV_CONFIG")"
  fi
  cd "$repo_root"

  if [ "$seed_only" = 1 ]; then
    [ -s "$key_file" ] || [ -n "${LENS_DEV_MASTER_KEY:-}" ] || die "start make lens-dev before --seed-only"
    load_master_key
    seed_data
    return
  fi

  [[ "$startup_timeout" =~ ^[1-9][0-9]*$ ]] || die "LENS_DEV_STARTUP_TIMEOUT_SECONDS must be a positive integer"
  [[ "$readiness_request_timeout" =~ ^[1-9][0-9]*$ ]] || die "LENS_DEV_READINESS_REQUEST_TIMEOUT_SECONDS must be a positive integer"
  listening "$proxy_port" && die "port $proxy_port is in use; set LENS_DEV_PROXY_PORT"
  listening "$ui_port" && die "port $ui_port is in use; set LENS_DEV_UI_PORT"
  listening "$lens_port" && die "port $lens_port is in use; set LENS_DEV_SERVICE_PORT"
  [ "$proxy_port" != "$ui_port" ] || die "proxy and UI ports must differ"
  [ "$lens_port" != "$proxy_port" ] && [ "$lens_port" != "$ui_port" ] || die "Lens service port must differ from proxy and UI ports"
  mkdir -p "$log_dir"
  load_master_key
  if [ ! -s "$service_key_file" ]; then
    (umask 077 && openssl rand -hex 32 > "$service_key_file")
  fi
  service_key="$(cat "$service_key_file")"

  uv sync --inexact --frozen --extra proxy --group proxy-dev --no-install-project
  ensure_services
  "$py" scripts/prisma_generate_if_needed.py

  # cargo/maturin already fingerprint every crate's sources, so re-running this on each
  # start is a no-op (a couple seconds) when nothing changed and only rebuilds the
  # subset that did. An import check can't tell content-stale from content-fresh: a
  # `.so` built from an older commit still imports fine, it just no longer matches
  # what the current Python bindings (e.g. the trace store protocol) expect.
  echo "lens-dev: checking the Rust bridge (litellm.rust_bridge._native) is current; the ClickHouse trace store uses it"
  PYO3_PYTHON="$py" VIRTUAL_ENV="$repo_root/.venv" uvx --from maturin==1.15.0 maturin develop \
    --release --manifest-path litellm-rust/crates/python-bridge/Cargo.toml --features extension-module
  cargo build --locked --manifest-path litellm-rust/Cargo.toml -p litellm-lens

  if [ ! -x ui/litellm-dashboard/node_modules/.bin/next ]; then
    (cd ui/litellm-dashboard && "$repo_root/scripts/with_dashboard_node.sh" npm ci)
  fi

  build_dashboard

  if [ -z "${config_file:-}" ]; then
    config_file="$state_dir/config.yaml"
    write_default_config "$config_file"
  fi
  exports="$(dotenv_exports)"

  trap cleanup EXIT
  trap 'exit 130' INT TERM
  set -m

  (
    proxy_env "$exports"
    export PROXY_BASE_URL="http://localhost:$ui_port"
    exec "$py" litellm/proxy/proxy_cli.py --config "$config_file" --host 127.0.0.1 --port "$proxy_port"
  ) < /dev/null > "$log_dir/proxy.log" 2>&1 &
  proxy_pid=$!
  pids+=("$proxy_pid")

  (
    cd ui/litellm-dashboard
    NEXT_PUBLIC_BASE_URL="" NEXT_PUBLIC_USE_REWRITES=true LENS_DEV_PROXY_URL="$proxy_url" \
      exec "$repo_root/scripts/with_dashboard_node.sh" npx next dev -p "$ui_port"
  ) < /dev/null > "$log_dir/ui.log" 2>&1 &
  ui_pid=$!
  pids+=("$ui_pid")

  wait_for_ui "$ui_pid"
  wait_for_proxy "$proxy_pid"
  ensure_worker_token
  LITELLM_RELEASE_TAG="$source_release_tag" \
    LITELLM_MODE=PRODUCTION LITELLM_URL="$proxy_url" LENS_WORKER_TOKEN="$(cat "$token_file")" \
    LITELLM_LENS_SERVICE_TOKEN="$service_key" LITELLM_LENS_LISTEN="127.0.0.1:$lens_port" \
    CLICKHOUSE_URL="$clickhouse_url" CLICKHOUSE_DATABASE=litellm \
    "$repo_root/litellm-rust/target/debug/litellm-lens" \
    < /dev/null > "$log_dir/worker.log" 2>&1 &
  lens_pid=$!
  pids+=("$lens_pid")
  wait_for_lens "$lens_pid"
  if [ -n "$seed_profile" ] || [ -n "$seed_logs_profile" ]; then seed_data; fi

  key_hint="password in $key_file"
  [ -z "${LENS_DEV_MASTER_KEY:-}" ] || key_hint="password from LENS_DEV_MASTER_KEY"
  cat <<EOF

Lens dev is up. Ctrl-C stops everything.
  Log in:   http://localhost:$ui_port/ui/login/  (admin / $key_hint)
  Lens:     http://localhost:$ui_port/ui/lens/  (hot-reloads)
  Logs:     http://localhost:$ui_port/ui/?page=logs
  API:      $proxy_url
  Traces:   http://localhost:$lens_port/v1/traces
  Logs:     $log_dir/proxy.log
            $log_dir/worker.log
            $log_dir/ui.log
Restart for backend or worker edits; UI edits hot-reload.
EOF

  while :; do
    for pid in "${pids[@]}"; do
      kill -0 "$pid" 2>/dev/null || die "a child process (pid $pid) exited; check the logs above"
    done
    sleep 2
  done
}

# Sourcing (tests) only defines the functions.
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  main "$@"
fi
