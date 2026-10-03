#!/usr/bin/env bash
# One-command Lens local dev loop: proxy + Lens worker + hot-reload dashboard.
#
#   LENS_DEV_PROXY_PORT   proxy port (default 4000)
#   LENS_DEV_UI_PORT      next dev port (default 3000)
#   LENS_DEV_MASTER_KEY   master key, also the admin UI password
#                         (default: random, generated once into .lens-dev/master_key)
#   LENS_DEV_CONFIG       proxy config to use instead of the generated one
#   LENS_DEV_DATABASE_URL Postgres URL (default: the tracing stack's litellm DB on :15432)
#   LENS_DEV_REBUILD_RUST=1  rebuild the Rust bridge even if it imports
#
# State (master key, worker token, generated config, logs) lives in .lens-dev/ (gitignored).
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
proxy_port="${LENS_DEV_PROXY_PORT:-4000}"
ui_port="${LENS_DEV_UI_PORT:-3000}"
state_dir="${LENS_DEV_STATE_DIR:-$repo_root/.lens-dev}"
log_dir="$state_dir/logs"
token_file="$state_dir/worker_token"
key_file="$state_dir/master_key"
proxy_url="http://localhost:$proxy_port"
py="${LENS_DEV_PYTHON:-$repo_root/.venv/bin/python}"
database_url="${LENS_DEV_DATABASE_URL:-postgresql://litellm:litellm@127.0.0.1:15432/litellm}"
clickhouse_url=http://default:local-tracing@127.0.0.1:18123
master_key=""
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
    docker compose -f docker/docker-compose.tracing.yml up -d --wait "${services[@]}"
  else
    echo "lens-dev: reusing running Postgres and ClickHouse"
  fi
}

write_default_config() {
  cat > "$1" <<'EOF'
model_list:
  - model_name: gpt-4.1-mini
    litellm_params:
      model: openai/gpt-4.1-mini
      api_key: os.environ/OPENAI_API_KEY
general_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
  store_prompts_in_spend_logs: true
  tracing:
    store:
      type: clickhouse
      url: os.environ/CLICKHOUSE_URL
      retention_days: 14
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
  export LITELLM_MODE=PRODUCTION
  export LITELLM_MASTER_KEY="$master_key"
  if [ "$master_key" = sk-1234 ]; then export LITELLM_DANGEROUSLY_PERMIT_WEAK_OR_UNSET_MASTER_KEY=true; fi
  export LITELLM_SALT_KEY=sk-local-tracing-salt-key
  export DATABASE_URL="$database_url"
  export STORE_MODEL_IN_DB=True
  export CLICKHOUSE_URL="$clickhouse_url"
  export CLICKHOUSE_DATABASE=litellm
  export LITELLM_LOCAL_MODEL_COST_MAP=True
  export PROXY_BASE_URL="$proxy_url"
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
  for _ in $(seq 1 300); do
    kill -0 "$proxy_pid" 2>/dev/null || die "proxy exited; see $log_dir/proxy.log"
    curl -fsS "$proxy_url/health/readiness" -H "Authorization: Bearer $master_key" >/dev/null 2>&1 && return
    sleep 1
  done
  die "proxy not ready after 300s; see $log_dir/proxy.log"
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

main() {
  local config_file exports proxy_pid pid key_hint
  if [ -n "${LENS_DEV_CONFIG:-}" ]; then
    [ -f "$LENS_DEV_CONFIG" ] || die "LENS_DEV_CONFIG not found: $LENS_DEV_CONFIG"
    config_file="$(cd "$(dirname "$LENS_DEV_CONFIG")" && pwd)/$(basename "$LENS_DEV_CONFIG")"
  fi
  cd "$repo_root"

  listening "$proxy_port" && die "port $proxy_port is in use; set LENS_DEV_PROXY_PORT"
  listening "$ui_port" && die "port $ui_port is in use; set LENS_DEV_UI_PORT"
  [ "$proxy_port" != "$ui_port" ] || die "proxy and UI ports must differ"
  mkdir -p "$log_dir"
  load_master_key

  uv sync --inexact --frozen --extra proxy --group proxy-dev --no-install-project
  ensure_services
  "$py" scripts/prisma_generate_if_needed.py

  if [ "${LENS_DEV_REBUILD_RUST:-0}" = "1" ] || ! "$py" -c "import litellm.rust_bridge._native" >/dev/null 2>&1; then
    echo "lens-dev: building the Rust bridge (litellm.rust_bridge._native); the ClickHouse trace store uses it"
    VIRTUAL_ENV="$repo_root/.venv" uvx --from maturin==1.15.0 maturin develop \
      --release --manifest-path litellm-rust/crates/python-bridge/Cargo.toml --features extension-module
  fi

  if [ ! -x ui/litellm-dashboard/node_modules/.bin/next ]; then
    (cd ui/litellm-dashboard && "$repo_root/scripts/with_dashboard_node.sh" npm ci)
  fi

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
    exec "$py" litellm/proxy/proxy_cli.py --config "$config_file" --host 127.0.0.1 --port "$proxy_port"
  ) < /dev/null > "$log_dir/proxy.log" 2>&1 &
  proxy_pid=$!
  pids+=("$proxy_pid")

  (
    cd ui/litellm-dashboard
    NEXT_PUBLIC_BASE_URL="$proxy_url" exec "$repo_root/scripts/with_dashboard_node.sh" npx next dev -p "$ui_port"
  ) < /dev/null > "$log_dir/ui.log" 2>&1 &
  pids+=("$!")

  wait_for_proxy "$proxy_pid"
  ensure_worker_token

  LITELLM_MODE=PRODUCTION LITELLM_URL="$proxy_url" LENS_WORKER_TOKEN="$(cat "$token_file")" \
    "$py" -c "import asyncio, logging; from litellm.proxy.lens.worker import main; logging.basicConfig(level=logging.INFO); asyncio.run(main())" \
    < /dev/null > "$log_dir/worker.log" 2>&1 &
  pids+=("$!")

  key_hint="password in $key_file"
  [ -z "${LENS_DEV_MASTER_KEY:-}" ] || key_hint="password from LENS_DEV_MASTER_KEY"
  cat <<EOF

Lens dev is up. Ctrl-C stops everything.
  Log in:   $proxy_url/ui/login  (admin / $key_hint)
  Lens:     http://localhost:$ui_port/lens
  Logs:     $log_dir/proxy.log
            $log_dir/worker.log
            $log_dir/ui.log
Restart (Ctrl-C, make lens-dev) to pick up backend or worker edits; the UI hot-reloads.
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
