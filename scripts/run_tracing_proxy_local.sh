#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

seed_fixtures=0
case "${1:-}" in
  --seed) seed_fixtures=1 ;;
  "") ;;
  *) echo "Usage: $0 [--seed]" >&2; exit 2 ;;
esac

if lsof -nP -iTCP:4002 -sTCP:LISTEN >/dev/null 2>&1; then
  echo "Port 4002 is already in use. Stop the existing proxy before starting this stack" >&2
  exit 1
fi

docker compose -f docker/docker-compose.tracing.yml up -d --wait db clickhouse
uv sync --inexact --frozen --extra proxy --group proxy-dev --no-install-project
"$repo_root/.venv/bin/python" scripts/prisma_generate_if_needed.py
VIRTUAL_ENV="$repo_root/.venv" uvx --from maturin==1.15.0 maturin develop \
  --release --manifest-path litellm-rust/crates/python-bridge/Cargo.toml --features extension-module

config_file="$(mktemp "${TMPDIR:-/tmp}/litellm-tracing-local.XXXXXX")"
proxy_pid=""
cleanup() {
  if [ -n "$proxy_pid" ]; then
    kill "$proxy_pid" 2>/dev/null || true
    wait "$proxy_pid" 2>/dev/null || true
  fi
  rm -f "$config_file"
}
trap cleanup EXIT
trap 'exit 130' INT TERM
cat > "$config_file" <<'EOF'
model_list:
  - model_name: openai/gpt-6-luna
    litellm_params:
      model: openai/gpt-6-luna
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

export LITELLM_MASTER_KEY=sk-1234
export LITELLM_DANGEROUSLY_PERMIT_WEAK_OR_UNSET_MASTER_KEY=true
export LITELLM_SALT_KEY=sk-local-tracing-salt-key
export DATABASE_URL=postgresql://litellm:litellm@127.0.0.1:15432/litellm
export STORE_MODEL_IN_DB=True
export CLICKHOUSE_URL=http://default:local-tracing@127.0.0.1:18123
export CLICKHOUSE_DATABASE=litellm
export LITELLM_LOCAL_MODEL_COST_MAP=True
export PROXY_BASE_URL=http://127.0.0.1:4002

(
  cd "$repo_root/ui/litellm-dashboard"
  "$repo_root/scripts/with_dashboard_node.sh" npm ci
  NEXT_PUBLIC_BASE_URL= "$repo_root/scripts/with_dashboard_node.sh" npm run build
)
export LITELLM_UI_PATH="$repo_root/ui/litellm-dashboard/out"

printf 'Dashboard: http://127.0.0.1:4002/ui/\nProxy: http://127.0.0.1:4002\nMaster key: %s\n' "$LITELLM_MASTER_KEY"
"$repo_root/.venv/bin/python" litellm/proxy/proxy_cli.py \
  --config "$config_file" --host 127.0.0.1 --port 4002 &
proxy_pid=$!

if [ "$seed_fixtures" = "1" ]; then
  ready=0
  for attempt in $(seq 1 180); do
    kill -0 "$proxy_pid" 2>/dev/null || { echo "Proxy exited before seeding" >&2; exit 1; }
    if curl --fail --silent "$PROXY_BASE_URL/health/readiness" \
      -H "Authorization: Bearer $LITELLM_MASTER_KEY" >/dev/null; then
      ready=1
      break
    fi
    sleep 1
  done
  [ "$ready" = "1" ] || { echo "Proxy did not become ready within 180 seconds" >&2; exit 1; }
  "$repo_root/.venv/bin/python" -m scripts.seed_tracing_fixtures
fi

wait "$proxy_pid"
