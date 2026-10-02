#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

docker compose -f docker/docker-compose.tracing.yml up -d --wait db clickhouse
uv sync --inexact --frozen --extra proxy --group proxy-dev --no-install-project
"$repo_root/.venv/bin/python" scripts/prisma_generate_if_needed.py
VIRTUAL_ENV="$repo_root/.venv" uvx --from maturin==1.15.0 maturin develop \
  --release --manifest-path litellm-rust/crates/python-bridge/Cargo.toml --features extension-module

config_file="$(mktemp "${TMPDIR:-/tmp}/litellm-tracing-local.XXXXXX.yaml")"
trap 'rm -f "$config_file"' EXIT
cat > "$config_file" <<'EOF'
model_list:
  - model_name: claude-sonnet
    litellm_params:
      model: anthropic/claude-sonnet-5-5
      api_key: os.environ/ANTHROPIC_API_KEY
general_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
  tracing:
    store:
      type: clickhouse
      url: os.environ/CLICKHOUSE_URL
      retention_days: 14
EOF

export LITELLM_MASTER_KEY=sk-local-tracing
export LITELLM_SALT_KEY=sk-local-tracing-salt-key
export DATABASE_URL=postgresql://litellm:litellm@127.0.0.1:15432/litellm
export STORE_MODEL_IN_DB=True
export CLICKHOUSE_URL=http://default:local-tracing@127.0.0.1:18123
export CLICKHOUSE_DATABASE=litellm
export LITELLM_LOCAL_MODEL_COST_MAP=True

(
  cd "$repo_root/ui/litellm-dashboard"
  "$repo_root/scripts/with_dashboard_node.sh" npm ci
  NEXT_PUBLIC_BASE_URL= "$repo_root/scripts/with_dashboard_node.sh" npm run build
)
export LITELLM_UI_PATH="$repo_root/ui/litellm-dashboard/out"

printf 'Dashboard: http://127.0.0.1:4002/ui/\nProxy: http://127.0.0.1:4002\nMaster key: %s\n' "$LITELLM_MASTER_KEY"
"$repo_root/.venv/bin/python" litellm/proxy/proxy_cli.py \
  --config "$config_file" --host 127.0.0.1 --port 4002
