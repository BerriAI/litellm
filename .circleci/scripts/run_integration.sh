#!/usr/bin/env bash
set -euo pipefail

if [ "${GITHUB_ACTIONS:-}" = true ]; then
  echo "Integration contracts are owned by CircleCI" >&2
  exit 1
fi

suite="${1:?integration suite required}"
results="test-results/integration-${suite}"
mkdir -p "$results"
shard_timeout=11m
if [ "$suite" = cost ]; then
  shard_timeout=20m
fi
integration_identity="$(.venv/bin/python -c 'import uuid; print(uuid.uuid4().hex)')"
upstream_pid=""
scripted_provider_pid=""
proxy_pid=""
peer_pid=""
launched_pid=""
guard_created=false
guard_installed=false
guard6_created=false
guard6_installed=false
cleanup() {
  original_status=$?
  trap - EXIT INT TERM
  sudo .venv/bin/python .circleci/scripts/stop_integration_processes.py \
    "$integration_identity" "$(id -u)" "$proxy_pid" "$peer_pid" "$upstream_pid" "$scripted_provider_pid" \
    > "$results/process-cleanup.txt" 2>&1 || original_status=1
  for owned_pid in "$peer_pid" "$proxy_pid" "$upstream_pid" "$scripted_provider_pid"; do
    if [ -n "$owned_pid" ]; then
      kill -- "-$owned_pid" 2>/dev/null || true
      for _ in {1..50}; do
        kill -0 -- "-$owned_pid" 2>/dev/null || break
        sleep 0.1
      done
      if kill -0 -- "-$owned_pid" 2>/dev/null; then
        kill -KILL -- "-$owned_pid" 2>/dev/null || true
        original_status=1
      fi
      wait "$owned_pid" 2>/dev/null || true
    fi
  done
  if [ "$guard_installed" = true ]; then
    sudo iptables -D OUTPUT -m owner --uid-owner "$(id -u)" -j integration_only || original_status=1
  fi
  if [ "$guard_created" = true ]; then
    sudo iptables -F integration_only || original_status=1
    sudo iptables -X integration_only || original_status=1
  fi
  if [ "$guard6_installed" = true ]; then
    sudo ip6tables -D OUTPUT -m owner --uid-owner "$(id -u)" -j integration_only || original_status=1
  fi
  if [ "$guard6_created" = true ]; then
    sudo ip6tables -F integration_only || original_status=1
    sudo ip6tables -X integration_only || original_status=1
  fi
  printf '%s\n' "$original_status" > "$results/exit-status.txt"
  exit "$original_status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

export PATH="$PWD/.venv/bin:$PATH"
export PYTHONPATH="$PWD:$PWD/tests:$PWD/tests/e2e"
export DATABASE_URL="postgresql://postgres:postgres@127.0.0.1:5432/circle_test"
export REDIS_HOST=127.0.0.1 REDIS_PORT=6379
export LITELLM_MASTER_KEY=sk-integration-master LITELLM_SALT_KEY=sk-integration-salt
export LITELLM_MODE=PRODUCTION LITELLM_LOCAL_MODEL_COST_MAP=True
export STORE_MODEL_IN_DB=True AWS_EC2_METADATA_DISABLED=true DO_NOT_TRACK=1
export INTEGRATION_PROXY_URL=http://127.0.0.1:4000
export INTEGRATION_PEER_URL=""
export INTEGRATION_UPSTREAM_URL=http://127.0.0.1:8190
export INTEGRATION_SCRIPTED_PROVIDER_URL=""
export INTEGRATION_MASTER_KEY="$LITELLM_MASTER_KEY"
export LITELLM_UI_PATH="$PWD/litellm/proxy/_experimental/out"
if [ "$suite" = browser ]; then
  export LITELLM_UI_PATH="$PWD/ui/litellm-dashboard/out"
  test -f "$LITELLM_UI_PATH/index.html"
fi
export INTEGRATION_SEED="$(.venv/bin/python -c 'import hashlib,os; print(int(hashlib.sha256((os.environ.get("CIRCLE_SHA1", "local") + os.environ.get("CIRCLE_WORKFLOW_ID", "local")).encode()).hexdigest()[:8],16))')"
export INTEGRATION_ORDER_SEED="$INTEGRATION_SEED"

uv run --no-sync prisma generate --schema litellm/proxy/schema.prisma > "$results/prisma-generate.log" 2>&1

sudo iptables -N integration_only
guard_created=true
sudo iptables -A integration_only -o lo -j ACCEPT
sudo iptables -A integration_only -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT
for service in postgres-db redis-cache; do
  address="$(docker inspect --format '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' "$service")"
  sudo iptables -A integration_only -d "$address" -j ACCEPT
done
sudo iptables -A integration_only -j REJECT
sudo iptables -I OUTPUT 1 -m owner --uid-owner "$(id -u)" -j integration_only
guard_installed=true
sudo ip6tables -N integration_only
guard6_created=true
sudo ip6tables -A integration_only -o lo -j ACCEPT
sudo ip6tables -A integration_only -j REJECT
sudo ip6tables -I OUTPUT 1 -m owner --uid-owner "$(id -u)" -j integration_only
guard6_installed=true

if curl --noproxy '*' --connect-timeout 2 -s http://198.51.100.1 >/dev/null 2>&1; then
  echo "Unexpected outbound network access" >&2
  exit 1
fi
sudo iptables -L integration_only -n -v -x > "$results/egress-guard.txt"
awk '$3 == "REJECT" && $1 > 0 { rejected=1 } END { exit !rejected }' "$results/egress-guard.txt"

setsid env -i PATH="$PATH" HOME="$HOME" PYTHONPATH="$PYTHONPATH" INTEGRATION_RUN_ID="$integration_identity" \
  .venv/bin/python -m integration._support.upstream > "$results/upstream.log" 2>&1 &
upstream_pid=$!
if [ "$suite" = cost ]; then
  export INTEGRATION_SCRIPTED_PROVIDER_URL=http://127.0.0.1:8191
  setsid env -i PATH="$PATH" HOME="$HOME" PYTHONPATH="$PYTHONPATH" INTEGRATION_RUN_ID="$integration_identity" \
    .venv/bin/python -m integration._support.scripted_provider --port 8191 \
    > "$results/scripted-provider.log" 2>&1 &
  scripted_provider_pid=$!
  for _ in {1..90}; do
    if curl --noproxy '*' -fsS "$INTEGRATION_SCRIPTED_PROVIDER_URL/health" >/dev/null 2>&1; then
      break
    fi
    sleep 1
  done
  curl --noproxy '*' -fsS "$INTEGRATION_SCRIPTED_PROVIDER_URL/health" >/dev/null
fi
start_proxy() {
  local port="$1"
  local log_name="$2"
  local -a cost_map_env
  if [ "$suite" = cost ]; then
    cost_map_env=(
      "LITELLM_MODEL_COST_MAP_URL=$INTEGRATION_SCRIPTED_PROVIDER_URL/_cost_map"
      "MODEL_COST_MAP_MIN_MODEL_COUNT=1"
      "MODEL_COST_MAP_MAX_SHRINK_RATIO=0"
    )
  else
    cost_map_env=("LITELLM_LOCAL_MODEL_COST_MAP=True")
  fi
  setsid env -i PATH="$PATH" HOME="$HOME" PYTHONPATH="$PYTHONPATH" INTEGRATION_RUN_ID="$integration_identity" \
    DATABASE_URL="$DATABASE_URL" REDIS_HOST="$REDIS_HOST" REDIS_PORT="$REDIS_PORT" \
    LITELLM_MASTER_KEY="$LITELLM_MASTER_KEY" LITELLM_SALT_KEY="$LITELLM_SALT_KEY" LITELLM_UI_PATH="$LITELLM_UI_PATH" PROXY_BASE_URL="http://127.0.0.1:$port" \
    LITELLM_MODE=PRODUCTION STORE_MODEL_IN_DB=True "${cost_map_env[@]}" \
    AWS_EC2_METADATA_DISABLED=true DO_NOT_TRACK=1 \
    .venv/bin/python -m integration._support.proxy --config tests/integration/proxy_config.yaml \
    --host 127.0.0.1 --port "$port" --num_workers 1 --telemetry False \
    --use_prisma_db_push --enforce_prisma_migration_check \
    > "$results/$log_name" 2>&1 &
  launched_pid=$!
}
start_proxy 4000 proxy.log
proxy_pid="$launched_pid"
.venv/bin/python .circleci/scripts/wait_integration_services.py
if [ "$suite" = management ]; then
  export INTEGRATION_PEER_URL=http://127.0.0.1:4001
  start_proxy 4001 peer.log
  peer_pid="$launched_pid"
  .venv/bin/python .circleci/scripts/wait_integration_services.py
fi

if [ "$suite" = providers ]; then
  INTEGRATION_RUN_ID="$integration_identity" .venv/bin/python -m pytest --noconftest -o addopts= \
    --strict-markers --strict-config -p no:pytest-retry -p no:rerunfailures --timeout=30 \
    tests/e2e/test_provider_edge.py::TestReplayMode::test_content_drift_returns_the_miss_status_naming_both_keys \
    tests/e2e/test_provider_edge.py::TestReplayMode::test_exhausted_key_returns_the_miss_status \
    tests/e2e/test_provider_edge.py::TestReplayLeftover::test_partially_consumed_recording_names_the_leftover \
    tests/e2e/test_provider_edge.py::TestStreamingFidelity::test_replay_of_a_stream_makes_no_provider_connection \
    --junitxml="$results/replay-controls.xml"
fi

if [ "$suite" = browser ]; then
  export E2E_UI_BASE_URL="$INTEGRATION_PROXY_URL" E2E_UI_ARTIFACT_DIR="$PWD/$results"
  export INTEGRATION_PYTHON="$PWD/.venv/bin/python"
  timeout --signal=TERM --kill-after=20s 3m env -i PATH="$PATH" HOME="$HOME" PYTHONPATH="$PYTHONPATH" \
    INTEGRATION_RUN_ID="$integration_identity" DATABASE_URL="$DATABASE_URL" \
    INTEGRATION_UPSTREAM_URL="$INTEGRATION_UPSTREAM_URL" INTEGRATION_PYTHON="$INTEGRATION_PYTHON" \
    E2E_UI_BASE_URL="$E2E_UI_BASE_URL" E2E_UI_ARTIFACT_DIR="$E2E_UI_ARTIFACT_DIR" \
    LITELLM_MASTER_KEY="$LITELLM_MASTER_KEY" CI=true \
    node tests/e2e/ui/node_modules/@playwright/test/cli.js test --config tests/e2e/ui/integration.config.ts
  .venv/bin/python .circleci/scripts/verify_integration_browser.py "$results/browser-results.json"
  exit 0
fi

timeout --signal=TERM --kill-after=20s "$shard_timeout" env -i PATH="$PATH" HOME="$HOME" PYTHONPATH="$PYTHONPATH" \
  INTEGRATION_RUN_ID="$integration_identity" \
  DATABASE_URL="$DATABASE_URL" REDIS_HOST="$REDIS_HOST" REDIS_PORT="$REDIS_PORT" \
  INTEGRATION_PROXY_URL="$INTEGRATION_PROXY_URL" INTEGRATION_PEER_URL="$INTEGRATION_PEER_URL" \
  INTEGRATION_UPSTREAM_URL="$INTEGRATION_UPSTREAM_URL" \
  INTEGRATION_SCRIPTED_PROVIDER_URL="$INTEGRATION_SCRIPTED_PROVIDER_URL" \
  INTEGRATION_MASTER_KEY="$INTEGRATION_MASTER_KEY" LITELLM_MODE=PRODUCTION \
  INTEGRATION_SEED="$INTEGRATION_SEED" \
  INTEGRATION_ORDER_SEED="$INTEGRATION_ORDER_SEED" \
  LITELLM_LOCAL_MODEL_COST_MAP=True AWS_EC2_METADATA_DISABLED=true DO_NOT_TRACK=1 \
  .venv/bin/python tests/integration/run.py "$suite" --results "$results"
