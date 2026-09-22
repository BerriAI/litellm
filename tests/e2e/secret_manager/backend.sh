#!/usr/bin/env bash
# Boot (or stop) one secret manager backend in Docker for a secret_manager lane.
#
#   bash tests/e2e/secret_manager/backend.sh up <system>
#   bash tests/e2e/secret_manager/backend.sh down <system>
#
# <system> is a key of secret_backends.BACKENDS. `up` replaces any earlier container
# of this script's for that system, waits for the manager to answer, and writes two
# env files under ${E2E_SECRET_MANAGER_DIR:-/tmp/litellm-e2e-secret-manager}/<system>/:
#   proxy.env  the manager's credentials for the proxy booted from
#              tests/e2e/gateway/secret_manager_<system>_ci_config.yml
#   tests.env  E2E_SECRET_MANAGER=<system> plus the store's E2E_* env for pytest
# The manager listens on 127.0.0.1:${E2E_SECRET_MANAGER_PORT:-<its usual port>}.
# Keep the directory private: both files hold a working admin credential.
set -euo pipefail

usage() {
  local systems
  systems=$(declare -F | sed -n 's/^declare -f up_//p' | paste -sd '|' -)
  echo "usage: $0 up|down $systems" >&2
  exit 2
}

action=${1:-}
system=${2:-}
dir=${E2E_SECRET_MANAGER_DIR:-/tmp/litellm-e2e-secret-manager}/$system
name=litellm-e2e-$system

wait_for() {
  local url=$1
  for _ in $(seq 1 90); do
    if curl -sf -o /dev/null "$url"; then
      return 0
    fi
    sleep 2
  done
  echo "$system did not answer at $url" >&2
  return 1
}

down() {
  docker rm -f "$name" "$name-db" >/dev/null 2>&1 || true
  docker network rm "$name" >/dev/null 2>&1 || true
  rm -rf "$dir"
}

up_hashicorp_vault() {
  local port=${E2E_SECRET_MANAGER_PORT:-8200}
  local token
  token=e2e-$(openssl rand -hex 16)
  docker run -d --name "$name" -p "127.0.0.1:$port:8200" --cap-add IPC_LOCK \
    -e VAULT_DEV_ROOT_TOKEN_ID="$token" hashicorp/vault:1.20 >/dev/null
  wait_for "http://127.0.0.1:$port/v1/sys/health"
  printf 'HCP_VAULT_ADDR=http://127.0.0.1:%s\nHCP_VAULT_TOKEN=%s\n' "$port" "$token" >"$dir/proxy.env"
  printf 'E2E_VAULT_ADDR=http://127.0.0.1:%s\nE2E_VAULT_TOKEN=%s\n' "$port" "$token" >"$dir/tests.env"
}

up_cyberark() {
  local port=${E2E_SECRET_MANAGER_PORT:-8080}
  local data_key api_key
  docker network create "$name" >/dev/null
  docker run -d --name "$name-db" --network "$name" -e POSTGRES_HOST_AUTH_METHOD=trust postgres:15 >/dev/null
  data_key=$(docker run --rm cyberark/conjur:1.24 data-key generate)
  docker run -d --name "$name" --network "$name" -p "127.0.0.1:$port:80" \
    -e DATABASE_URL="postgres://postgres@$name-db/postgres" -e CONJUR_DATA_KEY="$data_key" \
    -e CONJUR_AUTHENTICATORS=authn cyberark/conjur:1.24 server >/dev/null
  wait_for "http://127.0.0.1:$port/"
  docker exec "$name" conjurctl account create --name default >/dev/null
  api_key=$(docker exec "$name" conjurctl role retrieve-key default:user:admin | tr -d '\r\n')
  printf 'CYBERARK_API_BASE=http://127.0.0.1:%s\nCYBERARK_ACCOUNT=default\nCYBERARK_USERNAME=admin\nCYBERARK_API_KEY=%s\n' \
    "$port" "$api_key" >"$dir/proxy.env"
  printf 'E2E_CYBERARK_API_BASE=http://127.0.0.1:%s\nE2E_CYBERARK_ACCOUNT=default\nE2E_CYBERARK_USERNAME=admin\nE2E_CYBERARK_API_KEY=%s\n' \
    "$port" "$api_key" >"$dir/tests.env"
}

# A system is supported exactly when it has an up_<system> above.
[[ $# -eq 2 && -n $system ]] && declare -F "up_$system" >/dev/null || usage

case $action in
  up)
    down
    mkdir -p "$dir"
    chmod 700 "$dir"
    "up_$system"
    echo "E2E_SECRET_MANAGER=$system" >>"$dir/tests.env"
    chmod 600 "$dir/proxy.env" "$dir/tests.env"
    echo "$system is up; env in $dir/proxy.env (proxy) and $dir/tests.env (pytest)"
    ;;
  down) down ;;
  *) usage ;;
esac
