#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
KEYCLOAK_IMAGE="${E2E_KEYCLOAK_IMAGE:-quay.io/keycloak/keycloak@sha256:ff4257d0d64efbe99ed1ddfaf07765cc3c36dc7518bf8324d41961327f441c54}"
KEYCLOAK_PORT="${E2E_KEYCLOAK_PORT:-8081}"
POSTGRES_IMAGE="${E2E_POSTGRES_IMAGE:-postgres:16.6}"
: "${DATABASE_HOST:?}" "${DATABASE_PORT:?}" "${DATABASE_USER:?}" "${DATABASE_PASSWORD:?}" "${DATABASE_NAME:?}"

DB_HOST="${DATABASE_HOST}"
DB_NETWORK_ARGS=(--network bridge)
IDP_NETWORK_ARGS=(-p "127.0.0.1:${KEYCLOAK_PORT}:${KEYCLOAK_PORT}")
if [[ "$(uname)" == "Linux" ]]; then
  DB_NETWORK_ARGS=(--network host)
  IDP_NETWORK_ARGS=(--network host)
elif [[ "${DB_HOST}" == "127.0.0.1" || "${DB_HOST}" == "localhost" ]]; then
  DB_HOST=host.docker.internal
fi

docker run --rm "${DB_NETWORK_ARGS[@]}" -e "PGPASSWORD=${DATABASE_PASSWORD}" \
  "${POSTGRES_IMAGE}" psql -h "${DB_HOST}" -p "${DATABASE_PORT}" \
  -U "${DATABASE_USER}" -d "${DATABASE_NAME}" -v ON_ERROR_STOP=1 \
  -c 'CREATE SCHEMA IF NOT EXISTS keycloak' >/dev/null

docker rm -f e2e-keycloak >/dev/null 2>&1 || true
docker run -d --name e2e-keycloak "${IDP_NETWORK_ARGS[@]}" --memory 1536m \
  -v "${REPO_ROOT}/tests/e2e/idp_realm.json:/opt/keycloak/data/import/realm.json:ro" \
  -e KC_DB=postgres -e "KC_DB_URL_HOST=${DB_HOST}" -e "KC_DB_URL_PORT=${DATABASE_PORT}" \
  -e "KC_DB_URL_DATABASE=${DATABASE_NAME}" -e KC_DB_SCHEMA=keycloak \
  -e "KC_DB_USERNAME=${DATABASE_USER}" -e "KC_DB_PASSWORD=${DATABASE_PASSWORD}" \
  -e KC_DB_POOL_INITIAL_SIZE=2 -e KC_DB_POOL_MIN_SIZE=2 -e KC_DB_POOL_MAX_SIZE=10 \
  -e "KC_HTTP_PORT=${KEYCLOAK_PORT}" -e KC_BOOTSTRAP_ADMIN_USERNAME=admin \
  -e KC_BOOTSTRAP_ADMIN_PASSWORD=e2e-ephemeral-idp-not-a-secret \
  "${KEYCLOAK_IMAGE}" start-dev --import-realm >/dev/null

deadline=$((SECONDS + ${E2E_KEYCLOAK_STARTUP_TIMEOUT:-300}))
until curl -fsS --connect-timeout 2 --max-time 3 \
  "http://127.0.0.1:${KEYCLOAK_PORT}/realms/litellm-e2e/.well-known/openid-configuration" >/dev/null 2>&1; do
  if ((SECONDS >= deadline)); then
    echo 'e2e-stack: timed out waiting for the Keycloak realm' >&2
    exit 1
  fi
  sleep 2
done
echo 'e2e-stack: Keycloak realm is up'
