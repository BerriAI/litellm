#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STACK_DIR="${E2E_STACK_DIR:-${RUNNER_TEMP:-/tmp}/litellm-e2e-stack}"
CERTS_DIR="${STACK_DIR}/certs"
LOGS_DIR="${STACK_DIR}/logs"
PIDS_DIR="${STACK_DIR}/pids"

POSTGRES_IMAGE="${E2E_POSTGRES_IMAGE:-postgres:16.6}"
VALKEY_IMAGE="${E2E_VALKEY_IMAGE:-valkey/valkey:8.1}"
JAEGER_IMAGE="${E2E_JAEGER_IMAGE:-jaegertracing/jaeger:2.10.0}"
NGINX_IMAGE="${E2E_NGINX_IMAGE:-nginx:1.29-alpine}"

LB_PORT="${E2E_LB_PORT:-4000}"
GATEWAY_PORT_1="${E2E_GATEWAY_PORT_1:-4010}"
GATEWAY_PORT_2="${E2E_GATEWAY_PORT_2:-4011}"
BACKEND_PORT="${E2E_BACKEND_PORT:-4001}"
REDIS_PORT="${E2E_REDIS_PORT:-6379}"
DATABASE_HOST="${E2E_DATABASE_HOST:-127.0.0.1}"
DATABASE_PORT="${E2E_DATABASE_PORT:-5432}"
DATABASE_USER="${E2E_DATABASE_USER:-litellm}"
DATABASE_PASSWORD="${E2E_DATABASE_PASSWORD:-dbpassword9090}"
DATABASE_NAME="${E2E_DATABASE_NAME:-litellm}"
JAEGER_OTLP_PORT="${E2E_JAEGER_OTLP_PORT:-4318}"
JAEGER_QUERY_PORT="${E2E_JAEGER_QUERY_PORT:-16686}"

MASTER_KEY="${LITELLM_MASTER_KEY:-sk-e2e-$(openssl rand -hex 16)}"

mkdir -p "${CERTS_DIR}" "${LOGS_DIR}" "${PIDS_DIR}"

log() { printf 'e2e-stack: %s\n' "$*"; }

port_open() { (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null; }

wait_for() {
  local label="$1" check="$2" deadline=$((SECONDS + ${3:-120}))
  until eval "${check}"; do
    if ((SECONDS >= deadline)); then
      log "timed out waiting for ${label}"
      tail -n 60 "${LOGS_DIR}"/*.log 2>/dev/null || true
      exit 1
    fi
    sleep 2
  done
  log "${label} is up"
}

if [[ -f "${REPO_ROOT}/tests/e2e/.env" ]]; then
  set -a
  source "${REPO_ROOT}/tests/e2e/.env"
  set +a
fi

if ! port_open "${DATABASE_PORT}"; then
  docker run -d --name e2e-postgres -p "${DATABASE_PORT}:5432" \
    -e "POSTGRES_USER=${DATABASE_USER}" -e "POSTGRES_PASSWORD=${DATABASE_PASSWORD}" -e "POSTGRES_DB=${DATABASE_NAME}" \
    "${POSTGRES_IMAGE}" >/dev/null
fi
wait_for "postgres" "port_open ${DATABASE_PORT}"

if ! port_open "${JAEGER_QUERY_PORT}"; then
  docker run -d --name e2e-jaeger -p "${JAEGER_OTLP_PORT}:4318" -p "${JAEGER_QUERY_PORT}:16686" \
    "${JAEGER_IMAGE}" >/dev/null
fi
wait_for "jaeger" "curl -fs http://127.0.0.1:${JAEGER_QUERY_PORT}/api/services >/dev/null"

openssl genrsa -out "${CERTS_DIR}/ca.key" 2048 2>/dev/null
openssl req -x509 -new -nodes -key "${CERTS_DIR}/ca.key" -sha256 -days 7 \
  -subj "/CN=litellm-e2e-ca" -out "${CERTS_DIR}/ca.crt" 2>/dev/null
openssl genrsa -out "${CERTS_DIR}/server.key" 2048 2>/dev/null
openssl req -new -key "${CERTS_DIR}/server.key" -subj "/CN=localhost" -out "${CERTS_DIR}/server.csr" 2>/dev/null
openssl x509 -req -in "${CERTS_DIR}/server.csr" -CA "${CERTS_DIR}/ca.crt" -CAkey "${CERTS_DIR}/ca.key" \
  -CAcreateserial -days 7 -sha256 \
  -extfile <(printf 'subjectAltName=DNS:localhost,IP:127.0.0.1') \
  -out "${CERTS_DIR}/server.crt" 2>/dev/null
chmod 644 "${CERTS_DIR}"/*.key "${CERTS_DIR}"/*.crt

CERTIFI_BUNDLE="$(cd "${REPO_ROOT}" && uv run --no-sync python -c 'import certifi; print(certifi.where())')"
cat "${CERTIFI_BUNDLE}" "${CERTS_DIR}/ca.crt" > "${CERTS_DIR}/ca-bundle.pem"

docker rm -f e2e-valkey >/dev/null 2>&1 || true
docker run -d --name e2e-valkey -p "${REDIS_PORT}:${REDIS_PORT}" -v "${CERTS_DIR}:/certs:ro" \
  "${VALKEY_IMAGE}" valkey-server \
  --cluster-enabled yes --port 0 --tls-port "${REDIS_PORT}" \
  --tls-cert-file /certs/server.crt --tls-key-file /certs/server.key --tls-ca-cert-file /certs/ca.crt \
  --tls-auth-clients no --cluster-announce-ip 127.0.0.1 >/dev/null
VALKEY_CLI="docker exec e2e-valkey valkey-cli --tls --cacert /certs/ca.crt -h 127.0.0.1 -p ${REDIS_PORT}"
wait_for "valkey" "${VALKEY_CLI} ping 2>/dev/null | grep -q PONG"
${VALKEY_CLI} cluster addslotsrange 0 16383 >/dev/null
wait_for "valkey cluster" "${VALKEY_CLI} cluster info 2>/dev/null | grep -q cluster_state:ok"

CONFIG_PATH="${REPO_ROOT}/tests/e2e/gateway/litellm-config.yml"
if [[ "${REDIS_PORT}" != "6379" ]]; then
  CONFIG_PATH="${STACK_DIR}/litellm-config.yml"
  sed "s/port: 6379/port: ${REDIS_PORT}/" "${REPO_ROOT}/tests/e2e/gateway/litellm-config.yml" > "${CONFIG_PATH}"
fi

SERVER_ENV=(
  "LITELLM_MASTER_KEY=${MASTER_KEY}"
  "DATABASE_HOST=${DATABASE_HOST}"
  "DATABASE_PORT=${DATABASE_PORT}"
  "DATABASE_USER=${DATABASE_USER}"
  "DATABASE_PASSWORD=${DATABASE_PASSWORD}"
  "DATABASE_NAME=${DATABASE_NAME}"
  "DISABLE_SCHEMA_UPDATE=true"
  "REDIS_HOST=127.0.0.1"
  "REDIS_PORT=${REDIS_PORT}"
  "REDIS_CLUSTER_NODES=[{\"host\":\"127.0.0.1\",\"port\":${REDIS_PORT}}]"
  "CONFIG_FILE_PATH=${CONFIG_PATH}"
  "STORE_MODEL_IN_DB=True"
  "OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf"
  "OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:${JAEGER_OTLP_PORT}"
  "SSL_CERT_FILE=${CERTS_DIR}/ca-bundle.pem"
  "PYTHONPATH=${REPO_ROOT}"
)
if [[ -n "${VERTEXAI_CREDENTIALS:-}" ]]; then
  printf '%s' "${VERTEXAI_CREDENTIALS}" > "${STACK_DIR}/vertex-adc.json"
  SERVER_ENV+=("GOOGLE_APPLICATION_CREDENTIALS=${STACK_DIR}/vertex-adc.json")
fi

cd "${REPO_ROOT}"

log "running migrations"
env "${SERVER_ENV[@]}" uv run --no-sync python migrations/run.py >"${LOGS_DIR}/migrations.log" 2>&1

start_server() {
  local name="$1"; shift
  env "${SERVER_ENV[@]}" "$@" >"${LOGS_DIR}/${name}.log" 2>&1 &
  echo $! > "${PIDS_DIR}/${name}.pid"
}

start_server backend uv run --no-sync uvicorn backend.main:app --host 0.0.0.0 --port "${BACKEND_PORT}"
start_server gateway-1 uv run --no-sync uvicorn gateway.main:app --workers 1 --host 0.0.0.0 --port "${GATEWAY_PORT_1}"
start_server gateway-2 uv run --no-sync uvicorn gateway.main:app --workers 1 --host 0.0.0.0 --port "${GATEWAY_PORT_2}"

if [[ "$(uname)" == "Linux" ]]; then
  NGINX_UPSTREAM_HOST=127.0.0.1
  NGINX_DOCKER_ARGS=(--network host)
else
  NGINX_UPSTREAM_HOST=host.docker.internal
  NGINX_DOCKER_ARGS=(-p "${LB_PORT}:${LB_PORT}")
fi

cat > "${STACK_DIR}/nginx.conf" <<EOF
events {}
http {
  map \$http_upgrade \$connection_upgrade {
    default upgrade;
    '' close;
  }
  upstream litellm_gateways {
    server ${NGINX_UPSTREAM_HOST}:${GATEWAY_PORT_1};
    server ${NGINX_UPSTREAM_HOST}:${GATEWAY_PORT_2};
  }
  server {
    listen ${LB_PORT};
    client_max_body_size 100m;
    location / {
      proxy_pass http://litellm_gateways;
      proxy_http_version 1.1;
      proxy_set_header Host \$host;
      proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
      proxy_set_header X-Forwarded-Proto \$scheme;
      proxy_set_header Upgrade \$http_upgrade;
      proxy_set_header Connection \$connection_upgrade;
      proxy_buffering off;
      proxy_read_timeout 600s;
      proxy_send_timeout 600s;
    }
  }
}
EOF

docker rm -f e2e-nginx >/dev/null 2>&1 || true
docker run -d --name e2e-nginx "${NGINX_DOCKER_ARGS[@]}" \
  -v "${STACK_DIR}/nginx.conf:/etc/nginx/nginx.conf:ro" "${NGINX_IMAGE}" >/dev/null

wait_for "backend" "curl -fs http://127.0.0.1:${BACKEND_PORT}/health/liveliness >/dev/null" 300
wait_for "gateway-1" "curl -fs http://127.0.0.1:${GATEWAY_PORT_1}/health/liveliness >/dev/null" 300
wait_for "gateway-2" "curl -fs http://127.0.0.1:${GATEWAY_PORT_2}/health/liveliness >/dev/null" 300
wait_for "load balancer" "curl -fs http://127.0.0.1:${LB_PORT}/health/liveliness >/dev/null" 60

cat > "${STACK_DIR}/stack.env" <<EOF
LITELLM_PROXY_URL=http://127.0.0.1:${LB_PORT}
LITELLM_CONTROL_PLANE_URL=http://127.0.0.1:${BACKEND_PORT}
LITELLM_MASTER_KEY=${MASTER_KEY}
REDIS_HOST=127.0.0.1
REDIS_PORT=${REDIS_PORT}
E2E_OTEL_QUERY_URL=http://127.0.0.1:${JAEGER_QUERY_PORT}
SSL_CERT_FILE=${CERTS_DIR}/ca-bundle.pem
DATABASE_URL=postgresql://${DATABASE_USER}:${DATABASE_PASSWORD}@${DATABASE_HOST}:${DATABASE_PORT}/${DATABASE_NAME}
EOF

log "stack is up; pytest env written to ${STACK_DIR}/stack.env"
