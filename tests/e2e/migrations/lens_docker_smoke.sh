#!/usr/bin/env bash
set -euo pipefail

: "${LITELLM_IMAGE:?Set a published LiteLLM image}"
: "${LENS_TEST_WORKER_IMAGE:?Set the matching published Lens image}"
qa_dir=$(mktemp -d)
network=lens-docker-ci
containers=(lens-docker-proxy lens-docker-worker lens-docker-postgres lens-docker-clickhouse)
cleanup() {
  local status=$?
  if (( status != 0 )); then
    for container in "${containers[@]}"; do docker logs --tail 60 "$container" 2>&1 || true; done
  fi
  docker rm -fv "${containers[@]}" >/dev/null 2>&1 || true
  docker network rm "$network" >/dev/null 2>&1 || true
  docker volume rm lens-docker-storage >/dev/null 2>&1 || true
  rm -rf "$qa_dir"
  return "$status"
}
trap cleanup EXIT
umask 077
master_key="sk-$(openssl rand -hex 24)"
service_token=$(openssl rand -hex 32)
postgres_password=$(openssl rand -hex 24)
clickhouse_password=$(openssl rand -hex 24)
for value in "$master_key" "$service_token" "$postgres_password" "$clickhouse_password"; do
  if [[ -n "${GITHUB_ACTIONS:-}" ]]; then printf '::add-mask::%s\n' "$value"; fi
done
cat > "$qa_dir/proxy.env" <<ENV
DATABASE_URL=postgresql://litellm:$postgres_password@lens-docker-postgres:5432/litellm
LITELLM_MASTER_KEY=$master_key
LITELLM_SALT_KEY=sk-$(openssl rand -hex 32)
STORE_MODEL_IN_DB=True
LITELLM_LENS_URL=http://lens-docker-worker:4318
LITELLM_LENS_PUBLIC_URL=http://127.0.0.1:4419
LITELLM_LENS_SERVICE_TOKEN=$service_token
ENV
cat > "$qa_dir/worker.env" <<ENV
LITELLM_URL=http://lens-docker-proxy:4000
LITELLM_LENS_SERVICE_TOKEN=$service_token
CLICKHOUSE_URL=http://default:$clickhouse_password@lens-docker-clickhouse:8123
ENV
cat > "$qa_dir/postgres.env" <<ENV
POSTGRES_DB=litellm
POSTGRES_USER=litellm
POSTGRES_PASSWORD=$postgres_password
ENV
cat > "$qa_dir/clickhouse.env" <<ENV
CLICKHOUSE_USER=default
CLICKHOUSE_PASSWORD=$clickhouse_password
CLICKHOUSE_DEFAULT_ACCESS_MANAGEMENT=1
ENV
docker network create "$network"
docker volume create lens-docker-storage
docker run -d --name lens-docker-postgres --network "$network" \
  --env-file "$qa_dir/postgres.env" postgres:16
docker run -d --name lens-docker-clickhouse --network "$network" \
  --env-file "$qa_dir/clickhouse.env" -v lens-docker-storage:/var/lib/clickhouse \
  clickhouse/clickhouse-server:26.9.6.6@sha256:eb4870e7ca7ed70c259eebfcfbee6cf797017f6b5436c2926bbbfe3d4d28486e
for attempt in $(seq 1 60); do
  if docker exec lens-docker-postgres pg_isready -U litellm -d litellm >/dev/null 2>&1; then break; fi
  sleep 2
done
docker exec lens-docker-postgres pg_isready -U litellm -d litellm
docker run -d --name lens-docker-proxy --network "$network" \
  --env-file "$qa_dir/proxy.env" -p 127.0.0.1:4418:4000 \
  -v "$PWD/deploy/lens/config.yaml:/app/lens-config.yaml:ro" \
  "$LITELLM_IMAGE" --config /app/lens-config.yaml --port 4000
docker run -d --name lens-docker-worker --network "$network" \
  --env-file "$qa_dir/worker.env" -p 127.0.0.1:4419:4318 \
  --memory 2g --cpus 2 --pids-limit 64 --read-only \
  --tmpfs /tmp:rw,noexec,nosuid,size=1g --cap-drop ALL \
  --security-opt no-new-privileges:true "$LENS_TEST_WORKER_IMAGE"
api() {
  curl --fail-with-body --silent --show-error --max-time 20 \
    -H "Authorization: Bearer $master_key" -H 'Content-Type: application/json' \
    "http://127.0.0.1:4418$1" "${@:2}"
}
for attempt in $(seq 1 120); do
  if api /lens/service > "$qa_dir/service.json" 2>/dev/null && \
    jq -e '.connected and .status.storage_ready' "$qa_dir/service.json" >/dev/null; then break; fi
  sleep 2
done
jq -e '.connected and .status.storage_ready' "$qa_dir/service.json" >/dev/null
[[ "$(curl -sS -o /dev/null -w '%{http_code}' http://127.0.0.1:4419/internal/status)" == 401 ]]
[[ "$(docker exec lens-docker-worker id -u)" == 65532 ]]
trace_id=$(openssl rand -hex 16)
span_id=$(openssl rand -hex 8)
jq -n --arg trace "$trace_id" --arg span "$span_id" --arg at "$(date +%s)000000000" \
  '{resourceSpans:[{resource:{attributes:[{key:"service.name",value:{stringValue:"lens-docker-ci"}}]},
    scopeSpans:[{spans:[{traceId:$trace,spanId:$span,name:"Standalone Docker trace",kind:1,
      startTimeUnixNano:$at,endTimeUnixNano:$at,status:{code:1}}]}]}]}' > "$qa_dir/trace.json"
api /lens/tracing/keys -d '{"name":"Standalone Docker smoke"}' > "$qa_dir/key.json"
tracing_key=$(jq -r .key "$qa_dir/key.json")
if [[ -n "${GITHUB_ACTIONS:-}" ]]; then printf '::add-mask::%s\n' "$tracing_key"; fi
curl --fail-with-body --silent --show-error --retry 20 --retry-all-errors --retry-delay 2 \
  -H "Authorization: Bearer $tracing_key" -H 'Content-Type: application/json' \
  -d "@$qa_dir/trace.json" http://127.0.0.1:4419/v1/traces
saved_trace() {
  for attempt in $(seq 1 60); do
    if api "/v1/traces/$trace_id" > "$qa_dir/saved.json" 2>/dev/null && \
      jq -e --arg trace "$trace_id" --arg span "$span_id" \
        '.summary.trace_id == $trace and any(.spans[]; .span_id == $span)' "$qa_dir/saved.json" >/dev/null; then return 0; fi
    sleep 2
  done
  return 1
}
saved_trace
printf 'Standalone Docker: readiness, internal authentication and direct trace ingest/readback passed\n'
api /key/generate -d '{"key_alias":"Lens Docker CI","models":["lens-docker-ci"],"max_budget":1}' > "$qa_dir/analysis-key.json"
key_id=$(jq -r '.token_id // empty' "$qa_dir/analysis-key.json")
if [[ -z "$key_id" ]]; then
  key_id=$(jq -rj .key "$qa_dir/analysis-key.json" | openssl dgst -sha256 | awk '{print $NF}')
fi
jq -n --arg key "$key_id" '{name:"Lens Docker CI",analysis_key_id:$key,managed:true}' > "$qa_dir/registration.json"
api /lens/workers/register -d "@$qa_dir/registration.json" > "$qa_dir/worker.json"
jq -e --arg image "ghcr.io/berriai/litellm-lens-worker:v${LENS_TEST_RELEASE:?}" \
  '.managed == true and .token == "" and .image == $image' "$qa_dir/worker.json" >/dev/null
worker_id=$(jq -r .worker.id "$qa_dir/worker.json")
heartbeat_after=$(date -u +'%Y-%m-%dT%H:%M:%S')
connected() {
  for attempt in $(seq 1 90); do
    if api /lens > "$qa_dir/lens.json" 2>/dev/null && \
      jq -e --arg id "$worker_id" --arg since "$heartbeat_after" \
        '.workers[] | select(.id == $id and .last_seen > $since)' "$qa_dir/lens.json" >/dev/null; then return 0; fi
    sleep 2
  done
  return 1
}
connected
docker restart lens-docker-proxy lens-docker-worker lens-docker-clickhouse
heartbeat_after=$(date -u +'%Y-%m-%dT%H:%M:%S')
connected
saved_trace
printf 'Standalone Docker: managed worker heartbeat and trace persistence after restart passed\n'
