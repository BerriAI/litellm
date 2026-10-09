#!/usr/bin/env bash
set -euo pipefail

worker_image() {
  env -u LENS_WORKER_IMAGE -u LITELLM_VERSION \
    LITELLM_URL=http://litellm:4000 LITELLM_LENS_SERVICE_TOKEN=config-test-service-secret-32-characters \
    CLICKHOUSE_URL=http://clickhouse:8123 "$@" \
    docker compose --env-file /dev/null -f deploy/lens/compose.yaml config --images
}
[[ "$(worker_image LENS_WORKER_IMAGE=registry.example/lens:source)" == registry.example/lens:source ]]
[[ "$(worker_image LITELLM_VERSION=1.2.3)" == ghcr.io/berriai/litellm-lens-worker:v1.2.3 ]]
[[ "$(worker_image LENS_WORKER_IMAGE=registry.example/lens:source LITELLM_VERSION=1.2.3)" == registry.example/lens:source ]]
if worker_image > /dev/null 2>&1; then
  printf 'Worker Compose accepted neither an image nor a release version\n' >&2
  exit 1
fi

qa_dir=$(mktemp -d)
master_key="sk-$(openssl rand -hex 16)"
compose=(docker compose -p lens-compose-ci --env-file "$qa_dir/env" -f deploy/lens/stack.yaml)
cleanup() {
  "${compose[@]}" down -v --remove-orphans >/dev/null 2>&1 || true
  docker network rm lens-local-smoke_default >/dev/null 2>&1 || true
  rm -rf "$qa_dir"
}
trap cleanup EXIT
umask 077
printf 'LITELLM_VERSION=0.0.0-lens-ci\nLITELLM_PORT=4418\nLITELLM_MASTER_KEY=%s\nLITELLM_SALT_KEY=sk-%s\n' \
  "$master_key" "$(openssl rand -hex 32)" > "$qa_dir/env"
printf 'LITELLM_LENS_SERVICE_TOKEN=%s\nLENS_PORT=4419\n' "$(openssl rand -hex 32)" >> "$qa_dir/env"
printf 'POSTGRES_PASSWORD=%s:/?#@%%\nCLICKHOUSE_PASSWORD=%s:/?#@%%\n' \
  "$(openssl rand -hex 32)" "$(openssl rand -hex 32)" >> "$qa_dir/env"
docker tag "${LITELLM_IMAGE:?Set LITELLM_IMAGE to the built gateway image}" ghcr.io/berriai/litellm:0.0.0-lens-ci
docker build --build-arg LITELLM_RELEASE_TAG=v0.0.0-lens-ci -f deploy/lens/Dockerfile \
  -t ghcr.io/berriai/litellm-lens-worker:v0.0.0-lens-ci .
cat > "$qa_dir/local-worker.yaml" <<'YAML'
services:
  lens-worker:
    image: ghcr.io/berriai/litellm-lens-worker:v0.0.0-lens-ci
YAML
LITELLM_MASTER_KEY="$master_key" LITELLM_LENS_SERVICE_TOKEN="$(openssl rand -hex 32)" \
LITELLM_RELEASE_TAG=v0.0.0-lens-ci \
  docker compose --env-file /dev/null -p lens-local-smoke -f docker/docker-compose.tracing.yml \
    -f "$qa_dir/local-worker.yaml" run --rm --no-deps --pull never --entrypoint python3.13 lens-worker -I -S -c '
import os
import pathlib
import subprocess
capacity = os.statvfs("/tmp")
assert capacity.f_blocks * capacity.f_frsize >= 1024**3
probe = pathlib.Path("/tmp/noexec-probe")
probe.write_text("#!/bin/sh\nexit 0\n")
probe.chmod(0o700)
try:
    subprocess.run([str(probe)], check=True)
except PermissionError:
    pass
else:
    raise SystemExit("Local tracing stack permits executable scratch files")
'
docker network rm lens-local-smoke_default
printf 'Local tracing worker: at least 1 GiB scratch capacity and noexec enforced\n'
"${compose[@]}" up -d

api() {
  curl --fail-with-body --silent --show-error --max-time 30 \
    -H "Authorization: Bearer $master_key" -H 'Content-Type: application/json' \
    "http://127.0.0.1:4418$1" "${@:2}"
}
ready=false
for attempt in $(seq 1 90); do
  if api /health/liveliness > /dev/null 2>&1; then ready=true; break; fi
  sleep 2
done
if [[ "$ready" != true ]]; then "${compose[@]}" logs litellm; exit 1; fi
trace_id=$(openssl rand -hex 16)
span_id=$(openssl rand -hex 8)
start_ns="$(date +%s)000000000"
jq -n --arg trace "$trace_id" --arg span "$span_id" --arg at "$start_ns" \
  '{resourceSpans:[{resource:{attributes:[{key:"service.name",value:{stringValue:"lens-compose-ci"}}]},
    scopeSpans:[{scope:{name:"lens-compose-ci"},spans:[{traceId:$trace,spanId:$span,name:"Compose trace",
      kind:1,startTimeUnixNano:$at,endTimeUnixNano:$at,
      attributes:[{key:"openinference.span.kind",value:{stringValue:"AGENT"}}],status:{code:1}}]}]}]}' \
  > "$qa_dir/trace.json"
api /lens/tracing/keys -d '{"name":"Compose smoke"}' > "$qa_dir/tracing-key.json"
tracing_key=$(jq -r '.key' "$qa_dir/tracing-key.json")
trace_sent=false
for attempt in $(seq 1 60); do
  if curl --fail --silent --show-error --max-time 10 \
    -H "Authorization: Bearer $tracing_key" -H 'Content-Type: application/json' \
    -d "@$qa_dir/trace.json" http://127.0.0.1:4419/v1/traces > /dev/null 2>&1; then
    trace_sent=true; break
  fi
  sleep 2
done
[[ "$trace_sent" == true ]]
status=$(curl --silent -o /dev/null -w '%{http_code}' \
  -H "Authorization: Bearer $master_key" -H 'Content-Type: application/json' \
  -d "@$qa_dir/trace.json" http://127.0.0.1:4418/v1/traces)
[[ "$status" == 410 ]]
trace_saved() {
  for attempt in $(seq 1 60); do
    if api "/v1/traces/$trace_id" > "$qa_dir/saved-trace.json" 2>/dev/null && \
      jq -e --arg trace "$trace_id" --arg span "$span_id" \
        '.summary.trace_id == $trace and any(.spans[]; .span_id == $span)' "$qa_dir/saved-trace.json" > /dev/null; then
      return 0
    fi
    sleep 2
  done
  return 1
}
trace_saved
api /key/generate -d '{"key_alias":"Lens Compose CI","models":["lens-compose-ci"],"max_budget":1}' > "$qa_dir/key.json"
key_id=$(jq -r '.token_id // empty' "$qa_dir/key.json")
if [[ -z "$key_id" ]]; then
  key_id=$(jq -rj '.key' "$qa_dir/key.json" | openssl dgst -sha256 | awk '{print $NF}')
fi
jq -n --arg key "$key_id" '{name:"Lens Compose CI",analysis_key_id:$key,managed:true}' > "$qa_dir/registration.json"
api /lens/workers/register -d "@$qa_dir/registration.json" > "$qa_dir/worker.json"
jq -e '.image == "ghcr.io/berriai/litellm-lens-worker:v0.0.0-lens-ci"' "$qa_dir/worker.json" > /dev/null
jq -e '.managed == true and .token == ""' "$qa_dir/worker.json" > /dev/null
worker_id=$(jq -r '.worker.id' "$qa_dir/worker.json")
heartbeat_after=$(date -u +'%Y-%m-%dT%H:%M:%S')
"${compose[@]}" up -d

connected() {
  for attempt in $(seq 1 60); do
    if api /lens > "$qa_dir/lens.json" 2>/dev/null && \
      jq -e --arg id "$worker_id" --arg since "$heartbeat_after" \
        '.workers[] | select(.id == $id and .last_seen > $since)' "$qa_dir/lens.json" > /dev/null; then
      return 0
    fi
    sleep 2
  done
  "${compose[@]}" logs lens-worker
  return 1
}
connected
printf 'Fresh Compose stack: matching worker image and authenticated heartbeat passed\n'

database_address=$(docker inspect --format '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' "$("${compose[@]}" ps -q db)")
"${compose[@]}" exec -T lens-worker python3.13 -I -S -c '
import socket, sys
for host in ("db", sys.argv[1]):
    try:
        connection = socket.create_connection((host, 5432), timeout=2)
    except OSError:
        continue
    connection.close()
    raise SystemExit("Lens can reach PostgreSQL directly")
with socket.create_connection(("clickhouse", 8123), timeout=2):
    pass
' "$database_address"
"${compose[@]}" exec -T litellm python3 -c '
import socket
try:
    connection = socket.create_connection(("clickhouse", 8123), timeout=2)
except OSError:
    pass
else:
    connection.close()
    raise SystemExit("Gateway can reach ClickHouse directly")
'
printf 'Datastore isolation: Lens reaches ClickHouse, gateway reaches Postgres, neither reaches the other datastore\n'
service_token=$(sed -n 's/^LITELLM_LENS_SERVICE_TOKEN=//p' "$qa_dir/env")

status=$(curl --silent --show-error -o "$qa_dir/mismatch.json" -w '%{http_code}' -X POST \
  -H "Authorization: Bearer $service_token" \
  'http://127.0.0.1:4418/lens/worker/claim?protocol_version=4&worker_release=v0.0.0-old')
[[ "$status" == 409 ]]
jq -e '.detail | contains("Upgrade the Lens worker")' "$qa_dir/mismatch.json" > /dev/null

"${compose[@]}" restart litellm lens-worker
heartbeat_after=$(date -u +'%Y-%m-%dT%H:%M:%S')
connected
trace_saved
api /lens > "$qa_dir/restarted.json"
jq -e --arg id "$worker_id" --arg key "$key_id" \
  '.workers[] | select(.id == $id and .analysis_key_id == $key)' "$qa_dir/restarted.json" > /dev/null
printf 'Compose restart: trace, worker identity, token and billing assignment preserved; wrong release rejected\n'

"${compose[@]}" stop clickhouse
"${compose[@]}" restart litellm
for attempt in $(seq 1 90); do
  if api /health/liveliness > /dev/null 2>&1; then break; fi
  sleep 2
done
api /health/liveliness > /dev/null
"${compose[@]}" start clickhouse
trace_saved
printf 'Gateway cold startup succeeds with ClickHouse stopped; trace reads recover after storage restarts\n'
