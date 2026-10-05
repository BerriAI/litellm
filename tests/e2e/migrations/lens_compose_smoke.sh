#!/usr/bin/env bash
set -euo pipefail

worker_image() {
  env -u LENS_WORKER_IMAGE -u LITELLM_VERSION \
    LITELLM_URL=http://litellm:4000 LENS_WORKER_TOKEN=config-test "$@" \
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
master_key="sk-$(openssl rand -hex 32)"
compose=(docker compose -p lens-compose-ci --env-file "$qa_dir/env" -f deploy/lens/stack.yaml)
cleanup() {
  "${compose[@]}" --profile lens down -v --remove-orphans >/dev/null 2>&1 || true
  rm -rf "$qa_dir"
}
trap cleanup EXIT
umask 077
printf 'LITELLM_VERSION=0.0.0-lens-ci\nLITELLM_PORT=4418\nLITELLM_MASTER_KEY=%s\nLITELLM_SALT_KEY=sk-%s\n' \
  "$master_key" "$(openssl rand -hex 32)" > "$qa_dir/env"
printf 'POSTGRES_PASSWORD=%s:/?#@%%\nCLICKHOUSE_PASSWORD=%s:/?#@%%\n' \
  "$(openssl rand -hex 32)" "$(openssl rand -hex 32)" >> "$qa_dir/env"
docker tag "${LITELLM_IMAGE:?Set LITELLM_IMAGE to the built gateway image}" ghcr.io/berriai/litellm:0.0.0-lens-ci
docker build --build-arg LITELLM_RELEASE_TAG=v0.0.0-lens-ci -f deploy/lens/Dockerfile \
  -t ghcr.io/berriai/litellm-lens-worker:v0.0.0-lens-ci .
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
api /v1/traces -d "@$qa_dir/trace.json" > /dev/null
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
jq -n --arg key "$key_id" '{name:"Lens Compose CI",analysis_key_id:$key}' > "$qa_dir/registration.json"
api /lens/workers/register -d "@$qa_dir/registration.json" > "$qa_dir/worker.json"
jq -e '.image == "ghcr.io/berriai/litellm-lens-worker:v0.0.0-lens-ci"' "$qa_dir/worker.json" > /dev/null
printf 'LENS_WORKER_TOKEN=%s\n' "$(jq -r '.token' "$qa_dir/worker.json")" >> "$qa_dir/env"
worker_id=$(jq -r '.worker.id' "$qa_dir/worker.json")
heartbeat_after=$(date -u +'%Y-%m-%dT%H:%M:%S')
"${compose[@]}" --profile lens up -d

connected() {
  for attempt in $(seq 1 60); do
    if api /lens > "$qa_dir/lens.json" 2>/dev/null && \
      jq -e --arg id "$worker_id" --arg since "$heartbeat_after" \
        '.workers[] | select(.id == $id and .last_seen > $since)' "$qa_dir/lens.json" > /dev/null; then
      return 0
    fi
    sleep 2
  done
  "${compose[@]}" --profile lens logs lens-worker
  return 1
}
connected
printf 'Fresh Compose stack: matching worker image and authenticated heartbeat passed\n'

for target in db:5432 clickhouse:8123; do
  service=${target%:*}
  port=${target#*:}
  address=$(docker inspect --format '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' "$("${compose[@]}" ps -q "$service")")
  "${compose[@]}" exec -T lens-worker python -c '
import socket, sys
for host in (sys.argv[1], sys.argv[2]):
    try:
        connection = socket.create_connection((host, int(sys.argv[3])), timeout=2)
    except OSError:
        continue
    connection.close()
    raise SystemExit("Worker can reach a datastore directly")
' "$service" "$address" "$port"
done
printf 'Worker can reach the proxy but cannot connect directly to PostgreSQL or ClickHouse\n'

status=$(curl --silent --show-error -o "$qa_dir/mismatch.json" -w '%{http_code}' -X POST \
  -H "Authorization: Bearer $(jq -r '.token' "$qa_dir/worker.json")" \
  'http://127.0.0.1:4418/lens/worker/claim?protocol_version=4&worker_release=v0.0.0-old')
[[ "$status" == 409 ]]
jq -e '.detail | contains("Upgrade the Lens worker")' "$qa_dir/mismatch.json" > /dev/null

"${compose[@]}" --profile lens restart litellm lens-worker
heartbeat_after=$(date -u +'%Y-%m-%dT%H:%M:%S')
connected
trace_saved
api /lens > "$qa_dir/restarted.json"
jq -e --arg id "$worker_id" --arg key "$key_id" \
  '.workers[] | select(.id == $id and .analysis_key_id == $key)' "$qa_dir/restarted.json" > /dev/null
printf 'Compose restart: trace, worker identity, token and billing assignment preserved; wrong release rejected\n'

cat > "$qa_dir/unversioned.yaml" <<'EOF'
services:
  litellm:
    environment:
      LITELLM_RELEASE_TAG: ""
EOF
"${compose[@]}" -f "$qa_dir/unversioned.yaml" up -d litellm
for attempt in $(seq 1 90); do
  if api /health/liveliness > /dev/null 2>&1; then break; fi
  sleep 2
done
api /health/liveliness > /dev/null
status=$(curl --silent --show-error --max-time 30 -o "$qa_dir/unversioned-registration.json" -w '%{http_code}' \
  -H "Authorization: Bearer $master_key" -H 'Content-Type: application/json' \
  -d "@$qa_dir/registration.json" 'http://127.0.0.1:4418/lens/workers/register')
[[ "$status" == 503 ]]
jq -e '.detail | contains("no release identity")' "$qa_dir/unversioned-registration.json" > /dev/null
status=$(curl --silent --show-error --max-time 30 -o "$qa_dir/unversioned-claim.json" -w '%{http_code}' -X POST \
  -H "Authorization: Bearer $(jq -r '.token' "$qa_dir/worker.json")" \
  'http://127.0.0.1:4418/lens/worker/claim?protocol_version=4&worker_release=')
[[ "$status" == 503 ]]
jq -e '.detail | contains("no release identity")' "$qa_dir/unversioned-claim.json" > /dev/null
api /lens > "$qa_dir/unversioned-workers.json"
jq -e --arg id "$worker_id" '.workers | length == 1 and .[0].id == $id' "$qa_dir/unversioned-workers.json" > /dev/null
"${compose[@]}" up -d litellm
heartbeat_after=$(date -u +'%Y-%m-%dT%H:%M:%S')
connected
trace_saved
printf 'Unversioned gateway: setup and claims refused without guessing; original worker and trace recovered\n'
