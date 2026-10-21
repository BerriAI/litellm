#!/usr/bin/env bash
set -euo pipefail

qa_dir=$(mktemp -d)
cluster=lens-install-ci
forward_pids=()
cleanup() {
  local status=$?
  if (( status != 0 )); then
    for log in "$qa_dir"/*-forward.log; do
      if [[ -f "$log" ]]; then cat "$log" >&2; fi
    done
    if [[ -n "${namespace:-}" ]]; then diagnose || true; fi
  fi
  for pid in "${forward_pids[@]}"; do kill "$pid" 2>/dev/null || true; done
  kind delete cluster --name "$cluster" || true
  rm -rf "$qa_dir"
  return "$status"
}
trap cleanup EXIT
umask 077
export KUBECONFIG="$qa_dir/kubeconfig"
kind create cluster --name "$cluster" \
  --image kindest/node:v1.32.2@sha256:f226345927d7e348497136874b6d207e0b32cc52154ad8323129352923a3142f \
  --wait 120s
for component in gateway backend ui migrations monolith worker; do
  kind load docker-image --name "$cluster" "lens-ci-$component:v0.0.0-lens-ci"
done
helm dependency build helm/litellm-helm

api() {
  curl --fail-with-body --silent --show-error --max-time 20 \
    -H "Authorization: Bearer $master_key" -H 'Content-Type: application/json' \
    "http://127.0.0.1:14418$1" "${@:2}"
}

saved_trace() {
  for attempt in $(seq 1 30); do
    if api "/v1/traces/$trace_id" > "$qa_dir/saved.json" && \
      jq -e --arg span "$span_id" 'any(.spans[]; .span_id == $span)' "$qa_dir/saved.json" > /dev/null; then
      return 0
    fi
    sleep 2
  done
  return 1
}

diagnose() {
  kubectl -n "$namespace" get pods
  kubectl -n "$namespace" get services,endpoints
  kubectl -n "$namespace" get events --sort-by=.lastTimestamp | tail -30
  kubectl -n "$namespace" logs --all-containers -l app.kubernetes.io/instance=lens --tail=50 || true
  return 1
}

forward() {
  local service=$1 local_port=$2 remote_port=$3
  local log="$qa_dir/$service-forward.log"
  kubectl -n "$namespace" port-forward --address 127.0.0.1 --pod-running-timeout=30s \
    "service/$service" "$local_port:$remote_port" > "$log" 2>&1 &
  local pid=$!
  forward_pids+=("$pid")
  for attempt in $(seq 1 150); do
    if ! kill -0 "$pid" 2>/dev/null; then
      cat "$log" >&2
      return 1
    fi
    if grep -q "^Forwarding from 127\\.0\\.0\\.1:$local_port ->" "$log"; then return 0; fi
    sleep 0.2
  done
  cat "$log" >&2
  return 1
}

for chart in litellm-helm litellm; do
  namespace="lens-$chart"
  kubectl create namespace "$namespace"
  master_key="sk-$(openssl rand -hex 24)"
  kubectl -n "$namespace" create secret generic lens-secrets \
    --from-literal="master-key=$master_key" \
    --from-literal="service-token=$(openssl rand -hex 32)" \
    --from-literal="url=http://clickhouse:8123" \
    --from-literal=username=litellm --from-literal=password=isolated-helm-test
  kubectl -n "$namespace" apply -f - <<'YAML'
apiVersion: apps/v1
kind: Deployment
metadata: {name: postgres}
spec:
  selector: {matchLabels: {app: postgres}}
  template:
    metadata: {labels: {app: postgres}}
    spec:
      containers:
        - name: postgres
          image: postgres:16
          env:
            - {name: POSTGRES_DB, value: litellm}
            - {name: POSTGRES_USER, value: litellm}
            - {name: POSTGRES_PASSWORD, value: isolated-helm-test}
          readinessProbe:
            exec: {command: [pg_isready, -U, litellm, -d, litellm]}
---
apiVersion: v1
kind: Service
metadata: {name: postgres}
spec:
  selector: {app: postgres}
  ports: [{port: 5432}]
---
apiVersion: apps/v1
kind: Deployment
metadata: {name: clickhouse}
spec:
  selector: {matchLabels: {app: clickhouse}}
  template:
    metadata: {labels: {app: clickhouse}}
    spec:
      containers:
        - name: clickhouse
          image: clickhouse/clickhouse-server:26.9.6.6@sha256:eb4870e7ca7ed70c259eebfcfbee6cf797017f6b5436c2926bbbfe3d4d28486e
          env: [{name: CLICKHOUSE_SKIP_USER_SETUP, value: "1"}]
          readinessProbe:
            httpGet: {path: /ping, port: 8123}
---
apiVersion: v1
kind: Service
metadata: {name: clickhouse}
spec:
  selector: {app: clickhouse}
  ports: [{port: 8123}]
YAML
  kubectl -n "$namespace" rollout status deployment/postgres --timeout=180s
  kubectl -n "$namespace" rollout status deployment/clickhouse --timeout=180s
  cat > "$qa_dir/common.yaml" <<'YAML'
fullnameOverride: lens
lensWorker:
  enabled: true
  image: {repository: lens-ci-worker, tag: v0.0.0-lens-ci, pullPolicy: Never}
  serviceTokenSecret: {name: lens-secrets, key: service-token}
  clickhouseSecret: {name: lens-secrets, key: url}
  clickhouseDatabase: existing_traces
  retentionDays: 45
  publicUrl: http://127.0.0.1:14419
YAML
  if [[ "$chart" == litellm-helm ]]; then
    control=lens
    control_port=4000
    cat > "$qa_dir/chart.yaml" <<'YAML'
image: {repository: lens-ci-monolith, tag: v0.0.0-lens-ci, pullPolicy: Never}
masterkeySecretName: lens-secrets
masterkeySecretKey: master-key
envVars: {STORE_MODEL_IN_DB: "True"}
db:
  deployStandalone: false
  useExisting: true
  endpoint: postgres
  secret: {name: lens-secrets, usernameKey: username, passwordKey: password}
redis: {enabled: false}
proxy_config:
  model_list: []
  general_settings:
    master_key: os.environ/PROXY_MASTER_KEY
    store_model_in_db: true
    tracing: {enabled: true, store: {type: lens}}
YAML
  else
    control=lens-backend
    control_port=4001
    cat > "$qa_dir/chart.yaml" <<'YAML'
masterKey: {secretName: lens-secrets, secretKey: master-key}
database:
  writer:
    host: postgres
    dbname: litellm
    passwordSecret: {name: lens-secrets, usernameKey: username, passwordKey: password}
migrationJob:
  image: {repository: lens-ci-migrations, tag: v0.0.0-lens-ci, pullPolicy: Never}
gateway:
  image: {repository: lens-ci-gateway, tag: v0.0.0-lens-ci, pullPolicy: Never}
  numWorkers: 1
  extraEnv: [{name: STORE_MODEL_IN_DB, value: "True"}]
  hpa: {enabled: false}
  resources: {requests: {cpu: 100m, memory: 512Mi}, limits: {memory: 2Gi}}
  config:
    create: true
    proxy_config:
      model_list: []
      general_settings:
        store_model_in_db: true
        tracing: {enabled: true, store: {type: lens}}
backend:
  extraEnv: [{name: STORE_MODEL_IN_DB, value: "True"}]
  image: {repository: lens-ci-backend, tag: v0.0.0-lens-ci, pullPolicy: Never}
  hpa: {enabled: false}
  resources: {requests: {cpu: 100m, memory: 512Mi}, limits: {memory: 2Gi}}
ui:
  image: {repository: lens-ci-ui, tag: v0.0.0-lens-ci, pullPolicy: Never}
  hpa: {enabled: false}
YAML
  fi
  install=(helm upgrade --install lens "helm/$chart" -n "$namespace" \
    -f "$qa_dir/common.yaml" -f "$qa_dir/chart.yaml" --wait --wait-for-jobs --timeout 8m)
  "${install[@]}" || diagnose
  forward "$control" 14418 "$control_port"
  forward lens-lens-worker 14419 4318
  for attempt in $(seq 1 30); do
    if api /lens/service > "$qa_dir/status.json" && jq -e '.connected and .status.storage_ready' "$qa_dir/status.json"; then break; fi
    sleep 1
  done
  jq -e '.connected and .status.storage_ready' "$qa_dir/status.json"
  api /lens/tracing/keys -d '{"name":"Helm smoke"}' > "$qa_dir/key.json"
  tracing_key=$(jq -r .key "$qa_dir/key.json")
  trace_id=$(openssl rand -hex 16)
  span_id=$(openssl rand -hex 8)
  jq -n --arg trace "$trace_id" --arg span "$span_id" --arg at "$(date +%s)000000000" \
    '{resourceSpans:[{scopeSpans:[{spans:[{traceId:$trace,spanId:$span,name:"Helm trace",kind:1,
      startTimeUnixNano:$at,endTimeUnixNano:$at,status:{code:1}}]}]}]}' > "$qa_dir/trace.json"
  curl --fail-with-body --silent --show-error --retry 10 --retry-all-errors --retry-delay 1 \
    -H "Authorization: Bearer $tracing_key" -H 'Content-Type: application/json' \
    -d "@$qa_dir/trace.json" http://127.0.0.1:14419/v1/traces
  saved_trace
  kubectl -n "$namespace" exec deployment/clickhouse -- clickhouse-client --query \
    "SELECT count() FROM existing_traces.otel_traces WHERE TraceId = '$trace_id'" | grep -qx 1
  "${install[@]}" || diagnose
  saved_trace
  for pid in "${forward_pids[@]}"; do kill "$pid"; wait "$pid" 2>/dev/null || true; done
  forward_pids=()
  kubectl -n "$namespace" rollout restart "deployment/$control" deployment/lens-lens-worker
  kubectl -n "$namespace" rollout status "deployment/$control" --timeout=180s
  kubectl -n "$namespace" rollout status deployment/lens-lens-worker --timeout=180s
  forward "$control" 14418 "$control_port"
  saved_trace
  printf '%s: fresh install, direct ingestion, custom database, upgrade, and restart passed\n' "$chart"
  for pid in "${forward_pids[@]}"; do kill "$pid"; wait "$pid" 2>/dev/null || true; done
  forward_pids=()
  kubectl delete namespace "$namespace" --wait=true
done
