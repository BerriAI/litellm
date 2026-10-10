#!/usr/bin/env bash
set -euo pipefail

qualification_mode=${LENS_QUALIFICATION_MODE:-smoke}
layout_selection=${LENS_HELM_LAYOUT:-all}
case "$qualification_mode" in smoke|boundaries|release) ;; *) printf 'LENS_QUALIFICATION_MODE must be smoke, boundaries or release\n' >&2; exit 2 ;; esac
case "$layout_selection" in all|monolith|componentized) ;; *) printf 'LENS_HELM_LAYOUT must be all, monolith or componentized\n' >&2; exit 2 ;; esac
if [[ "$qualification_mode" == release ]]; then
  : "${LENS_RELEASE_PROVIDER_API_KEY:?A separately authorized provider credential is required}"
  : "${LENS_RELEASE_PROVIDER_API_BASE:?Set the authorized provider API base}"
  : "${LENS_RELEASE_PROVIDER_MODEL:?Set the provider model explicitly}"
  [[ "$LENS_RELEASE_PROVIDER_API_BASE" == https://* ]]
fi
qa_dir=$(mktemp -d)
cluster="lens-install-$(openssl rand -hex 6)"
cluster_owned=false
forward_pids=()
cleanup() {
  local status=$?
  if declare -F qualification_finish > /dev/null; then qualification_finish "$status" || true; fi
  if (( status != 0 )); then
    for log in "$qa_dir"/*-forward.log; do
      if [[ -f "$log" ]]; then cat "$log" >&2; fi
    done
    if [[ -n "${namespace:-}" ]]; then diagnose || true; fi
  fi
  for pid in "${forward_pids[@]}"; do kill "$pid" 2>/dev/null || true; done
  if [[ "$cluster_owned" == true ]]; then kind delete cluster --name "$cluster" || true; fi
  rm -rf "$qa_dir"
  return "$status"
}
trap cleanup EXIT
umask 077
export KUBECONFIG="$qa_dir/kubeconfig"
test "$(docker image inspect lens-ci-litellm:v0.0.0-lens-ci-baseline --format '{{.Id}}')" != \
  "$(docker image inspect lens-ci-litellm:v0.0.0-lens-ci --format '{{.Id}}')"
cluster_owned=true
kind create cluster --name "$cluster" \
  --image kindest/node:v1.32.2@sha256:f226345927d7e348497136874b6d207e0b32cc52154ad8323129352923a3142f \
  --wait 120s
kind load docker-image --name "$cluster" lens-ci-litellm:v0.0.0-lens-ci lens-ci-litellm:v0.0.0-lens-ci-baseline
test "$(docker image inspect lens-ci-worker:baseline --format '{{.Id}}')" != \
  "$(docker image inspect lens-ci-worker:upgrade --format '{{.Id}}')"
kind load docker-image --name "$cluster" lens-ci-worker:baseline lens-ci-worker:upgrade
test -f helm/litellm/charts/lens-0.1.0-dev.0.tgz

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
  if [[ "$qualification_mode" == smoke ]]; then
    kubectl -n "$namespace" logs --all-containers -l app.kubernetes.io/instance=lens --tail=50 || true
  fi
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

stop_forwards() {
  for pid in "${forward_pids[@]}"; do
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  done
  forward_pids=()
}

forward_bundled() {
  stop_forwards
  qualification_forward_control
  forward lens-lens-worker 14419 4318
}

deployment_pods() {
  local deployment selector deployment_uid replica_sets attempt
  local pods="$qa_dir/$1-pods.json" snapshot="$qa_dir/$1-snapshot.json"
  printf '%s: wait for deployment/%s before recording pod identities\n' "$namespace" "$1" >&2
  kubectl -n "$namespace" rollout status "deployment/$1" --timeout=180s >&2 || return
  deployment=$(kubectl -n "$namespace" get deployment "$1" -o json) || return
  selector=$(jq -r '.spec.selector.matchLabels | to_entries | map("\(.key)=\(.value)") | join(",")' <<< "$deployment")
  deployment_uid=$(jq -er .metadata.uid <<< "$deployment")
  test -n "$selector"
  replica_sets=$(kubectl -n "$namespace" get replicasets -l "$selector" -o json \
    | jq -ce --arg uid "$deployment_uid" '[.items[]
        | select(any(.metadata.ownerReferences[]?; .controller == true and .kind == "Deployment" and .uid == $uid))
        | .metadata.uid] | if length > 0 then . else error("Deployment ReplicaSet identity is missing") end') || return
  for attempt in $(seq 1 30); do
    kubectl -n "$namespace" get pods -l "$selector" -o json > "$pods" || return
    if jq -S -e --argjson replica_sets "$replica_sets" '[.items[]
        | select(any(.metadata.ownerReferences[]?; .controller == true and .kind == "ReplicaSet"
            and (.uid as $owner | $replica_sets | index($owner) != null)))
        | select(.metadata.deletionTimestamp == null)
        | {uid:.metadata.uid, images:.spec.containers | map({name,image}),
        containers:(.status.containerStatuses // []) | map({name,imageID,restartCount,ready})}] | sort_by(.uid)
      | if length > 0 and all(.[]; .containers | length > 0)
          and all(.[].containers[]; .imageID != null and .imageID != "" and .ready)
        then . else error("Ready pod runtime identity is missing") end' \
        "$pods" > "$snapshot" 2> "$qa_dir/$1-snapshot-error.log"; then
      cat "$snapshot"
      return 0
    fi
    sleep 1
  done
  printf '%s: deployment/%s pod identities did not converge\n' "$namespace" "$1" >&2
  jq '[.items[] | {name:.metadata.name, uid:.metadata.uid, deleting:.metadata.deletionTimestamp,
    phase:.status.phase, containers:[.status.containerStatuses[]? | {name,imageID,restartCount,ready}]}]' \
    "$pods" >&2
  return 1
}

gateway_pods() {
  local component=$1 tag=$2 deployment="lens-$1"
  if [[ "$component" == monolith ]]; then deployment=lens; fi
  deployment_pods "$deployment" \
    | jq -S -e --arg expected "lens-ci-litellm:$tag" '
      if all(.[]; any(.images[]; .image == $expected)) then
        [.[] as $pod | $pod.images[] | select(.image == $expected) | . as $requested
          | $pod.containers[] | select(.name == $requested.name)
          | {uid:$pod.uid, image:$requested.image, name, imageID, restartCount, ready}]
      else error("Gateway pod does not use the expected image") end'
}

migration_job() {
  kubectl -n "$namespace" get job lens-migrations -o json \
    | jq -e --arg image "$1" '
      select(.metadata.annotations["helm.sh/hook"] == "pre-install,pre-upgrade")
      | select(.metadata.annotations["helm.sh/hook-delete-policy"] == "before-hook-creation")
      | select(.metadata.annotations["argocd.argoproj.io/hook"] == null)
      | select(.status.succeeded == 1 and .spec.template.spec.containers[0].image == $image)
      | .metadata.uid | select(type == "string" and length > 0)'
}

source tests/e2e/migrations/lens_release_qualification.sh
qualification_initialize
for layout in monolith componentized; do
  if [[ "$layout_selection" != all && "$layout_selection" != "$layout" ]]; then continue; fi
  namespace="lens-$layout"
  kubectl create namespace "$namespace"
  master_key="sk-$(openssl rand -hex 24)"
  kubectl -n "$namespace" create secret generic lens-secrets \
    --from-literal="master-key=$master_key" \
    --from-literal="service-token=$(openssl rand -hex 32)" \
    --from-literal="gateway-secret=$(openssl rand -hex 32)" \
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
          image: postgres:16-alpine@sha256:721873c34ceb9f8d8fc265984940dc982404c105f19ad51be9fdc5970a6080ea
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
          volumeMounts:
            - {name: keeper, mountPath: /etc/clickhouse-server/config.d/lens-keeper.xml, subPath: lens-keeper.xml}
          readinessProbe:
            httpGet: {path: /ping, port: 8123}
      volumes:
        - name: keeper
          configMap: {name: clickhouse-keeper}
---
apiVersion: v1
kind: ConfigMap
metadata: {name: clickhouse-keeper}
data:
  lens-keeper.xml: |
    <clickhouse>
      <keeper_server>
        <tcp_port>9181</tcp_port><server_id>1</server_id>
        <log_storage_path>/var/lib/clickhouse/coordination/log</log_storage_path>
        <snapshot_storage_path>/var/lib/clickhouse/coordination/snapshots</snapshot_storage_path>
        <coordination_settings><operation_timeout_ms>10000</operation_timeout_ms><quorum_reads>true</quorum_reads></coordination_settings>
        <raft_configuration><server><id>1</id><hostname>127.0.0.1</hostname><port>9234</port></server></raft_configuration>
      </keeper_server>
      <zookeeper><node><host>127.0.0.1</host><port>9181</port></node></zookeeper>
      <keeper_map_path_prefix>/lens/keeper-map</keeper_map_path_prefix>
    </clickhouse>
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
migrationJob:
  ttlSecondsAfterFinished: 3600
  hooks:
    helm: {enabled: true}
    argocd: {enabled: false}
lensWorker:
  enabled: true
  image: {repository: lens-ci-worker, tag: baseline, pullPolicy: Never}
  serviceTokenSecret: {name: lens-secrets, key: service-token}
  gateway: {secretName: lens-secrets, secretKey: gateway-secret}
  clickhouseSecret: {name: lens-secrets, key: url}
  clickhouseDatabase: existing_traces
  retentionDays: 45
  publicUrl: http://127.0.0.1:14419
YAML
  cat > "$qa_dir/chart.yaml" <<'YAML'
image: {repository: lens-ci-litellm, tag: v0.0.0-lens-ci-baseline, pullPolicy: Never}
masterKey: {secretName: lens-secrets, secretKey: master-key}
database:
  writer:
    host: postgres
    dbname: litellm
    passwordSecret: {name: lens-secrets, usernameKey: username, passwordKey: password}
gateway:
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
  hpa: {enabled: false}
  resources: {requests: {cpu: 100m, memory: 512Mi}, limits: {memory: 2Gi}}
ui:
  hpa: {enabled: false}
YAML
  if [[ "$layout" == monolith ]]; then
    control=lens
    control_port=4000
    printf 'monolith: {enabled: true}\n' >> "$qa_dir/chart.yaml"
  else
    control=lens-backend
    control_port=4001
  fi
  install=(helm upgrade --install lens helm/litellm -n "$namespace" \
    -f "$qa_dir/common.yaml" -f "$qa_dir/chart.yaml" --wait --wait-for-jobs --timeout 8m)
  qualification_provider_values
  "${install[@]}" || diagnose
  forward_bundled
  for attempt in $(seq 1 30); do
    if api /lens/service > "$qa_dir/status.json" && jq -e '.connected and .status.storage_ready' "$qa_dir/status.json"; then break; fi
    sleep 1
  done
  jq -e '.connected and .status.storage_ready' "$qa_dir/status.json"
  qualification_setup
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
  qualification_read_boundaries baseline
  kubectl -n "$namespace" exec deployment/clickhouse -- clickhouse-client --query \
    "SELECT count() FROM existing_traces.otel_traces WHERE TraceId = '$trace_id'" | grep -qx 1
  "${install[@]}" || diagnose
  forward_bundled
  saved_trace
  qualification_read_boundaries reapplied
  kubectl -n "$namespace" get deployments -l app.kubernetes.io/instance=lens -o json \
    | jq -S '[.items[] | select(.metadata.name != "lens-lens-worker") | {name:.metadata.name,template:.spec.template}] | sort_by(.name)' \
    > "$qa_dir/gateway-before.json"
  "${install[@]}" --set lensWorker.image.tag=upgrade || diagnose
  kubectl -n "$namespace" get deployments -l app.kubernetes.io/instance=lens -o json \
    | jq -S '[.items[] | select(.metadata.name != "lens-lens-worker") | {name:.metadata.name,template:.spec.template}] | sort_by(.name)' \
    > "$qa_dir/gateway-after.json"
  cmp "$qa_dir/gateway-before.json" "$qa_dir/gateway-after.json"
  forward_bundled
  saved_trace
  qualification_read_boundaries lens-upgrade
  "${install[@]}" || diagnose
  forward_bundled
  saved_trace
  qualification_read_boundaries lens-rollback
  kubectl -n "$namespace" get deployment lens-lens-worker -o json \
    | jq -S .spec.template > "$qa_dir/lens-before.json"
  deployment_pods lens-lens-worker > "$qa_dir/lens-pods-before.json"
  gateway_components=(monolith)
  if [[ "$layout" == componentized ]]; then gateway_components=(gateway backend ui); fi
  gateway_upgrade=(--set image.tag=v0.0.0-lens-ci)
  migration_baseline=lens-ci-litellm:v0.0.0-lens-ci-baseline
  migration_current=lens-ci-litellm:v0.0.0-lens-ci
  migration_before=$(migration_job "$migration_baseline")
  for component in "${gateway_components[@]}"; do
    gateway_pods "$component" v0.0.0-lens-ci-baseline > "$qa_dir/$component-before.json"
  done
  install+=("${gateway_upgrade[@]}")
  "${install[@]}" || diagnose
  migration_after=$(migration_job "$migration_current")
  test "$migration_before" != "$migration_after"
  kubectl -n "$namespace" get deployment lens-lens-worker -o json \
    | jq -S .spec.template > "$qa_dir/lens-after.json"
  cmp "$qa_dir/lens-before.json" "$qa_dir/lens-after.json"
  deployment_pods lens-lens-worker > "$qa_dir/lens-pods-after.json"
  cmp "$qa_dir/lens-pods-before.json" "$qa_dir/lens-pods-after.json"
  for component in "${gateway_components[@]}"; do
    gateway_pods "$component" v0.0.0-lens-ci > "$qa_dir/$component-after.json"
    before_id=$(jq -ce '[.[].imageID] | unique | if length == 1 then .[0] else error("Mixed gateway images") end' \
      "$qa_dir/$component-before.json")
    after_id=$(jq -ce '[.[].imageID] | unique | if length == 1 then .[0] else error("Mixed gateway images") end' \
      "$qa_dir/$component-after.json")
    test "$before_id" != "$after_id"
  done
  forward_bundled
  saved_trace
  qualification_read_boundaries gateway-upgrade
  if [[ "$qualification_mode" != smoke ]]; then
    install+=(--set lensWorker.image.tag=upgrade)
    "${install[@]}" || diagnose
    forward_bundled
    qualification_read_boundaries both-candidates
  fi
  stop_forwards
  kubectl -n "$namespace" rollout restart "deployment/$control" deployment/lens-lens-worker
  kubectl -n "$namespace" rollout status "deployment/$control" --timeout=180s
  kubectl -n "$namespace" rollout status deployment/lens-lens-worker --timeout=180s
  qualification_forward_control
  saved_trace
  qualification_outage
  printf '%s: fresh install, ingestion, Lens upgrade and rollback, independent gateway image upgrade, and restart passed\n' "$layout"
  stop_forwards
  helm upgrade --install external-lens helm/litellm/charts/lens-0.1.0-dev.0.tgz \
    -n "$namespace" --wait --timeout 5m \
    --set fullnameOverride=external-lens \
    --set image.repository=lens-ci-worker --set image.tag=upgrade --set image.pullPolicy=Never \
    --set adminTokenSecret.name=lens-secrets --set adminTokenSecret.key=master-key \
    --set gateway.enabled=true --set gateway.secretName=lens-secrets \
    --set gateway.secretKey=gateway-secret --set serviceTokenSecret.name=lens-secrets \
    --set clickhouseSecret.name=lens-secrets --set clickhouseDatabase=existing_traces \
    --set publicUrl=http://127.0.0.1:14419
  "${install[@]}" --set lensWorker.mode=external \
    --set lensWorker.externalUrl=http://external-lens:4318 || diagnose
  test -z "$(kubectl -n "$namespace" get deployment lens-lens-worker --ignore-not-found -o name)"
  qualification_forward_control
  forward external-lens 14419 4318
  api /lens/service | jq -e '.configured and .connected and .status.storage_ready'
  saved_trace
  qualification_read_boundaries external-lens
  trace_id=$(openssl rand -hex 16)
  span_id=$(openssl rand -hex 8)
  jq --arg trace "$trace_id" --arg span "$span_id" \
    '.resourceSpans[0].scopeSpans[0].spans[0] |= (.traceId=$trace | .spanId=$span)' \
    "$qa_dir/trace.json" > "$qa_dir/external-trace.json"
  curl --fail-with-body --silent --show-error --max-time 20 \
    -H "Authorization: Bearer $tracing_key" -H 'Content-Type: application/json' \
    -d "@$qa_dir/external-trace.json" http://127.0.0.1:14419/v1/traces
  saved_trace
  kubectl -n "$namespace" get deployment external-lens -o json \
    | jq -S '{uid:.metadata.uid,template:.spec.template}' > "$qa_dir/external-before.json"
  stop_forwards
  "${install[@]}" --set lensWorker.mode=disabled \
    --set gateway.config.proxy_config.general_settings.tracing.enabled=false || diagnose
  test -z "$(kubectl -n "$namespace" get deployment lens-lens-worker --ignore-not-found -o name)"
  qualification_forward_control
  api /lens/service | jq -e '.configured == false and .connected == false'
  test "$(curl --silent --output /dev/null --write-out '%{http_code}' --max-time 20 \
    -H "Authorization: Bearer $master_key" http://127.0.0.1:14418/lens/datasets)" = 503
  kubectl -n "$namespace" get deployment external-lens -o json \
    | jq -S '{uid:.metadata.uid,template:.spec.template}' > "$qa_dir/external-after.json"
  cmp "$qa_dir/external-before.json" "$qa_dir/external-after.json"
  printf '%s: external Lens trace reads/writes and disabled gateway behavior passed\n' "$layout"
  qualification_record chart-lifecycle passed
  stop_forwards
  kubectl delete namespace "$namespace" --wait=true
done
