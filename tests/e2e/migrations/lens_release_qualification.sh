#!/usr/bin/env bash

qualification_initialize() {
  qualification_events="$qa_dir/qualification-events.jsonl"
  : > "$qualification_events"
  qualification_started=$(date -u +%FT%TZ)
  qualification_output=${LENS_QUALIFICATION_RESULTS_DIR:-}
  if [[ -n "$qualification_output" ]]; then
    mkdir -p "$qualification_output"
    test ! -e "$qualification_output/host-$chart_selection-$qualification_mode.json"
  fi
}

qualification_record() {
  if [[ "$qualification_mode" == smoke ]]; then return; fi
  jq -nc --arg chart "${chart:-bootstrap}" --arg case "$1" --arg result "$2" \
    --arg at "$(date -u +%FT%TZ)" '{chart:$chart,case:$case,result:$result,at:$at}' >> "$qualification_events"
}

qualification_finish() {
  if [[ "$qualification_mode" == smoke || -z "$qualification_output" ]]; then return; fi
  jq -n --slurpfile checks "$qualification_events" --arg mode "$qualification_mode" \
    --arg selection "$chart_selection" --arg started "$qualification_started" \
    --arg finished "$(date -u +%FT%TZ)" --arg source "$(git rev-parse HEAD)" --argjson exit_code "$1" \
    '{qualification:"isolated candidate host behavior",mode:$mode,chart_selection:$selection,
      started_at:$started,finished_at:$finished,harness_source:$source,exit_code:$exit_code,
      passed:($exit_code == 0),paid_provider_qualified:($mode == "release" and $exit_code == 0),
      signed_release_qualified:false,production_deployed:false,checks:$checks}' \
    > "$qualification_output/host-$chart_selection-$qualification_mode.json"
}

qualification_request() {
  local credential=$1 method=$2 route=$3 expected=$4 label=$5 input=${6:-}
  local port=14418
  if [[ "$chart" == litellm && "$route" == /v1/chat/completions ]]; then port=14420; fi
  local response="$qa_dir/release-response.json" status
  printf 'Authorization: Bearer %s\nContent-Type: application/json\n' "$credential" > "$qa_dir/release-headers"
  local arguments=(--silent --show-error --max-time 90 --request "$method" \
    --header "@$qa_dir/release-headers" --output "$response" --dump-header "$qa_dir/release-response.headers" \
    --write-out '%{http_code}')
  if [[ -n "$input" ]]; then arguments+=(--data-binary "@$input"); fi
  status=$(curl "${arguments[@]}" "http://127.0.0.1:$port$route")
  if [[ "$status" != "$expected" ]]; then
    qualification_record "$label" "failed-http-$status-expected-$expected"
    printf '%s: expected HTTP %s, received %s\n' "$label" "$expected" "$status" >&2
    return 1
  fi
  qualification_record "$label" "http-$status"
}

qualification_forward_control() {
  forward "$control" 14418 "$control_port"
  if [[ "$qualification_mode" == release && "$chart" == litellm ]]; then
    forward lens-gateway 14420 4000
  fi
}

qualification_provider_values() {
  if [[ "$qualification_mode" != release ]]; then return; fi
  printf '%s' "$LENS_RELEASE_PROVIDER_API_KEY" > "$qa_dir/provider-key"
  kubectl -n "$namespace" create secret generic lens-release-provider \
    --from-file="LENS_RELEASE_PROVIDER_API_KEY=$qa_dir/provider-key" > /dev/null
  if [[ "$chart" == litellm-helm ]]; then
    printf 'environmentSecrets: [lens-release-provider]\n' > "$qa_dir/provider.yaml"
  else
    cat > "$qa_dir/provider.yaml" <<'YAML'
gateway:
  extraEnv:
    - {name: STORE_MODEL_IN_DB, value: "True"}
    - name: LENS_RELEASE_PROVIDER_API_KEY
      valueFrom: {secretKeyRef: {name: lens-release-provider, key: LENS_RELEASE_PROVIDER_API_KEY}}
backend:
  extraEnv:
    - {name: STORE_MODEL_IN_DB, value: "True"}
    - name: LENS_RELEASE_PROVIDER_API_KEY
      valueFrom: {secretKeyRef: {name: lens-release-provider, key: LENS_RELEASE_PROVIDER_API_KEY}}
YAML
  fi
  install+=(-f "$qa_dir/provider.yaml")
}

qualification_setup() {
  if [[ "$qualification_mode" == smoke ]]; then return; fi
  local tenant user team credential trace span attempt status
  for tenant in a b; do
    user="lens-release-$tenant-$(openssl rand -hex 8)"
    team="lens-release-team-$tenant-$(openssl rand -hex 8)"
    jq -n --arg user "$user" '{user_id:$user,user_email:($user+"@example.test"),user_role:"internal_user",auto_create_key:false}' > "$qa_dir/request.json"
    qualification_request "$master_key" POST /user/new 200 "create-user-$tenant" "$qa_dir/request.json"
    jq -e --arg user "$user" '.user_id == $user' "$qa_dir/release-response.json" > /dev/null
    jq -n --arg user "$user" --arg team "$team" '{team_id:$team,team_alias:$team,members_with_roles:[{user_id:$user,role:"admin"}]}' > "$qa_dir/request.json"
    qualification_request "$master_key" POST /team/new 200 "create-team-$tenant" "$qa_dir/request.json"
    jq -e --arg team "$team" '.team_id == $team' "$qa_dir/release-response.json" > /dev/null
    jq -n --arg user "$user" --arg team "$team" '{user_id:$user,team_id:$team,key_alias:($team+"-isolation")}' > "$qa_dir/request.json"
    qualification_request "$master_key" POST /key/generate 200 "create-key-$tenant" "$qa_dir/request.json"
    credential=$(jq -er '.key | select(type == "string" and length > 0)' "$qa_dir/release-response.json")
    jq -n --arg user "$user" --arg team "$team" --arg key "$credential" '{user:$user,team:$team,key:$key}' > "$qa_dir/tenant-$tenant.json"
    jq -n --arg team "$team" '{name:"Release tenant trace",team_id:$team}' > "$qa_dir/request.json"
    qualification_request "$master_key" POST /lens/tracing/keys 200 "create-ingestion-key-$tenant" "$qa_dir/request.json"
    cp "$qa_dir/release-response.json" "$qa_dir/ingestion-$tenant.json"
    jq -e --arg team "$team" '.record.tenant.team_id == $team' "$qa_dir/ingestion-$tenant.json" > /dev/null
    credential=$(jq -er .key "$qa_dir/ingestion-$tenant.json")
    trace=$(openssl rand -hex 16)
    span=$(openssl rand -hex 8)
    jq -n --arg trace "$trace" --arg span "$span" --arg at "$(date +%s)000000000" \
      '{resourceSpans:[{scopeSpans:[{spans:[{traceId:$trace,spanId:$span,name:"Lens release tenant",kind:1,startTimeUnixNano:$at,endTimeUnixNano:$at,status:{code:1}}]}]}]}' > "$qa_dir/tenant-$tenant-trace.json"
    printf '%s' "$trace" > "$qa_dir/tenant-$tenant-trace-id"
    printf 'Authorization: Bearer %s\nContent-Type: application/json\n' "$credential" > "$qa_dir/ingestion-headers"
    for attempt in $(seq 1 30); do
      status=$(curl --silent --show-error --max-time 20 --output "$qa_dir/ingestion-response.json" --write-out '%{http_code}' \
        --header "@$qa_dir/ingestion-headers" --data-binary "@$qa_dir/tenant-$tenant-trace.json" http://127.0.0.1:14419/v1/traces)
      if [[ "$status" == 200 ]]; then break; fi
      test "$status" = 401
      sleep 1
    done
    test "$status" = 200
    qualification_record "ingest-team-$tenant" http-200
    for attempt in $(seq 1 30); do
      if api "/v1/traces/$trace" > "$qa_dir/tenant-$tenant-stored.json" 2> /dev/null; then break; fi
      sleep 1
    done
    jq -e --arg trace "$trace" '.summary.trace_id == $trace' "$qa_dir/tenant-$tenant-stored.json" > /dev/null
  done
  jq -n --arg user "$(jq -r .user "$qa_dir/tenant-a.json")" --arg team "$(jq -r .team "$qa_dir/tenant-a.json")" \
    '{user_id:$user,team_id:$team,key_alias:"Lens release revocation"}' > "$qa_dir/request.json"
  qualification_request "$master_key" POST /key/generate 200 create-revocation-key "$qa_dir/request.json"
  cp "$qa_dir/release-response.json" "$qa_dir/revoked-key.json"
  credential=$(jq -er .key "$qa_dir/revoked-key.json")
  qualification_request "$credential" GET /lens/service 200 key-valid-before-revocation
  jq -n --arg key "$credential" '{keys:[$key]}' > "$qa_dir/request.json"
  qualification_request "$master_key" POST /key/delete 200 revoke-gateway-key "$qa_dir/request.json"
  qualification_request "$credential" GET /lens/service 401 revoked-key-denied
  if [[ "$qualification_mode" == release ]]; then
    qualification_model="lens-release-$(openssl rand -hex 8)"
    jq -n --arg model "$qualification_model" --arg provider_model "$LENS_RELEASE_PROVIDER_MODEL" \
      --arg base "$LENS_RELEASE_PROVIDER_API_BASE" \
      '{model_name:$model,litellm_params:{model:$provider_model,api_base:$base,api_key:"os.environ/LENS_RELEASE_PROVIDER_API_KEY"}}' > "$qa_dir/request.json"
    qualification_request "$master_key" POST /model/new 200 register-provider-model "$qa_dir/request.json"
    jq -e '.model_info.id | type == "string" and length > 0' "$qa_dir/release-response.json" > /dev/null
    qualification_paid_calls=0
    qualification_total_cost=0
  fi
}

qualification_read_boundaries() {
  if [[ "$qualification_mode" == smoke ]]; then return; fi
  local phase=$1 tenant other credential own foreign
  for tenant in a b; do
    other=a
    if [[ "$tenant" == a ]]; then other=b; fi
    credential=$(jq -r .key "$qa_dir/tenant-$tenant.json")
    own=$(cat "$qa_dir/tenant-$tenant-trace-id")
    foreign=$(cat "$qa_dir/tenant-$other-trace-id")
    qualification_request "$credential" GET "/v1/traces/$own" 200 "$phase-$tenant-own-trace"
    jq -e --arg trace "$own" '.summary.trace_id == $trace' "$qa_dir/release-response.json" > /dev/null
    qualification_request "$credential" GET "/v1/traces/$foreign" 404 "$phase-$tenant-other-team-denied"
    qualification_request "$credential" GET "/v1/traces/$foreign?all_teams=1" 404 "$phase-$tenant-query-cannot-expand-scope"
    jq -n --arg sql "SELECT TraceId AS trace_id FROM otel_traces WHERE TraceId IN ('$own','$foreign') ORDER BY TraceId" '{sql:$sql}' > "$qa_dir/request.json"
    qualification_request "$credential" POST /v1/traces/query 200 "$phase-$tenant-scoped-sql" "$qa_dir/request.json"
    jq -e --arg trace "$own" '[.data[].trace_id] == [$trace]' "$qa_dir/release-response.json" > /dev/null
    qualification_request "$credential" GET /lens/signals 403 "$phase-$tenant-admin-surface-denied"
  done
  qualification_request "$(jq -r .key "$qa_dir/revoked-key.json")" GET /lens/service 401 "$phase-revocation-retained"
  qualification_request invalid-release-key GET "/v1/traces/$own" 401 "$phase-invalid-key-denied"
  qualification_record "$phase-auth-and-retained-data" passed
}

qualification_inference() {
  if [[ "$qualification_mode" != release ]]; then return; fi
  local phase=$1 credential response_id call_id cost attempt
  test "$qualification_paid_calls" -lt 3
  credential=$(jq -r .key "$qa_dir/tenant-a.json")
  jq -n --arg model "$qualification_model" --arg marker "$(openssl rand -hex 16)" \
    '{model:$model,messages:[{role:"user",content:("Reply with the single word ready. Test marker "+$marker)}],max_completion_tokens:64}' > "$qa_dir/request.json"
  qualification_request "$credential" POST /v1/chat/completions 200 "$phase-real-inference" "$qa_dir/request.json"
  qualification_paid_calls=$((qualification_paid_calls + 1))
  cp "$qa_dir/release-response.json" "$qa_dir/completion.json"
  jq -e '.choices[0].message.content | type == "string" and length > 0' "$qa_dir/completion.json" > /dev/null
  jq -e '.usage.prompt_tokens > 0 and .usage.completion_tokens > 0 and .usage.total_tokens == (.usage.prompt_tokens + .usage.completion_tokens)' "$qa_dir/completion.json" > /dev/null
  response_id=$(jq -er .id "$qa_dir/completion.json")
  call_id=$(awk 'tolower($1)=="x-litellm-call-id:" {gsub("\r", "", $2); print $2}' "$qa_dir/release-response.headers")
  cost=$(awk 'tolower($1)=="x-litellm-response-cost:" {gsub("\r", "", $2); print $2}' "$qa_dir/release-response.headers")
  [[ "$call_id" =~ ^[A-Za-z0-9_.:-]+$ ]]
  jq -en --arg cost "$cost" '($cost|tonumber) > 0' > /dev/null
  for attempt in $(seq 1 60); do
    qualification_request "$master_key" GET "/spend/logs?request_id=$call_id" 200 "$phase-billing-read"
    if jq -e --arg id "$response_id" '[.[] | select(.request_id == $id)] | length == 1' "$qa_dir/release-response.json" > /dev/null; then break; fi
    sleep 2
  done
  jq -e --arg id "$response_id" --arg cost "$cost" --arg team "$(jq -r .team "$qa_dir/tenant-a.json")" \
    --slurpfile completion "$qa_dir/completion.json" \
    '[.[] | select(.request_id == $id)] | length == 1 and (.[0] | .team_id == $team
      and .prompt_tokens == $completion[0].usage.prompt_tokens
      and .completion_tokens == $completion[0].usage.completion_tokens
      and .total_tokens == $completion[0].usage.total_tokens
      and ((.spend - ($cost|tonumber)) | fabs) < 0.00000001)' "$qa_dir/release-response.json" > /dev/null
  qualification_total_cost=$(jq -en --arg total "$qualification_total_cost" --arg cost "$cost" '($total|tonumber)+($cost|tonumber)')
  jq -en --arg total "$qualification_total_cost" '($total|tonumber) <= 0.10' > /dev/null
  qualification_record "$phase-billing-matches-usage-and-team" passed
  jq -nc --arg chart "$chart" --arg phase "$phase" --arg response_id "$response_id" \
    --arg call_id "$call_id" --argjson cost "$cost" '{chart:$chart,case:($phase+"-paid-receipt"),result:"passed",response_id:$response_id,request_id:$call_id,cost_usd:$cost}' >> "$qualification_events"
}

qualification_outage() {
  if [[ "$qualification_mode" == smoke ]]; then return; fi
  local replicas selector
  qualification_read_boundaries before-outage
  qualification_inference before-outage
  deployment_pods "$control" > "$qa_dir/outage-gateway-before.json"
  replicas=$(kubectl -n "$namespace" get deployment lens-lens-worker -o jsonpath='{.spec.replicas}')
  test "$replicas" = 1
  selector=$(kubectl -n "$namespace" get deployment lens-lens-worker -o json | jq -r '.spec.selector.matchLabels|to_entries|map("\(.key)=\(.value)")|join(",")')
  kubectl -n "$namespace" scale deployment/lens-lens-worker --replicas=0
  kubectl -n "$namespace" wait --for=delete pods -l "$selector" --timeout=90s
  qualification_request "$master_key" GET /lens/service 200 outage-service-status
  jq -e '.configured == true and .connected == false' "$qa_dir/release-response.json" > /dev/null
  qualification_request "$master_key" GET /lens/signals 503 outage-lens-unavailable
  qualification_request invalid-release-key GET /lens/signals 401 outage-invalid-key-still-denied
  qualification_inference during-outage
  kubectl -n "$namespace" scale deployment/lens-lens-worker --replicas="$replicas"
  kubectl -n "$namespace" rollout status deployment/lens-lens-worker --timeout=180s
  forward_bundled
  saved_trace
  qualification_read_boundaries after-outage
  qualification_inference after-outage
  deployment_pods "$control" > "$qa_dir/outage-gateway-after.json"
  cmp "$qa_dir/outage-gateway-before.json" "$qa_dir/outage-gateway-after.json"
  qualification_record lens-outage-without-gateway-restart passed
}

qualification_bundled_postgres() {
  local chart=bundled-postgresql release=lens-bootstrap
  namespace=lens-bundled-postgresql
  kubectl create namespace "$namespace"
  master_key="sk-$(openssl rand -hex 24)"
  printf '%s' "$master_key" > "$qa_dir/bootstrap-master"
  kubectl -n "$namespace" create secret generic bootstrap-master --from-file="key=$qa_dir/bootstrap-master" > /dev/null
  cat > "$qa_dir/bootstrap.yaml" <<YAML
fullnameOverride: $release
replicaCount: 0
image: {repository: lens-ci-monolith, tag: v0.0.0-lens-ci, pullPolicy: Never}
masterkeySecretName: bootstrap-master
masterkeySecretKey: key
envVars: {STORE_MODEL_IN_DB: "True"}
db: {deployStandalone: true, useExisting: false}
postgresql:
  auth: {username: litellm, database: litellm, password: "$(openssl rand -hex 24)", postgresPassword: "$(openssl rand -hex 24)"}
  primary: {persistence: {enabled: false}}
redis: {enabled: false}
lensWorker: {enabled: false}
migrationJob:
  ttlSecondsAfterFinished: 3600
  hooks: {helm: {enabled: false}, argocd: {enabled: false}}
proxy_config:
  model_list: []
  general_settings: {master_key: os.environ/PROXY_MASTER_KEY, store_model_in_db: true}
YAML
  helm upgrade --install "$release" helm/litellm-helm -n "$namespace" -f "$qa_dir/bootstrap.yaml" \
    --wait --wait-for-jobs --timeout 8m
  kubectl -n "$namespace" get deployment "$release" -o json | jq -e '.spec.replicas == 0 and (.status.replicas // 0) == 0' > /dev/null
  kubectl -n "$namespace" get job "$release-migrations" -o json \
    | jq -e '.status.succeeded == 1 and .metadata.annotations["helm.sh/hook"] == null and .metadata.annotations["argocd.argoproj.io/hook"] == null' > /dev/null
  qualification_record bundled-postgres-migrated-before-gateway-start passed
  kubectl -n "$namespace" scale deployment/"$release" --replicas=1
  kubectl -n "$namespace" rollout status deployment/"$release" --timeout=180s
  forward "$release" 14418 4000
  jq -n '{key_alias:"Bundled PostgreSQL persisted key"}' > "$qa_dir/request.json"
  qualification_request "$master_key" POST /key/generate 200 bundled-postgres-key-create "$qa_dir/request.json"
  local generated
  generated=$(jq -er .key "$qa_dir/release-response.json")
  qualification_request "$generated" GET /v1/models 200 bundled-postgres-key-auth
  kubectl -n "$namespace" rollout restart deployment/"$release"
  kubectl -n "$namespace" rollout status deployment/"$release" --timeout=180s
  stop_forwards
  forward "$release" 14418 4000
  qualification_request "$generated" GET /v1/models 200 bundled-postgres-key-survives-gateway-restart
  jq -n --arg key "$generated" '{keys:[$key]}' > "$qa_dir/request.json"
  qualification_request "$master_key" POST /key/delete 200 bundled-postgres-key-delete "$qa_dir/request.json"
  qualification_request "$generated" GET /v1/models 401 bundled-postgres-key-revoked
  qualification_record bundled-postgres-ordinary-job-two-phase-bootstrap passed
  stop_forwards
  kubectl delete namespace "$namespace" --wait=true
  namespace=
}
