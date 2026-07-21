#!/usr/bin/env bash
# Reproduce LiteLLM Azure PostgreSQL and Redis authentication from a real AKS
# Workload Identity. This creates billable resources and deletes them by default.
set -Eeuo pipefail

[[ "${AZURE_WI_E2E_ACK:-}" == "YES" ]] || {
  echo "Set AZURE_WI_E2E_ACK=YES to acknowledge temporary Azure charges." >&2
  exit 2
}

POST_PR_COMMENT="${POST_PR_COMMENT:-0}"
PR_NUMBER="${PR_NUMBER:-}"
if [[ "$POST_PR_COMMENT" == "1" && -z "$PR_NUMBER" ]]; then
  PR_NUMBER=30633
fi
if [[ "$POST_PR_COMMENT" == "1" && "${KEEP_RESOURCES:-0}" == "1" ]]; then
  echo "POST_PR_COMMENT=1 requires default cleanup; unset KEEP_RESOURCES." >&2
  exit 2
fi

for command_name in az git kubectl docker curl jq openssl rg; do
  command -v "$command_name" >/dev/null 2>&1 || {
    echo "Missing required command: $command_name" >&2
    exit 2
  }
done
if [[ -n "$PR_NUMBER" || "$POST_PR_COMMENT" == "1" ]]; then
  command -v gh >/dev/null 2>&1 || {
    echo "Missing required command for PR verification/publication: gh" >&2
    exit 2
  }
fi

SUBSCRIPTION_ID="${AZURE_SUBSCRIPTION_ID:-$(az account show --query id -o tsv)}"
TENANT_ID="$(az account show --query tenantId -o tsv)"
LOCATION="${AZURE_LOCATION:-westeurope}"
AKS_NODE_VM_SIZE="${AKS_NODE_VM_SIZE:-Standard_B2s}"
GIT_SHA="$(git rev-parse HEAD)"
SUFFIX="$(openssl rand -hex 4 | tr '[:upper:]' '[:lower:]')"
PREFIX="litellm-wi-e2e-${SUFFIX}"
RESOURCE_GROUP="$PREFIX"
NODE_RESOURCE_GROUP="${PREFIX}-nodes"
ACR_NAME="litellmwi${SUFFIX}"
AKS_NAME="${PREFIX}-aks"
IDENTITY_NAME="${PREFIX}-id"
POSTGRES_NAME="${PREFIX}-pg"
REDIS_NAME="${PREFIX}-redis"
NAMESPACE="litellm-wi-proof"
SERVICE_ACCOUNT="litellm-wi"
TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/litellm-wi-e2e.XXXXXX")"
EVIDENCE_FILE="${EVIDENCE_FILE:-${TMPDIR:-/tmp}/${PREFIX}-evidence.md}"
CLEANUP_STARTED=0
CLEANUP_OK=0
RUN_SUCCEEDED=0

die() {
  echo "ERROR: $*" >&2
  exit 1
}

retry_postgres_busy() {
  postgres_attempt=1
  while ! "$@" 2>"$TMP_DIR/postgres-operation.err"; do
    if ! rg -q 'ServerIsBusy|AnotherOperationInProgress' "$TMP_DIR/postgres-operation.err" || \
      [[ "$postgres_attempt" -ge 30 ]]; then
      sed -n '1,20p' "$TMP_DIR/postgres-operation.err" >&2
      return 1
    fi
    sleep 10
    postgres_attempt=$((postgres_attempt + 1))
  done
}

group_exists() {
  [[ "$(az group exists --name "$1" 2>/dev/null)" == "true" ]]
}

cleanup_resources() {
  [[ "$CLEANUP_STARTED" == "0" ]] || return 0
  CLEANUP_STARTED=1
  trap '' INT TERM
  set +e

  if [[ "${KEEP_RESOURCES:-0}" == "1" ]]; then
    {
      echo
      echo "## Cleanup"
      echo
      echo "- Resources retained by KEEP_RESOURCES=1: true"
      echo "- Delete with: \`az group delete -g $RESOURCE_GROUP -y\`"
    } >>"$EVIDENCE_FILE"
    echo "WARNING: Resources retained and may incur charges." >&2
    echo "Delete with: az group delete -g $RESOURCE_GROUP -y" >&2
    trap 'exit 130' INT
    trap 'exit 143' TERM
    return 0
  fi

  if group_exists "$RESOURCE_GROUP"; then
    az group delete --name "$RESOURCE_GROUP" --yes --no-wait --only-show-errors
    az group wait --name "$RESOURCE_GROUP" --deleted --interval 20 --timeout 1800
  fi
  if group_exists "$NODE_RESOURCE_GROUP"; then
    az group delete --name "$NODE_RESOURCE_GROUP" --yes --no-wait --only-show-errors
    az group wait --name "$NODE_RESOURCE_GROUP" --deleted --interval 20 --timeout 1800
  fi

  parent_exists="$(az group exists --name "$RESOURCE_GROUP" 2>/dev/null)"
  node_exists="$(az group exists --name "$NODE_RESOURCE_GROUP" 2>/dev/null)"
  {
    echo
    echo "## Cleanup"
    echo
    echo "- Parent resource group exists: $parent_exists"
    echo "- AKS node resource group exists: $node_exists"
  } >>"$EVIDENCE_FILE"
  if [[ "$parent_exists" == "false" && "$node_exists" == "false" ]]; then
    CLEANUP_OK=1
  fi
  set -e
  trap 'exit 130' INT
  trap 'exit 143' TERM
}

on_exit() {
  exit_code=$?
  trap - EXIT
  cleanup_resources
  if [[ "$RUN_SUCCEEDED" == "1" ]]; then
    rm -rf "$TMP_DIR"
  else
    echo "Failure artifacts preserved in: $TMP_DIR" >&2
  fi
  echo "Sanitized evidence: $EVIDENCE_FILE" >&2
  exit "$exit_code"
}
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

: >"$EVIDENCE_FILE"
chmod 600 "$EVIDENCE_FILE"

az account set --subscription "$SUBSCRIPTION_ID"
[[ "$(az account show --query state -o tsv)" == "Enabled" ]] || die "Azure subscription is not enabled."
[[ "$(az account show --query tenantId -o tsv)" == "$TENANT_ID" ]] || die "Azure tenant changed during preflight."
[[ -z "$(git status --porcelain)" ]] || die "Worktree must be clean so the image exactly matches Git HEAD."
if [[ -n "$PR_NUMBER" ]]; then
  PR_HEAD="$(gh pr view "$PR_NUMBER" --repo BerriAI/litellm --json headRefOid --jq .headRefOid)"
  [[ "$PR_HEAD" == "$GIT_SHA" ]] || die "Git HEAD does not match PR #$PR_NUMBER head."
fi

CREATOR="$(az account show --query user.name -o tsv)"
CREATED_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Resource group: $RESOURCE_GROUP"
echo "Emergency cleanup: az group delete -g $RESOURCE_GROUP -y"
echo "Evidence file: $EVIDENCE_FILE"

for provider in Microsoft.Cache Microsoft.ContainerRegistry Microsoft.ContainerService Microsoft.DBforPostgreSQL Microsoft.ManagedIdentity Microsoft.Network; do
  az provider register --namespace "$provider" --wait --only-show-errors
done

az group create --name "$RESOURCE_GROUP" --location "$LOCATION" \
  --tags purpose=litellm-workload-identity-e2e git_sha="$GIT_SHA" creator="$CREATOR" created_utc="$CREATED_UTC" \
  --only-show-errors >/dev/null
az acr create --resource-group "$RESOURCE_GROUP" --name "$ACR_NAME" --sku Basic \
  --admin-enabled false --only-show-errors >/dev/null
az identity create --resource-group "$RESOURCE_GROUP" --name "$IDENTITY_NAME" \
  --location "$LOCATION" --only-show-errors >/dev/null
IDENTITY_CLIENT_ID="$(az identity show -g "$RESOURCE_GROUP" -n "$IDENTITY_NAME" --query clientId -o tsv)"
IDENTITY_PRINCIPAL_ID="$(az identity show -g "$RESOURCE_GROUP" -n "$IDENTITY_NAME" --query principalId -o tsv)"

az aks create --resource-group "$RESOURCE_GROUP" --name "$AKS_NAME" --location "$LOCATION" \
  --tier free --node-count 1 --node-vm-size "$AKS_NODE_VM_SIZE" \
  --enable-oidc-issuer --enable-workload-identity \
  --node-resource-group "$NODE_RESOURCE_GROUP" --attach-acr "$ACR_NAME" \
  --no-ssh-key --only-show-errors >/dev/null
az aks get-credentials --resource-group "$RESOURCE_GROUP" --name "$AKS_NAME" \
  --overwrite-existing --only-show-errors
OIDC_ENABLED="$(az aks show -g "$RESOURCE_GROUP" -n "$AKS_NAME" --query oidcIssuerProfile.enabled -o tsv)"
WORKLOAD_IDENTITY_ENABLED="$(az aks show -g "$RESOURCE_GROUP" -n "$AKS_NAME" --query securityProfile.workloadIdentity.enabled -o tsv)"
OIDC_ISSUER="$(az aks show -g "$RESOURCE_GROUP" -n "$AKS_NAME" --query oidcIssuerProfile.issuerUrl -o tsv)"
[[ "$OIDC_ENABLED" == "true" && "$WORKLOAD_IDENTITY_ENABLED" == "true" ]] || die "AKS identity features are not enabled."

ADMIN_OBJECT_ID="$(az ad signed-in-user show --query id -o tsv)"
ADMIN_DISPLAY_NAME="$(az ad signed-in-user show --query userPrincipalName -o tsv)"
PUBLIC_IP="$(curl -fsS https://api.ipify.org)"
[[ "$PUBLIC_IP" =~ ^[0-9a-fA-F:.]+$ ]] || die "Could not resolve the operator public IP."

POSTGRES_BODY="$(jq -cn --arg location "$LOCATION" --arg tenant "$TENANT_ID" \
  '{location:$location,sku:{name:"Standard_B1ms",tier:"Burstable"},properties:{version:"16",storage:{storageSizeGB:32},authConfig:{activeDirectoryAuth:"Enabled",passwordAuth:"Disabled",tenantId:$tenant},network:{publicNetworkAccess:"Enabled"}}}')"
az rest --method put \
  --url "https://management.azure.com/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.DBforPostgreSQL/flexibleServers/$POSTGRES_NAME?api-version=2024-08-01" \
  --body "$POSTGRES_BODY" --only-show-errors >/dev/null
unset POSTGRES_BODY
az resource wait --resource-group "$RESOURCE_GROUP" --name "$POSTGRES_NAME" \
  --resource-type Microsoft.DBforPostgreSQL/flexibleServers \
  --custom "properties.state=='Ready'" --interval 20 --timeout 1800 --only-show-errors
retry_postgres_busy az postgres flexible-server firewall-rule create --resource-group "$RESOURCE_GROUP" \
  --server-name "$POSTGRES_NAME" --name operator-bootstrap \
  --start-ip-address "$PUBLIC_IP" --end-ip-address "$PUBLIC_IP" --only-show-errors >/dev/null

ADMIN_BODY="$(jq -cn --arg name "$ADMIN_DISPLAY_NAME" --arg tenant "$TENANT_ID" \
  '{properties:{principalName:$name,principalType:"User",tenantId:$tenant}}')"
retry_postgres_busy az rest --method put \
  --url "https://management.azure.com/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.DBforPostgreSQL/flexibleServers/$POSTGRES_NAME/administrators/$ADMIN_OBJECT_ID?api-version=2024-08-01" \
  --body "$ADMIN_BODY" --only-show-errors >/dev/null
unset ADMIN_BODY
POSTGRES_HOST="$(az postgres flexible-server show -g "$RESOURCE_GROUP" -n "$POSTGRES_NAME" --query fullyQualifiedDomainName -o tsv)"
POSTGRES_AUTH="$(az postgres flexible-server show -g "$RESOURCE_GROUP" -n "$POSTGRES_NAME" -o json)"
jq -e '.authConfig.activeDirectoryAuth == "Enabled" and .authConfig.passwordAuth == "Disabled"' \
  <<<"$POSTGRES_AUTH" >/dev/null || die "PostgreSQL is not Entra-only."

printf '%s\n' '{"aad-enabled":"true"}' >"$TMP_DIR/redis-configuration.json"
az redis create --resource-group "$RESOURCE_GROUP" --name "$REDIS_NAME" --location "$LOCATION" \
  --sku Basic --vm-size c0 --minimum-tls-version 1.2 \
  --disable-access-keys true --redis-configuration "@$TMP_DIR/redis-configuration.json" \
  --only-show-errors >/dev/null
REDIS_JSON="$(az redis show -g "$RESOURCE_GROUP" -n "$REDIS_NAME" -o json)"
REDIS_HOST="$(jq -r '.hostName' <<<"$REDIS_JSON")"
jq -e '((.redisConfiguration.aadEnabled // .redisConfiguration["aad-enabled"]) | tostring | ascii_downcase) == "true"
  and .disableAccessKeyAuthentication == true
  and .enableNonSslPort == false
  and .minimumTlsVersion == "1.2"' <<<"$REDIS_JSON" >/dev/null || die "Redis security settings are incorrect."

az identity federated-credential create --resource-group "$RESOURCE_GROUP" \
  --identity-name "$IDENTITY_NAME" --name litellm --issuer "$OIDC_ISSUER" \
  --subject "system:serviceaccount:$NAMESPACE:$SERVICE_ACCOUNT" \
  --audiences api://AzureADTokenExchange --only-show-errors >/dev/null

kubectl create namespace "$NAMESPACE"
kubectl create serviceaccount "$SERVICE_ACCOUNT" --namespace "$NAMESPACE"
kubectl annotate serviceaccount "$SERVICE_ACCOUNT" --namespace "$NAMESPACE" \
  "azure.workload.identity/client-id=$IDENTITY_CLIENT_ID"
SA_CLIENT_ID="$(kubectl get serviceaccount "$SERVICE_ACCOUNT" -n "$NAMESPACE" \
  -o jsonpath='{.metadata.annotations.azure\.workload\.identity/client-id}')"
[[ "$SA_CLIENT_ID" == "$IDENTITY_CLIENT_ID" ]] || die "Service-account client ID annotation does not match."

FEDERATION_JSON="$(az identity federated-credential show -g "$RESOURCE_GROUP" \
  --identity-name "$IDENTITY_NAME" -n litellm -o json)"
jq -e --arg subject "system:serviceaccount:$NAMESPACE:$SERVICE_ACCOUNT" \
  '.subject == $subject and (.audiences | index("api://AzureADTokenExchange") != null)' \
  <<<"$FEDERATION_JSON" >/dev/null || die "Federated credential does not match the service account."

POSTGRES_ADMIN_TOKEN="$(az account get-access-token --resource-type oss-rdbms --query accessToken -o tsv)"
PGPASSWORD="$POSTGRES_ADMIN_TOKEN" docker run --rm -e PGPASSWORD postgres:16-alpine \
  psql "host=$POSTGRES_HOST port=5432 dbname=postgres user=$ADMIN_DISPLAY_NAME sslmode=require" \
  --set=identity_name="$IDENTITY_NAME" --set=principal_id="$IDENTITY_PRINCIPAL_ID" \
  --command="SELECT * FROM pgaadauth_create_principal_with_oid(:'identity_name', :'principal_id', 'service', false);" \
  >/dev/null
PGPASSWORD="$POSTGRES_ADMIN_TOKEN" docker run --rm -e PGPASSWORD postgres:16-alpine \
  psql "host=$POSTGRES_HOST port=5432 dbname=postgres user=$ADMIN_DISPLAY_NAME sslmode=require" \
  --set=identity_name="$IDENTITY_NAME" \
  --command='CREATE DATABASE litellm OWNER :"identity_name";' >/dev/null
PG_PRINCIPAL_COUNT="$(PGPASSWORD="$POSTGRES_ADMIN_TOKEN" docker run --rm -e PGPASSWORD postgres:16-alpine \
  psql "host=$POSTGRES_HOST port=5432 dbname=postgres user=$ADMIN_DISPLAY_NAME sslmode=require" \
  --tuples-only --no-align --set=identity_name="$IDENTITY_NAME" \
  --command="SELECT count(*) FROM pg_roles WHERE rolname = :'identity_name';")"
unset POSTGRES_ADMIN_TOKEN PGPASSWORD
[[ "$PG_PRINCIPAL_COUNT" == "1" ]] || die "PostgreSQL managed-identity principal was not created."

AKS_OUTBOUND_IP_ID="$(az aks show -g "$RESOURCE_GROUP" -n "$AKS_NAME" \
  --query 'networkProfile.loadBalancerProfile.effectiveOutboundIPs[0].id' -o tsv)"
AKS_OUTBOUND_IP="$(az network public-ip show --ids "$AKS_OUTBOUND_IP_ID" --query ipAddress -o tsv)"
az postgres flexible-server firewall-rule create --resource-group "$RESOURCE_GROUP" \
  --server-name "$POSTGRES_NAME" --name aks-outbound \
  --start-ip-address "$AKS_OUTBOUND_IP" --end-ip-address "$AKS_OUTBOUND_IP" \
  --only-show-errors >/dev/null
OPERATOR_FIREWALL_RULES="$(az postgres flexible-server firewall-rule list -g "$RESOURCE_GROUP" \
  --server-name "$POSTGRES_NAME" --query "[?startIpAddress=='$PUBLIC_IP'].name" -o tsv)"
while IFS= read -r firewall_rule; do
  [[ -n "$firewall_rule" ]] || continue
  az postgres flexible-server firewall-rule delete -g "$RESOURCE_GROUP" --server-name "$POSTGRES_NAME" \
    --name "$firewall_rule" --yes --only-show-errors >/dev/null
done <<<"$OPERATOR_FIREWALL_RULES"
OPERATOR_FIREWALL_RULES="$(az postgres flexible-server firewall-rule list -g "$RESOURCE_GROUP" \
  --server-name "$POSTGRES_NAME" --query "[?startIpAddress=='$PUBLIC_IP'].name" -o tsv)"
[[ -z "$OPERATOR_FIREWALL_RULES" ]] || die "The operator PostgreSQL firewall rule was not removed."
unset OPERATOR_FIREWALL_RULES

az redis access-policy-assignment create --resource-group "$RESOURCE_GROUP" \
  --name "$REDIS_NAME" --object-id "$IDENTITY_PRINCIPAL_ID" \
  --object-id-alias "$IDENTITY_PRINCIPAL_ID" --access-policy-name "Data Contributor" \
  --policy-assignment-name litellm-workload-identity --only-show-errors >/dev/null
REDIS_POLICY_JSON="$(az redis access-policy-assignment show -g "$RESOURCE_GROUP" -n "$REDIS_NAME" \
  --policy-assignment-name litellm-workload-identity -o json)"
jq -e --arg object_id "$IDENTITY_PRINCIPAL_ID" \
  '.objectId == $object_id and .objectIdAlias == $object_id and .accessPolicyName == "Data Contributor"' \
  <<<"$REDIS_POLICY_JSON" >/dev/null || die "Redis access policy does not match the managed identity."

IMAGE_TAG="wi-e2e-${GIT_SHA}"
az acr build --registry "$ACR_NAME" --image "litellm:$IMAGE_TAG" . --only-show-errors
ACR_LOGIN_SERVER="$(az acr show -g "$RESOURCE_GROUP" -n "$ACR_NAME" --query loginServer -o tsv)"
IMAGE_DIGEST="$(az acr repository show -n "$ACR_NAME" --image "litellm:$IMAGE_TAG" --query digest -o tsv)"
[[ "$IMAGE_DIGEST" == sha256:* ]] || die "ACR did not return an immutable digest."
IMAGE_REFERENCE="${ACR_LOGIN_SERVER}/litellm@${IMAGE_DIGEST}"

cat >"$TMP_DIR/proxy.yaml" <<EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: litellm-config
  namespace: $NAMESPACE
data:
  config.yaml: |
    model_list: []
    litellm_settings:
      telemetry: false
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: litellm-postgres-proof
  namespace: $NAMESPACE
spec:
  replicas: 1
  selector:
    matchLabels:
      app: litellm-postgres-proof
  template:
    metadata:
      labels:
        app: litellm-postgres-proof
        azure.workload.identity/use: "true"
    spec:
      serviceAccountName: $SERVICE_ACCOUNT
      containers:
        - name: litellm
          image: $IMAGE_REFERENCE
          imagePullPolicy: IfNotPresent
          args: ["--config", "/config/config.yaml", "--port", "4000", "--azure_postgresql_auth"]
          ports:
            - name: http
              containerPort: 4000
          env:
            - name: DATABASE_HOST
              value: $POSTGRES_HOST
            - name: DATABASE_PORT
              value: "5432"
            - name: DATABASE_USER
              value: $IDENTITY_NAME
            - name: DATABASE_NAME
              value: litellm
            - name: STORE_MODEL_IN_DB
              value: "True"
          volumeMounts:
            - name: config
              mountPath: /config
              readOnly: true
      volumes:
        - name: config
          configMap:
            name: litellm-config
EOF
kubectl apply -f "$TMP_DIR/proxy.yaml"
kubectl rollout status deployment/litellm-postgres-proof -n "$NAMESPACE" --timeout=15m
PROXY_POD="$(kubectl get pods -n "$NAMESPACE" -l app=litellm-postgres-proof \
  -o jsonpath='{.items[0].metadata.name}')"
POD_IMAGE_ID="$(kubectl get pod "$PROXY_POD" -n "$NAMESPACE" \
  -o jsonpath='{.status.containerStatuses[0].imageID}')"
[[ "$POD_IMAGE_ID" == *"$IMAGE_DIGEST"* ]] || die "Running proxy image does not match the ACR digest."

kubectl exec -n "$NAMESPACE" "$PROXY_POD" -- python -c \
  'import json,os; required=("AZURE_CLIENT_ID","AZURE_TENANT_ID","AZURE_FEDERATED_TOKEN_FILE"); forbidden=("AZURE_CLIENT_SECRET","DATABASE_PASSWORD","DATABASE_URL","REDIS_PASSWORD"); print(json.dumps({"present":[x for x in required if x in os.environ],"absent":[x for x in forbidden if x not in os.environ]},sort_keys=True))' \
  >"$TMP_DIR/proxy-env.json"
jq -e '.present == ["AZURE_CLIENT_ID","AZURE_TENANT_ID","AZURE_FEDERATED_TOKEN_FILE"]
  and .absent == ["AZURE_CLIENT_SECRET","DATABASE_PASSWORD","DATABASE_URL","REDIS_PASSWORD"]' \
  "$TMP_DIR/proxy-env.json" >/dev/null || die "Proxy pod credential environment is incorrect."

READINESS_HTTP="$(kubectl exec -n "$NAMESPACE" "$PROXY_POD" -- python -c \
  'import pathlib,urllib.request; r=urllib.request.urlopen("http://127.0.0.1:4000/health/readiness",timeout=20); pathlib.Path("/tmp/readiness.json").write_bytes(r.read()); print(r.status)' | tr -d '\r')"
[[ "$READINESS_HTTP" == "200" ]] || die "Proxy readiness did not return HTTP 200."
kubectl cp "$NAMESPACE/$PROXY_POD:/tmp/readiness.json" "$TMP_DIR/readiness.json" >/dev/null
kubectl logs -n "$NAMESPACE" "$PROXY_POD" >"$TMP_DIR/proxy.log"
rg -q 'Azure PostgreSQL Entra token refresh loop started' "$TMP_DIR/proxy.log" || die "PostgreSQL token refresh loop did not start."
rg -q 'Azure PostgreSQL Entra token refresh scheduled in' "$TMP_DIR/proxy.log" || die "PostgreSQL token refresh was not scheduled."
rg -q 'Started Azure PostgreSQL Entra token proactive refresh background task' "$TMP_DIR/proxy.log" || \
  die "PostgreSQL proactive token refresh task did not start."
jq -e '.status == "healthy" and .db == "connected"' "$TMP_DIR/readiness.json" >/dev/null || \
  die "Proxy readiness does not report a connected database."

cat >"$TMP_DIR/redis-job.yaml" <<EOF
apiVersion: batch/v1
kind: Job
metadata:
  name: litellm-redis-proof
  namespace: $NAMESPACE
spec:
  backoffLimit: 0
  ttlSecondsAfterFinished: 600
  template:
    metadata:
      labels:
        app: litellm-redis-proof
        azure.workload.identity/use: "true"
    spec:
      restartPolicy: Never
      serviceAccountName: $SERVICE_ACCOUNT
      containers:
        - name: proof
          image: $IMAGE_REFERENCE
          imagePullPolicy: IfNotPresent
          command: ["python", "-c"]
          args:
            - |
              import asyncio
              import json
              import os
              import uuid
              from litellm._redis import get_redis_async_client, get_redis_client, get_redis_connection_pool

              async def main():
                  required = ("AZURE_CLIENT_ID", "AZURE_TENANT_ID", "AZURE_FEDERATED_TOKEN_FILE")
                  forbidden = ("AZURE_CLIENT_SECRET", "REDIS_PASSWORD")
                  assert all(name in os.environ for name in required)
                  assert all(name not in os.environ for name in forbidden)
                  key = f"litellm:wi-proof:{uuid.uuid4().hex}"
                  kwargs = {
                      "url": os.environ["REDIS_URL"],
                      "username": os.environ["REDIS_USERNAME"],
                      "azure_redis_ad_token": True,
                  }
                  sync_client = get_redis_client(**kwargs)
                  assert sync_client.ping() is True
                  pool = get_redis_connection_pool(**kwargs)
                  assert pool is not None
                  client = get_redis_async_client(connection_pool=pool, **kwargs)
                  assert await client.ping() is True
                  assert await client.set(key, "ok", ex=60) is True
                  assert await client.get(key) == b"ok"
                  assert await client.delete(key) == 1
                  assert await client.get(key) is None
                  print(json.dumps({"async_pool_ping": True, "credential_source": "AKS Workload Identity", "set_get_delete": True, "sync_ping": True}, sort_keys=True))
                  await client.aclose()
                  await pool.disconnect()
                  sync_client.close()

              asyncio.run(main())
          env:
            - name: REDIS_URL
              value: rediss://$REDIS_HOST:6380/0
            - name: REDIS_USERNAME
              value: $IDENTITY_PRINCIPAL_ID
            - name: REDIS_AZURE_AD_TOKEN
              value: "true"
            - name: REDIS_SOCKET_TIMEOUT
              value: "10"
EOF

REDIS_RESULT=""
attempt=1
while [[ "$attempt" -le 10 ]]; do
  kubectl delete job litellm-redis-proof -n "$NAMESPACE" --ignore-not-found --wait=true >/dev/null
  kubectl apply -f "$TMP_DIR/redis-job.yaml" >/dev/null
  if kubectl wait job/litellm-redis-proof -n "$NAMESPACE" --for=condition=complete --timeout=5m >/dev/null 2>&1; then
    kubectl logs job/litellm-redis-proof -n "$NAMESPACE" >"$TMP_DIR/redis.log"
    REDIS_RESULT="$(rg '^\{"async_pool_ping": true, "credential_source": "AKS Workload Identity", "set_get_delete": true, "sync_ping": true\}$' \
      "$TMP_DIR/redis.log" | tail -n 1)"
    [[ -n "$REDIS_RESULT" ]] || die "Redis proof completed without the expected result."
    break
  fi
  kubectl logs job/litellm-redis-proof -n "$NAMESPACE" >"$TMP_DIR/redis.log" 2>&1 || true
  if ! rg -qi 'WRONGPASS|NOAUTH|access.?policy|not authorized|authentication' "$TMP_DIR/redis.log"; then
    die "Redis proof failed for a non-propagation reason; see $TMP_DIR/redis.log"
  fi
  if [[ "$attempt" == "10" ]]; then
    die "Redis access policy did not propagate after ten attempts."
  fi
  echo "Redis identity access is still propagating (attempt $attempt/10)."
  sleep 30
  attempt=$((attempt + 1))
done

REDIS_POD="$(kubectl get pods -n "$NAMESPACE" -l job-name=litellm-redis-proof \
  -o jsonpath='{.items[0].metadata.name}')"
REDIS_IMAGE_ID="$(kubectl get pod "$REDIS_POD" -n "$NAMESPACE" \
  -o jsonpath='{.status.containerStatuses[0].imageID}')"
[[ "$REDIS_IMAGE_ID" == *"$IMAGE_DIGEST"* ]] || die "Running Redis proof image does not match the ACR digest."
REDIS_EXIT_CODE="$(kubectl get pod "$REDIS_POD" -n "$NAMESPACE" \
  -o jsonpath='{.status.containerStatuses[0].state.terminated.exitCode}')"
[[ "$REDIS_EXIT_CODE" == "0" ]] || die "Redis proof container did not exit successfully."

FORBIDDEN_ENV_NAMES="$(kubectl get deployment litellm-postgres-proof -n "$NAMESPACE" -o json | \
  jq -r '[.spec.template.spec.containers[].env[]?.name] | map(select(. == "AZURE_CLIENT_SECRET" or . == "DATABASE_PASSWORD" or . == "DATABASE_URL" or . == "REDIS_PASSWORD")) | length')"
[[ "$FORBIDDEN_ENV_NAMES" == "0" ]] || die "A forbidden credential variable exists in the proxy specification."

cat >"$EVIDENCE_FILE" <<EOF
# Azure Workload Identity live proof

- Git/PR head SHA: \`$GIT_SHA\`
- Immutable image digest: \`$IMAGE_DIGEST\`
- Proxy pod digest matches: true
- Redis job digest matches: true
- AKS OIDC issuer enabled: $OIDC_ENABLED
- AKS Workload Identity enabled: $WORKLOAD_IDENTITY_ENABLED
- Federated subject: \`system:serviceaccount:$NAMESPACE:$SERVICE_ACCOUNT\`
- Federated audience: \`api://AzureADTokenExchange\`
- Service-account client-ID annotation matches: true
- Injected variable names present: \`AZURE_CLIENT_ID\`, \`AZURE_TENANT_ID\`, \`AZURE_FEDERATED_TOKEN_FILE\`
- Forbidden variable names absent: \`AZURE_CLIENT_SECRET\`, \`DATABASE_PASSWORD\`, \`DATABASE_URL\`, \`REDIS_PASSWORD\`

## PostgreSQL

- SKU/storage: Burstable Standard_B1ms / 32 GiB
- Microsoft Entra authentication enabled: true
- Password authentication disabled: true
- Managed-identity database principal exists: true
- Readiness HTTP status: $READINESS_HTTP
- Readiness status: healthy
- Database status: connected
- Token refresh loop started: true
- Token refresh scheduled: true
- Proactive token refresh background task started: true
- No inference request made.

## Redis

- SKU: Azure Cache for Redis Basic C0
- Microsoft Entra authentication enabled: true
- Access-key authentication disabled: true
- Non-TLS port disabled: true
- Minimum TLS version: 1.2
- Access policy: Data Contributor
- Object-ID alias matches managed identity principal: true
- Job condition: Complete
- Container exit code: $REDIS_EXIT_CODE
- Production routes: sync URL client and async URL connection pool
- Result: \`$REDIS_RESULT\`
- Proof key deleted: true
- No inference request made.
EOF

SECRET_PATTERN='eyJ[A-Za-z0-9_-]+\.|password[=:][^[:space:]<]+|postgresql://[^[:space:]<]+:[^@[:space:]<]+@|client_secret[=:][^[:space:]<]+|refresh_token[=:][^[:space:]<]+|access_token[=:][^[:space:]<]+'
if rg -qi "$SECRET_PATTERN" "$EVIDENCE_FILE" "$TMP_DIR/proxy.log" "$TMP_DIR/redis.log" >/dev/null; then
  die "Potential secret found; evidence will not be published."
fi

cleanup_resources
if [[ "${KEEP_RESOURCES:-0}" != "1" && "$CLEANUP_OK" != "1" ]]; then
  die "Azure cleanup could not be confirmed."
fi
if rg -qi "$SECRET_PATTERN" "$EVIDENCE_FILE" >/dev/null; then
  die "Potential secret found after cleanup; evidence will not be published."
fi

if [[ "$POST_PR_COMMENT" == "1" ]]; then
  PR_HEAD="$(gh pr view "$PR_NUMBER" --repo BerriAI/litellm --json headRefOid --jq .headRefOid)"
  [[ "$PR_HEAD" == "$GIT_SHA" ]] || die "PR #$PR_NUMBER advanced during the live proof; evidence was not published."
  gh pr comment "$PR_NUMBER" --repo BerriAI/litellm --body-file "$EVIDENCE_FILE"
  gh pr comment "$PR_NUMBER" --repo BerriAI/litellm \
    --body "@greptileai please re-review commit $GIT_SHA and update the confidence score."
fi

RUN_SUCCEEDED=1
echo "PostgreSQL and Redis Workload Identity proofs passed."
echo "Sanitized evidence: $EVIDENCE_FILE"
