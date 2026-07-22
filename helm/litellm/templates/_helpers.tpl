{{/*
Common naming + label helpers shared by gateway, backend, and ui templates.
*/}}

{{- define "litellm.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "litellm.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- $name := default .Chart.Name .Values.nameOverride -}}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}

{{- define "litellm.gateway.fullname" -}}
{{- printf "%s-gateway" (include "litellm.fullname" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "litellm.backend.fullname" -}}
{{- printf "%s-backend" (include "litellm.fullname" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "litellm.ui.fullname" -}}
{{- printf "%s-ui" (include "litellm.fullname" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "litellm.commonLabels" -}}
app.kubernetes.io/name: {{ include "litellm.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" }}
{{- if .Values.extraLabels }}
{{ toYaml .Values.extraLabels }}
{{- end }}
{{- end -}}

{{/*
Enterprise billable-request metering. Wired into gateway and backend, not the
migrations job. The gateway serves nearly all billable traffic, but the backend
keeps the named-server MCP transport (/{mcp_server_name}/mcp), which writes a
SpendLogs row, so metering only the gateway would silently drop that traffic.
The client certificate identifies the deployment to LiteLLM's collector, so it is
mounted read-only from an existing Secret rather than passed through the
environment.
*/}}
{{- define "litellm.billingMetrics.certDir" -}}/etc/litellm/billing-mtls{{- end -}}
{{- define "litellm.billingMetrics.caDir" -}}/etc/litellm/billing-mtls-ca{{- end -}}

{{- define "litellm.billingMetricsEnv" -}}
- name: LITELLM_BILLING_METRICS_ENDPOINT
  value: {{ required "billingMetrics.endpoint is required when billingMetrics.enabled is true" .Values.billingMetrics.endpoint | quote }}
- name: LITELLM_BILLING_METRICS_CLIENT_CERT
  value: {{ printf "%s/tls.crt" (include "litellm.billingMetrics.certDir" .) | quote }}
- name: LITELLM_BILLING_METRICS_CLIENT_KEY
  value: {{ printf "%s/tls.key" (include "litellm.billingMetrics.certDir" .) | quote }}
{{- if .Values.billingMetrics.caSecretName }}
- name: LITELLM_BILLING_METRICS_CA_CERT
  value: {{ printf "%s/ca.crt" (include "litellm.billingMetrics.caDir" .) | quote }}
{{- end }}
{{- with .Values.billingMetrics.exportIntervalMs }}
- name: LITELLM_BILLING_METRICS_EXPORT_INTERVAL_MS
  value: {{ . | quote }}
{{- end }}
{{- end -}}

{{- define "litellm.billingMetricsVolumes" -}}
- name: billing-metrics-mtls
  secret:
    secretName: {{ required "billingMetrics.secretName is required when billingMetrics.enabled is true (an existing Secret with tls.crt and tls.key)" .Values.billingMetrics.secretName }}
{{- if .Values.billingMetrics.caSecretName }}
- name: billing-metrics-mtls-ca
  secret:
    secretName: {{ .Values.billingMetrics.caSecretName }}
{{- end }}
{{- end -}}

{{- define "litellm.billingMetricsVolumeMounts" -}}
- name: billing-metrics-mtls
  mountPath: {{ include "litellm.billingMetrics.certDir" . }}
  readOnly: true
{{- if .Values.billingMetrics.caSecretName }}
- name: billing-metrics-mtls-ca
  mountPath: {{ include "litellm.billingMetrics.caDir" . }}
  readOnly: true
{{- end }}
{{- end -}}

{{/*
Per-component selector labels — used in both Service selectors and Deployment matchLabels.
*/}}
{{- define "litellm.gateway.selectorLabels" -}}
app.kubernetes.io/name: {{ include "litellm.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/component: gateway
{{- end -}}

{{- define "litellm.backend.selectorLabels" -}}
app.kubernetes.io/name: {{ include "litellm.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/component: backend
{{- end -}}

{{- define "litellm.ui.selectorLabels" -}}
app.kubernetes.io/name: {{ include "litellm.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/component: ui
{{- end -}}

{{/*
Per-component ServiceAccount name helpers.

Each component (gateway, backend, ui) has its own SA config under
.Values.serviceAccounts.<component>. When `create` is true and `name` is
empty the chart defaults to "<release>-litellm-<component>". When `create`
is false the chart uses the provided name, or the namespace `default` SA.
*/}}
{{- define "litellm.gateway.serviceAccountName" -}}
{{- if .Values.serviceAccounts.gateway.create -}}
{{ default (include "litellm.gateway.fullname" .) .Values.serviceAccounts.gateway.name }}
{{- else -}}
{{ default "default" .Values.serviceAccounts.gateway.name }}
{{- end -}}
{{- end -}}

{{- define "litellm.backend.serviceAccountName" -}}
{{- if .Values.serviceAccounts.backend.create -}}
{{ default (include "litellm.backend.fullname" .) .Values.serviceAccounts.backend.name }}
{{- else -}}
{{ default "default" .Values.serviceAccounts.backend.name }}
{{- end -}}
{{- end -}}

{{- define "litellm.ui.serviceAccountName" -}}
{{- if .Values.serviceAccounts.ui.create -}}
{{ default (include "litellm.ui.fullname" .) .Values.serviceAccounts.ui.name }}
{{- else -}}
{{ default "default" .Values.serviceAccounts.ui.name }}
{{- end -}}
{{- end -}}

{{/*
Master-key + database + redis env block — shared by gateway, backend, and the
migrations Job.

Invoke with a dict: `(dict "root" $ "component" .Values.gateway)`. `root` is
the chart context (needed for .Values), `component` selects which component's
`extraEnv` / `logLevel` to render.

Sensitive values (master key, DB username + password, Redis password) come
only from referenced Secrets; the chart never accepts inline values for them.

The chart never assembles DATABASE_URL itself. It emits only the discrete
DATABASE_HOST/PORT/USER/NAME/SCHEMA (+ DATABASE_PASSWORD for password auth)
vars; the proxy's entrypoint (DatabaseURLSettings in
litellm/proxy/db/db_url_settings.py) builds the URL from them and
percent-encodes the credentials. Assembling the URL here via Kubernetes
`$(VAR)` substitution would embed the raw secret value, corrupting the URL
whenever the password contains a URL-reserved character (@, /, ?, %, +,
...) — as AWS RDS auto-generated passwords routinely do.

When `database.writer.useIAMAuth: true`, the chart injects
IAM_TOKEN_DB_AUTH=true and omits DATABASE_PASSWORD — the entrypoint mints
the URL from DATABASE_HOST/PORT/USER/NAME plus a short-lived IAM token
instead of a static password.

The read replica is opt-in via `database.reader.host`. The chart emits
DATABASE_HOST_READ_REPLICA / DATABASE_PORT_READ_REPLICA /
DATABASE_NAME_READ_REPLICA (+ DATABASE_SCHEMA_READ_REPLICA) for both auth
modes, plus DATABASE_USER_READ_REPLICA / DATABASE_PASSWORD_READ_REPLICA for
password auth. When `database.reader.useIAMAuth: true` it omits
DATABASE_PASSWORD_READ_REPLICA and the entrypoint mints the reader URL the
same way. Reader IAM only takes effect when the writer also uses IAM auth
(the proxy gates URL minting on IAM_TOKEN_DB_AUTH, which only the writer
sets).
*/}}
{{- define "litellm.serverEnv" -}}
{{- $root := .root -}}
{{- $component := .component -}}
- name: LITELLM_MASTER_KEY
  valueFrom:
    secretKeyRef:
      name: {{ required "masterKey.secretName is required (the chart no longer accepts an inline master key)" $root.Values.masterKey.secretName }}
      key: {{ $root.Values.masterKey.secretKey | default "master-key" }}
{{- if $component.logLevel }}
- name: LITELLM_LOG
  value: {{ $component.logLevel | quote }}
{{- end }}
{{- with $root.Values.database.writer }}
- name: DATABASE_HOST
  value: {{ required "database.writer.host is required" .host | quote }}
- name: DATABASE_PORT
  value: {{ .port | default 5432 | quote }}
- name: DATABASE_USER
  valueFrom:
    secretKeyRef:
      name: {{ required "database.writer.passwordSecret.name is required" .passwordSecret.name }}
      key: {{ .passwordSecret.usernameKey | default "username" }}
- name: DATABASE_NAME
  value: {{ required "database.writer.dbname is required" .dbname | quote }}
{{- if .schema }}
- name: DATABASE_SCHEMA
  value: {{ .schema | quote }}
{{- end }}
{{- if .useIAMAuth }}
- name: IAM_TOKEN_DB_AUTH
  value: "true"
{{- else }}
- name: DATABASE_PASSWORD
  valueFrom:
    secretKeyRef:
      name: {{ .passwordSecret.name }}
      key: {{ .passwordSecret.passwordKey | default "password" }}
{{- end }}
{{- end }}
{{- with $root.Values.database.reader }}
{{- if .host }}
{{- if and .useIAMAuth (not $root.Values.database.writer.useIAMAuth) }}
{{- fail "database.reader.useIAMAuth requires database.writer.useIAMAuth: true (the proxy gates IAM URL minting on IAM_TOKEN_DB_AUTH, which is only set by the writer)" }}
{{- end }}
- name: DATABASE_HOST_READ_REPLICA
  value: {{ .host | quote }}
- name: DATABASE_PORT_READ_REPLICA
  value: {{ .port | default 5432 | quote }}
- name: DATABASE_NAME_READ_REPLICA
  value: {{ required "database.reader.dbname is required when database.reader.host is set" .dbname | quote }}
{{- if .schema }}
- name: DATABASE_SCHEMA_READ_REPLICA
  value: {{ .schema | quote }}
{{- end }}
{{- if .useIAMAuth }}
{{- if .passwordSecret.name }}
- name: DATABASE_USER_READ_REPLICA
  valueFrom:
    secretKeyRef:
      name: {{ .passwordSecret.name }}
      key: {{ .passwordSecret.usernameKey | default "username" }}
{{- end }}
{{- else }}
{{- if not .passwordSecret.name }}
{{- fail "database.reader.passwordSecret.name is required when database.reader.host is set" }}
{{- end }}
- name: DATABASE_USER_READ_REPLICA
  valueFrom:
    secretKeyRef:
      name: {{ .passwordSecret.name }}
      key: {{ .passwordSecret.usernameKey | default "username" }}
- name: DATABASE_PASSWORD_READ_REPLICA
  valueFrom:
    secretKeyRef:
      name: {{ .passwordSecret.name }}
      key: {{ .passwordSecret.passwordKey | default "password" }}
{{- end }}
{{- end }}
{{- end }}
{{/*
The migrations Job (helm.sh/hook: pre-upgrade) is the single owner of
`prisma migrate deploy`. Without this, every gateway/backend pod also runs
Prisma schema-update on startup and contends with the Job — and with each
other — for Prisma's Postgres advisory lock on the writer, which makes the
Job's `migrate deploy` intermittently block until its per-attempt timeout
and retry-exhaust. The Job's entrypoint (migrations/run.py) does not import
proxy_server and never reads DISABLE_SCHEMA_UPDATE, so emitting it here is a
harmless no-op for the Job and authoritative for the app pods.
*/}}
- name: DISABLE_SCHEMA_UPDATE
  value: "true"
{{/* These feed the proxy's coordination Redis (cross-pod rate limits, spend
     tracking, pod lock manager) via its REDIS_* env fallback. An explicit
     `general_settings.coordination_redis` block in proxy_config takes
     precedence over anything emitted here. */}}
{{- if $root.Values.redis.host }}
- name: REDIS_HOST
  value: {{ $root.Values.redis.host | quote }}
- name: REDIS_PORT
  value: {{ $root.Values.redis.port | quote }}
{{- if $root.Values.redis.passwordSecret.name }}
- name: REDIS_PASSWORD
  valueFrom:
    secretKeyRef:
      name: {{ $root.Values.redis.passwordSecret.name }}
      key: {{ $root.Values.redis.passwordSecret.passwordKey | default "password" }}
{{- end }}
{{- if $root.Values.redis.cluster }}
{{/* The proxy falls back to REDIS_CLUSTER_NODES (JSON) to build a cluster-mode
     coordination client when `general_settings.coordination_redis` is absent
     and no plain-Redis response cache is configured. We seed with the single
     configured endpoint; the cluster client discovers the remaining nodes from
     CLUSTER SLOTS at startup. */}}
- name: REDIS_CLUSTER_NODES
  value: {{ printf "[{\"host\":%q,\"port\":%v}]" $root.Values.redis.host (int $root.Values.redis.port) | quote }}
{{- end }}
{{- end }}
{{- with $component.extraEnv }}
{{ toYaml . }}
{{- end }}
{{- end -}}

{{/*
PodDisruptionBudget shared by gateway, backend, and ui.

Invoke with a dict:
  (dict "root" $ "component" .Values.gateway "componentName" "gateway"
        "fullname" (include "litellm.gateway.fullname" .)
        "selectorLabels" (include "litellm.gateway.selectorLabels" .))

Renders nothing unless both the component and its `pdb.enabled` are on.
Only one of minAvailable / maxUnavailable should be set; if both are,
minAvailable wins. If neither is set, falls back to `maxUnavailable: 1` so
an enabled-but-unconfigured PDB still permits node drains.

"Set" means non-nil and non-empty-string, so an explicit 0 (e.g.
`maxUnavailable: 0` to forbid all voluntary disruptions) is honored rather
than silently replaced by the fallback.
*/}}
{{- define "litellm.pdb" -}}
{{- $root := .root -}}
{{- $component := .component -}}
{{- $min := $component.pdb.minAvailable -}}
{{- $max := $component.pdb.maxUnavailable -}}
{{- $minSet := not (or (kindIs "invalid" $min) (eq (printf "%v" $min) "")) -}}
{{- $maxSet := not (or (kindIs "invalid" $max) (eq (printf "%v" $max) "")) -}}
{{- if and $component.enabled $component.pdb $component.pdb.enabled }}
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: {{ .fullname }}
  labels:
    {{- include "litellm.commonLabels" $root | nindent 4 }}
    app.kubernetes.io/component: {{ .componentName }}
spec:
  selector:
    matchLabels:
      {{- .selectorLabels | nindent 6 }}
  {{- if $minSet }}
  minAvailable: {{ $min }}
  {{- else if $maxSet }}
  maxUnavailable: {{ $max }}
  {{- else }}
  maxUnavailable: 1
  {{- end }}
{{- end }}
{{- end -}}

{{/*
Renders `envFrom:` block for a component's `envConfigMaps` / `envSecrets`
lists. Each entry is a resource name; the chart wires the whole ConfigMap /
Secret into the container's env via configMapRef / secretRef.

Invoke with just the component dict, e.g. `.Values.gateway`. Emits nothing
when both lists are empty so the container spec stays clean.
*/}}
{{- define "litellm.envFrom" -}}
{{- $component := . -}}
{{- if or $component.envConfigMaps $component.envSecrets }}
envFrom:
{{- range $component.envConfigMaps }}
  - configMapRef:
      name: {{ . }}
{{- end }}
{{- range $component.envSecrets }}
  - secretRef:
      name: {{ . }}
{{- end }}
{{- end }}
{{- end -}}
