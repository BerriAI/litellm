{{/*
Expand the name of the chart.
*/}}
{{- define "litellm.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "litellm.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "litellm.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "litellm.labels" -}}
helm.sh/chart: {{ include "litellm.chart" . }}
{{ include "litellm.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "litellm.selectorLabels" -}}
app.kubernetes.io/name: {{ include "litellm.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Enterprise billable-request metering. The client certificate identifies the
deployment to LiteLLM's collector, so it is mounted read-only from an existing
Secret rather than passed through the environment.
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
Create the name of the service account to use
*/}}
{{- define "litellm.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "litellm.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Create the service account name used by migration jobs.
When Helm hooks are enabled, pre-install/pre-upgrade hooks run before normal resources.
If this chart is creating the ServiceAccount, it is not yet available for the hook job,
so fall back to "default" (or an explicit override) to avoid a cyclic dependency.
*/}}
{{- define "litellm.migrationServiceAccountName" -}}
{{- if and .Values.migrationJob.hooks.helm.enabled .Values.serviceAccount.create }}
{{- default "default" .Values.migrationJob.serviceAccountName }}
{{- else }}
{{- include "litellm.serviceAccountName" . }}
{{- end }}
{{- end }}

{{/*
Get redis service name.
The bundled Redis subchart only serves sentinel in "replication" architecture
(it rejects standalone + sentinel outright), and in that mode the sentinel
Service is named "<release>-redis", not "<release>-redis-master".
*/}}
{{- define "litellm.redis.serviceName" -}}
{{- if .Values.redis.sentinel.enabled -}}
{{- printf "%s-%s" .Release.Name (default "redis" .Values.redis.nameOverride | trunc 63 | trimSuffix "-") -}}
{{- else -}}
{{- printf "%s-%s-master" .Release.Name (default "redis" .Values.redis.nameOverride | trunc 63 | trimSuffix "-") -}}
{{- end -}}
{{- end -}}

{{/*
Get redis service port
*/}}
{{- define "litellm.redis.port" -}}
{{- if .Values.redis.sentinel.enabled -}}
{{ .Values.redis.sentinel.service.ports.sentinel }}
{{- else -}}
{{ .Values.redis.master.service.ports.redis }}
{{- end -}}
{{- end -}}

{{/*
Reject an unpinned image tag for the bundled PostgreSQL.
A floating tag lets a chart upgrade start a newer PostgreSQL major against the
existing PersistentVolumeClaim. The server then refuses to start on a data
directory written by another major version, and the only way back is a dump
taken before the change, which by that point no longer exists.
*/}}
{{- define "litellm.validateBundledPostgresImageTag" -}}
{{- $tag := .Values.postgresql.image.tag | default "" | toString -}}
{{- $digest := .Values.postgresql.image.digest | default "" | toString -}}
{{- if and (eq $digest "") (or (eq $tag "") (eq $tag "latest")) -}}
{{- fail (printf "postgresql.image.tag must be pinned to an explicit version when db.deployStandalone is true (got %q). An unpinned tag can start a different PostgreSQL major against the existing data directory, which makes the database unreadable and is not recoverable in place. Crossing a major version requires a dump and restore." $tag) -}}
{{- end -}}
{{- end -}}

{{/*
Environment shared by the proxy container and the opt-in collector sidecar:
database, pgbouncer, master key, redis, user envVars. Both containers must see
the same DATABASE_URL and REDIS_* so the sidecar reaches the pod's pgbouncer
and the same spend transaction buffer.
*/}}
{{- define "litellm.proxyEnv" -}}
- name: HOST
  value: "{{ .Values.listen | default "0.0.0.0" }}"
- name: PORT
  value: {{ .Values.service.port | quote}}
{{- if .Values.db.deployStandalone }}
- name: DATABASE_USERNAME
  valueFrom:
    secretKeyRef:
      name: {{ include "litellm.fullname" . }}-dbcredentials
      key: username
- name: DATABASE_PASSWORD
  valueFrom:
    secretKeyRef:
      name: {{ include "litellm.fullname" . }}-dbcredentials
      key: password
- name: DATABASE_HOST
  value: {{ .Release.Name }}-postgresql
- name: DATABASE_NAME
  value: litellm
{{- else if .Values.db.useExisting }}
- name: DATABASE_USERNAME
  valueFrom:
    secretKeyRef:
      name: {{ .Values.db.secret.name }}
      key: {{ .Values.db.secret.usernameKey }}
- name: DATABASE_PASSWORD
  valueFrom:
    secretKeyRef:
      name: {{ .Values.db.secret.name }}
      key: {{ .Values.db.secret.passwordKey }}
- name: DATABASE_HOST
  {{- if .Values.db.secret.endpointKey }}
  valueFrom:
    secretKeyRef:
      name: {{ .Values.db.secret.name }}
      key: {{ .Values.db.secret.endpointKey }}
  {{- else }}
  value: {{ .Values.db.endpoint }}
  {{- end }}
- name: DATABASE_NAME
  value: {{ .Values.db.database }}
- name: DATABASE_URL
  value: {{ .Values.db.url | quote }}
{{- end }}
{{- if and .Values.db.useExisting .Values.db.readReplicaUrl .Values.db.secret.readReplicaEndpointKey (not .Values.db.secret.readReplicaUrlKey) }}
- name: DATABASE_READER_HOST
  valueFrom:
    secretKeyRef:
      name: {{ .Values.db.secret.name }}
      key: {{ .Values.db.secret.readReplicaEndpointKey }}
{{- end }}
{{- if and .Values.db.useExisting .Values.db.secret.readReplicaUrlKey }}
- name: DATABASE_URL_READ_REPLICA
  valueFrom:
    secretKeyRef:
      name: {{ .Values.db.secret.name }}
      key: {{ .Values.db.secret.readReplicaUrlKey }}
{{- else if .Values.db.readReplicaUrl }}
- name: DATABASE_URL_READ_REPLICA
  value: {{ .Values.db.readReplicaUrl | quote }}
{{- end }}
{{- if .Values.db.connectionPool.enabled }}
- name: LITELLM_PGBOUNCER_ENABLED
  value: "true"
- name: LITELLM_PGBOUNCER_MAX_DB_CONNECTIONS
  value: {{ .Values.db.connectionPool.maxDbConnections | quote }}
- name: LITELLM_PGBOUNCER_MAX_CLIENT_CONN
  value: {{ .Values.db.connectionPool.maxClientConn | quote }}
{{- end }}
- name: PROXY_MASTER_KEY
  valueFrom:
    secretKeyRef:
      name: {{ .Values.masterkeySecretName | default (printf "%s-masterkey" (include "litellm.fullname" .)) }}
      key: {{ .Values.masterkeySecretKey | default "masterkey" }}
{{- if .Values.redis.enabled }}
- name: REDIS_HOST
  value: {{ include "litellm.redis.serviceName" . }}
- name: REDIS_PORT
  value: {{ include "litellm.redis.port" . | quote }}
- name: REDIS_PASSWORD
  valueFrom:
    secretKeyRef:
      name: {{ include "redis.secretName" .Subcharts.redis }}
      key: {{include "redis.secretPasswordKey" .Subcharts.redis }}
{{- end }}
{{- /*
  Inject LITELLM_LOG only when envVars does not already define it.
*/}}
{{- if and .Values.logLevel (not (hasKey (default dict .Values.envVars) "LITELLM_LOG")) }}
- name: LITELLM_LOG
  value: {{ .Values.logLevel | quote }}
{{- end }}
{{- if .Values.envVars }}
{{- range $key, $val := .Values.envVars }}
- name: {{ $key }}
  value: {{ $val | quote }}
{{- end }}
{{- end }}
{{- with .Values.extraEnvVars }}
{{ toYaml . }}
{{- end }}
{{- if .Values.migrationJob.enabled }}
# Schema updates are owned by the dedicated migrations Job; skip
# the proxy's startup `prisma db push` so N replicas don't race
# one DB on every rollout. Placed last (after envVars and
# extraEnvVars) so this override can't be silently shadowed by a
# user-supplied DISABLE_SCHEMA_UPDATE under last-wins duplicate-env
# semantics — same pattern the migrations Job uses.
- name: DISABLE_SCHEMA_UPDATE
  value: "true"
{{- end }}
{{- end -}}

{{/*
Proxy-only metering and metrics env. The collector sidecar serves no HTTP
traffic, so it gets neither.
*/}}
{{- define "litellm.proxyMetricsEnv" -}}
{{- if .Values.billingMetrics.enabled }}
{{ include "litellm.billingMetricsEnv" . }}
{{- end }}
{{- if .Values.metricsServer.enabled }}
{{- if eq (int .Values.metricsServer.port) (int .Values.service.port) }}
{{- fail "metricsServer.port must differ from service.port" }}
{{- end }}
- name: PROMETHEUS_METRICS_PORT
  value: {{ .Values.metricsServer.port | quote }}
{{- end }}
{{- end -}}

{{/*
Directory of the collector's unix socket, shared between the two containers
through an emptyDir. Empty when the sidecar is off or uses 127.0.0.1 TCP.
*/}}
{{- define "litellm.collector.socketDir" -}}
{{- if and .Values.collector.enabled (hasPrefix "unix://" .Values.collector.address) -}}
{{- dir (trimPrefix "unix://" .Values.collector.address) -}}
{{- end -}}
{{- end -}}

{{- define "litellm.collectorEnv" -}}
- name: LITELLM_COLLECTOR_ENABLED
  value: "true"
- name: LITELLM_COLLECTOR_ADDRESS
  value: {{ .Values.collector.address | quote }}
- name: LITELLM_COLLECTOR_BUFFER_SIZE
  value: {{ .Values.collector.bufferSize | quote }}
- name: LITELLM_COLLECTOR_ON_UNAVAILABLE
  value: {{ .Values.collector.onUnavailable | quote }}
- name: LITELLM_COLLECTOR_DRAIN_TIMEOUT_SECONDS
  value: {{ .Values.collector.drainTimeoutSeconds | quote }}
{{- end -}}
