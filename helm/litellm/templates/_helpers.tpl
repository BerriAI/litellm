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
Shared ServiceAccount name used by all three component Deployments. When
`serviceAccount.create` is true and `serviceAccount.name` is empty, default
to the chart fullname. When `create` is false, fall back to the provided
name or the namespace's `default` SA.
*/}}
{{- define "litellm.serviceAccountName" -}}
{{- if .Values.serviceAccount.create -}}
{{ default (include "litellm.fullname" .) .Values.serviceAccount.name }}
{{- else -}}
{{ default "default" .Values.serviceAccount.name }}
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

DATABASE_URL is assembled at pod startup via Kubernetes `$(VAR)` env
substitution, so the unencoded password is never written to the Pod spec.
Kubernetes only substitutes vars declared earlier in the same container's
env list, so DATABASE_USER / DATABASE_PASSWORD must precede DATABASE_URL.

When `database.writer.useIAMAuth: true`, the chart injects
IAM_TOKEN_DB_AUTH=true and omits DATABASE_PASSWORD / DATABASE_URL — the
proxy's entrypoint (litellm/proxy/auth/rds_iam_token.py:190) then mints
the URL from DATABASE_HOST/PORT/USER/NAME plus an IAM token.

When `database.reader.useIAMAuth: true`, the chart emits
DATABASE_HOST_READ_REPLICA / DATABASE_PORT_READ_REPLICA /
DATABASE_NAME_READ_REPLICA (plus DATABASE_USER_READ_REPLICA when a reader
secret is supplied) and omits DATABASE_PASSWORD_READ_REPLICA /
DATABASE_URL_READ_REPLICA — the proxy mints the reader URL the same way.
Reader IAM only takes effect when the writer also uses IAM auth (the
proxy gates URL minting on IAM_TOKEN_DB_AUTH, which only the writer
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
- name: DATABASE_URL
  value: "postgresql://$(DATABASE_USER):$(DATABASE_PASSWORD)@$(DATABASE_HOST):$(DATABASE_PORT)/$(DATABASE_NAME){{ if .schema }}?schema=$(DATABASE_SCHEMA){{ end }}"
{{- end }}
{{- end }}
{{- with $root.Values.database.reader }}
{{- if .host }}
{{- if and .useIAMAuth (not $root.Values.database.writer.useIAMAuth) }}
{{- fail "database.reader.useIAMAuth requires database.writer.useIAMAuth: true (the proxy gates IAM URL minting on IAM_TOKEN_DB_AUTH, which is only set by the writer)" }}
{{- end }}
{{- if .useIAMAuth }}
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
- name: DATABASE_URL_READ_REPLICA
  value: "postgresql://$(DATABASE_USER_READ_REPLICA):$(DATABASE_PASSWORD_READ_REPLICA)@{{ .host }}:{{ .port | default 5432 }}/{{ required "database.reader.dbname is required when database.reader.host is set" .dbname }}{{ if .schema }}?schema={{ .schema }}{{ end }}"
{{- end }}
{{- end }}
{{- end }}
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
{{/* The proxy's Cache() reads REDIS_CLUSTER_NODES as JSON and constructs a
     RedisClusterCache when it's set (litellm/caching/caching.py:169-192).
     We seed with the single configured endpoint — the cluster client
     discovers the remaining nodes from CLUSTER SLOTS at startup. */}}
- name: REDIS_CLUSTER_NODES
  value: {{ printf "[{\"host\":%q,\"port\":%v}]" $root.Values.redis.host (int $root.Values.redis.port) | quote }}
{{- end }}
{{- end }}
{{- with $component.extraEnv }}
{{ toYaml . }}
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
