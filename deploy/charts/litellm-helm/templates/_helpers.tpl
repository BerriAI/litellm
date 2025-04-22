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
Get redis service name
*/}}
{{- define "litellm.redis.serviceName" -}}
{{- if and (eq .Values.redis.architecture "standalone") .Values.redis.sentinel.enabled -}}
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
