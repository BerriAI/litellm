{{/*
The pod that runs the LLM data plane. In componentized mode it is the gateway
Deployment (`args: [gateway, ...]`), in monolith mode the proxy Deployment
(`args: [proxy, ...]`) that also serves the management API and the Admin UI.
Both are configured by .Values.gateway; the name, component label, selector
and args come from the litellm.workload.* helpers.
*/}}
{{- define "litellm.workload.deployment" -}}
{{- $component := include "litellm.workload.componentName" . -}}
{{- $fullname := include "litellm.workload.fullname" . -}}
{{- if and .Values.gateway.hpa.enabled .Values.gateway.keda.enabled }}
{{- fail "gateway.hpa.enabled and gateway.keda.enabled are mutually exclusive: two autoscalers on one Deployment fight over the replica count, so set gateway.hpa.enabled: false when using KEDA" }}
{{- end }}
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ $fullname }}
  labels:
    {{- include "litellm.commonLabels" . | nindent 4 }}
    app.kubernetes.io/component: {{ $component }}
    {{- with .Values.gateway.deploymentLabels }}
    {{- toYaml . | nindent 4 }}
    {{- end }}
  {{- with .Values.gateway.deploymentAnnotations }}
  annotations:
    {{- toYaml . | nindent 4 }}
  {{- end }}
spec:
  {{- if and (not .Values.gateway.hpa.enabled) (not .Values.gateway.keda.enabled) (not (kindIs "invalid" .Values.gateway.replicaCount)) }}
  replicas: {{ .Values.gateway.replicaCount }}
  {{- end }}
  {{- $minReady := .Values.gateway.minReadySeconds }}
  {{- if not (or (kindIs "invalid" $minReady) (eq (printf "%v" $minReady) "")) }}
  minReadySeconds: {{ $minReady }}
  {{- end }}
  {{- with .Values.gateway.strategy }}
  strategy:
    {{- toYaml . | nindent 4 }}
  {{- end }}
  selector:
    matchLabels:
      {{- include "litellm.workload.selectorLabels" . | nindent 6 }}
  template:
    metadata:
      annotations:
        {{- if .Values.gateway.config.create }}
        checksum/config: {{ include (print $.Template.BasePath "/gateway/configmap.yaml") . | sha256sum }}
        {{- end }}
        {{- with .Values.gateway.podAnnotations }}
        {{- toYaml . | nindent 8 }}
        {{- end }}
      labels:
        {{- include "litellm.workload.selectorLabels" . | nindent 8 }}
        {{- with .Values.gateway.podLabels }}
        {{- include "litellm.podLabels" (dict "podLabels" . "componentName" "gateway") | nindent 8 }}
        {{- end }}
    spec:
      serviceAccountName: {{ include "litellm.gateway.serviceAccountName" . }}
      automountServiceAccountToken: {{ .Values.serviceAccounts.gateway.automount }}
      {{- with .Values.gateway.podSecurityContext }}
      securityContext:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      {{- with .Values.imagePullSecrets }}
      imagePullSecrets:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      {{- with .Values.gateway.extraInitContainers }}
      initContainers:
        {{- tpl (toYaml .) $ | nindent 8 }}
      {{- end }}
      containers:
        - name: {{ $component }}
          image: {{ include "litellm.image" . | quote }}
          imagePullPolicy: {{ .Values.image.pullPolicy }}
          {{- with .Values.gateway.securityContext }}
          securityContext:
            {{- toYaml . | nindent 12 }}
          {{- end }}
          args:
            {{- if .Values.monolith.enabled }}
            {{- include "litellm.proxy.args" . | nindent 12 }}
            {{- else }}
            - gateway
            - --host
            - 0.0.0.0
            - --port
            - "4000"
            {{- end }}
          ports:
            - name: http
              containerPort: 4000
              protocol: TCP
          env:
            {{- include "litellm.lensConnectionEnv" . | nindent 12 }}
            {{- include "litellm.masterKeyEnv" $ | nindent 12 }}
            {{- if .Values.monolith.enabled }}
            - name: LENS_WORKER_IMAGE
              value: {{ include "litellm.lensWorker.image" . | quote }}
            {{- include "litellm.liteadminEnv" (dict "root" $ "component" .Values.gateway) | nindent 12 }}
            {{- end }}
            {{- include "litellm.serverEnv" (dict "root" $ "component" .Values.gateway) | nindent 12 }}
            {{- if .Values.gateway.config.create }}
            - name: CONFIG_FILE_PATH
              value: /app/config/config.yaml
            {{- end }}
            {{- if .Values.gateway.numWorkers }}
            - name: NUM_WORKERS
              value: {{ .Values.gateway.numWorkers | quote }}
            {{- end }}
            {{- if .Values.database.connectionPool.enabled }}
            {{- include "litellm.connectionPoolEnv" $ | nindent 12 }}
            {{- end }}
            {{- if .Values.billingMetrics.enabled }}
            {{- include "litellm.billingMetricsEnv" . | nindent 12 }}
            {{- end }}
            {{- if .Values.gateway.metricsServer.enabled }}
            {{- if eq (int .Values.gateway.metricsServer.port) 4000 }}
            {{- fail (printf "gateway.metricsServer.port must differ from the %s port 4000" $component) }}
            {{- end }}
            - name: PROMETHEUS_MULTIPROC_DIR
              value: {{ include "litellm.gateway.prometheusMultiprocDir" . }}
            {{- end }}
            {{- if .Values.gateway.collector.enabled }}
            {{- include "litellm.gateway.collectorEnv" . | nindent 12 }}
            {{- end }}
          {{- include "litellm.envFrom" .Values.gateway | nindent 10 }}
          {{- if or .Values.gateway.config.create .Values.gateway.volumeMounts .Values.billingMetrics.enabled .Values.gateway.metricsServer.enabled (include "litellm.gateway.collectorSocketDir" .) }}
          volumeMounts:
            {{- if .Values.gateway.config.create }}
            - name: gateway-config
              mountPath: /app/config/config.yaml
              subPath: config.yaml
            {{- end }}
            {{- if .Values.gateway.metricsServer.enabled }}
            - name: prometheus-multiproc
              mountPath: {{ include "litellm.gateway.prometheusMultiprocDir" . }}
            {{- end }}
            {{- if include "litellm.gateway.collectorSocketDir" . }}
            - name: collector-socket
              mountPath: {{ include "litellm.gateway.collectorSocketDir" . }}
            {{- end }}
            {{- if .Values.billingMetrics.enabled }}
            {{- include "litellm.billingMetricsVolumeMounts" . | nindent 12 }}
            {{- end }}
            {{- with .Values.gateway.volumeMounts }}
            {{- toYaml . | nindent 12 }}
            {{- end }}
          {{- end }}
          {{- with .Values.gateway.livenessProbe }}
          livenessProbe:
            {{- toYaml . | nindent 12 }}
          {{- end }}
          {{- with .Values.gateway.readinessProbe }}
          readinessProbe:
            {{- toYaml . | nindent 12 }}
          {{- end }}
          {{- with .Values.gateway.startupProbe }}
          startupProbe:
            {{- toYaml . | nindent 12 }}
          {{- end }}
          {{- with .Values.gateway.lifecycle }}
          lifecycle:
            {{- toYaml . | nindent 12 }}
          {{- end }}
          resources:
            {{- toYaml .Values.gateway.resources | nindent 12 }}
        {{- if .Values.gateway.metricsServer.enabled }}
        - name: metrics
          image: {{ include "litellm.image" . | quote }}
          imagePullPolicy: {{ .Values.image.pullPolicy }}
          {{- with .Values.gateway.securityContext }}
          securityContext:
            {{- toYaml . | nindent 12 }}
          {{- end }}
          args:
            - metrics
            - --port
            - {{ .Values.gateway.metricsServer.port | quote }}
          env:
            - name: PROMETHEUS_MULTIPROC_DIR
              value: {{ include "litellm.gateway.prometheusMultiprocDir" . }}
          ports:
            - name: metrics
              containerPort: {{ .Values.gateway.metricsServer.port }}
              protocol: TCP
          volumeMounts:
            - name: prometheus-multiproc
              mountPath: {{ include "litellm.gateway.prometheusMultiprocDir" . }}
          readinessProbe:
            tcpSocket: { port: metrics }
            periodSeconds: 10
          livenessProbe:
            tcpSocket: { port: metrics }
            periodSeconds: 15
            failureThreshold: 6
          resources:
            {{- toYaml .Values.gateway.metricsServer.resources | nindent 12 }}
        {{- end }}
        {{- if .Values.gateway.collector.enabled }}
        - name: collector
          image: {{ include "litellm.image" . | quote }}
          imagePullPolicy: {{ .Values.image.pullPolicy }}
          {{- with .Values.gateway.securityContext }}
          securityContext:
            {{- toYaml . | nindent 12 }}
          {{- end }}
          args:
            - collector
          env:
            {{- include "litellm.masterKeyEnv" $ | nindent 12 }}
            {{- include "litellm.serverEnv" (dict "root" $ "component" .Values.gateway) | nindent 12 }}
            {{- if .Values.gateway.config.create }}
            - name: CONFIG_FILE_PATH
              value: /app/config/config.yaml
            {{- end }}
            {{- if .Values.database.connectionPool.enabled }}
            {{- include "litellm.connectionPoolEnv" $ | nindent 12 }}
            {{- end }}
            {{- include "litellm.gateway.collectorEnv" . | nindent 12 }}
            - name: LITELLM_JOB_ROLE
              value: collector
          {{- include "litellm.envFrom" .Values.gateway | nindent 10 }}
          {{- if or .Values.gateway.config.create .Values.gateway.volumeMounts (include "litellm.gateway.collectorSocketDir" .) }}
          volumeMounts:
            {{- if .Values.gateway.config.create }}
            - name: gateway-config
              mountPath: /app/config/config.yaml
              subPath: config.yaml
            {{- end }}
            {{- if include "litellm.gateway.collectorSocketDir" . }}
            - name: collector-socket
              mountPath: {{ include "litellm.gateway.collectorSocketDir" . }}
            {{- end }}
            {{- with .Values.gateway.volumeMounts }}
            {{- toYaml . | nindent 12 }}
            {{- end }}
          {{- end }}
          resources:
            {{- toYaml .Values.gateway.collector.resources | nindent 12 }}
        {{- end }}
      {{- with .Values.gateway.extraContainers }}
        {{- tpl (toYaml .) $ | nindent 8 }}
      {{- end }}
      {{- if or .Values.gateway.config.create .Values.gateway.volumes .Values.billingMetrics.enabled .Values.gateway.metricsServer.enabled (include "litellm.gateway.collectorSocketDir" .) }}
      volumes:
        {{- if .Values.gateway.config.create }}
        - name: gateway-config
          configMap:
            name: {{ include "litellm.workload.fullname" . }}-config
        {{- end }}
        {{- if .Values.gateway.metricsServer.enabled }}
        - name: prometheus-multiproc
          emptyDir: {}
        {{- end }}
        {{- if include "litellm.gateway.collectorSocketDir" . }}
        - name: collector-socket
          emptyDir:
            sizeLimit: 1Mi
        {{- end }}
        {{- if .Values.billingMetrics.enabled }}
        {{- include "litellm.billingMetricsVolumes" . | nindent 8 }}
        {{- end }}
        {{- with .Values.gateway.volumes }}
        {{- toYaml . | nindent 8 }}
        {{- end }}
      {{- end }}
      {{- with .Values.gateway.nodeSelector }}
      nodeSelector:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      {{- with .Values.gateway.affinity }}
      affinity:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      {{- with .Values.gateway.tolerations }}
      tolerations:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      {{- with .Values.gateway.topologySpreadConstraints }}
      topologySpreadConstraints:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      {{- $gracePeriod := .Values.gateway.terminationGracePeriodSeconds }}
      {{- if not (or (kindIs "invalid" $gracePeriod) (eq (printf "%v" $gracePeriod) "")) }}
      terminationGracePeriodSeconds: {{ $gracePeriod }}
      {{- end }}
{{- end -}}

{{- define "litellm.workload.service" -}}
{{- $component := include "litellm.workload.componentName" . -}}
apiVersion: v1
kind: Service
metadata:
  name: {{ include "litellm.workload.fullname" . }}
  labels:
    {{- include "litellm.commonLabels" . | nindent 4 }}
    app.kubernetes.io/component: {{ $component }}
  {{- with .Values.gateway.service.annotations }}
  annotations:
    {{- toYaml . | nindent 4 }}
  {{- end }}
spec:
  type: {{ .Values.gateway.service.type }}
  {{- with include "litellm.service.extras" .Values.gateway.service }}
  {{- . | nindent 2 }}
  {{- end }}
  ports:
    - port: {{ .Values.gateway.service.port }}
      targetPort: http
      protocol: TCP
      name: http
  selector:
    {{- include "litellm.workload.selectorLabels" . | nindent 4 }}
{{- end -}}

{{- define "litellm.workload.hpa" -}}
{{- $component := include "litellm.workload.componentName" . -}}
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: {{ include "litellm.workload.fullname" . }}
  labels:
    {{- include "litellm.commonLabels" . | nindent 4 }}
    app.kubernetes.io/component: {{ $component }}
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: {{ include "litellm.workload.fullname" . }}
  minReplicas: {{ .Values.gateway.hpa.minReplicas }}
  maxReplicas: {{ .Values.gateway.hpa.maxReplicas }}
  metrics:
    {{- if .Values.gateway.hpa.targetCPUUtilizationPercentage }}
    {{- if and .Values.gateway.collector.enabled .Values.gateway.collector.scaleOnGatewayContainerCpu }}
    - type: ContainerResource
      containerResource:
        name: cpu
        container: {{ $component }}
        target:
          type: Utilization
          averageUtilization: {{ .Values.gateway.hpa.targetCPUUtilizationPercentage }}
    {{- else }}
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: {{ .Values.gateway.hpa.targetCPUUtilizationPercentage }}
    {{- end }}
    {{- end }}
    {{- if .Values.gateway.hpa.targetMemoryUtilizationPercentage }}
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: {{ .Values.gateway.hpa.targetMemoryUtilizationPercentage }}
    {{- end }}
    {{- with .Values.gateway.hpa.targetRequestsPerSecond }}
    - type: Pods
      pods:
        metric:
          name: litellm_requests_per_second
        target:
          type: AverageValue
          averageValue: {{ toJson . | trimAll "\"" | quote }}
    {{- end }}
    {{- with .Values.gateway.hpa.targetTokensPerSecond }}
    - type: Pods
      pods:
        metric:
          name: litellm_tokens_per_second
        target:
          type: AverageValue
          averageValue: {{ toJson . | trimAll "\"" | quote }}
    {{- end }}
  {{- with .Values.gateway.hpa.behavior }}
  behavior:
    {{- toYaml . | nindent 4 }}
  {{- end }}
{{- end -}}

{{/*
KEDA ScaledObject targeting the workload Deployment. The optional Prometheus
triggers query the same per pod counters the HPA workload metrics use,
scoped to this release's metrics Service through the `job` label the
ServiceMonitor gives every scrape.
*/}}
{{- define "litellm.workload.keda" -}}
{{- $component := include "litellm.workload.componentName" . -}}
{{- $keda := .Values.gateway.keda -}}
{{- $prom := $keda.prometheus -}}
{{- $wantsPrometheus := or $prom.targetRequestsPerSecond $prom.targetTokensPerSecond -}}
{{- if and $wantsPrometheus (not $prom.serverAddress) }}
{{- fail "gateway.keda.prometheus.serverAddress is required when gateway.keda.prometheus.targetRequestsPerSecond or targetTokensPerSecond is set" }}
{{- end }}
{{- if and $wantsPrometheus (not .Values.gateway.metricsServer.enabled) }}
{{- fail "gateway.metricsServer.enabled is required when gateway.keda.prometheus.targetRequestsPerSecond or targetTokensPerSecond is set" }}
{{- end }}
{{- $selector := printf "namespace=%q,job=%q" .Release.Namespace (printf "%s-metrics" (include "litellm.workload.fullname" .)) -}}
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: {{ include "litellm.workload.fullname" . }}
  labels:
    {{- include "litellm.commonLabels" . | nindent 4 }}
    app.kubernetes.io/component: {{ $component }}
  {{- with (dig "scaledObject" "annotations" (dict) $keda) }}
  annotations:
    {{- toYaml . | nindent 4 }}
  {{- end }}
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: {{ include "litellm.workload.fullname" . }}
  minReplicaCount: {{ $keda.minReplicaCount }}
  maxReplicaCount: {{ $keda.maxReplicaCount }}
  pollingInterval: {{ $keda.pollingInterval }}
  cooldownPeriod: {{ $keda.cooldownPeriod }}
  {{- with $keda.advanced }}
  advanced:
    {{- toYaml . | nindent 4 }}
  {{- end }}
  {{- with $keda.fallback }}
  {{- $fallback := deepCopy . }}
  {{- if not (hasKey $fallback "failureThreshold") }}
  {{- $_ := set $fallback "failureThreshold" 3 }}
  {{- end }}
  {{- if not (hasKey $fallback "replicas") }}
  {{- $_ := set $fallback "replicas" $keda.maxReplicaCount }}
  {{- end }}
  fallback:
    {{- toYaml $fallback | nindent 4 }}
  {{- end }}
  triggers:
    {{- with $keda.triggers }}
    {{- toYaml . | nindent 4 }}
    {{- end }}
    {{- with $prom.targetRequestsPerSecond }}
    - type: prometheus
      metricType: AverageValue
      metadata:
        serverAddress: {{ $prom.serverAddress | quote }}
        query: {{ printf "sum(rate(litellm_proxy_total_requests_metric_total{%s}[1m]))" $selector | quote }}
        threshold: {{ toJson . | trimAll "\"" | quote }}
    {{- end }}
    {{- with $prom.targetTokensPerSecond }}
    - type: prometheus
      metricType: AverageValue
      metadata:
        serverAddress: {{ $prom.serverAddress | quote }}
        query: {{ printf "sum(rate(litellm_total_tokens_metric_total{%s}[1m]))" $selector | quote }}
        threshold: {{ toJson . | trimAll "\"" | quote }}
    {{- end }}
{{- end -}}

{{- define "litellm.workload.serviceMetrics" -}}
{{- $component := include "litellm.workload.componentName" . -}}
apiVersion: v1
kind: Service
metadata:
  name: {{ include "litellm.workload.fullname" . }}-metrics
  labels:
    {{- include "litellm.commonLabels" . | nindent 4 }}
    app.kubernetes.io/component: {{ $component }}
spec:
  type: ClusterIP
  ports:
    - port: {{ .Values.gateway.metricsServer.port }}
      targetPort: metrics
      protocol: TCP
      name: metrics
  selector:
    {{- include "litellm.workload.selectorLabels" . | nindent 4 }}
{{- end -}}

{{- define "litellm.workload.serviceMonitor" -}}
{{- $component := include "litellm.workload.componentName" . -}}
{{- if not .Values.gateway.metricsServer.enabled }}
{{- fail "gateway.serviceMonitor.enabled requires gateway.metricsServer.enabled: the http port serves /metrics/ behind virtual-key auth, so an unauthenticated scrape gets 401" }}
{{- end }}
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: {{ include "litellm.workload.fullname" . }}
  labels:
    {{- include "litellm.commonLabels" . | nindent 4 }}
    app.kubernetes.io/component: {{ $component }}
    {{- with .Values.gateway.serviceMonitor.labels }}
    {{- toYaml . | nindent 4 }}
    {{- end }}
  {{- with .Values.gateway.serviceMonitor.annotations }}
  annotations:
    {{- toYaml . | nindent 4 }}
  {{- end }}
spec:
  selector:
    matchLabels:
      {{- include "litellm.workload.selectorLabels" . | nindent 6 }}
  namespaceSelector:
    matchNames:
      {{- with .Values.gateway.serviceMonitor.namespaceSelector.matchNames }}
      {{- toYaml . | nindent 6 }}
      {{- else }}
      - {{ .Release.Namespace | quote }}
      {{- end }}
  endpoints:
    - port: metrics
      path: /metrics/
      interval: {{ .Values.gateway.serviceMonitor.interval }}
      scrapeTimeout: {{ .Values.gateway.serviceMonitor.scrapeTimeout }}
      scheme: http
      {{- with .Values.gateway.serviceMonitor.relabelings }}
      relabelings:
        {{- toYaml . | nindent 8 }}
      {{- end }}
{{- end -}}
