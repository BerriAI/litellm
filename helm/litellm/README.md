# litellm Helm chart

Deploys [LiteLLM](https://github.com/BerriAI/litellm) from one image, `ghcr.io/berriai/litellm` (mirrored at `docker.litellm.ai/berriai/litellm`), in one of two layouts:

- componentized (default): the `gateway` (LLM data plane, port 4000), the `backend` (management API, port 4001) and the `ui` (static dashboard, port 3000) each get their own Deployment, Service, HPA and PDB so they scale independently
- monolith: `monolith.enabled: true` renders one Deployment and Service running the full proxy (`args: [proxy, ...]`), which serves the gateway routes, the management routes and the Admin UI from a single process

Both layouts run the same image. The image entrypoint dispatches on the first container argument (`proxy`, `gateway`, `backend`, `ui`, `migrations`, `metrics`, `collector`), so the chart only ever sets `args` and never `command`

## Requirements

Kubernetes 1.25+ and Helm 3.8+. The chart has no subcharts: bring your own PostgreSQL (`database.writer.*`) and, optionally, Redis (`redis.*`). The proxy also needs a master key, either an existing Secret named by `masterKey.secretName` or one the chart generates with `masterKey.generate: true` and `masterKey.secretName: ""`

## Install

```bash
kubectl create secret generic litellm-master-key-secret --from-literal=master-key=sk-change-me
helm install litellm helm/litellm \
  --set database.writer.host=postgres.example.com \
  --set database.writer.dbname=litellm \
  --set database.writer.passwordSecret.name=litellm-db-secret
```

The image tag defaults to the chart's `appVersion`. Override it with `image.tag`, or pin the bytes with `image.digest`, which renders as `repository:tag@digest`

## Monolith quickstart

```bash
helm install litellm helm/litellm \
  --set monolith.enabled=true \
  --set masterKey.secretName="" \
  --set masterKey.generate=true \
  --set database.writer.host=postgres.example.com \
  --set database.writer.dbname=litellm \
  --set database.writer.passwordSecret.name=litellm-db-secret
kubectl port-forward svc/litellm-litellm 4000:4000
```

With `monolith.enabled: true`:

- one Deployment and one Service named `<release>-litellm` render, running `args: [proxy, --port, 4000, --config, /app/config/config.yaml]` plus `monolith.extraArgs`
- the gateway, backend and ui Deployments, Services, HPAs, PDBs and ServiceMonitors are not rendered, whatever `gateway.enabled`, `backend.enabled` and `ui.enabled` say
- every `gateway.*` value configures the monolith pod: `config`, `numWorkers`, `resources`, probes, `securityContext`, `hpa`, `keda`, `pdb`, `metricsServer`, `collector`, `volumes`, `extraEnv`, scheduling. The monolith runs as `serviceAccounts.gateway`
- every `backend.*` and `ui.*` value is ignored
- the Ingress sends every path, built in or from `ingress.extraPaths`, to the monolith Service
- the migrations Job renders exactly as in componentized mode

## Testing

```bash
helm lint helm/litellm
helm unittest -f 'tests/*.yaml' helm/litellm
helm test <release> --logs
```

## Migrating from the litellm-helm chart

The `litellm-helm` chart (`oci://ghcr.io/berriai/litellm-helm`) is retired; its published packages stay available for a grace period. Its flat values described one monolith Deployment, so the equivalent install here is `monolith.enabled: true` with the values moved under `gateway.*`. The per-component `gateway.image`, `backend.image`, `ui.image` and `migrations.image` blocks of earlier `helm/litellm` versions are gone too: the chart is a major bump to `1.0.0` and every container uses the top-level `image`

| litellm-helm value | litellm value |
|---|---|
| (implicit single Deployment) | `monolith.enabled: true` |
| `image.repository` / `image.tag` / `image.pullPolicy` | `image.repository` / `image.tag` / `image.pullPolicy` (`image.digest` is new) |
| `replicaCount` | `gateway.replicaCount` |
| `args` | `monolith.extraArgs` (appended after the chart's proxy arguments) |
| `command` | removed: the image entrypoint dispatcher must stay in place |
| `proxy_config` | `gateway.config.proxy_config` |
| `proxyConfigMap.create: false` + `proxyConfigMap.name` | `gateway.config.create: false` and mount your ConfigMap with `gateway.volumes` / `gateway.volumeMounts`, or pass `--config` in `monolith.extraArgs` |
| `masterkeySecretName` / `masterkeySecretKey` | `masterKey.secretName` / `masterKey.secretKey` |
| `masterkeySecretName: ""` (auto generated Secret) | `masterKey.generate: true` with `masterKey.secretName: ""` (creates a new key; copy the old key first, see below, to keep existing credentials valid) |
| `db.useExisting`, `db.endpoint`, `db.database`, `db.secret.*` | `database.writer.host`, `database.writer.port`, `database.writer.dbname`, `database.writer.passwordSecret.*` |
| `db.readReplicaUrl` / `db.secret.readReplica*` | `database.reader.*` |
| `db.connectionPool.*` | `database.connectionPool.*` |
| `db.deployStandalone: true` / `postgresql.*` | removed: the chart ships no PostgreSQL, point `database.writer.*` at your own |
| `redis.enabled: true` (bundled) | removed: the chart ships no Redis, point `redis.host` at your own |
| external Redis via `envVars` | `redis.host`, `redis.port`, `redis.passwordSecret.*`, `redis.cluster` |
| `envVars` / `extraEnvVars` | `gateway.extraEnv` (list of `name` / `value` or `valueFrom` entries) |
| `environmentSecrets` | `gateway.envSecrets` |
| `environmentConfigMaps` | `gateway.envConfigMaps` |
| `logLevel` | `gateway.logLevel` |
| `resources` | `gateway.resources` |
| `livenessProbe` / `readinessProbe` / `startupProbe` | `gateway.livenessProbe` / `gateway.readinessProbe` / `gateway.startupProbe` |
| `securityContext` / `podSecurityContext` | `gateway.securityContext` / `gateway.podSecurityContext` |
| `service.*` | `gateway.service.*` |
| `ingress.*` | `ingress.*` (routes to the monolith Service in monolith mode) |
| `autoscaling.*` | `gateway.hpa.*` |
| `keda.*` | `gateway.keda.*` (`keda.prometheus.requestsPerSecond` / `tokensPerSecond` are `gateway.keda.prometheus.targetRequestsPerSecond` / `targetTokensPerSecond`) |
| `pdb.*` | `gateway.pdb.*` |
| `metricsServer.*` | `gateway.metricsServer.*` |
| `serviceMonitor.*` | `gateway.serviceMonitor.*` |
| `collector.*` | `gateway.collector.*` |
| `billingMetrics.*` | `billingMetrics.*` |
| `migrationJob.*` | `migrationJob.*` |
| `volumes` / `volumeMounts` | `gateway.volumes` / `gateway.volumeMounts` |
| `extraContainers` / `extraInitContainers` | `gateway.extraContainers` / `gateway.extraInitContainers` |
| `lifecycle` | `gateway.lifecycle` |
| `strategy` | `gateway.strategy` |
| `deploymentAnnotations` / `deploymentLabels` / `deploymentMinReadySeconds` | `gateway.deploymentAnnotations` / `gateway.deploymentLabels` / `gateway.minReadySeconds` |
| `podAnnotations` / `podLabels` | `gateway.podAnnotations` / `gateway.podLabels` |
| `nodeSelector` / `tolerations` / `affinity` / `topologySpreadConstraints` | `gateway.nodeSelector` / `gateway.tolerations` / `gateway.affinity` / `gateway.topologySpreadConstraints` |
| `terminationGracePeriodSeconds` | `gateway.terminationGracePeriodSeconds` |
| `serviceAccount.*` | `serviceAccounts.gateway.*` |
| `extraResources` | `extraResources` |
| `nameOverride: "litellm"` | `nameOverride: ""` (the chart name is already `litellm`) |

A minimal migration:

```yaml
monolith:
  enabled: true
masterKey:
  secretName: litellm-master-key-secret
database:
  writer:
    host: postgres.example.com
    dbname: litellm
    passwordSecret:
      name: litellm-db-secret
      usernameKey: username
      passwordKey: password
gateway:
  replicaCount: 2
  config:
    proxy_config:
      model_list:
        - model_name: gpt-4o
          litellm_params:
            model: openai/gpt-4o
            api_key: os.environ/OPENAI_API_KEY
  envSecrets:
    - litellm-provider-keys
```

The monolith Service keeps the `<release>-litellm` name the old chart produced through its `nameOverride: "litellm"` default, so an existing Ingress or port-forward keeps working after `helm uninstall` of the old release and `helm install` of this one. The generated Secret in this chart has `helm.sh/resource-policy: keep` and is reused across upgrades of this chart. The retired chart stored its generated key under a different Secret and data key, and `helm uninstall` deletes that Secret, so copy it into a Secret you own before uninstalling the old release to keep existing credentials valid, for example:

```bash
kubectl create secret generic litellm-master-key-secret \
  --from-literal=master-key="$(kubectl get secret <release>-litellm-masterkey -o jsonpath='{.data.masterkey}' | base64 -d)"
```

Then set `masterKey.secretName: litellm-master-key-secret`, as in the minimal example above. Releases that set `masterkeySecretName` already point `masterKey.secretName` and `masterKey.secretKey` at that Secret
