import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import Image from '@theme/IdealImage';

# ⚡ Best Practices for Production

## 1. Use this config.yaml
Use this config.yaml in production (with your own LLMs)

```yaml
model_list:
  - model_name: fake-openai-endpoint
    litellm_params:
      model: openai/fake
      api_key: fake-key
      api_base: https://exampleopenaiendpoint-production.up.railway.app/

general_settings:
  master_key: sk-1234      # enter your own master key, ensure it starts with 'sk-'
  alerting: ["slack"]      # Setup slack alerting - get alerts on LLM exceptions, Budget Alerts, Slow LLM Responses
  proxy_batch_write_at: 60 # Batch write spend updates every 60s
  database_connection_pool_limit: 10 # connection pool limit per worker process. Total connections = limit × workers × instances. Calculate: MAX_DB_CONNECTIONS / (instances × workers). Default: 10.

  # OPTIONAL Best Practices
  disable_error_logs: True # turn off writing LLM Exceptions to DB
  allow_requests_on_db_unavailable: True # Only USE when running LiteLLM on your VPC. Allow requests to still be processed even if the DB is unavailable. We recommend doing this if you're running LiteLLM on VPC that cannot be accessed from the public internet.

litellm_settings:
  request_timeout: 600    # raise Timeout error if call takes longer than 600 seconds. Default value is 6000seconds if not set
  set_verbose: False      # Switch off Debug Logging, ensure your logs do not have any debugging on
  json_logs: true         # Get debug logs in json format
```
:::warning Multiple instances

If running multiple LiteLLM instances (e.g., Kubernetes pods), remember each instance multiplies your total connections. Example: 3 instances × 4 workers × 10 connections = 120 total connections.

:::

Set slack webhook url in your env
```shell
export SLACK_WEBHOOK_URL="example-slack-webhook-url"
```

Turn off FASTAPI's default info logs
```bash
export LITELLM_LOG="ERROR"
```

:::info

Need Help or want dedicated support ? Talk to a founder [here](https://enterprise.litellm.ai/demo).

:::


## 2. Recommended Machine Specifications

For optimal performance in production, we recommend the following resource configuration.

**1. Memory `requests` and `limits`**

```yaml
resources:
  requests:
    cpu: "1" # should be 1*num_workers
    memory: "4Gi" # should be 4*num_workers
  limits:
    cpu: "1"
    memory: "4Gi"
```

**2. HPA thresholds**

```yaml
targetCPUUtilizationPercentage: 60
targetMemoryUtilizationPercentage: 80
```


## 3. Choose your server: Uvicorn vs. Gunicorn

LiteLLM Proxy runs on [Uvicorn](https://uvicorn.dev/) by default. Passing `--run_gunicorn` instead starts [Gunicorn](https://gunicorn.org/) as a process manager that supervises [Uvicorn worker processes](https://uvicorn.dev/deployment/#gunicorn) (`uvicorn.workers.UvicornWorker`). In both cases your application code still runs on Uvicorn; the difference is which process manages and recycles the workers.

| | Uvicorn (default) | Gunicorn (`--run_gunicorn`) |
|---|---|---|
| **When to use** | Recommended for almost all deployments, especially Kubernetes with one worker per pod. | Choose when you run **multiple workers in a single container** and want a mature process manager to supervise and recycle them. |
| **Worker recycling** | Uvicorn's [`limit_max_requests`](https://uvicorn.dev/settings/#resource-limits). | Gunicorn's [`max_requests`](https://gunicorn.org/reference/settings/#max_requests), the battle-tested mechanism Gunicorn has shipped for years. |
| **Process supervision** | Uvicorn's built-in multiprocess manager. | Gunicorn's [arbiter](https://gunicorn.org/design/#arbiter), which restarts workers one at a time as they exit. |

:::tip Recommendation

On Kubernetes, run **one Uvicorn worker per pod** and scale **horizontally** (more pods) rather than vertically (more workers per pod). One process per pod keeps latency predictable under load, lets the Horizontal Pod Autoscaler use the [thresholds above](#2-recommended-machine-specifications) accurately, and makes rolling restarts hitless because Kubernetes drains one pod at a time. Reach for Gunicorn only when you must pack multiple workers into one container.

:::

### 3a. Recommended: one Uvicorn worker per pod

This is the default server, so you only need to set `--num_workers 1` (the default is already `1`):

```shell
CMD ["--port", "4000", "--config", "./proxy_server_config.yaml", "--num_workers", "1"]
```

### 3b. Recycle workers with `--max_requests_before_restart`

If you observe gradual memory growth under sustained load, recycle each worker after a fixed number of requests to bound memory usage. `--max_requests_before_restart` maps to Uvicorn's [`limit_max_requests`](https://uvicorn.dev/settings/#resource-limits) (default server) and to Gunicorn's [`max_requests`](https://gunicorn.org/reference/settings/#max_requests) under `--run_gunicorn`. Configure it via CLI flag or environment variable:

```shell
# CLI
CMD ["--port", "4000", "--config", "./proxy_server_config.yaml", "--num_workers", "1", "--max_requests_before_restart", "10000"]

# or ENV (for deployment manifests / containers)
export MAX_REQUESTS_BEFORE_RESTART=10000
```

:::tip

When you run **multiple workers in one container** and rely on `--max_requests_before_restart`, prefer `--run_gunicorn`. Gunicorn's [`max_requests`](https://gunicorn.org/reference/settings/#max_requests) recycling is more mature than Uvicorn's, and its [arbiter](https://gunicorn.org/design/#arbiter) restarts workers one at a time so the pod keeps serving traffic while a worker is replaced.

:::

```shell
# Multiple workers in one container, with Gunicorn-managed recycling
CMD ["--port", "4000", "--config", "./proxy_server_config.yaml", "--num_workers", "4", "--run_gunicorn", "--max_requests_before_restart", "10000"]
```

### 3c. Keep restarts hitless

A restart is "hitless" when in-flight requests finish before the process exits, so no client sees a dropped connection. Two cases matter in production:

**Worker recycling (from `--max_requests_before_restart`).** Both servers stop accepting new connections on the recycled worker and let outstanding requests drain before it exits, then a replacement worker starts. Gunicorn additionally guarantees in-flight requests up to its [`graceful_timeout`](https://gunicorn.org/reference/settings/#graceful_timeout) (30s by default) on [`SIGTERM`](https://gunicorn.org/signals/). With one worker per pod, recycling briefly reduces that pod's capacity, which is why we recommend scaling horizontally so the load balancer can route around it.

**Rolling deploys and pod restarts (Kubernetes).** Make restarts hitless at the orchestration layer rather than relying on the server alone:

- Use a [`RollingUpdate`](https://kubernetes.io/docs/concepts/workloads/controllers/deployment/#rolling-update-deployment) strategy (the Deployment default) so new pods become Ready before old pods are terminated.
- Keep a [readiness probe](https://kubernetes.io/docs/concepts/configuration/liveness-readiness-startup-probes/) on `/health/readiness` so Kubernetes only sends traffic to pods that can serve it, and stops routing to a pod as soon as termination begins.
- Set [`terminationGracePeriodSeconds`](https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/#pod-termination) to comfortably exceed your longest expected request (LiteLLM's request timeout defaults to 600s; see [Section 1](#1-use-this-configyaml)). On termination Kubernetes sends `SIGTERM`, and both Uvicorn and Gunicorn shut down [gracefully](https://uvicorn.dev/deployment/) by draining in-flight requests before exiting.
- Optionally add a small [`preStop` hook](https://kubernetes.io/docs/concepts/containers/container-lifecycle-hooks/#container-hooks) (for example `sleep 5`) to give the load balancer time to deregister the pod before the server begins shutting down, eliminating the brief window where traffic can still arrive at a terminating pod.

```yaml title="Kubernetes Deployment snippet for hitless rolling restarts"
spec:
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 0   # never drop below desired replica count
      maxSurge: 1         # add one new pod at a time
  template:
    spec:
      terminationGracePeriodSeconds: 620   # > your longest request (request_timeout: 600)
      containers:
        - name: litellm
          readinessProbe:
            httpGet:
              path: /health/readiness
              port: 4000
          lifecycle:
            preStop:
              exec:
                command: ["sh", "-c", "sleep 5"]
```


## 4. Use Redis 'port','host', 'password'. NOT 'redis_url'

If you decide to use Redis, DO NOT use 'redis_url'. We recommend using redis port, host, and password params. 

`redis_url`is 80 RPS slower

This is still something we're investigating. Keep track of it [here](https://github.com/BerriAI/litellm/issues/3188)

### Redis Version Requirement

| Component | Minimum Version |
|-----------|-----------------|
| Redis     | 7.0+            |

Recommended to do this for prod:

```yaml
router_settings:
  routing_strategy: simple-shuffle # (default) - recommended for best performance
  # redis_url: "os.environ/REDIS_URL"
  redis_host: os.environ/REDIS_HOST
  redis_port: os.environ/REDIS_PORT
  redis_password: os.environ/REDIS_PASSWORD

litellm_settings:
  cache: True
  cache_params:
    type: redis
    host: os.environ/REDIS_HOST
    port: os.environ/REDIS_PORT
    password: os.environ/REDIS_PASSWORD
```

> **WARNING**
**Usage-based routing is not recommended for production due to performance impacts.** Use `simple-shuffle` (default) for optimal performance in high-traffic scenarios.

## 5. Disable 'load_dotenv'

Set `export LITELLM_MODE="PRODUCTION"`

This disables the load_dotenv() functionality, which will automatically load your environment credentials from the local `.env`. 

## 6. If running LiteLLM on VPC, gracefully handle DB unavailability

When running LiteLLM on a VPC (and inaccessible from the public internet), you can enable graceful degradation so that request processing continues even if the database is temporarily unavailable.


**WARNING: Only do this if you're running LiteLLM on VPC, that cannot be accessed from the public internet.**

#### Configuration

```yaml showLineNumbers title="litellm config.yaml"
general_settings:
  allow_requests_on_db_unavailable: True
```

#### Expected Behavior

When `allow_requests_on_db_unavailable` is set to `true`, LiteLLM will handle errors as follows:

| Type of Error | Expected Behavior | Details |
|---------------|-------------------|----------------|
| Prisma Errors | ✅ Request will be allowed | Covers issues like DB connection resets or rejections from the DB via Prisma, the ORM used by LiteLLM. |
| Httpx Errors | ✅ Request will be allowed | Occurs when the database is unreachable, allowing the request to proceed despite the DB outage. |
| Pod Startup Behavior | ✅ Pods start regardless | LiteLLM Pods will start even if the database is down or unreachable, ensuring higher uptime guarantees for deployments. |
| Health/Readiness Check | ✅ Always returns 200 OK | The /health/readiness endpoint returns a 200 OK status to ensure that pods remain operational even when the database is unavailable.
| LiteLLM Budget Errors or Model Errors | ❌ Request will be blocked | Triggered when the DB is reachable but the authentication token is invalid, lacks access, or exceeds budget limits. |


[More information about what the Database is used for here](db_info)

## 7. Use Helm PreSync Hook for Database Migrations [BETA]

To ensure only one service manages database migrations, use our [Helm PreSync hook for Database Migrations](https://github.com/BerriAI/litellm/blob/main/deploy/charts/litellm-helm/templates/migrations-job.yaml). This ensures migrations are handled during `helm upgrade` or `helm install`, while LiteLLM pods explicitly disable migrations.


1. **Helm PreSync Hook**:
   - The Helm PreSync hook is configured in the chart to run database migrations during deployments.
   - The hook always sets `DISABLE_SCHEMA_UPDATE=false`, ensuring migrations are executed reliably.
  
  Reference Settings to set on ArgoCD for `values.yaml`

  ```yaml
  db:
    useExisting: true # use existing Postgres DB
    url: postgresql://ishaanjaffer0324:... # url of existing Postgres DB
  ```

2. **LiteLLM Pods**:
   - Set `DISABLE_SCHEMA_UPDATE=true` in LiteLLM pod configurations to prevent them from running migrations.
   
   Example configuration for LiteLLM pod:
   ```yaml
   env:
     - name: DISABLE_SCHEMA_UPDATE
       value: "true"
   ```


## 8. Set LiteLLM Salt Key 

If you plan on using the DB, set a salt key for encrypting/decrypting variables in the DB. 

Do not change this after adding a model. It is used to encrypt / decrypt your LLM API Key credentials

We recommend - https://1password.com/password-generator/ password generator to get a random hash for litellm salt key.

```bash
export LITELLM_SALT_KEY="sk-1234"
```

[**See Code**](https://github.com/BerriAI/litellm/blob/036a6821d588bd36d170713dcf5a72791a694178/litellm/proxy/common_utils/encrypt_decrypt_utils.py#L15)


## 9. Use `prisma migrate deploy`

Use this to handle db migrations across LiteLLM versions in production

<Tabs>
<TabItem value="env" label="ENV">

```bash
USE_PRISMA_MIGRATE="True"
```

</TabItem>

<TabItem value="cli" label="CLI">

```bash
litellm
```

</TabItem>
</Tabs>

Benefits:

The migrate deploy command:

- **Does not** issue a warning if an already applied migration is missing from migration history
- **Does not** detect drift (production database schema differs from migration history end state - for example, due to a hotfix)
- **Does not** reset the database or generate artifacts (such as Prisma Client)
- **Does not** rely on a shadow database


### How does LiteLLM handle DB migrations in production?

1. A new migration file is written to our `litellm-proxy-extras` package. [See all](https://github.com/BerriAI/litellm/tree/main/litellm-proxy-extras/litellm_proxy_extras/migrations)

2. The core litellm pip package is bumped to point to the new `litellm-proxy-extras` package. This ensures, older versions of LiteLLM will continue to use the old migrations. [See code](https://github.com/BerriAI/litellm/blob/52b35cd8093b9ad833987b24f494586a1e923209/pyproject.toml#L58)

3. When you upgrade to a new version of LiteLLM, the migration file is applied to the database. [See code](https://github.com/BerriAI/litellm/blob/52b35cd8093b9ad833987b24f494586a1e923209/litellm-proxy-extras/litellm_proxy_extras/utils.py#L42)


### Read-only File System

Running LiteLLM with `readOnlyRootFilesystem: true` is a Kubernetes security best practice that prevents container processes from writing to the root filesystem. LiteLLM fully supports this configuration.

#### Quick Fix for Permission Errors

If you see a `Permission denied` error, it means the LiteLLM pod is running with a read-only file system. LiteLLM needs writable directories for:
- **Database migrations**: Set `LITELLM_MIGRATION_DIR="/path/to/writable/directory"`
- **Admin UI**: Set `LITELLM_UI_PATH="/path/to/writable/directory"`
- **UI assets/logos**: Set `LITELLM_ASSETS_PATH="/path/to/writable/directory"`

#### Complete Read-Only Filesystem Setup (Kubernetes)

For production deployments with enhanced security, use this configuration:

**Option 1: Using EmptyDir Volumes with InitContainer (Recommended)**

This approach copies the pre-built UI from the Docker image to writable emptyDir volumes at pod startup.

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: litellm-proxy
spec:
  template:
    spec:
      initContainers:
        - name: setup-ui
          image: ghcr.io/berriai/litellm:main-stable
          command:
            - sh
            - -c
            - |
              cp -r /var/lib/litellm/ui/* /app/var/litellm/ui/ && \
              cp -r /var/lib/litellm/assets/* /app/var/litellm/assets/
          volumeMounts:
            - name: ui-volume
              mountPath: /app/var/litellm/ui
            - name: assets-volume
              mountPath: /app/var/litellm/assets

      containers:
        - name: litellm
          image: ghcr.io/berriai/litellm:main-stable
          env:
            - name: LITELLM_NON_ROOT
              value: "true"
            - name: LITELLM_UI_PATH
              value: "/app/var/litellm/ui"
            - name: LITELLM_ASSETS_PATH
              value: "/app/var/litellm/assets"
            - name: LITELLM_MIGRATION_DIR
              value: "/app/migrations"
            - name: PRISMA_BINARY_CACHE_DIR
              value: "/app/cache/prisma-python/binaries"
            - name: XDG_CACHE_HOME
              value: "/app/cache"
          securityContext:
            readOnlyRootFilesystem: true
            runAsNonRoot: true
            runAsUser: 101
            capabilities:
              drop:
                - ALL
          volumeMounts:
            - name: config
              mountPath: /app/config.yaml
              subPath: config.yaml
              readOnly: true
            - name: ui-volume
              mountPath: /app/var/litellm/ui
            - name: assets-volume
              mountPath: /app/var/litellm/assets
            - name: cache
              mountPath: /app/cache
            - name: migrations
              mountPath: /app/migrations

      volumes:
        - name: config
          configMap:
            name: litellm-config
        - name: ui-volume
          emptyDir:
            sizeLimit: 100Mi
        - name: assets-volume
          emptyDir:
            sizeLimit: 10Mi
        - name: cache
          emptyDir:
            sizeLimit: 500Mi
        - name: migrations
          emptyDir:
            sizeLimit: 64Mi
```

**Option 2: Without UI (API-only deployment)**

If you don't need the admin UI, you can run with minimal configuration:

```yaml
env:
  - name: LITELLM_NON_ROOT
    value: "true"
  - name: LITELLM_MIGRATION_DIR
    value: "/app/migrations"
securityContext:
  readOnlyRootFilesystem: true
```

The proxy will log a warning about the UI but API endpoints will work normally.

#### Environment Variables for Read-Only Filesystems

| Variable | Purpose | Default |
|----------|---------|---------|
| `LITELLM_UI_PATH` | Admin UI directory | `/var/lib/litellm/ui` (Docker) |
| `LITELLM_ASSETS_PATH` | UI assets/logos | `/var/lib/litellm/assets` (Docker) |
| `LITELLM_MIGRATION_DIR` | Database migrations | Package directory |
| `PRISMA_BINARY_CACHE_DIR` | Prisma binary cache | System default |
| `XDG_CACHE_HOME` | General cache directory | System default |

#### Important Notes

1. **Migrations**: Always set `LITELLM_MIGRATION_DIR` to a writable emptyDir path
2. **Prisma Cache**: Set `PRISMA_BINARY_CACHE_DIR` and `XDG_CACHE_HOME` to writable paths
3. **Server Root Path**: If using a custom `server_root_path`, you must pre-process UI files in your Dockerfile as the proxy cannot modify files at runtime with read-only filesystem
4. **Automatic Detection**: The UI is automatically detected as pre-restructured if it contains a `.litellm_ui_ready` marker file (created by the official Docker images)

## Extras
### Expected Performance in Production

See benchmarks [here](../benchmarks#performance-metrics)

### Verifying Debugging logs are off

You should only see the following level of details in logs on the proxy server
```shell
# INFO:     192.168.2.205:11774 - "POST /chat/completions HTTP/1.1" 200 OK
# INFO:     192.168.2.205:34717 - "POST /chat/completions HTTP/1.1" 200 OK
# INFO:     192.168.2.205:29734 - "POST /chat/completions HTTP/1.1" 200 OK
```
