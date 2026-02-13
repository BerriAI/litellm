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

:::warning
**Multiple instances:** If running multiple LiteLLM instances (e.g., Kubernetes pods), remember each instance multiplies your total connections. Example: 3 instances × 4 workers × 10 connections = 120 total connections.
:::

  # OPTIONAL Best Practices
  disable_error_logs: True # turn off writing LLM Exceptions to DB
  allow_requests_on_db_unavailable: True # Only USE when running LiteLLM on your VPC. Allow requests to still be processed even if the DB is unavailable. We recommend doing this if you're running LiteLLM on VPC that cannot be accessed from the public internet.

litellm_settings:
  request_timeout: 600    # raise Timeout error if call takes longer than 600 seconds. Default value is 6000seconds if not set
  set_verbose: False      # Switch off Debug Logging, ensure your logs do not have any debugging on
  json_logs: true         # Get debug logs in json format
```

Set slack webhook url in your env
```shell
export SLACK_WEBHOOK_URL="example-slack-webhook-url"
```

Turn off FASTAPI's default info logs
```bash
export LITELLM_LOG="ERROR"
```

:::info

Need Help or want dedicated support ? Talk to a founder [here]: (https://calendly.com/d/4mp-gd3-k5k/litellm-1-1-onboarding-chat)

:::


## 2. Recommended Machine Specifications

For optimal performance in production, we recommend the following minimum machine specifications:

| Resource | Recommended Value |
|----------|------------------|
| CPU      | 4 vCPU           |
| Memory   | 8 GB RAM         |

These specifications provide:
- Sufficient compute power for handling concurrent requests
- Adequate memory for request processing and caching


## 3. On Kubernetes — Match Uvicorn Workers to CPU Count [Suggested CMD]

Use this Docker `CMD`. It automatically matches Uvicorn workers to the pod’s CPU count, ensuring each worker uses one core efficiently for better throughput and stable latency.

```shell
CMD ["--port", "4000", "--config", "./proxy_server_config.yaml", "--num_workers", "$(nproc)"]
```

> **Optional:** If you observe gradual memory growth under sustained load, consider recycling workers after a fixed number of requests to mitigate leaks.
> You can configure this either via CLI or environment variable:

```shell
# CLI
CMD ["--port", "4000", "--config", "./proxy_server_config.yaml", "--num_workers", "$(nproc)", "--max_requests_before_restart", "10000"]

# or ENV (for deployment manifests / containers)
export MAX_REQUESTS_BEFORE_RESTART=10000
```

> **Tip:** When using `--max_requests_before_restart`, the `--run_gunicorn` flag is more stable and mature as it uses Gunicorn's battle-tested worker recycling mechanism instead of Uvicorn's implementation.

```shell
# Use Gunicorn for more stable worker recycling
CMD ["--port", "4000", "--config", "./proxy_server_config.yaml", "--num_workers", "$(nproc)", "--run_gunicorn", "--max_requests_before_restart", "10000"]
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

## 10. Use a Separate Health Check App
:::info
The Separate Health Check App only runs when running via the the LiteLLM Docker Image and using Docker and setting the SEPARATE_HEALTH_APP env var to "1"
:::

Using a separate health check app ensures that your liveness and readiness probes remain responsive even when the main application is under heavy load. 

**Why is this important?**

- If your health endpoints share the same process as your main app, high traffic or resource exhaustion can cause health checks to hang or fail.
- When Kubernetes liveness probes hang or time out, it may incorrectly assume your pod is unhealthy and restart it—even if the main app is just busy, not dead.
- By running health endpoints on a separate lightweight FastAPI app (with its own port), you guarantee that health checks remain fast and reliable, preventing unnecessary pod restarts during traffic spikes or heavy workloads.
- The way it works is, if either of the health or main proxy app dies due to whatever reason, it will kill the pod and which would be marked as unhealthy prompting the orchestrator to restart the pod
- Since the proxy and health app are running in the same pod, if the pod dies the health check probe fails, it signifies that the pod is unhealthy and needs to restart/have action taken upon.

**How to enable:**

Set the following environment variable(s):
```bash
SEPARATE_HEALTH_APP="1" # Default "0" 
SEPARATE_HEALTH_PORT="8001" # Default "4001", Works only if `SEPARATE_HEALTH_APP` is "1"
SUPERVISORD_STOPWAITSECS="3600" # Optional: Upper bound timeout in seconds for graceful shutdown. Default: 3600 (1 hour). Only used when SEPARATE_HEALTH_APP=1.
```

**Graceful Shutdown:**

Previously, `stopwaitsecs` was not set, defaulting to 10 seconds and causing in-flight requests to fail. `SUPERVISORD_STOPWAITSECS` (default: 3600) provides an upper bound for graceful shutdown, allowing uvicorn to wait for all in-flight requests to complete.

<video controls width="100%" style={{ borderRadius: '8px', marginBottom: '1em' }}>
  <source src="https://cdn.loom.com/sessions/thumbnails/b08be303331246b88fdc053940d03281-1718990992822.mp4" type="video/mp4" />
  Your browser does not support the video tag.
</video>

Or [watch on Loom](https://www.loom.com/share/b08be303331246b88fdc053940d03281?sid=a145ec66-d55f-41f7-aade-a9f41fbe752d).


### High Level Architecture

<Image alt="Separate Health App Architecture" img={require('../../img/separate_health_app_architecture.png')} style={{ borderRadius: '8px', marginBottom: '1em', maxWidth: '100%' }} />


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