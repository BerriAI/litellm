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
  database_connection_pool_limit: 10 # limit the number of database connections to = MAX Number of DB Connections/Number of instances of litellm proxy (Around 10-20 is good number)

  # OPTIONAL Best Practices
  disable_spend_logs: True # turn off writing each transaction to the db. We recommend doing this is you don't need to see Usage on the LiteLLM UI and are tracking metrics via Prometheus
  allow_requests_on_db_unavailable: True # Only USE when running LiteLLM on your VPC. Allow requests to still be processed even if the DB is unavailable. We recommend doing this if you're running LiteLLM on VPC that cannot be accessed from the public internet.

litellm_settings:
  request_timeout: 600    # raise Timeout error if call takes longer than 600 seconds. Default value is 6000seconds if not set
  set_verbose: False      # Switch off Debug Logging, ensure your logs do not have any debugging on
  json_logs: true         # Get debug logs in json format
```

Set slack webhook url in your env
```shell
export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/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"
```

Turn off FASTAPI's default info logs
```bash
export LITELLM_LOG="ERROR"
```

:::info

Need Help or want dedicated support ? Talk to a founder [here]: (https://calendly.com/d/4mp-gd3-k5k/litellm-1-1-onboarding-chat)

:::


## 2. On Kubernetes - Use 1 Uvicorn worker [Suggested CMD]

Use this Docker `CMD`. This will start the proxy with 1 Uvicorn Async Worker

(Ensure that you're not setting `run_gunicorn` or `num_workers` in the CMD). 
```shell
CMD ["--port", "4000", "--config", "./proxy_server_config.yaml"]
```


## 3. Use Redis 'port','host', 'password'. NOT 'redis_url'

If you decide to use Redis, DO NOT use 'redis_url'. We recommend usig redis port, host, and password params. 

`redis_url`is 80 RPS slower

This is still something we're investigating. Keep track of it [here](https://github.com/BerriAI/litellm/issues/3188)

Recommended to do this for prod: 

```yaml
router_settings:
  routing_strategy: usage-based-routing-v2 
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

## 4. Disable 'load_dotenv'

Set `export LITELLM_MODE="PRODUCTION"`

This disables the load_dotenv() functionality, which will automatically load your environment credentials from the local `.env`. 

## 5. If running LiteLLM on VPC, gracefully handle DB unavailability

This will allow LiteLLM to continue to process requests even if the DB is unavailable. This is better handling for DB unavailability.

**WARNING: Only do this if you're running LiteLLM on VPC, that cannot be accessed from the public internet.**

```yaml
general_settings:
  allow_requests_on_db_unavailable: True
```

## 6. Disable spend_logs if you're not using the LiteLLM UI

By default LiteLLM will write every request to the `LiteLLM_SpendLogs` table. This is used for viewing Usage on the LiteLLM UI. 

If you're not viewing Usage on the LiteLLM UI (most users use Prometheus when this is disabled), you can disable spend_logs by setting `disable_spend_logs` to `True`.

```yaml
general_settings:
  disable_spend_logs: True
```

## 7. Use Helm PreSync Hook for Database Migrations [BETA]

To ensure only one service manages database migrations, use our [Helm PreSync hook for Database Migrations](https://github.com/BerriAI/litellm/blob/main/deploy/charts/litellm-helm/templates/migrations-job.yaml). This ensures migrations are handled during `helm upgrade` or `helm install`, while LiteLLM pods explicitly disable migrations.


1. **Helm PreSync Hook**:
   - The Helm PreSync hook is configured in the chart to run database migrations during deployments.
   - The hook always sets `DISABLE_SCHEMA_UPDATE=false`, ensuring migrations are executed reliably.
  
  Reference Settings to set on ArgoCD for `values.yaml`

  ```yaml
  db:
    useExisting: true # use existing Postgres DB
    url: postgresql://ishaanjaffer0324:3rnwpOBau6hT@ep-withered-mud-a5dkdpke.us-east-2.aws.neon.tech/test-argo-cd?sslmode=require # url of existing Postgres DB
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

We recommned - https://1password.com/password-generator/ password generator to get a random hash for litellm salt key.

```bash
export LITELLM_SALT_KEY="sk-1234"
```

[**See Code**](https://github.com/BerriAI/litellm/blob/036a6821d588bd36d170713dcf5a72791a694178/litellm/proxy/common_utils/encrypt_decrypt_utils.py#L15)

## Extras
### Expected Performance in Production

1 LiteLLM Uvicorn Worker on Kubernetes

| Description | Value |
|--------------|-------|
| Avg latency | `50ms` |
| Median latency | `51ms` |
| `/chat/completions` Requests/second | `100` |
| `/chat/completions` Requests/minute | `6000` |
| `/chat/completions` Requests/hour | `360K` |


### Verifying Debugging logs are off

You should only see the following level of details in logs on the proxy server
```shell
# INFO:     192.168.2.205:11774 - "POST /chat/completions HTTP/1.1" 200 OK
# INFO:     192.168.2.205:34717 - "POST /chat/completions HTTP/1.1" 200 OK
# INFO:     192.168.2.205:29734 - "POST /chat/completions HTTP/1.1" 200 OK
```


### Machine Specifications to Deploy LiteLLM

| Service | Spec | CPUs | Memory | Architecture | Version|
| --- | --- | --- | --- | --- | --- | 
| Server | `t2.small`. | `1vCPUs` | `8GB` | `x86` |
| Redis Cache | - | - | - | - | 7.0+ Redis Engine|


### Reference Kubernetes Deployment YAML

Reference Kubernetes `deployment.yaml` that was load tested by us

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: litellm-deployment
spec:
  replicas: 3
  selector:
    matchLabels:
      app: litellm
  template:
    metadata:
      labels:
        app: litellm
    spec:
      containers:
        - name: litellm-container
          image: ghcr.io/berriai/litellm:main-latest
          imagePullPolicy: Always
          env:
            - name: AZURE_API_KEY
              value: "d6******"
            - name: AZURE_API_BASE
              value: "https://ope******"
            - name: LITELLM_MASTER_KEY
              value: "sk-1234"
            - name: DATABASE_URL
              value: "po**********"
          args:
            - "--config"
            - "/app/proxy_config.yaml"  # Update the path to mount the config file
          volumeMounts:                 # Define volume mount for proxy_config.yaml
            - name: config-volume
              mountPath: /app
              readOnly: true
          livenessProbe:
            httpGet:
              path: /health/liveliness
              port: 4000
            initialDelaySeconds: 120
            periodSeconds: 15
            successThreshold: 1
            failureThreshold: 3
            timeoutSeconds: 10
          readinessProbe:
            httpGet:
              path: /health/readiness
              port: 4000
            initialDelaySeconds: 120
            periodSeconds: 15
            successThreshold: 1
            failureThreshold: 3
            timeoutSeconds: 10
      volumes:  # Define volume to mount proxy_config.yaml
        - name: config-volume
          configMap:
            name: litellm-config  

```


Reference Kubernetes `service.yaml` that was load tested by us
```yaml
apiVersion: v1
kind: Service
metadata:
  name: litellm-service
spec:
  selector:
    app: litellm
  ports:
    - protocol: TCP
      port: 4000
      targetPort: 4000
  type: LoadBalancer
```
