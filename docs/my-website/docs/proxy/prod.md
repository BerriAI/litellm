import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# ⚡ Best Practices for Production

Expected Performance in Production

1 LiteLLM Uvicorn Worker on Kubernetes

| Description | Value |
|--------------|-------|
| Avg latency | `50ms` |
| Median latency | `51ms` |
| `/chat/completions` Requests/second | `35` |
| `/chat/completions` Requests/minute | `2100` |
| `/chat/completions` Requests/hour | `126K` |


## 1. Switch off Debug Logging

Remove `set_verbose: True` from your config.yaml
```yaml
litellm_settings:
  set_verbose: True
```

You should only see the following level of details in logs on the proxy server
```shell
# INFO:     192.168.2.205:11774 - "POST /chat/completions HTTP/1.1" 200 OK
# INFO:     192.168.2.205:34717 - "POST /chat/completions HTTP/1.1" 200 OK
# INFO:     192.168.2.205:29734 - "POST /chat/completions HTTP/1.1" 200 OK
```

## 2. On Kubernetes - Use 1 Uvicorn worker [Suggested CMD]

Use this Docker `CMD`. This will start the proxy with 1 Uvicorn Async Worker

(Ensure that you're not setting `run_gunicorn` or `num_workers` in the CMD). 
```shell
CMD ["--port", "4000", "--config", "./proxy_server_config.yaml"]
```

## 3. Batch write spend updates every 60s

The default proxy batch write is 10s. This is to make it easy to see spend when debugging locally. 

In production, we recommend using a longer interval period of 60s. This reduces the number of connections used to make DB writes. 

```yaml
general_settings:
  master_key: sk-1234
  proxy_batch_write_at: 60 # 👈 Frequency of batch writing logs to server (in seconds)
```

## 4. use Redis 'port','host', 'password'. NOT 'redis_url'

When connecting to Redis use redis port, host, and password params. Not 'redis_url'. We've seen a 80 RPS difference between these 2 approaches when using the async redis client. 

This is still something we're investigating. Keep track of it [here](https://github.com/BerriAI/litellm/issues/3188)

Recommended to do this for prod: 

```yaml
router_settings:
  routing_strategy: usage-based-routing-v2 
  # redis_url: "os.environ/REDIS_URL"
  redis_host: os.environ/REDIS_HOST
  redis_port: os.environ/REDIS_PORT
  redis_password: os.environ/REDIS_PASSWORD
```

## 5. Switch off resetting budgets

Add this to your config.yaml. (Only spend per Key, User and Team will be tracked - spend per API Call will not be written to the LiteLLM Database)
```yaml
general_settings:
  disable_reset_budget: true
```

## 6. Move spend logs to separate server (BETA)

Writing each spend log to the db can slow down your proxy. In testing we saw a 70% improvement in median response time, by moving writing spend logs to a separate server. 

👉 [LiteLLM Spend Logs Server](https://github.com/BerriAI/litellm/tree/main/litellm-js/spend-logs)


**Spend Logs**  
This is a log of the key, tokens, model, and latency for each call on the proxy. 

[**Full Payload**](https://github.com/BerriAI/litellm/blob/8c9623a6bc4ad9da0a2dac64249a60ed8da719e8/litellm/proxy/utils.py#L1769)


**1. Start the spend logs server**

```bash
docker run -p 3000:3000 \
  -e DATABASE_URL="postgres://.." \
  ghcr.io/berriai/litellm-spend_logs:main-latest

# RUNNING on http://0.0.0.0:3000
```

**2. Connect to proxy**


Example litellm_config.yaml

```yaml
model_list:
- model_name: fake-openai-endpoint
  litellm_params:
    model: openai/my-fake-model
    api_key: my-fake-key
    api_base: https://exampleopenaiendpoint-production.up.railway.app/

general_settings:
  master_key: sk-1234
  proxy_batch_write_at: 5 # 👈 Frequency of batch writing logs to server (in seconds)
```

Add `SPEND_LOGS_URL` as an environment variable when starting the proxy 

```bash
docker run \
    -v $(pwd)/litellm_config.yaml:/app/config.yaml \
    -e DATABASE_URL="postgresql://.." \
    -e SPEND_LOGS_URL="http://host.docker.internal:3000" \ # 👈 KEY CHANGE
    -p 4000:4000 \
    ghcr.io/berriai/litellm:main-latest \
    --config /app/config.yaml --detailed_debug

# Running on http://0.0.0.0:4000
```

**3. Test Proxy!**


```bash
curl --location 'http://0.0.0.0:4000/v1/chat/completions' \
--header 'Content-Type: application/json' \
--header 'Authorization: Bearer sk-1234' \
--data '{
    "model": "fake-openai-endpoint", 
    "messages": [
        {"role": "system", "content": "Be helpful"},
        {"role": "user", "content": "What do you know?"}
    ]
}'
```

In your LiteLLM Spend Logs Server, you should see

**Expected Response**

```
Received and stored 1 logs. Total logs in memory: 1
...
Flushed 1 log to the DB.
```


### Machine Specification

A t2.micro should be sufficient to handle 1k logs / minute on this server. 

This consumes at max 120MB, and <0.1 vCPU. 

## Machine Specifications to Deploy LiteLLM

| Service | Spec | CPUs | Memory | Architecture | Version|
| --- | --- | --- | --- | --- | --- | 
| Server | `t2.small`. | `1vCPUs` | `8GB` | `x86` |
| Redis Cache | - | - | - | - | 7.0+ Redis Engine|


## Reference Kubernetes Deployment YAML

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
