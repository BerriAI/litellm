# Health Checks

Use this to health check all LLMs defined in your config.yaml

## Summary

The proxy exposes:

* A `/health` endpoint which returns the health of the LLM APIs  
* A `/health/readiness` endpoint for returning if the proxy is ready to accept requests
* A `/health/liveliness` endpoint for returning if the proxy is alive
* A `/health/db` endpoint for returning info and metrics about the database

## `/health`
#### Request
Make a GET Request to `/health` on the proxy
```shell
curl --location 'http://0.0.0.0:4000/health' -H "Authorization: Bearer sk-1234"
```

You can also run `litellm -health` it makes a `get` request to `http://0.0.0.0:4000/health` for you
```
litellm --health
```
#### Response
```shell
{
    "healthy_endpoints": [
        {
            "model": "azure/gpt-35-turbo",
            "api_base": "https://my-endpoint-canada-berri992.openai.azure.com/"
        },
        {
            "model": "azure/gpt-35-turbo",
            "api_base": "https://my-endpoint-europe-berri-992.openai.azure.com/"
        }
    ],
    "unhealthy_endpoints": [
        {
            "model": "azure/gpt-35-turbo",
            "api_base": "https://openai-france-1234.openai.azure.com/"
        }
    ]
}
```

### Background Health Checks 

You can enable model health checks being run in the background, to prevent each model from being queried too frequently via `/health`.

Here's how to use it: 
1. in the config.yaml add:
```
general_settings: 
  background_health_checks: True # enable background health checks
  health_check_interval: 300 # frequency of background health checks
```

2. Start server 
```
$ litellm /path/to/config.yaml
```

3. Query health endpoint: 
```
curl --location 'http://0.0.0.0:4000/health'
```

### Embedding Models 

We need some way to know if the model is an embedding model when running checks, if you have this in your config, specifying mode it makes an embedding health check

```yaml
model_list:
  - model_name: azure-embedding-model
    litellm_params:
      model: azure/azure-embedding-model
      api_base: os.environ/AZURE_API_BASE
      api_key: os.environ/AZURE_API_KEY
      api_version: "2023-07-01-preview"
    model_info:
      mode: embedding # 👈 ADD THIS
```

### Image Generation Models 

We need some way to know if the model is an image generation model when running checks, if you have this in your config, specifying mode it makes an image generation health check

```yaml
model_list:
  - model_name: dall-e-3
    litellm_params:
      model: azure/dall-e-3
      api_base: os.environ/AZURE_API_BASE
      api_key: os.environ/AZURE_API_KEY
      api_version: "2023-07-01-preview"
    model_info:
      mode: image_generation # 👈 ADD THIS
```


### Text Completion Models 

We need some way to know if the model is a text completion model when running checks, if you have this in your config, specifying mode it makes an embedding health check

```yaml
model_list:
  - model_name: azure-text-completion
    litellm_params:
      model: azure/text-davinci-003
      api_base: os.environ/AZURE_API_BASE
      api_key: os.environ/AZURE_API_KEY
      api_version: "2023-07-01-preview"
    model_info:
      mode: completion # 👈 ADD THIS
```

## `/health/readiness`

Unprotected endpoint for checking if proxy is ready to accept requests

Example Request: 

```bash
curl http://0.0.0.0:4000/health/readiness
```

Example Response:  

```json
{
  "status": "connected",
  "db": "connected",
  "cache": null,
  "litellm_version": "1.40.21",
  "success_callbacks": [
    "langfuse",
    "_PROXY_track_cost_callback",
    "response_taking_too_long_callback",
    "_PROXY_MaxParallelRequestsHandler",
    "_PROXY_MaxBudgetLimiter",
    "_PROXY_CacheControlCheck",
    "ServiceLogging"
  ],
  "last_updated": "2024-07-10T18:59:10.616968"
}
```

If the proxy is not connected to a database, then the `"db"` field will be `"Not
connected"` instead of `"connected"` and the `"last_updated"` field will not be present.

## `/health/liveliness`

Unprotected endpoint for checking if proxy is alive


Example Request: 

```
curl -X 'GET' \
  'http://0.0.0.0:4000/health/liveliness' \
  -H 'accept: application/json'
```

Example Response: 

```json
"I'm alive!"
```

## `/health/db`

Unprotected endpoint for viewing info about the database, including metrics like
the number of open connections

Example Request:

```shell
curl -sSL http://0.0.0.0:4000/health/db | jq '.'
```

Example Response:

```json
{
  "metrics": {
    "counters": [
      {
        "key": "prisma_client_queries_total",
        "value": 8,
        "labels": {},
        "description": "The total number of Prisma Client queries executed"
      },
      {
        "key": "prisma_datasource_queries_total",
        "value": 13,
        "labels": {},
        "description": "The total number of datasource queries executed"
      },
      {
        "key": "prisma_pool_connections_closed_total",
        "value": 0,
        "labels": {},
        "description": "The total number of pool connections closed"
      },
      {
        "key": "prisma_pool_connections_opened_total",
        "value": 2,
        "labels": {},
        "description": "The total number of pool connections opened"
      }
    ],
    "gauges": [
      {
        "key": "prisma_client_queries_active",
        "value": 0.0,
        "labels": {},
        "description": "The number of currently active Prisma Client queries"
      },
      {
        "key": "prisma_client_queries_wait",
        "value": 0.0,
        "labels": {},
        "description": "The number of datasource queries currently waiting for an free connection"
      },
      {
        "key": "prisma_pool_connections_busy",
        "value": 0.0,
        "labels": {},
        "description": "The number of pool connections currently executing datasource queries"
      },
      {
        "key": "prisma_pool_connections_idle",
        "value": 100.0,
        "labels": {},
        "description": "The number of pool connections that are not busy running a query"
      },
      {
        "key": "prisma_pool_connections_open",
        "value": 2.0,
        "labels": {},
        "description": "The number of pool connections currently open"
      }
    ],
    "histograms": [
      {
        "key": "prisma_client_queries_duration_histogram_ms",
        "value": {
          "sum": 24731.272958,
          "count": 8,
          "buckets": [
            ...
          ]
        },
        "labels": {},
        "description": "The distribution of the time Prisma Client queries took to run end to end"
      },
      {
        "key": "prisma_client_queries_wait_histogram_ms",
        "value": {
          "sum": 0.024627,
          "count": 9,
          "buckets": [
            ...
          ]
        },
        "labels": {},
        "description": "The distribution of the time all datasource queries spent waiting for a free connection"
      },
      {
        "key": "prisma_datasource_queries_duration_histogram_ms",
        "value": {
          "sum": 24804.927917,
          "count": 13,
          "buckets": [
            ...
          ]
        },
        "labels": {},
        "description": "The distribution of the time datasource queries took to run"
      }
    ]
  },
  "db_health_status": {
    "status": "connected",
    "last_updated": "2024-07-11T12:46:35.904744"
  }
}
```

## Advanced - Call specific models 

To check health of specific models, here's how to call them: 

### 1. Get model id via `/model/info` 

```bash
curl -X GET 'http://0.0.0.0:4000/v1/model/info' \
--header 'Authorization: Bearer sk-1234' \
```

**Expected Response**

```bash
{
    "model_name": "bedrock-anthropic-claude-3",
    "litellm_params": {
        "model": "anthropic.claude-3-sonnet-20240229-v1:0"
    },
    "model_info": {
        "id": "634b87c444..", # 👈 UNIQUE MODEL ID
}
```

### 2. Call specific model via `/chat/completions` 

```bash
curl -X POST 'http://localhost:4000/chat/completions' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer sk-1234' \
-D '{
  "model": "634b87c444.." # 👈 UNIQUE MODEL ID
  "messages": [
    {
      "role": "user",
      "content": "ping"
    }
  ],
}
'
```