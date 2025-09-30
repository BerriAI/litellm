# Health Checks
Use this to health check all LLMs defined in your config.yaml

## When to Use Each Endpoint

| Endpoint | Use Case | Purpose |
|----------|----------|---------|
| `/health/liveliness` | **Container liveness probes** | Basic alive check - use for container restart decisions |
| `/health/readiness` | **Load balancer health checks** | Ready to accept traffic - includes DB connection status |
| `/health` | **Model health monitoring** | Comprehensive LLM model health - makes actual API calls |
| `/health/services` | **Service debugging** | Check specific integrations (datadog, langfuse, etc.) |

## Summary 

The proxy exposes: 
* a /health endpoint which returns the health of the LLM APIs  
* a /health/readiness endpoint for returning if the proxy is ready to accept requests 
* a /health/liveliness endpoint for returning if the proxy is alive 

## `/health`
#### Request
Make a GET Request to `/health` on the proxy 

:::info
**This endpoint makes an LLM API call to each model to check if it is healthy.**
:::

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

### Embedding Models 

To run embedding health checks, specify the mode as "embedding" in your config for the relevant model.

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

To run image generation health checks, specify the mode as "image_generation" in your config for the relevant model.

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


To run `/completions` health checks, specify the mode as "completion" in your config for the relevant model.

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

### Speech to Text Models 

```yaml
model_list:
  - model_name: whisper
    litellm_params:
      model: whisper-1
      api_key: os.environ/OPENAI_API_KEY
    model_info:
      mode: audio_transcription
```


### Text to Speech Models 

```yaml
# OpenAI Text to Speech Models
  - model_name: tts
    litellm_params:
      model: openai/tts-1
      api_key: "os.environ/OPENAI_API_KEY"
    model_info:
      mode: audio_speech
      health_check_voice: alloy
```

You can specify a `health_check_voice` if you need to use a voice other than "alloy".

### Rerank Models 

To run rerank health checks, specify the mode as "rerank" in your config for the relevant model.

```yaml
model_list:
  - model_name: rerank-english-v3.0
    litellm_params:
      model: cohere/rerank-english-v3.0
      api_key: os.environ/COHERE_API_KEY
    model_info:
      mode: rerank
```

### Batch Models (Azure Only)

For Azure models deployed as 'batch' models, set `mode: batch`. 

```yaml
model_list:
  - model_name: "batch-gpt-4o-mini"
    litellm_params:
      model: "azure/batch-gpt-4o-mini"
      api_key: os.environ/AZURE_API_KEY
      api_base: os.environ/AZURE_API_BASE
    model_info:
      mode: batch
```

Expected Response 


```bash
{
    "healthy_endpoints": [
        {
            "api_base": "https://...",
            "model": "azure/gpt-4o-mini",
            "x-ms-region": "East US"
        }
    ],
    "unhealthy_endpoints": [],
    "healthy_count": 1,
    "unhealthy_count": 0
}
```

### Realtime Models 

To run realtime health checks, specify the mode as "realtime" in your config for the relevant model.

```yaml
model_list:
  - model_name: openai/gpt-4o-realtime-audio
    litellm_params:
      model: openai/gpt-4o-realtime-audio
      api_key: os.environ/OPENAI_API_KEY
    model_info:
      mode: realtime
```

### Wildcard Routes

For wildcard routes, you can specify a `health_check_model` in your config.yaml. This model will be used for health checks for that wildcard route.

In this example, when running a health check for `openai/*`, the health check will make a `/chat/completions` request to `openai/gpt-4o-mini`.

```yaml
model_list:
  - model_name: openai/*
    litellm_params:
      model:  openai/*
      api_key: os.environ/OPENAI_API_KEY
    model_info:
      health_check_model: openai/gpt-4o-mini
  - model_name: anthropic/*
    litellm_params:
      model: anthropic/*
      api_key: os.environ/ANTHROPIC_API_KEY
    model_info:
      health_check_model: anthropic/claude-3-5-sonnet-20240620
```

## Background Health Checks 

You can enable model health checks being run in the background, to prevent each model from being queried too frequently via `/health`. 

:::info

**This makes an LLM API call to each model to check if it is healthy.**

:::

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

### Disable Background Health Checks For Specific Models

Use this if you want to disable background health checks for specific models.

If `background_health_checks` is enabled you can skip individual models by
setting `disable_background_health_check: true` in the model's `model_info`.

```yaml
model_list:
  - model_name: openai/gpt-4o
    litellm_params:
      model: openai/gpt-4o
      api_key: os.environ/OPENAI_API_KEY
    model_info:
      disable_background_health_check: true
```

### Hide details

The health check response contains details like endpoint URLs, error messages,
and other LiteLLM params. While this is useful for debugging, it can be
problematic when exposing the proxy server to a broad audience.

You can hide these details by setting the `health_check_details` setting to `False`.

```yaml
general_settings: 
  health_check_details: False
```

## Health Check Timeout

The health check timeout is set in `litellm/constants.py` and defaults to 60 seconds.

This can be overridden in the config.yaml by setting `health_check_timeout` in the model_info section.

```yaml
model_list:
  - model_name: openai/gpt-4o
    litellm_params:
      model: openai/gpt-4o
      api_key: os.environ/OPENAI_API_KEY
    model_info:
      health_check_timeout: 10 # 👈 OVERRIDE HEALTH CHECK TIMEOUT
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

## `/health/services`

Use this admin-only endpoint to check if a connected service (datadog/slack/langfuse/etc.) is healthy.

```bash
curl -L -X GET 'http://0.0.0.0:4000/health/services?service=datadog'     -H 'Authorization: Bearer sk-1234'
```

[**API Reference**](https://litellm-api.up.railway.app/#/health/health_services_endpoint_health_services_get)


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

