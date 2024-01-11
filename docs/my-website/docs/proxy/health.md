# Health Checks
Use this to health check all LLMs defined in your config.yaml

## Summary 

The proxy exposes: 
* a /health endpoint which returns the health of the LLM APIs  
* a /test endpoint which makes a ping to the litellm server

#### Request
Make a GET Request to `/health` on the proxy
```shell
curl --location 'http://0.0.0.0:8000/health'
```

You can also run `litellm -health` it makes a `get` request to `http://0.0.0.0:8000/health` for you
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

## Background Health Checks 

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
curl --location 'http://0.0.0.0:8000/health'
```

## Embedding Models 

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

## Text Completion Models 

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
