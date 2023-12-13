
# Load Balancing - Multiple Instances of 1 model
Load balance multiple instances of the same model

The proxy will handle routing requests (using LiteLLM's Router). **Set `rpm` in the config if you want maximize throughput**
## Quick Start - Load Balancing
### Step 1 - Set deployments on config

**Example config below**. Here requests with `model=gpt-3.5-turbo` will be routed across multiple instances of `azure/gpt-3.5-turbo`
```yaml
model_list:
  - model_name: gpt-3.5-turbo
    litellm_params:
      model: azure/<your-deployment-name>
      api_base: <your-azure-endpoint>
      api_key: <your-azure-api-key>
      rpm: 6      # Rate limit for this deployment: in requests per minute (rpm)
  - model_name: gpt-3.5-turbo
    litellm_params:
      model: azure/gpt-turbo-small-ca
      api_base: https://my-endpoint-canada-berri992.openai.azure.com/
      api_key: <your-azure-api-key>
      rpm: 6
  - model_name: gpt-3.5-turbo
    litellm_params:
      model: azure/gpt-turbo-large
      api_base: https://openai-france-1234.openai.azure.com/
      api_key: <your-azure-api-key>
      rpm: 1440
```

### Step 2: Start Proxy with config

```shell
$ litellm --config /path/to/config.yaml
```

### Step 3: Use proxy - Call a model group [Load Balancing]
Curl Command
```shell
curl --location 'http://0.0.0.0:8000/chat/completions' \
--header 'Content-Type: application/json' \
--data ' {
      "model": "gpt-3.5-turbo",
      "messages": [
        {
          "role": "user",
          "content": "what llm are you"
        }
      ],
    }
'
```

### Usage - Call a specific model deployment
If you want to call a specific model defined in the `config.yaml`, you can call the `litellm_params: model`

In this example it will call `azure/gpt-turbo-small-ca`. Defined in the config on Step 1

```bash
curl --location 'http://0.0.0.0:8000/chat/completions' \
--header 'Content-Type: application/json' \
--data ' {
      "model": "azure/gpt-turbo-small-ca",
      "messages": [
        {
          "role": "user",
          "content": "what llm are you"
        }
      ],
    }
'
```

## Router settings on config - routing_strategy, model_group_alias

litellm.Router() settings can be set under `router_settings`. You can set `model_group_alias`, `routing_strategy`, `num_retries`,`timeout` . See all Router supported params [here](https://github.com/BerriAI/litellm/blob/1b942568897a48f014fa44618ec3ce54d7570a46/litellm/router.py#L64)

Example config with `router_settings`
```yaml
model_list:
  - model_name: gpt-3.5-turbo
    litellm_params:
      model: azure/<your-deployment-name>
      api_base: <your-azure-endpoint>
      api_key: <your-azure-api-key>
      rpm: 6      # Rate limit for this deployment: in requests per minute (rpm)
  - model_name: gpt-3.5-turbo
    litellm_params:
      model: azure/gpt-turbo-small-ca
      api_base: https://my-endpoint-canada-berri992.openai.azure.com/
      api_key: <your-azure-api-key>
      rpm: 6
router_settings:
  model_group_alias: {"gpt-4": "gpt-3.5-turbo"} # all requests with `gpt-4` will be routed to models with `gpt-3.5-turbo`
  routing_strategy: least-busy                  # Literal["simple-shuffle", "least-busy", "usage-based-routing", "latency-based-routing"]
  num_retries: 2
  timeout: 30                                  # 30 seconds
```

## Fallbacks + Cooldowns + Retries + Timeouts 

If a call fails after num_retries, fall back to another model group.

If the error is a context window exceeded error, fall back to a larger model group (if given).

[**See Code**](https://github.com/BerriAI/litellm/blob/main/litellm/router.py)

**Set via config**
```yaml
model_list:
  - model_name: zephyr-beta
    litellm_params:
        model: huggingface/HuggingFaceH4/zephyr-7b-beta
        api_base: http://0.0.0.0:8001
  - model_name: zephyr-beta
    litellm_params:
        model: huggingface/HuggingFaceH4/zephyr-7b-beta
        api_base: http://0.0.0.0:8002
  - model_name: zephyr-beta
    litellm_params:
        model: huggingface/HuggingFaceH4/zephyr-7b-beta
        api_base: http://0.0.0.0:8003
  - model_name: gpt-3.5-turbo
    litellm_params:
        model: gpt-3.5-turbo
        api_key: <my-openai-key>
  - model_name: gpt-3.5-turbo-16k
    litellm_params:
        model: gpt-3.5-turbo-16k
        api_key: <my-openai-key>

litellm_settings:
  num_retries: 3 # retry call 3 times on each model_name (e.g. zephyr-beta)
  request_timeout: 10 # raise Timeout error if call takes longer than 10s. Sets litellm.request_timeout 
  fallbacks: [{"zephyr-beta": ["gpt-3.5-turbo"]}] # fallback to gpt-3.5-turbo if call fails num_retries 
  context_window_fallbacks: [{"zephyr-beta": ["gpt-3.5-turbo-16k"]}, {"gpt-3.5-turbo": ["gpt-3.5-turbo-16k"]}] # fallback to gpt-3.5-turbo-16k if context window error
  allowed_fails: 3 # cooldown model if it fails > 1 call in a minute. 
```

**Set dynamically**

```bash
curl --location 'http://0.0.0.0:8000/chat/completions' \
--header 'Content-Type: application/json' \
--data ' {
      "model": "zephyr-beta",
      "messages": [
        {
          "role": "user",
          "content": "what llm are you"
        }
      ],
      "fallbacks": [{"zephyr-beta": ["gpt-3.5-turbo"]}],
      "context_window_fallbacks": [{"zephyr-beta": ["gpt-3.5-turbo"]}],
      "num_retries": 2,
      "timeout": 10
    }
'
```

## Custom Timeouts, Stream Timeouts - Per Model
For each model you can set `timeout` & `stream_timeout` under `litellm_params`
```yaml
model_list:
  - model_name: gpt-3.5-turbo
    litellm_params:
      model: azure/gpt-turbo-small-eu
      api_base: https://my-endpoint-europe-berri-992.openai.azure.com/
      api_key: <your-key>
      timeout: 0.1                      # timeout in (seconds)
      stream_timeout: 0.01              # timeout for stream requests (seconds)
      max_retries: 5
  - model_name: gpt-3.5-turbo
    litellm_params:
      model: azure/gpt-turbo-small-ca
      api_base: https://my-endpoint-canada-berri992.openai.azure.com/
      api_key: 
      timeout: 0.1                      # timeout in (seconds)
      stream_timeout: 0.01              # timeout for stream requests (seconds)
      max_retries: 5

```

#### Start Proxy 
```shell
$ litellm --config /path/to/config.yaml
```



## Health Check LLMs on Proxy
Use this to health check all LLMs defined in your config.yaml
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