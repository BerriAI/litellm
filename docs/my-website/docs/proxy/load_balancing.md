
# Load Balancing - Multiple Instances of 1 model
Load balance multiple instances of the same model

The proxy will handle routing requests (using LiteLLM's Router). **Set `rpm` in the config if you want maximize throughput**

#### Example config
requests with `model=gpt-3.5-turbo` will be routed across multiple instances of `azure/gpt-3.5-turbo`
```yaml
model_list:
  - model_name: gpt-3.5-turbo
    litellm_params:
      model: azure/gpt-turbo-small-eu
      api_base: https://my-endpoint-europe-berri-992.openai.azure.com/
      api_key: 
      rpm: 6      # Rate limit for this deployment: in requests per minute (rpm)
  - model_name: gpt-3.5-turbo
    litellm_params:
      model: azure/gpt-turbo-small-ca
      api_base: https://my-endpoint-canada-berri992.openai.azure.com/
      api_key: 
      rpm: 6
  - model_name: gpt-3.5-turbo
    litellm_params:
      model: azure/gpt-turbo-large
      api_base: https://openai-france-1234.openai.azure.com/
      api_key: 
      rpm: 1440
```

#### Step 2: Start Proxy with config

```shell
$ litellm --config /path/to/config.yaml
```

#### Step 3: Use proxy
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

### Fallbacks + Cooldowns + Retries + Timeouts 

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
  request_timeout: 10 # raise Timeout error if call takes longer than 10s
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
      "request_timeout": 10
    }
'
```
