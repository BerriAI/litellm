# Proxy Config.yaml
Set model list, `api_base`, `api_key`, `temperature` & proxy server settings (`master-key`) on the config.yaml. 

| Param Name           | Description                                                   |
|----------------------|---------------------------------------------------------------|
| `model_list`         | List of supported models on the server, with model-specific configs |
| `litellm_settings`   | litellm Module settings, example `litellm.drop_params=True`, `litellm.set_verbose=True`, `litellm.api_base`, `litellm.cache` |
| `general_settings`   | Server settings, example setting `master_key: sk-my_special_key` |
| `environment_variables`   | Environment Variables example, `REDIS_HOST`, `REDIS_PORT` |

#### Example Config
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

litellm_settings:
  drop_params: True
  set_verbose: True

general_settings: 
  master_key: sk-1234 # [OPTIONAL] Only use this if you to require all calls to contain this key (Authorization: Bearer sk-1234)


environment_variables:
  OPENAI_API_KEY: sk-123
  REPLICATE_API_KEY: sk-cohere-is-okay
  REDIS_HOST: redis-16337.c322.us-east-1-2.ec2.cloud.redislabs.com
  REDIS_PORT: "16337"
  REDIS_PASSWORD: 
```

### Config for Multiple Models - GPT-4, Claude-2

Here's how you can use multiple llms with one proxy `config.yaml`. 

#### Step 1: Setup Config
```yaml
model_list:
  - model_name: zephyr-alpha # the 1st model is the default on the proxy
    litellm_params: # params for litellm.completion() - https://docs.litellm.ai/docs/completion/input#input---request-body
      model: huggingface/HuggingFaceH4/zephyr-7b-alpha
      api_base: http://0.0.0.0:8001
  - model_name: gpt-4
    litellm_params:
      model: gpt-4
      api_key: sk-1233
  - model_name: claude-2
    litellm_params:
      model: claude-2
      api_key: sk-claude    
```

:::info

The proxy uses the first model in the config as the default model - in this config the default model is `zephyr-alpha`
:::


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
      "model": "zephyr-alpha",
      "messages": [
        {
          "role": "user",
          "content": "what llm are you"
        }
      ],
    }
'
```

### Config for Embedding Models - xorbitsai/inference

Here's how you can use multiple llms with one proxy `config.yaml`. 
Here is how [LiteLLM calls OpenAI Compatible Embedding models](https://docs.litellm.ai/docs/embedding/supported_embedding#openai-compatible-embedding-models)

#### Config
```yaml
model_list:
  - model_name: custom_embedding_model
    litellm_params:
      model: openai/custom_embedding  # the `openai/` prefix tells litellm it's openai compatible
      api_base: http://0.0.0.0:8000/
  - model_name: custom_embedding_model
    litellm_params:
      model: openai/custom_embedding  # the `openai/` prefix tells litellm it's openai compatible
      api_base: http://0.0.0.0:8001/
```

Run the proxy using this config
```shell
$ litellm --config /path/to/config.yaml
```

### Save Model-specific params (API Base, API Keys, Temperature, Headers etc.)
You can use the config to save model-specific information like api_base, api_key, temperature, max_tokens, etc. 

**Step 1**: Create a `config.yaml` file
```yaml
model_list:
  - model_name: gpt-4-team1
    litellm_params: # params for litellm.completion() - https://docs.litellm.ai/docs/completion/input#input---request-body
      model: azure/chatgpt-v-2
      api_base: https://openai-gpt-4-test-v-1.openai.azure.com/
      api_version: "2023-05-15"
      azure_ad_token: eyJ0eXAiOiJ
  - model_name: gpt-4-team2
    litellm_params:
      model: azure/gpt-4
      api_key: sk-123
      api_base: https://openai-gpt-4-test-v-2.openai.azure.com/
  - model_name: mistral-7b
    litellm_params:
      model: ollama/mistral
      api_base: your_ollama_api_base
      headers: {
        "HTTP-Referer": "litellm.ai",  
        "X-Title": "LiteLLM Server"
      }
```

**Step 2**: Start server with config

```shell
$ litellm --config /path/to/config.yaml
```

### Load API Keys from Vault 

If you have secrets saved in Azure Vault, etc. and don't want to expose them in the config.yaml, here's how to load model-specific keys from the environment. 

```python
os.environ["AZURE_NORTH_AMERICA_API_KEY"] = "your-azure-api-key"
```

```yaml 
model_list:
  - model_name: gpt-4-team1
    litellm_params: # params for litellm.completion() - https://docs.litellm.ai/docs/completion/input#input---request-body
      model: azure/chatgpt-v-2
      api_base: https://openai-gpt-4-test-v-1.openai.azure.com/
      api_version: "2023-05-15"
      api_key: os.environ/AZURE_NORTH_AMERICA_API_KEY
```

[**See Code**](https://github.com/BerriAI/litellm/blob/c12d6c3fe80e1b5e704d9846b246c059defadce7/litellm/utils.py#L2366)

s/o to [@David Manouchehri](https://www.linkedin.com/in/davidmanouchehri/) for helping with this. 

### Config for setting Model Aliases

Set a model alias for your deployments. 

In the `config.yaml` the model_name parameter is the user-facing name to use for your deployment. 

In the config below requests with `model=gpt-4` will route to `ollama/llama2`

```yaml
model_list:
  - model_name: text-davinci-003
    litellm_params:
        model: ollama/zephyr
  - model_name: gpt-4
    litellm_params:
        model: ollama/llama2
  - model_name: gpt-3.5-turbo
    litellm_params:
        model: ollama/llama2
```

### Set Custom Prompt Templates

LiteLLM by default checks if a model has a [prompt template and applies it](./completion/prompt_formatting.md) (e.g. if a huggingface model has a saved chat template in it's tokenizer_config.json). However, you can also set a custom prompt template on your proxy in the `config.yaml`: 

**Step 1**: Save your prompt template in a `config.yaml`
```yaml
# Model-specific parameters
model_list:
  - model_name: mistral-7b # model alias
    litellm_params: # actual params for litellm.completion()
      model: "huggingface/mistralai/Mistral-7B-Instruct-v0.1" 
      api_base: "<your-api-base>"
      api_key: "<your-api-key>" # [OPTIONAL] for hf inference endpoints
      initial_prompt_value: "\n"
      roles: {"system":{"pre_message":"<|im_start|>system\n", "post_message":"<|im_end|>"}, "assistant":{"pre_message":"<|im_start|>assistant\n","post_message":"<|im_end|>"}, "user":{"pre_message":"<|im_start|>user\n","post_message":"<|im_end|>"}}
      final_prompt_value: "\n"
      bos_token: "<s>"
      eos_token: "</s>"
      max_tokens: 4096
```

**Step 2**: Start server with config

```shell
$ litellm --config /path/to/config.yaml
```