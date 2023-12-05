import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Proxy Config.yaml
Set model list, `api_base`, `api_key`, `temperature` & proxy server settings (`master-key`) on the config.yaml. 

| Param Name           | Description                                                   |
|----------------------|---------------------------------------------------------------|
| `model_list`         | List of supported models on the server, with model-specific configs |
| `litellm_settings`   | litellm Module settings, example `litellm.drop_params=True`, `litellm.set_verbose=True`, `litellm.api_base`, `litellm.cache` |
| `general_settings`   | Server settings, example setting `master_key: sk-my_special_key` |
| `environment_variables`   | Environment Variables example, `REDIS_HOST`, `REDIS_PORT` |

## Quick Start 

Set a model alias for your deployments. 

In the `config.yaml` the model_name parameter is the user-facing name to use for your deployment. 

In the config below requests with:
- `model=vllm-models` will route to `openai/facebook/opt-125m`. 
- `model=gpt-3.5-turbo` will load balance between `azure/gpt-turbo-small-eu` and `azure/gpt-turbo-small-ca`

```yaml
model_list:
  - model_name: gpt-3.5-turbo # user-facing model alias
    litellm_params: # all params accepted by litellm.completion() - https://docs.litellm.ai/docs/completion/input
      model: azure/gpt-turbo-small-eu
      api_base: https://my-endpoint-europe-berri-992.openai.azure.com/
      api_key: "os.environ/AZURE_API_KEY_EU" # does os.getenv("AZURE_API_KEY_EU")
      rpm: 6      # Rate limit for this deployment: in requests per minute (rpm)
  - model_name: bedrock-claude-v1 
    litellm_params:
      model: bedrock/anthropic.claude-instant-v1
  - model_name: gpt-3.5-turbo
    litellm_params:
      model: azure/gpt-turbo-small-ca
      api_base: https://my-endpoint-canada-berri992.openai.azure.com/
      api_key: "os.environ/AZURE_API_KEY_CA"
      rpm: 6
  - model_name: vllm-models
    litellm_params:
      model: openai/facebook/opt-125m # the `openai/` prefix tells litellm it's openai compatible
      api_base: http://0.0.0.0:8000
      rpm: 1440
    model_info: 
      version: 2

litellm_settings: # module level litellm settings - https://github.com/BerriAI/litellm/blob/main/litellm/__init__.py
  drop_params: True
  set_verbose: True

general_settings: 
  master_key: sk-1234 # [OPTIONAL] Only use this if you to require all calls to contain this key (Authorization: Bearer sk-1234)
```

#### Step 2: Start Proxy with config

```shell
$ litellm --config /path/to/config.yaml
```


### Using Proxy - Curl Request, OpenAI Package, Langchain, Langchain JS
Calling a model group 

<Tabs>
<TabItem value="Curl" label="Curl Request">

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
</TabItem>
<TabItem value="openai" label="OpenAI v1.0.0+">

```python
import openai
client = openai.OpenAI(
    api_key="anything",
    base_url="http://0.0.0.0:8000"
)

# request sent to model set on litellm proxy, `litellm --model`
response = client.chat.completions.create(model="gpt-3.5-turbo", messages = [
    {
        "role": "user",
        "content": "this is a test request, write a short poem"
    }
])

print(response)


```

</TabItem>
<TabItem value="langchain" label="Langchain">

```python
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.schema import HumanMessage, SystemMessage

chat = ChatOpenAI(
    openai_api_base="http://0.0.0.0:8000", # set openai_api_base to the LiteLLM Proxy
    model = "gpt-3.5-turbo",
    temperature=0.1
)

messages = [
    SystemMessage(
        content="You are a helpful assistant that im using to make a test request to."
    ),
    HumanMessage(
        content="test from litellm. tell me why it's amazing in 1 sentence"
    ),
]
response = chat(messages)

print(response)
```

</TabItem>
</Tabs>


## Save Model-specific params (API Base, API Keys, Temperature, Headers etc.)
You can use the config to save model-specific information like api_base, api_key, temperature, max_tokens, etc. 

[**All input params**](https://docs.litellm.ai/docs/completion/input#input-params-1)

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

## Load API Keys

### Load API Keys from Environment 

If you have secrets saved in your environment, and don't want to expose them in the config.yaml, here's how to load model-specific keys from the environment. 

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

### Load API Keys from Azure Vault 

1. Install Proxy dependencies 
```bash
$ pip install litellm[proxy] litellm[extra_proxy]
```

2. Save Azure details in your environment
```bash 
export["AZURE_CLIENT_ID"]="your-azure-app-client-id"
export["AZURE_CLIENT_SECRET"]="your-azure-app-client-secret"
export["AZURE_TENANT_ID"]="your-azure-tenant-id"
export["AZURE_KEY_VAULT_URI"]="your-azure-key-vault-uri"
```

3. Add to proxy config.yaml 
```yaml
model_list: 
    - model_name: "my-azure-models" # model alias 
        litellm_params:
            model: "azure/<your-deployment-name>"
            api_key: "os.environ/AZURE-API-KEY" # reads from key vault - get_secret("AZURE_API_KEY")
            api_base: "os.environ/AZURE-API-BASE" # reads from key vault - get_secret("AZURE_API_BASE")

general_settings:
  use_azure_key_vault: True
```

You can now test this by starting your proxy: 
```bash
litellm --config /path/to/config.yaml
```

### Set Custom Prompt Templates

LiteLLM by default checks if a model has a [prompt template and applies it](../completion/prompt_formatting.md) (e.g. if a huggingface model has a saved chat template in it's tokenizer_config.json). However, you can also set a custom prompt template on your proxy in the `config.yaml`: 

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