import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# 🔥 Fallbacks, Retries, Timeouts, Load Balancing

Retry call with multiple instances of the same model.

If a call fails after num_retries, fall back to another model group.

If the error is a context window exceeded error, fall back to a larger model group (if given).

[**See Code**](https://github.com/BerriAI/litellm/blob/main/litellm/router.py)

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
curl --location 'http://0.0.0.0:4000/chat/completions' \
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
curl --location 'http://0.0.0.0:4000/chat/completions' \
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

## Fallbacks + Retries + Timeouts + Cooldowns

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
curl --location 'http://0.0.0.0:4000/chat/completions' \
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

### Test it!


```bash
curl --location 'http://0.0.0.0:4000/chat/completions' \
     --header 'Content-Type: application/json' \
     --data-raw '{
        "model": "zephyr-beta", # 👈 MODEL NAME to fallback from
        "messages": [
            {"role": "user", "content": "what color is red"}
        ],
        "mock_testing_fallbacks": true
     }'
```

## Advanced - Context Window Fallbacks 

**Before call is made** check if a call is within model context window with  **`enable_pre_call_checks: true`**.

[**See Code**](https://github.com/BerriAI/litellm/blob/c9e6b05cfb20dfb17272218e2555d6b496c47f6f/litellm/router.py#L2163)

**1. Setup config**

For azure deployments, set the base model. Pick the base model from [this list](https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json), all the azure models start with azure/.


<Tabs>
<TabItem value="same-group" label="Same Group">

Filter older instances of a model (e.g. gpt-3.5-turbo) with smaller context windows

```yaml
router_settings:
	enable_pre_call_checks: true # 1. Enable pre-call checks

model_list:
	- model_name: gpt-3.5-turbo
	  litellm_params:
		model: azure/chatgpt-v-2
		api_base: os.environ/AZURE_API_BASE
		api_key: os.environ/AZURE_API_KEY
		api_version: "2023-07-01-preview"
	  model_info:
		base_model: azure/gpt-4-1106-preview # 2. 👈 (azure-only) SET BASE MODEL
	
	- model_name: gpt-3.5-turbo
	  litellm_params:
		model: gpt-3.5-turbo-1106
		api_key: os.environ/OPENAI_API_KEY
```

**2. Start proxy**

```bash
litellm --config /path/to/config.yaml

# RUNNING on http://0.0.0.0:4000
```

**3. Test it!**

```python
import openai
client = openai.OpenAI(
    api_key="anything",
    base_url="http://0.0.0.0:4000"
)

text = "What is the meaning of 42?" * 5000

# request sent to model set on litellm proxy, `litellm --model`
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages = [
        {"role": "system", "content": text},
		{"role": "user", "content": "Who was Alexander?"},
    ],
)

print(response)
```

</TabItem>

<TabItem value="different-group" label="Context Window Fallbacks (Different Groups)">

Fallback to larger models if current model is too small.

```yaml
router_settings:
	enable_pre_call_checks: true # 1. Enable pre-call checks

model_list:
	- model_name: gpt-3.5-turbo-small
	  litellm_params:
		model: azure/chatgpt-v-2
		api_base: os.environ/AZURE_API_BASE
		api_key: os.environ/AZURE_API_KEY
		api_version: "2023-07-01-preview"
	  model_info:
		base_model: azure/gpt-4-1106-preview # 2. 👈 (azure-only) SET BASE MODEL
	
	- model_name: gpt-3.5-turbo-large
	  litellm_params:
		model: gpt-3.5-turbo-1106
		api_key: os.environ/OPENAI_API_KEY

  - model_name: claude-opus
    litellm_params:
      model: claude-3-opus-20240229
      api_key: os.environ/ANTHROPIC_API_KEY

litellm_settings:
  context_window_fallbacks: [{"gpt-3.5-turbo-small": ["gpt-3.5-turbo-large", "claude-opus"]}]
```

**2. Start proxy**

```bash
litellm --config /path/to/config.yaml

# RUNNING on http://0.0.0.0:4000
```

**3. Test it!**

```python
import openai
client = openai.OpenAI(
    api_key="anything",
    base_url="http://0.0.0.0:4000"
)

text = "What is the meaning of 42?" * 5000

# request sent to model set on litellm proxy, `litellm --model`
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages = [
        {"role": "system", "content": text},
		{"role": "user", "content": "Who was Alexander?"},
    ],
)

print(response)
```

</TabItem>
</Tabs>


## Advanced - Custom Timeouts, Stream Timeouts - Per Model
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


## Advanced - Setting Dynamic Timeouts - Per Request

LiteLLM Proxy supports setting a `timeout` per request 

**Example Usage**
<Tabs>
<TabItem value="Curl" label="Curl Request">

```shell
curl --location 'http://0.0.0.0:4000/chat/completions' \
     --header 'Content-Type: application/json' \
     --data-raw '{
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "user", "content": "what color is red"}
        ],
        "logit_bias": {12481: 100},
        "timeout": 1
     }'
```
</TabItem>
<TabItem value="openai" label="OpenAI v1.0.0+">

```python
import openai


client = openai.OpenAI(
    api_key="anything",
    base_url="http://0.0.0.0:4000"
)

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": "what color is red"}
    ],
    logit_bias={12481: 100},
    timeout=1
)

print(response)
```
</TabItem>
</Tabs>
