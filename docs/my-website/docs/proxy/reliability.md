import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Fallbacks, Load Balancing, Retries

- Quick Start [load balancing](#test---load-balancing)
- Quick Start [client side fallbacks](#test---client-side-fallbacks)

## Quick Start - Load Balancing
#### Step 1 - Set deployments on config

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

router_settings:
  routing_strategy: simple-shuffle # Literal["simple-shuffle", "least-busy", "usage-based-routing","latency-based-routing"], default="simple-shuffle"
  model_group_alias: {"gpt-4": "gpt-3.5-turbo"} # all requests with `gpt-4` will be routed to models with `gpt-3.5-turbo`
  num_retries: 2
  timeout: 30                                  # 30 seconds
  redis_host: <your redis host>                # set this when using multiple litellm proxy deployments, load balancing state stored in redis
  redis_password: <your redis password>
  redis_port: 1992
```

:::info
Detailed information about [routing strategies can be found here](../routing)
:::

#### Step 2: Start Proxy with config

```shell
$ litellm --config /path/to/config.yaml
```

### Test - Simple Call

Here requests with model=gpt-3.5-turbo will be routed across multiple instances of azure/gpt-3.5-turbo

👉 Key Change: `model="gpt-3.5-turbo"`

**Check the `model_id` in Response Headers to make sure the requests are being load balanced**

<Tabs>

<TabItem value="openai" label="OpenAI Python v1.0.0+">

```python
import openai
client = openai.OpenAI(
    api_key="anything",
    base_url="http://0.0.0.0:4000"
)

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages = [
        {
            "role": "user",
            "content": "this is a test request, write a short poem"
        }
    ]
)

print(response)
```
</TabItem>

<TabItem value="Curl" label="Curl Request">

```shell
curl --location 'http://0.0.0.0:4000/chat/completions' \
    --header 'Content-Type: application/json' \
    --data '{
    "model": "gpt-3.5-turbo",
    "messages": [
        {
        "role": "user",
        "content": "what llm are you"
        }
    ]
}'
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
import os 

os.environ["OPENAI_API_KEY"] = "anything"

chat = ChatOpenAI(
    openai_api_base="http://0.0.0.0:4000",
    model="gpt-3.5-turbo",
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


### Test - Loadbalancing

In this request, the following will occur:
1. A rate limit exception will be raised 
2. LiteLLM proxy will retry the request on the model group (default is 3).

```bash
curl -X POST 'http://0.0.0.0:4000/chat/completions' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer sk-1234' \
-d '{
  "model": "gpt-3.5-turbo",
  "messages": [
        {"role": "user", "content": "Hi there!"}
    ],
    "mock_testing_rate_limit_error": true
}'
```

[**See Code**](https://github.com/BerriAI/litellm/blob/6b8806b45f970cb2446654d2c379f8dcaa93ce3c/litellm/router.py#L2535)

### Test - Client Side Fallbacks
In this request the following will occur:
1. The request to `model="zephyr-beta"` will fail
2. litellm proxy will loop through all the model_groups specified in `fallbacks=["gpt-3.5-turbo"]`
3. The request to `model="gpt-3.5-turbo"` will succeed and the client making the request will get a response from gpt-3.5-turbo 

👉 Key Change: `"fallbacks": ["gpt-3.5-turbo"]`

<Tabs>

<TabItem value="openai" label="OpenAI Python v1.0.0+">

```python
import openai
client = openai.OpenAI(
    api_key="anything",
    base_url="http://0.0.0.0:4000"
)

response = client.chat.completions.create(
    model="zephyr-beta",
    messages = [
        {
            "role": "user",
            "content": "this is a test request, write a short poem"
        }
    ],
    extra_body={
        "fallbacks": ["gpt-3.5-turbo"]
    }
)

print(response)
```
</TabItem>

<TabItem value="Curl" label="Curl Request">

Pass `metadata` as part of the request body

```shell
curl --location 'http://0.0.0.0:4000/chat/completions' \
    --header 'Content-Type: application/json' \
    --data '{
    "model": "zephyr-beta"",
    "messages": [
        {
        "role": "user",
        "content": "what llm are you"
        }
    ],
    "fallbacks": ["gpt-3.5-turbo"]
}'
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
import os 

os.environ["OPENAI_API_KEY"] = "anything"

chat = ChatOpenAI(
    openai_api_base="http://0.0.0.0:4000",
    model="zephyr-beta",
    extra_body={
        "fallbacks": ["gpt-3.5-turbo"]
    }
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



<!-- 
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
``` -->

## Advanced
### Fallbacks + Retries + Timeouts + Cooldowns

To set fallbacks, just do: 

```
litellm_settings:
  fallbacks: [{"zephyr-beta": ["gpt-3.5-turbo"]}] 
```

**Covers all errors (429, 500, etc.)**

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
  allowed_fails: 3 # cooldown model if it fails > 1 call in a minute. 
  cooldown_time: 30 # how long to cooldown model if fails/min > allowed_fails
```

### Test Fallbacks! 

Check if your fallbacks are working as expected. 

#### **Regular Fallbacks**
```bash
curl -X POST 'http://0.0.0.0:4000/chat/completions' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer sk-1234' \
-D '{
  "model": "my-bad-model",
  "messages": [
    {
      "role": "user",
      "content": "ping"
    }
  ],
  "mock_testing_fallbacks": true # 👈 KEY CHANGE
}
'
```

#### **Content Policy Fallbacks**
```bash
curl -X POST 'http://0.0.0.0:4000/chat/completions' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer sk-1234' \
-D '{
  "model": "my-bad-model",
  "messages": [
    {
      "role": "user",
      "content": "ping"
    }
  ],
  "mock_testing_content_policy_fallbacks": true # 👈 KEY CHANGE
}
'
```

#### **Context Window Fallbacks**

```bash
curl -X POST 'http://0.0.0.0:4000/chat/completions' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer sk-1234' \
-D '{
  "model": "my-bad-model",
  "messages": [
    {
      "role": "user",
      "content": "ping"
    }
  ],
  "mock_testing_context_window_fallbacks": true # 👈 KEY CHANGE
}
'
```


### Context Window Fallbacks (Pre-Call Checks + Fallbacks)

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


### Content Policy Fallbacks

Fallback across providers (e.g. from Azure OpenAI to Anthropic) if you hit content policy violation errors. 

```yaml
model_list:
	- model_name: gpt-3.5-turbo-small
	  litellm_params:
		model: azure/chatgpt-v-2
        api_base: os.environ/AZURE_API_BASE
        api_key: os.environ/AZURE_API_KEY
        api_version: "2023-07-01-preview"

    - model_name: claude-opus
      litellm_params:
        model: claude-3-opus-20240229
        api_key: os.environ/ANTHROPIC_API_KEY

litellm_settings:
  content_policy_fallbacks: [{"gpt-3.5-turbo-small": ["claude-opus"]}]
```



### Default Fallbacks 

You can also set default_fallbacks, in case a specific model group is misconfigured / bad.


```yaml
model_list:
	- model_name: gpt-3.5-turbo-small
	  litellm_params:
		model: azure/chatgpt-v-2
        api_base: os.environ/AZURE_API_BASE
        api_key: os.environ/AZURE_API_KEY
        api_version: "2023-07-01-preview"

    - model_name: claude-opus
      litellm_params:
        model: claude-3-opus-20240229
        api_key: os.environ/ANTHROPIC_API_KEY

litellm_settings:
  default_fallbacks: ["claude-opus"]
```

This will default to claude-opus in case any model fails.

A model-specific fallbacks (e.g. {"gpt-3.5-turbo-small": ["claude-opus"]}) overrides default fallback.

### EU-Region Filtering (Pre-Call Checks)

**Before call is made** check if a call is within model context window with  **`enable_pre_call_checks: true`**.

Set 'region_name' of deployment. 

**Note:** LiteLLM can automatically infer region_name for Vertex AI, Bedrock, and IBM WatsonxAI based on your litellm params. For Azure, set `litellm.enable_preview = True`.

**1. Set Config**

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
    region_name: "eu" # 👈 SET EU-REGION

- model_name: gpt-3.5-turbo
  litellm_params:
    model: gpt-3.5-turbo-1106
    api_key: os.environ/OPENAI_API_KEY

- model_name: gemini-pro
  litellm_params:
    model: vertex_ai/gemini-pro-1.5
    vertex_project: adroit-crow-1234
    vertex_location: us-east1 # 👈 AUTOMATICALLY INFERS 'region_name'
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

# request sent to model set on litellm proxy, `litellm --model`
response = client.chat.completions.with_raw_response.create(
    model="gpt-3.5-turbo",
    messages = [{"role": "user", "content": "Who was Alexander?"}]
)

print(response)

print(f"response.headers.get('x-litellm-model-api-base')")
```

### Custom Timeouts, Stream Timeouts - Per Model
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


### Setting Dynamic Timeouts - Per Request

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

### Setting Fallbacks for Wildcard Models

You can set fallbacks for wildcard models (e.g. `azure/*`) in your config file.

1. Setup config
```yaml
model_list:
  - model_name: "gpt-4o"
    litellm_params:
      model: "openai/gpt-4o"
      api_key: os.environ/OPENAI_API_KEY
  - model_name: "azure/*"
    litellm_params:
      model: "azure/*"
      api_key: os.environ/AZURE_API_KEY
      api_base: os.environ/AZURE_API_BASE

litellm_settings:
  fallbacks: [{"gpt-4o": ["azure/gpt-4o"]}]
```

2. Start Proxy
```bash
litellm --config /path/to/config.yaml
```

3. Test it!

```bash
curl -L -X POST 'http://0.0.0.0:4000/v1/chat/completions' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer sk-1234' \
-d '{
    "model": "gpt-4o",
    "messages": [
      {
        "role": "user",
        "content": [    
          {
            "type": "text",
            "text": "what color is red"
          }
        ]
      }
    ],
    "max_tokens": 300,
    "mock_testing_fallbacks": true
}'
```

### Disable Fallbacks per key

You can disable fallbacks per key by setting `disable_fallbacks: true` in your key metadata.

```bash
curl -L -X POST 'http://0.0.0.0:4000/key/generate' \
-H 'Authorization: Bearer sk-1234' \
-H 'Content-Type: application/json' \
-d '{
    "metadata": {
        "disable_fallbacks": true
    }
}'
```