import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Fallbacks

If a call fails after num_retries, fallback to another model group. 

- Quick Start [load balancing](./load_balancing.md)
- Quick Start [client side fallbacks](#client-side-fallbacks)


Fallbacks are typically done from one `model_name` to another `model_name`. 

## Quick Start 

### 1. Setup fallbacks

Key change: 

```python
fallbacks=[{"gpt-3.5-turbo": ["gpt-4"]}]
```

<Tabs>
<TabItem value="sdk" label="SDK">

```python
from litellm import Router 
router = Router(
  model_list=[
    {
      "model_name": "gpt-3.5-turbo",
      "litellm_params": {
        "model": "azure/<your-deployment-name>",
        "api_base": "<your-azure-endpoint>",
        "api_key": "<your-azure-api-key>",
        "rpm": 6
      }
    },
    {
      "model_name": "gpt-4",
      "litellm_params": {
        "model": "azure/gpt-4-ca",
        "api_base": "https://my-endpoint-canada-berri992.openai.azure.com/",
        "api_key": "<your-azure-api-key>",
        "rpm": 6
      }
    }
  ],
  fallbacks=[{"gpt-3.5-turbo": ["gpt-4"]}] # 👈 KEY CHANGE
)

```

</TabItem>
<TabItem value="proxy" label="PROXY">


```yaml
model_list:
  - model_name: gpt-3.5-turbo
    litellm_params:
      model: azure/<your-deployment-name>
      api_base: <your-azure-endpoint>
      api_key: <your-azure-api-key>
      rpm: 6      # Rate limit for this deployment: in requests per minute (rpm)
  - model_name: gpt-4
    litellm_params:
      model: azure/gpt-4-ca
      api_base: https://my-endpoint-canada-berri992.openai.azure.com/
      api_key: <your-azure-api-key>
      rpm: 6

router_settings:
  fallbacks: [{"gpt-3.5-turbo": ["gpt-4"]}]
```


</TabItem>
</Tabs>


### 2. Start Proxy

```bash
litellm --config /path/to/config.yaml
```

### 3. Test Fallbacks

Pass `mock_testing_fallbacks=true` in request body, to trigger fallbacks.

<Tabs>
<TabItem value="sdk" label="SDK">


```python

from litellm import Router

model_list = [{..}, {..}] # defined in Step 1.

router = Router(model_list=model_list, fallbacks=[{"bad-model": ["my-good-model"]}])

response = router.completion(
  model="bad-model",
  messages=[{"role": "user", "content": "Hey, how's it going?"}],
  mock_testing_fallbacks=True,
)
```

</TabItem>
<TabItem value="proxy" label="PROXY">

```bash
curl -X POST 'http://0.0.0.0:4000/chat/completions' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer sk-1234' \
-d '{
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

</TabItem>
</Tabs>




### Explanation

Fallbacks are done in-order - ["gpt-3.5-turbo, "gpt-4", "gpt-4-32k"], will do 'gpt-3.5-turbo' first, then 'gpt-4', etc.

You can also set [`default_fallbacks`](#default-fallbacks), in case a specific model group is misconfigured / bad.

There are 3 types of fallbacks: 
- `content_policy_fallbacks`: For litellm.ContentPolicyViolationError - LiteLLM maps content policy violation errors across providers [**See Code**](https://github.com/BerriAI/litellm/blob/89a43c872a1e3084519fb9de159bf52f5447c6c4/litellm/utils.py#L8495C27-L8495C54)
- `context_window_fallbacks`: For litellm.ContextWindowExceededErrors - LiteLLM maps context window error messages across providers [**See Code**](https://github.com/BerriAI/litellm/blob/89a43c872a1e3084519fb9de159bf52f5447c6c4/litellm/utils.py#L8469)
- `fallbacks`: For all remaining errors - e.g. litellm.RateLimitError


## Client Side Fallbacks

Set fallbacks in the `.completion()` call for SDK and client-side for proxy. 

In this request the following will occur:
1. The request to `model="zephyr-beta"` will fail
2. litellm proxy will loop through all the model_groups specified in `fallbacks=["gpt-3.5-turbo"]`
3. The request to `model="gpt-3.5-turbo"` will succeed and the client making the request will get a response from gpt-3.5-turbo 

👉 Key Change: `"fallbacks": ["gpt-3.5-turbo"]`

<Tabs>
<TabItem value="sdk" label="SDK">

```python
from litellm import Router

router = Router(model_list=[..]) # defined in Step 1.

resp = router.completion(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hey, how's it going?"}],
    mock_testing_fallbacks=True, # 👈 trigger fallbacks
    fallbacks=[
        {
            "model": "claude-3-haiku",
            "messages": [{"role": "user", "content": "What is LiteLLM?"}],
        }
    ],
)

print(resp)
```

</TabItem>
<TabItem value="proxy" label="PROXY">

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
</TabItem>

</Tabs>

### Control Fallback Prompts  

Pass in messages/temperature/etc. per model in fallback (works for embedding/image generation/etc. as well).

Key Change:

```
fallbacks = [
  {
    "model": <model_name>,
    "messages": <model-specific-messages>
    ... # any other model-specific parameters
  }
]
```

<Tabs>
<TabItem value="sdk" label="SDK">

```python
from litellm import Router

router = Router(model_list=[..]) # defined in Step 1.

resp = router.completion(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hey, how's it going?"}],
    mock_testing_fallbacks=True, # 👈 trigger fallbacks
    fallbacks=[
        {
            "model": "claude-3-haiku",
            "messages": [{"role": "user", "content": "What is LiteLLM?"}],
        }
    ],
)

print(resp)
```

</TabItem>
<TabItem value="proxy" label="PROXY">

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
      "fallbacks": [{
          "model": "claude-3-haiku",
          "messages": [{"role": "user", "content": "What is LiteLLM?"}]
      }]
    }
)

print(response)
```
</TabItem>

<TabItem value="Curl" label="Curl Request">

```bash
curl -L -X POST 'http://0.0.0.0:4000/v1/chat/completions' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer sk-1234' \
-d '{
    "model": "gpt-3.5-turbo",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "Hi, how are you ?"
          }
        ]
      }
    ],
    "fallbacks": [{
        "model": "claude-3-haiku",
        "messages": [{"role": "user", "content": "What is LiteLLM?"}]
    }],
    "mock_testing_fallbacks": true
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
      "fallbacks": [{
          "model": "claude-3-haiku",
          "messages": [{"role": "user", "content": "What is LiteLLM?"}]
      }]
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

</TabItem>
</Tabs>

## Content Policy Violation Fallback

Key change: 

```python
content_policy_fallbacks=[{"claude-2": ["my-fallback-model"]}]
```

<Tabs>
<TabItem value="sdk" label="SDK">

```python
from litellm import Router 

router = Router(
  model_list=[
    {
      "model_name": "claude-2",
      "litellm_params": {
        "model": "claude-2",
        "api_key": "",
        "mock_response": Exception("content filtering policy"),
      },
    },
    {
      "model_name": "my-fallback-model",
      "litellm_params": {
        "model": "claude-2",
        "api_key": "",
        "mock_response": "This works!",
      },
    },
  ],
  content_policy_fallbacks=[{"claude-2": ["my-fallback-model"]}], # 👈 KEY CHANGE
  # fallbacks=[..], # [OPTIONAL]
  # context_window_fallbacks=[..], # [OPTIONAL]
)

response = router.completion(
  model="claude-2",
  messages=[{"role": "user", "content": "Hey, how's it going?"}],
)
```
</TabItem>
<TabItem value="proxy" label="PROXY">

In your proxy config.yaml just add this line 👇

```yaml
router_settings:
  content_policy_fallbacks=[{"claude-2": ["my-fallback-model"]}]
```

Start proxy 

```bash
litellm --config /path/to/config.yaml

# RUNNING on http://0.0.0.0:4000
```

</TabItem>
</Tabs>

## Context Window Exceeded Fallback

Key change: 

```python
context_window_fallbacks=[{"claude-2": ["my-fallback-model"]}]
```

<Tabs>
<TabItem value="sdk" label="SDK">

```python
from litellm import Router 

router = Router(
  model_list=[
    {
      "model_name": "claude-2",
      "litellm_params": {
        "model": "claude-2",
        "api_key": "",
        "mock_response": Exception("prompt is too long"),
      },
    },
    {
      "model_name": "my-fallback-model",
      "litellm_params": {
        "model": "claude-2",
        "api_key": "",
        "mock_response": "This works!",
      },
    },
  ],
  context_window_fallbacks=[{"claude-2": ["my-fallback-model"]}], # 👈 KEY CHANGE
  # fallbacks=[..], # [OPTIONAL]
  # content_policy_fallbacks=[..], # [OPTIONAL]
)

response = router.completion(
  model="claude-2",
  messages=[{"role": "user", "content": "Hey, how's it going?"}],
)
```
</TabItem>
<TabItem value="proxy" label="PROXY">

In your proxy config.yaml just add this line 👇

```yaml
router_settings:
  context_window_fallbacks=[{"claude-2": ["my-fallback-model"]}]
```

Start proxy 

```bash
litellm --config /path/to/config.yaml

# RUNNING on http://0.0.0.0:4000
```

</TabItem>
</Tabs>

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

### Fallback to Specific Model ID

If all models in a group are in cooldown (e.g. rate limited), LiteLLM will fallback to the model with the specific model ID.

This skips any cooldown check for the fallback model.

1. Specify the model ID in `model_info`
```yaml
model_list:
  - model_name: gpt-4
    litellm_params:
      model: openai/gpt-4
    model_info:
      id: my-specific-model-id # 👈 KEY CHANGE
  - model_name: gpt-4
    litellm_params:
      model: azure/chatgpt-v-2
      api_base: os.environ/AZURE_API_BASE
      api_key: os.environ/AZURE_API_KEY
  - model_name: anthropic-claude
    litellm_params:
      model: anthropic/claude-3-opus-20240229
      api_key: os.environ/ANTHROPIC_API_KEY
```

**Note:** This will only fallback to the model with the specific model ID. If you want to fallback to another model group, you can set `fallbacks=[{"gpt-4": ["anthropic-claude"]}]`

2. Set fallbacks in config

```yaml
litellm_settings:
  fallbacks: [{"gpt-4": ["my-specific-model-id"]}]
```

3. Test it!

```bash
curl -X POST 'http://0.0.0.0:4000/chat/completions' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer sk-1234' \
-d '{
  "model": "gpt-4",
  "messages": [
    {
      "role": "user",
      "content": "ping"
    }
  ],
  "mock_testing_fallbacks": true
}'
```

Validate it works, by checking the response header `x-litellm-model-id`

```bash
x-litellm-model-id: my-specific-model-id
```

### Test Fallbacks! 

Check if your fallbacks are working as expected. 

#### **Regular Fallbacks**
```bash
curl -X POST 'http://0.0.0.0:4000/chat/completions' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer sk-1234' \
-d '{
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
-d '{
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
-d '{
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

A model-specific fallbacks (e.g. `{"gpt-3.5-turbo-small": ["claude-opus"]}`) overrides default fallback.

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

### Disable Fallbacks (Per Request/Key)


<Tabs>

<TabItem value="request" label="Per Request">

You can disable fallbacks per key by setting `disable_fallbacks: true` in your request body.

```bash
curl -L -X POST 'http://0.0.0.0:4000/v1/chat/completions' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer sk-1234' \
-d '{
    "messages": [
        {
            "role": "user",
            "content": "List 5 important events in the XIX century"
        }
    ],
    "model": "gpt-3.5-turbo",
    "disable_fallbacks": true # 👈 DISABLE FALLBACKS
}'
```

</TabItem>

<TabItem value="key" label="Per Key">

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

</TabItem>
</Tabs>
