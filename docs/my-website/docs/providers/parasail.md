import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Parasail

## Overview

| Property | Details |
|-------|-------|
| Description | Parasail provides serverless inference for popular open-source large language models with an OpenAI-compatible API. |
| Provider Route on LiteLLM | `parasail/` |
| Link to Provider Doc | [Parasail Documentation ↗](https://docs.parasail.io/parasail-docs) |
| Base URL | `https://api.parasail.io/v1` |
| Supported Operations | [`/chat/completions`](#non-streaming), [`/responses`](#responses-api) |

<br />
<br />

**We support all Parasail models; set `parasail/` as the prefix when sending requests.**

## Required Variables

```python showLineNumbers title="Environment Variables"
os.environ["PARASAIL_API_KEY"] = ""  # your Parasail API key
```

## Usage - LiteLLM Python SDK

### Non-streaming

```python showLineNumbers title="Parasail Non-streaming Completion"
import os
import litellm
from litellm import completion

os.environ["PARASAIL_API_KEY"] = ""

messages = [{"content": "Hello, how are you?", "role": "user"}]

response = completion(
    model="parasail/parasail-llama-33-70b-fp8",
    messages=messages,
)

print(response)
```

### Streaming

```python showLineNumbers title="Parasail Streaming Completion"
import os
import litellm
from litellm import completion

os.environ["PARASAIL_API_KEY"] = ""

messages = [{"content": "Write a short story about AI", "role": "user"}]

response = completion(
    model="parasail/parasail-llama-33-70b-fp8",
    messages=messages,
    stream=True,
)

for chunk in response:
    print(chunk)
```

### Function Calling

```python showLineNumbers title="Parasail Function Calling"
import os
import litellm
from litellm import completion

os.environ["PARASAIL_API_KEY"] = ""

tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather in a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                },
            },
            "required": ["location"],
        },
    },
}]

messages = [{"role": "user", "content": "What's the weather in Boston?"}]

response = completion(
    model="parasail/parasail-llama-33-70b-fp8",
    messages=messages,
    tools=tools,
    tool_choice="auto",
)

print(response)
```

### Responses API

Parasail does not store server-side response state, so it rejects `store: true`. LiteLLM forces `store=False` on every Responses API call to Parasail; you do not need to set it manually.

```python showLineNumbers title="Parasail Responses API"
import os
import litellm

os.environ["PARASAIL_API_KEY"] = ""

response = litellm.responses(
    model="parasail/parasail-llama-33-70b-fp8",
    input="Summarize the plot of Hamlet in two sentences.",
)

print(response)
```

## Usage - LiteLLM Proxy Server

```yaml showLineNumbers title="config.yaml"
model_list:
  - model_name: parasail-llama-33-70b
    litellm_params:
      model: parasail/parasail-llama-33-70b-fp8
      api_key: os.environ/PARASAIL_API_KEY
  - model_name: parasail-deepseek-r1
    litellm_params:
      model: parasail/parasail-deepseek-r1
      api_key: os.environ/PARASAIL_API_KEY
  - model_name: parasail-mistral-small
    litellm_params:
      model: parasail/parasail-mistral-small-32-24b
      api_key: os.environ/PARASAIL_API_KEY
```

## Custom API Base

**Option 1: Environment variable**

```python showLineNumbers title="Custom API Base via env var"
import os
from litellm import completion

os.environ["PARASAIL_API_BASE"] = "https://custom.parasail.io/v1"
os.environ["PARASAIL_API_KEY"] = ""

response = completion(
    model="parasail/parasail-llama-33-70b-fp8",
    messages=[{"content": "Hello!", "role": "user"}],
)
```

**Option 2: Pass directly**

```python showLineNumbers title="Custom API Base via parameter"
from litellm import completion

response = completion(
    model="parasail/parasail-llama-33-70b-fp8",
    messages=[{"content": "Hello!", "role": "user"}],
    api_base="https://custom.parasail.io/v1",
    api_key="your-api-key",
)
```

## Supported OpenAI Parameters

- `temperature`
- `max_tokens`
- `max_completion_tokens`
- `top_p`
- `frequency_penalty`
- `presence_penalty`
- `stop`
- `n`
- `stream`
- `tools`
- `tool_choice`
- `response_format`
- `seed`
- `user`
- `logit_bias`
- `logprobs`
- `top_logprobs`
