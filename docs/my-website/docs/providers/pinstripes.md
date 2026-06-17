import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Pinstripes

## Overview

| Property | Details |
|-------|-------|
| Description | Pinstripes provides fast, cost-effective inference for frontier open-source models with transparent per-token pricing and an OpenAI-compatible API. |
| Provider Route on LiteLLM | `pinstripes/` |
| Link to Provider Doc | [Pinstripes Documentation ↗](https://pinstripes.io) |
| Base URL | `https://pinstripes.io/v1` |
| Supported Operations | [`/chat/completions`](#sample-usage) |

<br />
<br />

**We support ALL Pinstripes models, just set `pinstripes/` as a prefix when sending completion requests**

## Available Models

| Model | Description | Context Window |
|-------|-------------|----------------|
| `pinstripes/ps/glm-4.5-air` | GLM-4.5 Air — reasoning-capable MoE | 128,000 tokens |
| `pinstripes/ps/qwen3.6-35b-a3b` | Qwen3.6 35B A3B — reasoning MoE | 131,072 tokens |
| `pinstripes/ps/qwen3-30b-a3b` | Qwen3 30B A3B — efficient reasoning MoE | 131,072 tokens |
| `pinstripes/ps/qwen3-coder-30b-a3b` | Qwen3 Coder 30B A3B — code-specialized MoE | 131,072 tokens |
| `pinstripes/ps/deepseek-v4-flash` | DeepSeek V4 Flash — fast reasoning model | 163,840 tokens |
| `pinstripes/ps/minimax-m2.7` | MiniMax M2.7 — 1M context window model | 1,000,192 tokens |

## Required Variables

```python showLineNumbers title="Environment Variables"
os.environ["PINSTRIPES_API_KEY"] = ""  # your Pinstripes API key
```

## Usage - LiteLLM Python SDK

### Non-streaming

```python showLineNumbers title="Pinstripes Non-streaming Completion"
import os
import litellm
from litellm import completion

os.environ["PINSTRIPES_API_KEY"] = ""  # your Pinstripes API key

messages = [{"content": "Hello, how are you?", "role": "user"}]

# Pinstripes call
response = completion(
    model="pinstripes/ps/glm-4.5-air",
    messages=messages
)

print(response)
```

### Streaming

```python showLineNumbers title="Pinstripes Streaming Completion"
import os
import litellm
from litellm import completion

os.environ["PINSTRIPES_API_KEY"] = ""  # your Pinstripes API key

messages = [{"content": "Write a short story about AI", "role": "user"}]

# Pinstripes call with streaming
response = completion(
    model="pinstripes/ps/glm-4.5-air",
    messages=messages,
    stream=True
)

for chunk in response:
    print(chunk)
```

### Function Calling

```python showLineNumbers title="Pinstripes Function Calling"
import os
import litellm
from litellm import completion

os.environ["PINSTRIPES_API_KEY"] = ""  # your Pinstripes API key

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
                    "description": "The city and state, e.g. San Francisco, CA"
                }
            },
            "required": ["location"]
        }
    }
}]

messages = [{"role": "user", "content": "What's the weather in Boston?"}]

response = completion(
    model="pinstripes/ps/glm-4.5-air",
    messages=messages,
    tools=tools,
    tool_choice="auto"
)

print(response)
```

## Usage - LiteLLM Proxy Server

```yaml showLineNumbers title="config.yaml"
model_list:
  - model_name: glm-4.5-air
    litellm_params:
      model: pinstripes/ps/glm-4.5-air
      api_key: os.environ/PINSTRIPES_API_KEY
  - model_name: qwen3-35b
    litellm_params:
      model: pinstripes/ps/qwen3.6-35b-a3b
      api_key: os.environ/PINSTRIPES_API_KEY
  - model_name: deepseek-v4-flash
    litellm_params:
      model: pinstripes/ps/deepseek-v4-flash
      api_key: os.environ/PINSTRIPES_API_KEY
  - model_name: minimax-m2
    litellm_params:
      model: pinstripes/ps/minimax-m2.7
      api_key: os.environ/PINSTRIPES_API_KEY
```

## Custom API Base

**Option 1: Environment variable**

```python showLineNumbers title="Custom API Base via env var"
import os
from litellm import completion

os.environ["PINSTRIPES_API_BASE"] = "https://custom.pinstripes.io/v1"
os.environ["PINSTRIPES_API_KEY"] = ""  # your API key

response = completion(
    model="pinstripes/ps/glm-4.5-air",
    messages=[{"content": "Hello!", "role": "user"}],
)
```

**Option 2: Pass directly**

```python showLineNumbers title="Custom API Base via parameter"
from litellm import completion

response = completion(
    model="pinstripes/ps/glm-4.5-air",
    messages=[{"content": "Hello!", "role": "user"}],
    api_base="https://custom.pinstripes.io/v1",
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
