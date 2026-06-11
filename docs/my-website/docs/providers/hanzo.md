import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Hanzo

## Overview

| Property | Details |
|-------|-------|
| Description | Hanzo Cloud is an OpenAI-compatible AI gateway that serves Hanzo's native Zen model family alongside 444 models across 56 underlying providers through a unified `/v1/chat/completions` surface. |
| Provider Route on LiteLLM | `hanzo/` |
| Link to Provider Doc | [Hanzo API Documentation ↗](https://api.hanzo.ai) |
| Base URL | `https://api.hanzo.ai/v1` |
| Supported Operations | [`/chat/completions`](#sample-usage) |

<br />
<br />

**This entry covers Hanzo's native Zen family. Other upstream models reachable through the Hanzo gateway can be invoked by passing the upstream model id under the `hanzo/` prefix.**

## Available Models

| Model | Description | Context Window |
|-------|-------------|----------------|
| `hanzo/zen4` | Hanzo Zen4 chat model | 200,000 tokens |
| `hanzo/zen4-max` | Hanzo Zen4 Max — extended-reasoning long-context | 1,000,000 tokens |

## Required Variables

```python showLineNumbers title="Environment Variables"
os.environ["HANZO_API_KEY"] = ""  # your Hanzo API key
```

## Usage - LiteLLM Python SDK

### Non-streaming

```python showLineNumbers title="Hanzo Non-streaming Completion"
import os
import litellm
from litellm import completion

os.environ["HANZO_API_KEY"] = ""  # your Hanzo API key

messages = [{"content": "Hello, how are you?", "role": "user"}]

# Hanzo call
response = completion(
    model="hanzo/zen4",
    messages=messages
)

print(response)
```

### Streaming

```python showLineNumbers title="Hanzo Streaming Completion"
import os
import litellm
from litellm import completion

os.environ["HANZO_API_KEY"] = ""  # your Hanzo API key

messages = [{"content": "Write a short story about AI", "role": "user"}]

# Hanzo call with streaming
response = completion(
    model="hanzo/zen4-max",
    messages=messages,
    stream=True
)

for chunk in response:
    print(chunk)
```

### Function Calling

```python showLineNumbers title="Hanzo Function Calling"
import os
import litellm
from litellm import completion

os.environ["HANZO_API_KEY"] = ""  # your Hanzo API key

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
    model="hanzo/zen4",
    messages=messages,
    tools=tools,
    tool_choice="auto"
)

print(response)
```

## Usage - LiteLLM Proxy Server

```yaml showLineNumbers title="config.yaml"
model_list:
  - model_name: zen4
    litellm_params:
      model: hanzo/zen4
      api_key: os.environ/HANZO_API_KEY
  - model_name: zen4-max
    litellm_params:
      model: hanzo/zen4-max
      api_key: os.environ/HANZO_API_KEY
```

## Custom API Base

**Option 1: Environment variable**

```python showLineNumbers title="Custom API Base via env var"
import os
from litellm import completion

os.environ["HANZO_API_BASE"] = "https://custom.hanzo.example/v1"
os.environ["HANZO_API_KEY"] = ""  # your API key

response = completion(
    model="hanzo/zen4",
    messages=[{"content": "Hello!", "role": "user"}],
)
```

**Option 2: Pass directly**

```python showLineNumbers title="Custom API Base via parameter"
from litellm import completion

response = completion(
    model="hanzo/zen4",
    messages=[{"content": "Hello!", "role": "user"}],
    api_base="https://custom.hanzo.example/v1",
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
