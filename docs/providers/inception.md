import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Inception

## Overview

| Property | Details |
|-------|-------|
| Description | Inception serves the Mercury family of diffusion LLMs (dLLMs). The API is OpenAI-compatible. |
| Provider Route on LiteLLM | `inception/` (chat), `text-completion-inception/` (fill-in-the-middle) |
| Link to Provider Doc | [Inception Platform Documentation ↗](https://docs.inceptionlabs.ai/) |
| Base URL | `https://api.inceptionlabs.ai/v1` |
| Supported Operations | [`/chat/completions`](#usage---litellm-python-sdk), [`/fim/completions`](#fill-in-the-middle-fim) |

<br />
<br />

## Available Models

| Model | Description | Context Window |
|-------|-------------|----------------|
| `inception/mercury-2` | Fast reasoning chat model; supports tool calling and structured outputs | 128,000 tokens |
| `text-completion-inception/mercury-edit-2` | Code model for fill-in-the-middle (FIM) autocomplete | 32,000 tokens |

## Required Variables

```python showLineNumbers title="Environment Variables"
os.environ["INCEPTION_API_KEY"] = ""  # your Inception API key
```

## Usage - LiteLLM Python SDK

### Non-streaming

```python showLineNumbers title="Inception Non-streaming Completion"
import os
import litellm
from litellm import completion

os.environ["INCEPTION_API_KEY"] = ""  # your Inception API key

messages = [{"content": "Hello, how are you?", "role": "user"}]

# Inception call
response = completion(
    model="inception/mercury-2",
    messages=messages
)

print(response)
```

### Streaming

```python showLineNumbers title="Inception Streaming Completion"
import os
import litellm
from litellm import completion

os.environ["INCEPTION_API_KEY"] = ""  # your Inception API key

messages = [{"content": "Write a short story about AI", "role": "user"}]

# Inception call with streaming
response = completion(
    model="inception/mercury-2",
    messages=messages,
    stream=True
)

for chunk in response:
    print(chunk)
```

### Reasoning Effort and Reasoning Summary

Mercury exposes a `reasoning_effort` control with an Inception-specific `instant` value for near real-time responses, alongside the standard `low`, `medium`, and `high`. Set `reasoning_summary=True` to receive a summary of the model's reasoning on the response.

```python showLineNumbers title="Inception Reasoning"
import os
from litellm import completion

os.environ["INCEPTION_API_KEY"] = ""  # your Inception API key

response = completion(
    model="inception/mercury-2",
    messages=[{"role": "user", "content": "If a bat and ball cost $1.10 and the bat is $1 more than the ball, how much is the ball?"}],
    reasoning_effort="high",
    reasoning_summary=True,
)

print(response.choices[0].message.content)
print(response.reasoning_summary)  # {"content": "...", "status": "complete"}
```

### Function Calling

```python showLineNumbers title="Inception Function Calling"
import os
from litellm import completion

os.environ["INCEPTION_API_KEY"] = ""  # your Inception API key

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
    model="inception/mercury-2",
    messages=messages,
    tools=tools,
    tool_choice="auto"
)

print(response)
```

### Fill-in-the-Middle (FIM)

`mercury-edit-2` provides code autocomplete through Inception's `/v1/fim/completions` endpoint. Use `text_completion` with the `text-completion-inception/` route and pass a `prompt` (prefix) plus an optional `suffix`.

```python showLineNumbers title="Inception FIM"
import os
from litellm import text_completion

os.environ["INCEPTION_API_KEY"] = ""  # your Inception API key

response = text_completion(
    model="text-completion-inception/mercury-edit-2",
    prompt="def add(a, b):\n    return ",
    suffix="\n",
    max_tokens=64,
)

print(response.choices[0].text)
```

## Usage - LiteLLM Proxy Server

```yaml showLineNumbers title="config.yaml"
model_list:
  - model_name: mercury-2
    litellm_params:
      model: inception/mercury-2
      api_key: os.environ/INCEPTION_API_KEY
  - model_name: mercury-edit-2
    litellm_params:
      model: text-completion-inception/mercury-edit-2
      api_key: os.environ/INCEPTION_API_KEY
```

## Supported OpenAI Parameters

- `max_tokens`
- `max_completion_tokens`
- `temperature`
- `stop`
- `tools`
- `tool_choice`
- `stream`
- `stream_options`
- `response_format`

## Inception-specific Parameters

These are passed through to the Inception chat API:

- `reasoning_effort` (`instant` | `low` | `medium` | `high`)
- `reasoning_summary` (bool) — return a summary of the model's reasoning
- `reasoning_summary_wait` (bool) — wait for the summary to complete before returning
- `diffusing` (bool) — stream intermediate denoising steps
- `realtime` (bool) — optimize for lowest latency
