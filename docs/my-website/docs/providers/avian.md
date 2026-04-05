# Avian

## Overview

| Property | Details |
|-------|-------|
| Description | Avian is an AI inference platform providing fast access to top open-source models through an OpenAI-compatible API. |
| Provider Route on LiteLLM | `avian/` |
| Link to Provider Doc | [Avian ↗](https://avian.io) |
| Base URL | `https://api.avian.io/v1` |
| Supported Operations | [`/chat/completions`](#usage---litellm-python-sdk), [`/models`](#supported-models) |

<br />

## Required Variables

```python showLineNumbers title="Environment Variables"
os.environ["AVIAN_API_KEY"] = ""  # your Avian API key
```

Get your Avian API key from [avian.io](https://avian.io).

## Usage - LiteLLM Python SDK

### Non-streaming

```python showLineNumbers title="Avian Non-streaming Completion"
import os
import litellm
from litellm import completion

os.environ["AVIAN_API_KEY"] = ""  # your Avian API key

messages = [{"content": "What is the capital of France?", "role": "user"}]

# Avian call
response = completion(
    model="avian/deepseek/deepseek-v3.2",
    messages=messages
)

print(response)
```

### Streaming

```python showLineNumbers title="Avian Streaming Completion"
import os
import litellm
from litellm import completion

os.environ["AVIAN_API_KEY"] = ""  # your Avian API key

messages = [{"content": "What is the capital of France?", "role": "user"}]

# Avian streaming call
response = completion(
    model="avian/deepseek/deepseek-v3.2",
    messages=messages,
    stream=True
)

for chunk in response:
    print(chunk)
```

## Usage - LiteLLM Proxy Server

Add to your `config.yaml`:

```yaml showLineNumbers title="config.yaml"
model_list:
  - model_name: deepseek-v3
    litellm_params:
      model: avian/deepseek/deepseek-v3.2
      api_key: os.environ/AVIAN_API_KEY
  - model_name: kimi-k2.5
    litellm_params:
      model: avian/moonshotai/kimi-k2.5
      api_key: os.environ/AVIAN_API_KEY
  - model_name: glm-5
    litellm_params:
      model: avian/z-ai/glm-5
      api_key: os.environ/AVIAN_API_KEY
  - model_name: minimax-m2.5
    litellm_params:
      model: avian/minimax/minimax-m2.5
      api_key: os.environ/AVIAN_API_KEY
```

Start the proxy:

```bash
litellm --config config.yaml
```

Send a request:

```bash
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -d '{
    "model": "deepseek-v3",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## Supported Models

| Model | Model ID |
|-------|----------|
| DeepSeek V3.2 | `avian/deepseek/deepseek-v3.2` |
| Kimi K2.5 | `avian/moonshotai/kimi-k2.5` |
| GLM-5 | `avian/z-ai/glm-5` |
| MiniMax M2.5 | `avian/minimax/minimax-m2.5` |

## Supported OpenAI Parameters

Avian supports standard OpenAI chat completion parameters including:

| Parameter | Supported |
|-----------|-----------|
| `temperature` | Yes |
| `top_p` | Yes |
| `max_tokens` / `max_completion_tokens` | Yes |
| `stream` | Yes |
| `stop` | Yes |
| `tools` / `tool_choice` | Yes |
| `response_format` | Yes |
