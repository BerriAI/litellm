# Poe

## Overview

| Property | Details |
|-------|-------|
| Description | Poe is Quora's AI platform that provides access to more than 100 models across text, image, video, and voice modalities through a developer-friendly API. |
| Provider Route on LiteLLM | `poe/` |
| Link to Provider Doc | [Poe Website ↗](https://poe.com) |
| Base URL | `https://api.poe.com/v1` |
| Supported Operations | [`/chat/completions`](#sample-usage) |

<br />

## What is Poe?

Poe is Quora's comprehensive AI platform that offers:
- **100+ Models**: Access to a wide variety of AI models
- **Multiple Modalities**: Text, image, video, and voice AI
- **Popular Models**: Including OpenAI's GPT series and Anthropic's Claude
- **Developer API**: Easy integration for applications
- **Extensive Reach**: Benefits from Quora's 400M monthly unique visitors

## Required Variables

```python showLineNumbers title="Environment Variables"
os.environ["POE_API_KEY"] = ""  # your Poe API key
```

Get your Poe API key from the [Poe platform](https://poe.com).

## Usage - LiteLLM Python SDK

### Non-streaming

```python showLineNumbers title="Poe Non-streaming Completion"
import os
import litellm
from litellm import completion

os.environ["POE_API_KEY"] = ""  # your Poe API key

messages = [{"content": "What is the capital of France?", "role": "user"}]

# Poe call
response = completion(
    model="poe/model-name",  # Replace with actual model name
    messages=messages
)

print(response)
```

### Streaming

```python showLineNumbers title="Poe Streaming Completion"
import os
import litellm
from litellm import completion

os.environ["POE_API_KEY"] = ""  # your Poe API key

messages = [{"content": "Write a short poem about AI", "role": "user"}]

# Poe call with streaming
response = completion(
    model="poe/model-name",  # Replace with actual model name
    messages=messages,
    stream=True
)

for chunk in response:
    print(chunk)
```

## Usage - LiteLLM Proxy Server

### 1. Save key in your environment

```bash
export POE_API_KEY=""
```

### 2. Start the proxy

```yaml
model_list:
  - model_name: poe-model
    litellm_params:
      model: poe/model-name  # Replace with actual model name
      api_key: os.environ/POE_API_KEY
```

## Supported OpenAI Parameters

Poe supports all standard OpenAI-compatible parameters:

| Parameter | Type | Description |
|-----------|------|-------------|
| `messages` | array | **Required**. Array of message objects with 'role' and 'content' |
| `model` | string | **Required**. Model ID from 100+ available models |
| `stream` | boolean | Optional. Enable streaming responses |
| `temperature` | float | Optional. Sampling temperature |
| `top_p` | float | Optional. Nucleus sampling parameter |
| `max_tokens` | integer | Optional. Maximum tokens to generate |
| `frequency_penalty` | float | Optional. Penalize frequent tokens |
| `presence_penalty` | float | Optional. Penalize tokens based on presence |
| `stop` | string/array | Optional. Stop sequences |
| `tools` | array | Optional. List of available tools/functions |
| `tool_choice` | string/object | Optional. Control tool/function calling |
| `response_format` | object | Optional. Response format specification |
| `user` | string | Optional. User identifier |

## Available Model Categories

Poe provides access to models across multiple providers:
- **OpenAI Models**: Including GPT-4, GPT-4 Turbo, GPT-3.5 Turbo
- **Anthropic Models**: Including Claude 3 Opus, Sonnet, Haiku
- **Other Popular Models**: Various provider models available
- **Multi-Modal**: Text, image, video, and voice models

## Platform Benefits

Using Poe through LiteLLM offers several advantages:
- **Unified Access**: Single API for many different models
- **Quora Integration**: Access to large user base and content ecosystem
- **Content Sharing**: Capabilities to share model outputs with followers
- **Content Distribution**: Best AI content distributed to all users
- **Model Discovery**: Efficient way to explore new AI models

## Developer Resources

Poe is actively building developer features and welcomes early access requests for API integration.

## Additional Resources

- [Poe Website](https://poe.com)
- [Poe AI Quora Space](https://poeai.quora.com)
- [Quora Blog Post about Poe](https://quorablog.quora.com/Poe)
