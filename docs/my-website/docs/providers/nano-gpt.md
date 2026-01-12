# NanoGPT

## Overview

| Property | Details |
|-------|-------|
| Description | NanoGPT is a pay-per-prompt and subscription based AI service providing instant access to over 200+ powerful AI models with no subscriptions or registration required. |
| Provider Route on LiteLLM | `nano-gpt/` |
| Link to Provider Doc | [NanoGPT Website ↗](https://nano-gpt.com) |
| Base URL | `https://nano-gpt.com/api/v1` |
| Supported Operations | [`/chat/completions`](#sample-usage), [`/completions`](#text-completion), [`/embeddings`](#embeddings) |

<br />

## What is NanoGPT?

NanoGPT is a flexible AI API service that offers:
- **Pay-Per-Prompt Pricing**: No subscriptions, pay only for what you use
- **200+ AI Models**: Access to text, image, and video generation models
- **No Registration Required**: Get started instantly
- **OpenAI-Compatible API**: Easy integration with existing code
- **Streaming Support**: Real-time response streaming
- **Tool Calling**: Support for function calling

## Required Variables

```python showLineNumbers title="Environment Variables"
os.environ["NANOGPT_API_KEY"] = ""  # your NanoGPT API key
```

Get your NanoGPT API key from [nano-gpt.com](https://nano-gpt.com).

## Usage - LiteLLM Python SDK

### Non-streaming

```python showLineNumbers title="NanoGPT Non-streaming Completion"
import os
import litellm
from litellm import completion

os.environ["NANOGPT_API_KEY"] = ""  # your NanoGPT API key

messages = [{"content": "What is the capital of France?", "role": "user"}]

# NanoGPT call
response = completion(
    model="nano-gpt/model-name",  # Replace with actual model name
    messages=messages
)

print(response)
```

### Streaming

```python showLineNumbers title="NanoGPT Streaming Completion"
import os
import litellm
from litellm import completion

os.environ["NANOGPT_API_KEY"] = ""  # your NanoGPT API key

messages = [{"content": "Write a short poem about AI", "role": "user"}]

# NanoGPT call with streaming
response = completion(
    model="nano-gpt/model-name",  # Replace with actual model name
    messages=messages,
    stream=True
)

for chunk in response:
    print(chunk)
```

### Tool Calling

```python showLineNumbers title="NanoGPT Tool Calling"
import os
import litellm

os.environ["NANOGPT_API_KEY"] = ""

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                }
            }
        }
    }
]

response = litellm.completion(
    model="nano-gpt/model-name",
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    tools=tools
)
```

## Usage - LiteLLM Proxy Server

### 1. Save key in your environment

```bash
export NANOGPT_API_KEY=""
```

### 2. Start the proxy

```yaml
model_list:
  - model_name: nano-gpt-model
    litellm_params:
      model: nano-gpt/model-name  # Replace with actual model name
      api_key: os.environ/NANOGPT_API_KEY
```

## Supported OpenAI Parameters

NanoGPT supports all standard OpenAI-compatible parameters:

| Parameter | Type | Description |
|-----------|------|-------------|
| `messages` | array | **Required**. Array of message objects with 'role' and 'content' |
| `model` | string | **Required**. Model ID from 200+ available models |
| `stream` | boolean | Optional. Enable streaming responses |
| `temperature` | float | Optional. Sampling temperature |
| `top_p` | float | Optional. Nucleus sampling parameter |
| `max_tokens` | integer | Optional. Maximum tokens to generate |
| `frequency_penalty` | float | Optional. Penalize frequent tokens |
| `presence_penalty` | float | Optional. Penalize tokens based on presence |
| `stop` | string/array | Optional. Stop sequences |
| `n` | integer | Optional. Number of completions to generate |
| `tools` | array | Optional. List of available tools/functions |
| `tool_choice` | string/object | Optional. Control tool/function calling |
| `response_format` | object | Optional. Response format specification |
| `user` | string | Optional. User identifier |

## Model Categories

NanoGPT provides access to multiple model categories:
- **Text Generation**: 200+ LLMs for chat, completion, and analysis
- **Image Generation**: AI models for creating images
- **Video Generation**: AI models for video creation
- **Embedding Models**: Text embedding models for vector search

## Pricing Model

NanoGPT offers a flexible pricing structure:
- **Pay-Per-Prompt**: No subscription required
- **No Registration**: Get started immediately
- **Transparent Pricing**: Pay only for what you use

## API Documentation

For detailed API documentation, visit [docs.nano-gpt.com](https://docs.nano-gpt.com).

## Additional Resources

- [NanoGPT Website](https://nano-gpt.com)
- [NanoGPT API Documentation](https://nano-gpt.com/api)
- [NanoGPT Model List](https://docs.nano-gpt.com/api-reference/endpoint/models)
