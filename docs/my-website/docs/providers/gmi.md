# GMI Cloud

## Overview

| Property | Details |
|-------|-------|
| Description | GMI Cloud is a GPU cloud infrastructure provider offering access to top AI models including Claude, GPT, DeepSeek, Gemini, and more through OpenAI-compatible APIs. |
| Provider Route on LiteLLM | `gmi/` |
| Link to Provider Doc | [GMI Cloud Docs ↗](https://docs.gmicloud.ai) |
| Base URL | `https://api.gmi-serving.com/v1` |
| Supported Operations | [`/chat/completions`](#sample-usage), [`/models`](#supported-models) |

<br />

## What is GMI Cloud?

GMI Cloud is a venture-backed digital infrastructure company ($82M+ funding) providing:
- **Top-tier GPU Access**: NVIDIA H100 GPUs for AI workloads
- **Multiple AI Models**: Claude, GPT, DeepSeek, Gemini, Kimi, Qwen, and more
- **OpenAI-Compatible API**: Drop-in replacement for OpenAI SDK
- **Global Infrastructure**: Data centers in US (Colorado) and APAC (Taiwan)

## Required Variables

```python showLineNumbers title="Environment Variables"
os.environ["GMI_API_KEY"] = ""  # your GMI Cloud API key
```

Get your GMI Cloud API key from [console.gmicloud.ai](https://console.gmicloud.ai).

## Usage - LiteLLM Python SDK

### Non-streaming

```python showLineNumbers title="GMI Cloud Non-streaming Completion"
import os
import litellm
from litellm import completion

os.environ["GMI_API_KEY"] = ""  # your GMI Cloud API key

messages = [{"content": "What is the capital of France?", "role": "user"}]

# GMI Cloud call
response = completion(
    model="gmi/deepseek-ai/DeepSeek-V3.2",
    messages=messages
)

print(response)
```

### Streaming

```python showLineNumbers title="GMI Cloud Streaming Completion"
import os
import litellm
from litellm import completion

os.environ["GMI_API_KEY"] = ""  # your GMI Cloud API key

messages = [{"content": "Write a short poem about AI", "role": "user"}]

# GMI Cloud call with streaming
response = completion(
    model="gmi/anthropic/claude-sonnet-4.5",
    messages=messages,
    stream=True
)

for chunk in response:
    print(chunk)
```

## Usage - LiteLLM Proxy Server

### 1. Save key in your environment

```bash
export GMI_API_KEY=""
```

### 2. Start the proxy

```yaml
model_list:
  - model_name: deepseek-v3
    litellm_params:
      model: gmi/deepseek-ai/DeepSeek-V3.2
      api_key: os.environ/GMI_API_KEY
  - model_name: claude-sonnet
    litellm_params:
      model: gmi/anthropic/claude-sonnet-4.5
      api_key: os.environ/GMI_API_KEY
```

## Supported Models

| Model | Model ID | Context Length |
|-------|----------|----------------|
| Claude Opus 4.5 | `gmi/anthropic/claude-opus-4.5` | 409K |
| Claude Sonnet 4.5 | `gmi/anthropic/claude-sonnet-4.5` | 409K |
| Claude Sonnet 4 | `gmi/anthropic/claude-sonnet-4` | 409K |
| Claude Opus 4 | `gmi/anthropic/claude-opus-4` | 409K |
| GPT-5.2 | `gmi/openai/gpt-5.2` | 409K |
| GPT-5.1 | `gmi/openai/gpt-5.1` | 409K |
| GPT-5 | `gmi/openai/gpt-5` | 409K |
| GPT-4o | `gmi/openai/gpt-4o` | 131K |
| GPT-4o-mini | `gmi/openai/gpt-4o-mini` | 131K |
| DeepSeek V3.2 | `gmi/deepseek-ai/DeepSeek-V3.2` | 163K |
| DeepSeek V3 0324 | `gmi/deepseek-ai/DeepSeek-V3-0324` | 163K |
| Gemini 3 Pro | `gmi/google/gemini-3-pro-preview` | 1M |
| Gemini 3 Flash | `gmi/google/gemini-3-flash-preview` | 1M |
| Kimi K2 Thinking | `gmi/moonshotai/Kimi-K2-Thinking` | 262K |
| MiniMax M2.1 | `gmi/MiniMaxAI/MiniMax-M2.1` | 196K |
| Qwen3-VL 235B | `gmi/Qwen/Qwen3-VL-235B-A22B-Instruct-FP8` | 262K |
| GLM-4.7 | `gmi/zai-org/GLM-4.7-FP8` | 202K |

## Supported OpenAI Parameters

GMI Cloud supports all standard OpenAI-compatible parameters:

| Parameter | Type | Description |
|-----------|------|-------------|
| `messages` | array | **Required**. Array of message objects with 'role' and 'content' |
| `model` | string | **Required**. Model ID from available models |
| `stream` | boolean | Optional. Enable streaming responses |
| `temperature` | float | Optional. Sampling temperature |
| `top_p` | float | Optional. Nucleus sampling parameter |
| `max_tokens` | integer | Optional. Maximum tokens to generate |
| `frequency_penalty` | float | Optional. Penalize frequent tokens |
| `presence_penalty` | float | Optional. Penalize tokens based on presence |
| `stop` | string/array | Optional. Stop sequences |
| `response_format` | object | Optional. JSON mode with `{"type": "json_object"}` |

## Additional Resources

- [GMI Cloud Website](https://www.gmicloud.ai)
- [GMI Cloud Documentation](https://docs.gmicloud.ai)
- [GMI Cloud Console](https://console.gmicloud.ai)
