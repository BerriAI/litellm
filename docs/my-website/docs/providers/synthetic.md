# Synthetic

## Overview

| Property | Details |
|-------|-------|
| Description | Synthetic runs open-source AI models in secure datacenters within the US and EU, with a focus on privacy. They never train on your data and auto-delete API data within 14 days. |
| Provider Route on LiteLLM | `synthetic/` |
| Link to Provider Doc | [Synthetic Website ↗](https://synthetic.new) |
| Base URL | `https://api.synthetic.new/openai/v1` |
| Supported Operations | [`/chat/completions`](#sample-usage) |

<br />

## What is Synthetic?

Synthetic is a privacy-focused AI platform that provides access to open-source LLMs with the following guarantees:
- **Privacy-First**: Data never used for training
- **Secure Hosting**: Models run in secure datacenters in US and EU
- **Auto-Deletion**: API data automatically deleted within 14 days
- **Open Source**: Runs open-source AI models

## Required Variables

```python showLineNumbers title="Environment Variables"
os.environ["SYNTHETIC_API_KEY"] = ""  # your Synthetic API key
```

Get your Synthetic API key from [synthetic.new](https://synthetic.new).

## Usage - LiteLLM Python SDK

### Non-streaming

```python showLineNumbers title="Synthetic Non-streaming Completion"
import os
import litellm
from litellm import completion

os.environ["SYNTHETIC_API_KEY"] = ""  # your Synthetic API key

messages = [{"content": "What is the capital of France?", "role": "user"}]

# Synthetic call
response = completion(
    model="synthetic/model-name",  # Replace with actual model name
    messages=messages
)

print(response)
```

### Streaming

```python showLineNumbers title="Synthetic Streaming Completion"
import os
import litellm
from litellm import completion

os.environ["SYNTHETIC_API_KEY"] = ""  # your Synthetic API key

messages = [{"content": "Write a short poem about AI", "role": "user"}]

# Synthetic call with streaming
response = completion(
    model="synthetic/model-name",  # Replace with actual model name
    messages=messages,
    stream=True
)

for chunk in response:
    print(chunk)
```

## Usage - LiteLLM Proxy Server

### 1. Save key in your environment

```bash
export SYNTHETIC_API_KEY=""
```

### 2. Start the proxy

```yaml
model_list:
  - model_name: synthetic-model
    litellm_params:
      model: synthetic/model-name  # Replace with actual model name
      api_key: os.environ/SYNTHETIC_API_KEY
```

## Supported OpenAI Parameters

Synthetic supports all standard OpenAI-compatible parameters:

| Parameter | Type | Description |
|-----------|------|-------------|
| `messages` | array | **Required**. Array of message objects with 'role' and 'content' |
| `model` | string | **Required**. Model ID |
| `stream` | boolean | Optional. Enable streaming responses |
| `temperature` | float | Optional. Sampling temperature |
| `top_p` | float | Optional. Nucleus sampling parameter |
| `max_tokens` | integer | Optional. Maximum tokens to generate |
| `frequency_penalty` | float | Optional. Penalize frequent tokens |
| `presence_penalty` | float | Optional. Penalize tokens based on presence |
| `stop` | string/array | Optional. Stop sequences |

## Privacy & Security

Synthetic provides enterprise-grade privacy protections:
- Data auto-deleted within 14 days
- No data used for model training
- Secure hosting in US and EU datacenters
- Compliance-friendly architecture

## Additional Resources

- [Synthetic Website](https://synthetic.new)
