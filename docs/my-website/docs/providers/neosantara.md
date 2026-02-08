import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Neosantara

## Overview

| Property | Details |
|-------|-------|
| Description | Neosantara is a unified LLM gateway designed for developers in Indonesia, providing a single OpenAI-compatible interface to multiple top-tier AI models (OpenAI, Anthropic, Gemini, etc.). |
| Provider Route on LiteLLM | `neosantara/` |
| Link to Provider Doc | [Neosantara Dashboard ↗](https://app.neosantara.xyz) |
| Base URL | `https://api.neosantara.xyz/v1` |
| Supported Operations | [`/chat/completions`](#sample-usage), [`/embeddings`](#embeddings) |

<br />

## What is Neosantara?

Neosantara is a unified gateway that lets developers:
- **Access Multiple LLM Providers**: Unified interface for OpenAI, Anthropic, Gemini, and more.
- **Optimized for Indonesia**: Designed specifically for the needs of developers in the region.
- **Unified Billing**: Pay-As-You-Go system with local payment support.
- **OpenAI Compatible**: Seamlessly drop into existing OpenAI-based workflows.

## Required Variables

```python showLineNumbers title="Environment Variables"
os.environ["NEOSANTARA_API_KEY"] = "your-neosantara-api-key"
```

Get your Neosantara API key from [app.neosantara.xyz](https://app.neosantara.xyz).

## Usage - LiteLLM Python SDK

<Tabs>
<TabItem value="non-streaming" label="Non-streaming">

```python showLineNumbers title="Neosantara Non-streaming Completion"
import os
import litellm
from litellm import completion

os.environ["NEOSANTARA_API_KEY"] = "your-neosantara-api-key"

messages = [{"content": "What is the capital of Indonesia?", "role": "user"}]

# Neosantara call
response = completion(
    model="neosantara/claude-3-haiku",
    messages=messages
)

print(response)
```

</TabItem>
<TabItem value="streaming" label="Streaming">

```python showLineNumbers title="Neosantara Streaming Completion"
import os
import litellm
from litellm import completion

os.environ["NEOSANTARA_API_KEY"] = "your-neosantara-api-key"

messages = [{"content": "Write a short poem about Jakarta", "role": "user"}]

# Neosantara call with streaming
response = completion(
    model="neosantara/claude-3-haiku",
    messages=messages,
    stream=True
)

for chunk in response:
    print(chunk)
```

</TabItem>
<TabItem value="embeddings" label="Embeddings">

```python showLineNumbers title="Neosantara Embeddings"
import os
import litellm
from litellm import embedding

os.environ["NEOSANTARA_API_KEY"] = "your-neosantara-api-key"

# Neosantara call
response = embedding(
    model="neosantara/nusa-embedding-0001",
    input=["Hello, how are you?"]
)

print(response)
```

</TabItem>
</Tabs>

## Usage - LiteLLM Proxy Server

### 1. Set Neosantara Models on `config.yaml`

```yaml
model_list:
  - model_name: neosantara-claude-3-haiku
    litellm_params:
      model: neosantara/claude-3-haiku
      api_key: os.environ/NEOSANTARA_API_KEY
```

### 2. Start Proxy

```bash
litellm --config config.yaml
```

### 3. Test it

<Tabs>
<TabItem value="Curl" label="Curl Request">

```shell
curl --location 'http://0.0.0.0:4000/chat/completions' \
--header 'Content-Type: application/json' \
--header 'Authorization: Bearer sk-1234' \
--data ' {
      "model": "neosantara-claude-3-haiku",
      "messages": [
        {
          "role": "user",
          "content": "what llm are you"
        }
      ]
    }
'
```
</TabItem>
<TabItem value="openai" label="OpenAI v1.0.0+">

```python
import openai
client = openai.OpenAI(
    api_key="anything",
    base_url="http://0.0.0.0:4000"
)

response = client.chat.completions.create(
    model="neosantara-claude-3-haiku", 
    messages = [
        {
            "role": "user",
            "content": "this is a test request, write a short poem"
        }
    ]
)

print(response)
```
</TabItem>
</Tabs>

## Supported Models

We support a wide range of models optimized for the Indonesian context and high-performance tasks.

| Model Name | Model ID (for LiteLLM) | Provider | Description |
|------------|------------------------|----------|-------------|
| **Nusantara Base** | `neosantara/nusantara-base` | Gemini | Flagship balanced model |
| **Archipelago 70B** | `neosantara/archipelago-70b` | Llama 3.3 | Cultural context awareness |
| **Garda Beta Mini** | `neosantara/garda-beta-mini` | Groq/Paxsenix | Fast & efficient Indonesian understanding |
| **Claude 3 Haiku** | `neosantara/claude-3-haiku` | Bedrock | Near-instant responsiveness |
| **Claude 3 Sonnet** | `neosantara/claude-3-sonnet` | Bedrock | Balance of intelligence and speed |
| **Sahabat AI Llama v4** | `neosantara/sahabat-ai-llama-v4` | SahabatAI | Fine-tuned for Sahabat AI ecosystem |
| **Nusa Embedding 0001**| `neosantara/nusa-embedding-0001` | Embedding | Optimized for Indonesian search |

:::info
**Note:** You can use any model supported by Neosantara by adding the `neosantara/` prefix to the model name in your LiteLLM calls.
:::

## Supported OpenAI Parameters

Neosantara supports all standard OpenAI-compatible parameters:

| Parameter | Type | Description |
|-----------|------|-------------|
| `messages` | array | **Required**. Array of message objects with 'role' and 'content' |
| `model` | string | **Required**. Model ID (e.g., `claude-3-haiku`, `archipelago-70b`) |
| `stream` | boolean | Optional. Enable streaming responses |
| `temperature` | float | Optional. Sampling temperature |
| `top_p` | float | Optional. Nucleus sampling parameter |
| `max_tokens` | integer | Optional. Maximum tokens to generate |
| `tools` | array | Optional. List of available tools/functions |
| `tool_choice` | string/object | Optional. Control tool/function calling |

## Additional Resources

- [Neosantara Dashboard](https://app.neosantara.xyz)
- [API Documentation](https://docs.neosantara.xyz)