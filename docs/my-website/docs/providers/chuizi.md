# Chuizi.AI

## Overview

| Property | Details |
|-------|-------|
| Description | Chuizi.AI is a unified AI gateway providing access to 100+ models across 16 providers through a single OpenAI-compatible endpoint. |
| Provider Route on LiteLLM | `chuizi/` |
| Link to Provider Doc | [Chuizi.AI Docs ↗](https://chuizi.ai/docs) |
| Base URL | `https://api.chuizi.ai/v1` |
| Supported Operations | [`/chat/completions`](#sample-usage), [`/completions`](#sample-usage), [`/embeddings`](#sample-usage) |

<br />

## Required Variables

```python showLineNumbers title="Environment Variables"
os.environ["CHUIZI_API_KEY"] = ""  # your Chuizi.AI API key
```

Get your API key from [app.chuizi.ai](https://app.chuizi.ai/login).

## Usage - LiteLLM Python SDK

### Non-streaming

```python showLineNumbers title="Chuizi.AI Non-streaming"
import os
import litellm
from litellm import completion

os.environ["CHUIZI_API_KEY"] = ""  # your Chuizi.AI API key

messages = [{"content": "What is the capital of France?", "role": "user"}]

# Claude via Chuizi.AI
response = completion(
    model="chuizi/anthropic/claude-sonnet-4-6",
    messages=messages
)
print(response)
```

### Streaming

```python showLineNumbers title="Chuizi.AI Streaming"
import os
import litellm
from litellm import completion

os.environ["CHUIZI_API_KEY"] = ""

messages = [{"content": "Write a haiku about AI", "role": "user"}]

response = completion(
    model="chuizi/openai/gpt-4.1",
    messages=messages,
    stream=True
)

for chunk in response:
    print(chunk.choices[0].delta.content or "", end="")
```

## Usage - LiteLLM Proxy

### Config

```yaml showLineNumbers title="config.yaml"
model_list:
  - model_name: claude-sonnet
    litellm_params:
      model: chuizi/anthropic/claude-sonnet-4-6
      api_key: os.environ/CHUIZI_API_KEY
  - model_name: gpt-4.1
    litellm_params:
      model: chuizi/openai/gpt-4.1
      api_key: os.environ/CHUIZI_API_KEY
  - model_name: deepseek-chat
    litellm_params:
      model: chuizi/deepseek/deepseek-chat
      api_key: os.environ/CHUIZI_API_KEY
```

### Start Proxy

```bash
litellm --config /path/to/config.yaml
```

## Available Models

Models use `provider/model` naming. Some popular models:

| Model | LiteLLM Route |
|-------|--------------|
| Claude Sonnet 4.6 | `chuizi/anthropic/claude-sonnet-4-6` |
| Claude Opus 4.6 | `chuizi/anthropic/claude-opus-4-6` |
| Claude Haiku 4.5 | `chuizi/anthropic/claude-haiku-4-5` |
| GPT-4.1 | `chuizi/openai/gpt-4.1` |
| o4-mini | `chuizi/openai/o4-mini` |
| Gemini 2.5 Pro | `chuizi/google/gemini-2.5-pro` |
| DeepSeek V3.2 | `chuizi/deepseek/deepseek-chat` |
| DeepSeek R1 | `chuizi/deepseek/deepseek-r1` |

See the full model list at [chuizi.ai/models](https://chuizi.ai/models).
