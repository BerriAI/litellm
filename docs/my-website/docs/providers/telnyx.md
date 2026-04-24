# Telnyx

Telnyx provides an OpenAI-compatible Inference API for LLM access to hosted models.

## Overview

| Property | Details |
|-------|--------|
| Provider ID | `telnyx` |
| API Base | `https://api.telnyx.com/v2/ai` |
| API Key Env | `TELNYX_API_KEY` |
| OpenAI Compatible | Yes |
| Website | [telnyx.com](https://telnyx.com) |
| API Docs | [developers.telnyx.com/docs/inference](https://developers.telnyx.com/docs/inference/getting-started) |
| Sign Up | [telnyx.com/sign-up](https://telnyx.com/sign-up) |

## Quick Start

### Installation

```bash
pip install litellm
```

### Usage

```python
import os
import litellm

os.environ["TELNYX_API_KEY"] = "your-api-key"

# Chat completion
response = litellm.completion(
    model="telnyx/moonshotai/Kimi-K2.6",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)

# With streaming
response = litellm.completion(
    model="telnyx/moonshotai/Kimi-K2.6",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True,
)
for chunk in response:
    print(chunk.choices[0].delta.content or "", end="")
```

### Using with OpenAI SDK

Since Telnyx is OpenAI-compatible, you can also use the OpenAI SDK directly:

```python
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["TELNYX_API_KEY"],
    base_url="https://api.telnyx.com/v2/ai/openai",
)

response = client.chat.completions.create(
    model="moonshotai/Kimi-K2.6",
    messages=[{"role": "user", "content": "Hello!"}],
)
```

## Available Models

| Model ID | Parameters | Context Length | Best For |
|----------|-----------|---------------|----------|
| `moonshotai/Kimi-K2.6` | 1.0T | 256K | Highest intelligence, voice AI |
| `zai-org/GLM-5.1-FP8` | 753.9B | 202K | Efficient reasoning, function calling |
| `MiniMaxAI/MiniMax-M2.7` | — | 2M | Cheapest, high intelligence |

See [Telnyx Available Models](https://developers.telnyx.com/docs/inference/models) for the full list.

## Embeddings

```python
response = litellm.embedding(
    model="telnyx/thenlper/gte-large",
    input=["Hello world"],
)
```

## Proxy Server (LiteLLM Proxy)

Add Telnyx to your LiteLLM proxy config:

```yaml
model_list:
  - model_name: kimi-k2.6
    litellm_params:
      model: telnyx/moonshotai/Kimi-K2.6
      api_key: os.environ/TELNYX_API_KEY
```

## Getting an API Key

1. Sign up at [telnyx.com/sign-up](https://telnyx.com/sign-up)
2. Navigate to the [Telnyx Portal](https://portal.telnyx.com/)
3. Create an API key under **Auth > API Keys**
