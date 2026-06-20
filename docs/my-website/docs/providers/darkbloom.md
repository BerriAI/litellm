import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Darkbloom

## Overview

| Property | Details |
|-------|-------|
| Description | Darkbloom provides OpenAI-compatible private inference on verified Apple Silicon providers. |
| Provider Route on LiteLLM | `darkbloom/` |
| Link to Provider Doc | [Darkbloom API Documentation](https://www.darkbloom.dev/) |
| Base URL | `https://api.darkbloom.dev/v1` |
| Supported Operations | [`/chat/completions`](#sample-usage) |

LiteLLM supports Darkbloom models by prefixing the model id from Darkbloom's `/v1/models` endpoint with `darkbloom/`.

## Available Models

| Model | Description | Context Window |
|-------|-------------|----------------|
| `darkbloom/gemma-4-26b` | Gemma 4 26B MoE | 128K tokens |
| `darkbloom/gpt-oss-20b` | GPT-OSS 20B MoE | 128K tokens |

## Pricing

Prices are per 1M tokens from the Darkbloom public pricing page.

| Model | Input | Output |
|-------|-------|--------|
| `darkbloom/gemma-4-26b` | $0.03 | $0.165 |
| `darkbloom/gpt-oss-20b` | $0.015 | $0.07 |

## Required Variables

```python showLineNumbers title="Environment Variables"
os.environ["DARKBLOOM_API_KEY"] = ""  # your Darkbloom API key
```

## Usage - LiteLLM Python SDK

### Non-streaming

```python showLineNumbers title="Darkbloom Non-streaming Completion"
import os
from litellm import completion

os.environ["DARKBLOOM_API_KEY"] = ""  # your Darkbloom API key

response = completion(
    model="darkbloom/gemma-4-26b",
    messages=[{"role": "user", "content": "Explain quantum computing"}],
    max_tokens=1024,
)

print(response)
```

### Streaming

```python showLineNumbers title="Darkbloom Streaming Completion"
import os
from litellm import completion

os.environ["DARKBLOOM_API_KEY"] = ""  # your Darkbloom API key

response = completion(
    model="darkbloom/gemma-4-26b",
    messages=[{"role": "user", "content": "Explain quantum computing"}],
    stream=True,
    max_tokens=1024,
)

for chunk in response:
    print(chunk)
```

## Usage - LiteLLM Proxy Server

```yaml showLineNumbers title="config.yaml"
model_list:
  - model_name: darkbloom-gemma
    litellm_params:
      model: darkbloom/gemma-4-26b
      api_key: os.environ/DARKBLOOM_API_KEY
```

## Custom API Base

```python showLineNumbers title="Custom API Base"
from litellm import completion

response = completion(
    model="darkbloom/gemma-4-26b",
    messages=[{"role": "user", "content": "Hello!"}],
    api_base="https://api.darkbloom.dev/v1",
    api_key="your-api-key",
)
```

## List Models

```bash
curl https://api.darkbloom.dev/v1/models \
  -H "Authorization: Bearer $DARKBLOOM_API_KEY"
```
