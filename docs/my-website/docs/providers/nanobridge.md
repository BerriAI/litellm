# Nanobridge

| | |
|---|---|
| **Provider** | `nanobridge` |
| **API key env** | `NANOBRIDGE_API_KEY` |
| **Base override env** | `NANOBRIDGE_API_BASE` (optional) |

Nanobridge is an OpenAI-compatible API gateway for Tencent Hunyuan model routes (DeepSeek, GLM, MiniMax, etc.). It also exposes an Anthropic Messages-compatible base for Claude Code and similar agents.

## Quick Start

```bash
export NANOBRIDGE_API_KEY="your-api-key"
```

```python
import litellm

response = litellm.completion(
    model="nanobridge/deepseek-v4-flash",
    messages=[{"role": "user", "content": "Hello"}],
)
print(response.choices[0].message.content)
```

## Regional endpoints

Three regions share the same API key and billing. Pick the closest node for lower latency.

| Region | OpenAI base (`/v1`) | Anthropic base |
|--------|---------------------|----------------|
| Germany (default) | `https://api.nanobridge.net/v1` | `https://api.nanobridge.net/anthropic` |
| Singapore | `https://api-sg.nanobridge.net/v1` | `https://api-sg.nanobridge.net/anthropic` |
| United States | `https://api-us.nanobridge.net/v1` | `https://api-us.nanobridge.net/anthropic` |

Override the default Germany URL:

```bash
export NANOBRIDGE_API_BASE="https://api-sg.nanobridge.net/v1"
```

## Supported models

Use `nanobridge/<model_id>` in LiteLLM:

| Model ID | Notes |
|----------|--------|
| `deepseek-v4-flash` | Recommended default |
| `deepseek-v4-pro` | |
| `deepseek-v3.2` | |
| `glm-5.1` | Alias: `glm-5-1` |
| `minimax-m2.7` | Alias: `minimax-m-2-7` |

## API paths

- Chat: `POST /chat/completions` or `POST /v1/chat/completions`
- Models: `GET /v1/models`
- Anthropic: `POST /v1/messages` on the Anthropic base
- Streaming: `stream: true` (SSE)
- Auth: `Authorization: Bearer <API_KEY>`

## LiteLLM Proxy

```yaml
model_list:
  - model_name: nanobridge-deepseek-flash
    litellm_params:
      model: nanobridge/deepseek-v4-flash
      api_key: os.environ/NANOBRIDGE_API_KEY
      # api_base: os.environ/NANOBRIDGE_API_BASE
```

## OpenAI Python SDK

```python
from openai import OpenAI
import os

client = OpenAI(
    api_key=os.environ["NANOBRIDGE_API_KEY"],
    base_url="https://api.nanobridge.net/v1",
)
client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[{"role": "user", "content": "Hello"}],
)
```

## Links

- [Nanobridge API documentation](https://platform.nanobridge.net/#/api-docs)
- [Platform console](https://platform.nanobridge.net)
