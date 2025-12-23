# MiniMax

## Overview

| Property | Details |
|-------|-------|
| Description | MiniMax is an AI company providing large language models with OpenAI-compatible API endpoints |
| Provider Route on LiteLLM | `minimax/` |
| Supported Endpoints | `/chat/completions`, `/completions`, `/responses`, `/messages` |
| Link to Provider Doc | [MiniMax API Documentation](https://platform.minimax.io/docs/api-reference/text-openai-api) |

## Quick Start

### LiteLLM Python SDK

```python showLineNumbers title="SDK Usage"
import litellm
import os

# Set environment variables
os.environ["MINIMAX_API_KEY"] = "your-api-key"

# Make a completion request
response = litellm.completion(
    model="minimax/MiniMax-Text-01",
    messages=[{"role": "user", "content": "Hello, how are you?"}]
)

print(response.choices[0].message.content)
```

### LiteLLM Proxy

```yaml showLineNumbers title="proxy_config.yaml"
model_list:
  - model_name: minimax-text
    litellm_params:
      model: minimax/MiniMax-Text-01
      api_key: os.environ/MINIMAX_API_KEY
```

```bash showLineNumbers title="Start Proxy"
litellm --config proxy_config.yaml
```

```python showLineNumbers title="Test Proxy"
import openai

client = openai.OpenAI(
    api_key="your-litellm-api-key",
    base_url="http://localhost:4000"
)

response = client.chat.completions.create(
    model="minimax-text",
    messages=[{"role": "user", "content": "Hello!"}]
)

print(response.choices[0].message.content)
```

## Authentication

Set your MiniMax API key as an environment variable:

```bash
export MINIMAX_API_KEY="your-api-key"
```

Or pass it directly in the request:

```python showLineNumbers title="Direct API Key"
response = litellm.completion(
    model="minimax/MiniMax-Text-01",
    messages=[{"role": "user", "content": "Hello!"}],
    api_key="your-api-key"
)
```

## Supported Models

All MiniMax models are supported.

Use the MiniMax provider prefix: `minimax/<model-name>`

## Streaming

```python showLineNumbers title="Streaming Example"
import litellm

response = litellm.completion(
    model="minimax/MiniMax-Text-01",
    messages=[{"role": "user", "content": "Write a short story"}],
    stream=True
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

## Async Support

```python showLineNumbers title="Async Usage"
import litellm
import asyncio

async def main():
    response = await litellm.acompletion(
        model="minimax/MiniMax-Text-01",
        messages=[{"role": "user", "content": "Hello!"}]
    )
    print(response.choices[0].message.content)

asyncio.run(main())
```

## Important Notes

- **API Base URL**: The default base URL is `https://api.minimax.io/v1` for international users
- **China Users**: For users in China, the base URL is `https://api.minimaxi.com/v1`
- **OpenAI Compatible**: MiniMax follows the OpenAI API format, making it easy to integrate with existing OpenAI-based applications

