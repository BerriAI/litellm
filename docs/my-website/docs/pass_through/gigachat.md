# GigaChat Passthrough

Pass-through endpoints for direct GigaChat API access via LiteLLM Proxy.

## Overview

| Feature | Supported | Notes |
|-------|-------|-------|
| Cost Tracking | ✅ | Works with proxy cost metadata and router models |
| Logging | ✅ | Logs requests and responses across LiteLLM integrations |
| Streaming | ✅ | Supported for streaming GigaChat chat completions |

## When to use this

- Use the native LiteLLM GigaChat provider for standard chat and embedding calls when possible.
- Use `/gigachat` passthrough when you need provider-specific GigaChat endpoints or raw GigaChat request shapes.
- This is useful for newer or less common GigaChat API endpoints that LiteLLM does not yet expose natively.

## How it works

Any path under `/gigachat` is treated as a provider-specific route and routed through LiteLLM's GigaChat passthrough path.
The proxy accepts the same request body shape as GigaChat and forwards it to the GigaChat backend.

### Proxy base URL mapping

| Original GigaChat URL | Proxy URL |
|-----------------------|-----------|
| `https://gigachat.devices.sberbank.ru/api/v1` | `http://0.0.0.0:4000/gigachat/api/v1` |

## Request format

The proxy requires a `model` field in the request body. For GigaChat passthrough, use the LightLLM model prefix format such as `gigachat/GigaChat-2-Max`.

### Example: Chat completion

```bash
curl --request POST \
  --url http://0.0.0.0:4000/gigachat/api/v1/chat/completions \
  --header 'accept: application/json' \
  --header 'content-type: application/json' \
  --header 'x-api-key: $LITELLM_API_KEY' \
  --data '{
    "model": "gigachat/GigaChat-2-Max",
    "messages": [
      {"role": "user", "content": "Hello, world"}
    ]
  }'
```

### Python example

```python
import requests
import os

response = requests.post(
    "http://0.0.0.0:4000/gigachat/api/v1/chat/completions",
    headers={
        "Content-Type": "application/json",
        "x-api-key": os.environ["LITELLM_API_KEY"],
    },
    json={
        "model": "gigachat/GigaChat-2-Max",
        "messages": [
            {"role": "user", "content": "Hello, world"}
        ],
    },
)
print(response.json())
```

## Authentication

- Authenticate to the proxy with `x-api-key: $LITELLM_API_KEY` or `Authorization: Bearer $LITELLM_API_KEY`.
- The proxy then uses the configured GigaChat credentials to authenticate with the upstream GigaChat API.

## Notes

- GigaChat uses OAuth-style credentials. Configure your GigaChat credentials in LiteLLM using `GIGACHAT_CREDENTIALS` or `GIGACHAT_API_KEY` as described in the main GigaChat provider docs.
- The proxy automatically handles GigaChat's self-signed SSL setup when forwarding requests, so you do not need to disable SSL verification from the client side.
- The `model` field is required for passthrough requests.

## Advanced

### Use with router-backed GigaChat models

If you define router models in `config.yaml`, you can use the passthrough endpoint with a router-backed GigaChat model:

```bash
curl --request POST \
  --url http://0.0.0.0:4000/gigachat/api/v1/chat/completions \
  --header 'Content-Type: application/json' \
  --header 'x-api-key: $LITELLM_API_KEY' \
  --data '{
    "model": "gigachat/GigaChat-2-Max",
    "messages": [
      {"role": "user", "content": "Hello, world"}
    ]
  }'
```

### Sending metadata

You can attach LiteLLM metadata for cost tracking and tags using `litellm_metadata` in the request body:

```bash
curl --request POST \
  --url http://0.0.0.0:4000/gigachat/api/v1/chat/completions \
  --header 'accept: application/json' \
  --header 'content-type: application/json' \
  --header 'x-api-key: $LITELLM_API_KEY' \
  --data '{
    "model": "gigachat/GigaChat-2-Max",
    "messages": [
      {"role": "user", "content": "Hello, world"}
    ],
    "litellm_metadata": {
      "tags": ["test-tag"],
      "user": "test-user"
    }
  }'
```
