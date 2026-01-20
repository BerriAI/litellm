# Qwen AI (OAuth)

Use Qwen OAuth device flow to access Qwen models through LiteLLM.

| Property | Details |
|-------|-------|
| Description | Qwen OAuth access via the Qwen portal with OpenAI-compatible endpoints |
| Provider Route on LiteLLM | `qwen_ai/` |
| Supported Endpoints | `/v1/chat/completions` |
| API Reference | https://qwen.ai |

This provider is OAuth-only. For API key access, use the DashScope provider: /docs/providers/dashscope.

## Free Tier (Qwen OAuth)

Qwen offers a free tier for OAuth users with **2,000 requests/day**, **60 requests/minute**, and **no token limits**.
For details, see https://aiengineerguide.com/blog/qwen-code-cli-free-tier/.

## Authentication

Qwen OAuth uses a device code flow:

1. LiteLLM prints a device authorization URL
2. Open the URL, sign in, and approve the request
3. Tokens are cached locally for reuse

## Supported Models (OAuth)

- `qwen_ai/coder-model`
- `qwen_ai/qwen3-coder-plus`
- `qwen_ai/qwen3-coder-flash`
- `qwen_ai/vision-model`

> Note: Embeddings are not currently supported for the `qwen_ai` provider. For API key access (including embeddings), use the `dashscope/` provider: /docs/providers/dashscope.

## Usage - LiteLLM Python SDK

### Chat Completions

```python showLineNumbers title="Qwen AI Chat"
import litellm

response = litellm.completion(
    model="qwen_ai/qwen3-coder-plus",
    messages=[{"role": "user", "content": "Write a short hello world."}],
)

print(response)
```

### Vision

```python showLineNumbers title="Qwen AI Vision"
import litellm

response = litellm.completion(
    model="qwen_ai/vision-model",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe the image in one sentence."},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3c/Shaki_waterfall.jpg/320px-Shaki_waterfall.jpg"
                    },
                },
            ],
        }
    ],
)

print(response)
```

## Usage - LiteLLM Proxy

```yaml showLineNumbers title="config.yaml"
model_list:
  - model_name: qwen_ai/qwen3-coder-plus
    litellm_params:
      model: qwen_ai/qwen3-coder-plus
  - model_name: qwen_ai/vision-model
    litellm_params:
      model: qwen_ai/vision-model
```

```bash showLineNumbers title="Start LiteLLM Proxy"
litellm --config config.yaml
```

## Configuration

### Environment Variables

- `QWEN_TOKEN_DIR`: Custom token storage directory
- `QWEN_AUTH_FILE`: Auth file name (default: `oauth_creds.json`)
- `QWEN_API_BASE`: Override the API base URL (default: `https://dashscope.aliyuncs.com/compatible-mode/v1`)
