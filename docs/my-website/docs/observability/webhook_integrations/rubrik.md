# Rubrik

Rubrik is a data security platform. Send LLM logs to Rubrik for data governance and compliance monitoring.
 [Learn more about Rubrik](https://www.rubrik.com/).

:::info Auto-Generated Documentation

This documentation is auto-generated from [`generic_api_compatible_callbacks.json`](https://github.com/BerriAI/litellm/blob/main/litellm/integrations/generic_api/generic_api_compatible_callbacks.json).

:::

## Supported Events

- ✅ **Success events** (`llm_api_success`)

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `RUBRIK_API_KEY` | API key or authentication token | Yes |
| `RUBRIK_WEBHOOK_URL` | Webhook endpoint URL | Yes |

:::tip Setup Note

Configure your Rubrik webhook URL in the Rubrik console to receive LLM usage data.

:::

## Quick Start

### LiteLLM Proxy (config.yaml)

```yaml
model_list:
  - model_name: gpt-4
    litellm_params:
      model: openai/gpt-4
      api_key: os.environ/OPENAI_API_KEY

litellm_settings:
  callbacks: ["rubrik"]

environment_variables:
  RUBRIK_API_KEY: "your-value-here"
  RUBRIK_WEBHOOK_URL: "your-value-here"
```

Start the proxy:

```bash
litellm --config config.yaml
```

Test with a request:

```bash
curl -X POST http://localhost:4000/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -d '{
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### LiteLLM Python SDK

```python
import os
import litellm
from litellm import completion

# Set environment variables
os.environ["RUBRIK_API_KEY"] = "your-value-here"
os.environ["RUBRIK_WEBHOOK_URL"] = "your-value-here"
os.environ["OPENAI_API_KEY"] = "your-openai-key"

# Enable the callback
litellm.success_callback = ["rubrik"]

# Make a request - logs will be sent automatically
response = completion(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response)
```

## Logged Payload

The [LiteLLM Standard Logging Payload](https://docs.litellm.ai/docs/proxy/logging_spec) is sent to your endpoint. This includes:

- Request ID and timestamps
- Model and provider information
- Token usage and cost
- Request/response content (unless redacted)
- Metadata and user information

## Request Headers

The following headers are sent with each request:

| Header | Value |
|--------|-------|
| `Content-Type` | application/json |
| `Authorization` | Bearer `$RUBRIK_API_KEY` |

## See Also

- [Generic API Callback](../generic_api.md) - Custom webhook configuration
- [Logging Spec](../../proxy/logging_spec.md) - Payload format details
- [Contribute Custom Webhook API](../../contribute_integration/custom_webhook_api.md) - Add your own integration
