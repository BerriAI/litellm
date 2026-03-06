import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Azure Anthropic (Claude via Azure Foundry)

LiteLLM supports Claude models deployed via Microsoft Azure Foundry, including Claude Sonnet 4.5, Claude Haiku 4.5, and Claude Opus 4.1.

## Available Models

Azure Foundry supports the following Claude models:

- `claude-sonnet-4-5` - Anthropic's most capable model for building real-world agents and handling complex, long-horizon tasks
- `claude-haiku-4-5` - Near-frontier performance with the right speed and cost for high-volume use cases
- `claude-opus-4-1` - Industry leader for coding, delivering sustained performance on long-running tasks

| Property | Details |
|-------|-------|
| Description | Claude models deployed via Microsoft Azure Foundry. Uses the same API as Anthropic's Messages API but with Azure authentication. |
| Provider Route on LiteLLM | `azure_ai/` (add this prefix to Claude model names - e.g. `azure_ai/claude-sonnet-4-5`) |
| Provider Doc | [Azure Foundry Claude Models ↗](https://learn.microsoft.com/en-us/azure/ai-services/foundry-models/claude) |
| API Endpoint | `https://<resource-name>.services.ai.azure.com/anthropic/v1/messages` |
| Supported Endpoints | `/chat/completions`, `/anthropic/v1/messages`|

## Key Features

- **Extended thinking**: Enhanced reasoning capabilities for complex tasks
- **Image and text input**: Strong vision capabilities for analyzing charts, graphs, technical diagrams, and reports
- **Code generation**: Advanced thinking with code generation, analysis, and debugging (Claude Sonnet 4.5 and Claude Opus 4.1)
- **Same API as Anthropic**: All request/response transformations are identical to the main Anthropic provider

## Authentication

Azure Anthropic supports two authentication methods:

1. **API Key**: Use the `api-key` header
2. **Azure AD Token**: Use `Authorization: Bearer <token>` header (Microsoft Entra ID)

## API Keys and Configuration

```python
import os

# Option 1: API Key authentication
os.environ["AZURE_API_KEY"] = "your-azure-api-key"
os.environ["AZURE_API_BASE"] = "https://<resource-name>.services.ai.azure.com/anthropic"

# Option 2: Azure AD Token authentication
os.environ["AZURE_AD_TOKEN"] = "your-azure-ad-token"
os.environ["AZURE_API_BASE"] = "https://<resource-name>.services.ai.azure.com/anthropic"

# Optional: Azure AD Token Provider (for automatic token refresh)
os.environ["AZURE_TENANT_ID"] = "your-tenant-id"
os.environ["AZURE_CLIENT_ID"] = "your-client-id"
os.environ["AZURE_CLIENT_SECRET"] = "your-client-secret"
os.environ["AZURE_SCOPE"] = "https://cognitiveservices.azure.com/.default"
```

## Usage - LiteLLM Python SDK

### Basic Completion

```python
from litellm import completion

# Set environment variables
os.environ["AZURE_API_KEY"] = "your-azure-api-key"
os.environ["AZURE_API_BASE"] = "https://<resource-name>.services.ai.azure.com/anthropic"

# Make a completion request
response = completion(
    model="azure_ai/claude-sonnet-4-5",
    messages=[
        {"role": "user", "content": "What are 3 things to visit in Seattle?"}
    ],
    max_tokens=1000,
    temperature=0.7,
)

print(response)
```

### Completion with API Key Parameter

```python
import litellm

response = litellm.completion(
    model="azure_ai/claude-sonnet-4-5",
    api_base="https://<resource-name>.services.ai.azure.com/anthropic",
    api_key="your-azure-api-key",
    messages=[
        {"role": "user", "content": "Hello!"}
    ],
    max_tokens=1000,
)
```

### Completion with Azure AD Token

```python
import litellm

response = litellm.completion(
    model="azure_ai/claude-sonnet-4-5",
    api_base="https://<resource-name>.services.ai.azure.com/anthropic",
    azure_ad_token="your-azure-ad-token",
    messages=[
        {"role": "user", "content": "Hello!"}
    ],
    max_tokens=1000,
)
```

### Streaming

```python
from litellm import completion

response = completion(
    model="azure_ai/claude-sonnet-4-5",
    messages=[
        {"role": "user", "content": "Write a short story"}
    ],
    stream=True,
    max_tokens=1000,
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### Tool Calling

```python
from litellm import completion

response = completion(
    model="azure_ai/claude-sonnet-4-5",
    messages=[
        {"role": "user", "content": "What's the weather in Seattle?"}
    ],
    tools=[
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA"
                        }
                    },
                    "required": ["location"]
                }
            }
        }
    ],
    tool_choice="auto",
    max_tokens=1000,
)

print(response)
```

## Usage - LiteLLM Proxy Server

### 1. Save key in your environment

```bash
export AZURE_API_KEY="your-azure-api-key"
export AZURE_API_BASE="https://<resource-name>.services.ai.azure.com/anthropic"
```

### 2. Configure the proxy

```yaml
model_list:
  - model_name: claude-sonnet-4-5
    litellm_params:
      model: azure_ai/claude-sonnet-4-5
      api_base: https://<resource-name>.services.ai.azure.com/anthropic
      api_key: os.environ/AZURE_API_KEY
```

### 3. Test it

<Tabs>
<TabItem value="curl" label="curl">

```bash
curl --location 'http://0.0.0.0:4000/chat/completions' \
--header 'Content-Type: application/json' \
--data '{
    "model": "claude-sonnet-4-5",
    "messages": [
        {
            "role": "user",
            "content": "Hello!"
        }
    ],
    "max_tokens": 1000
}'
```

</TabItem>
<TabItem value="openai" label="OpenAI Python SDK">

```python
from openai import OpenAI

client = OpenAI(
    api_key="anything",
    base_url="http://0.0.0.0:4000"
)

response = client.chat.completions.create(
    model="claude-sonnet-4-5",
    messages=[
        {"role": "user", "content": "Hello!"}
    ],
    max_tokens=1000
)

print(response)
```

</TabItem>
</Tabs>

## Messages API

Azure Anthropic also supports the native Anthropic Messages API. The endpoint structure is the same as Anthropic's `/v1/messages` API.

### Using Anthropic SDK

```python
from anthropic import Anthropic

client = Anthropic(
    api_key="your-azure-api-key",
    base_url="https://<resource-name>.services.ai.azure.com/anthropic"
)

response = client.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=1000,
    messages=[
        {"role": "user", "content": "Hello, world"}
    ]
)

print(response)
```

### Using LiteLLM Proxy

```bash
curl --request POST \
  --url http://0.0.0.0:4000/anthropic/v1/messages \
  --header 'accept: application/json' \
  --header 'content-type: application/json' \
  --header "Authorization: bearer sk-anything" \
  --data '{
    "model": "claude-sonnet-4-5",
    "max_tokens": 1024,
    "messages": [
        {"role": "user", "content": "Hello, world"}
    ]
}'
```

## Supported OpenAI Parameters

Azure Anthropic supports the same parameters as the main Anthropic provider:

```
"stream",
"stop",
"temperature",
"top_p",
"max_tokens",
"max_completion_tokens",
"tools",
"tool_choice",
"extra_headers",
"parallel_tool_calls",
"response_format",
"user",
"thinking",
"reasoning_effort"
```

:::info

Azure Anthropic API requires `max_tokens` to be passed. LiteLLM automatically passes `max_tokens=4096` when no `max_tokens` are provided.

:::

## Differences from Standard Anthropic Provider

The only difference between Azure Anthropic and the standard Anthropic provider is authentication:

- **Standard Anthropic**: Uses `x-api-key` header
- **Azure Anthropic**: Uses `api-key` header or `Authorization: Bearer <token>` for Azure AD authentication

All other request/response transformations, tool calling, streaming, and feature support are identical.

## API Base URL Format

The API base URL should follow this format:

```
https://<resource-name>.services.ai.azure.com/anthropic
```

LiteLLM will automatically append `/v1/messages` if not already present in the URL.

## Example: Full Configuration

```python
import os
from litellm import completion

# Configure Azure Anthropic
os.environ["AZURE_API_KEY"] = "your-azure-api-key"
os.environ["AZURE_API_BASE"] = "https://my-resource.services.ai.azure.com/anthropic"

# Make a request
response = completion(
    model="azure_ai/claude-sonnet-4-5",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum computing in simple terms."}
    ],
    max_tokens=1000,
    temperature=0.7,
    stream=False,
)

print(response.choices[0].message.content)
```

## Troubleshooting

### Missing API Base Error

If you see an error about missing API base, ensure you've set:

```python
os.environ["AZURE_API_BASE"] = "https://<resource-name>.services.ai.azure.com/anthropic"
```

Or pass it directly:

```python
response = completion(
    model="azure_ai/claude-sonnet-4-5",
    api_base="https://<resource-name>.services.ai.azure.com/anthropic",
    # ...
)
```

### Authentication Errors

- **API Key**: Ensure `AZURE_API_KEY` is set or passed as `api_key` parameter
- **Azure AD Token**: Ensure `AZURE_AD_TOKEN` is set or passed as `azure_ad_token` parameter
- **Token Provider**: For automatic token refresh, configure `AZURE_TENANT_ID`, `AZURE_CLIENT_ID`, and `AZURE_CLIENT_SECRET`

## Related Documentation

- [Anthropic Provider Documentation](./anthropic.md) - For standard Anthropic API usage
- [Azure OpenAI Documentation](./azure.md) - For Azure OpenAI models
- [Azure Authentication Guide](../secret_managers/azure_key_vault.md) - For Azure AD token setup

