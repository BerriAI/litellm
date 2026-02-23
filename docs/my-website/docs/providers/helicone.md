import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Helicone

## Overview

| Property | Details |
|-------|-------|
| Description | Helicone is an AI gateway and observability platform that provides OpenAI-compatible endpoints with advanced monitoring, caching, and analytics capabilities. |
| Provider Route on LiteLLM | `helicone/` |
| Link to Provider Doc | [Helicone Documentation ↗](https://docs.helicone.ai) |
| Base URL | `https://ai-gateway.helicone.ai/` |
| Supported Operations | [`/chat/completions`](#sample-usage), [`/completions`](#text-completion), [`/embeddings`](#embeddings) |

<br />

**We support [ALL models available](https://helicone.ai/models) through Helicone's AI Gateway. Use `helicone/` as a prefix when sending requests.**

## What is Helicone?

Helicone is an open-source observability platform for LLM applications that provides:
- **Request Monitoring**: Track all LLM requests with detailed metrics
- **Caching**: Reduce costs and latency with intelligent caching
- **Rate Limiting**: Control request rates per user/key
- **Cost Tracking**: Monitor spend across models and users
- **Custom Properties**: Tag requests with metadata for filtering and analysis
- **Prompt Management**: Version control for prompts

## Required Variables

```python showLineNumbers title="Environment Variables"
os.environ["HELICONE_API_KEY"] = ""  # your Helicone API key
```

Get your Helicone API key from your [Helicone dashboard](https://helicone.ai).

## Usage - LiteLLM Python SDK

### Non-streaming

```python showLineNumbers title="Helicone Non-streaming Completion"
import os
import litellm
from litellm import completion

os.environ["HELICONE_API_KEY"] = ""  # your Helicone API key

messages = [{"content": "What is the capital of France?", "role": "user"}]

# Helicone call - routes through Helicone gateway to OpenAI
response = completion(
    model="helicone/gpt-4",
    messages=messages
)

print(response)
```

### Streaming

```python showLineNumbers title="Helicone Streaming Completion"
import os
import litellm
from litellm import completion

os.environ["HELICONE_API_KEY"] = ""  # your Helicone API key

messages = [{"content": "Write a short poem about AI", "role": "user"}]

# Helicone call with streaming
response = completion(
    model="helicone/gpt-4",
    messages=messages,
    stream=True
)

for chunk in response:
    print(chunk)
```

### With Metadata (Helicone Custom Properties)

```python showLineNumbers title="Helicone with Custom Properties"
import os
import litellm
from litellm import completion

os.environ["HELICONE_API_KEY"] = ""  # your Helicone API key

response = completion(
    model="helicone/gpt-4o-mini",
    messages=[{"role": "user", "content": "What's the weather like?"}],
    metadata={
        "Helicone-Property-Environment": "production",
        "Helicone-Property-User-Id": "user_123",
        "Helicone-Property-Session-Id": "session_abc"
    }
)

print(response)
```

### Text Completion

```python showLineNumbers title="Helicone Text Completion"
import os
import litellm

os.environ["HELICONE_API_KEY"] = ""  # your Helicone API key

response = litellm.completion(
    model="helicone/gpt-4o-mini",  # text completion model
    prompt="Once upon a time"
)

print(response)
```


## Retry and Fallback Mechanisms

```python
import litellm

litellm.api_base = "https://ai-gateway.helicone.ai/"
litellm.metadata = {
    "Helicone-Retry-Enabled": "true",
    "helicone-retry-num": "3",
    "helicone-retry-factor": "2",
}

response = litellm.completion(
    model="helicone/gpt-4o-mini/openai,claude-3-5-sonnet-20241022/anthropic", # Try OpenAI first, then fallback to Anthropic, then continue with other models,
    messages=[{"role": "user", "content": "Hello"}]
)
```

## Supported OpenAI Parameters

Helicone supports all standard OpenAI-compatible parameters:

| Parameter | Type | Description |
|-----------|------|-------------|
| `messages` | array | **Required**. Array of message objects with 'role' and 'content' |
| `model` | string | **Required**. Model ID (e.g., gpt-4, claude-3-opus, etc.) |
| `stream` | boolean | Optional. Enable streaming responses |
| `temperature` | float | Optional. Sampling temperature |
| `top_p` | float | Optional. Nucleus sampling parameter |
| `max_tokens` | integer | Optional. Maximum tokens to generate |
| `frequency_penalty` | float | Optional. Penalize frequent tokens |
| `presence_penalty` | float | Optional. Penalize tokens based on presence |
| `stop` | string/array | Optional. Stop sequences |
| `n` | integer | Optional. Number of completions to generate |
| `tools` | array | Optional. List of available tools/functions |
| `tool_choice` | string/object | Optional. Control tool/function calling |
| `response_format` | object | Optional. Response format specification |
| `user` | string | Optional. User identifier |

## Helicone-Specific Headers

Pass these as metadata to leverage Helicone features:

| Header | Description |
|--------|-------------|
| `Helicone-Property-*` | Custom properties for filtering (e.g., `Helicone-Property-User-Id`) |
| `Helicone-Cache-Enabled` | Enable caching for this request |
| `Helicone-User-Id` | User identifier for tracking |
| `Helicone-Session-Id` | Session identifier for grouping requests |
| `Helicone-Prompt-Id` | Prompt identifier for versioning |
| `Helicone-Rate-Limit-Policy` | Rate limiting policy name |

Example with headers:

```python showLineNumbers title="Helicone with Custom Headers"
import litellm

response = litellm.completion(
    model="helicone/gpt-4",
    messages=[{"role": "user", "content": "Hello"}],
    metadata={
        "Helicone-Cache-Enabled": "true",
        "Helicone-Property-Environment": "production",
        "Helicone-Property-User-Id": "user_123",
        "Helicone-Session-Id": "session_abc",
        "Helicone-Prompt-Id": "prompt_v1"
    }
)
```

## Advanced Usage

### Using with Different Providers

Helicone acts as a gateway and supports multiple providers:

```python showLineNumbers title="Helicone with Anthropic"
import litellm

# Set both Helicone and Anthropic keys
os.environ["HELICONE_API_KEY"] = "your-helicone-key"

response = litellm.completion(
    model="helicone/claude-3.5-haiku/anthropic",
    messages=[{"role": "user", "content": "Hello"}]
)
```

### Caching

Enable caching to reduce costs and latency:

```python showLineNumbers title="Helicone Caching"
import litellm

response = litellm.completion(
    model="helicone/gpt-4",
    messages=[{"role": "user", "content": "What is 2+2?"}],
    metadata={
        "Helicone-Cache-Enabled": "true"
    }
)

# Subsequent identical requests will be served from cache
response2 = litellm.completion(
    model="helicone/gpt-4",
    messages=[{"role": "user", "content": "What is 2+2?"}],
    metadata={
        "Helicone-Cache-Enabled": "true"
    }
)
```

## Features

### Request Monitoring
- Track all requests with detailed metrics
- View request/response pairs
- Monitor latency and errors
- Filter by custom properties

### Cost Tracking
- Per-model cost tracking
- Per-user cost tracking
- Cost alerts and budgets
- Historical cost analysis

### Rate Limiting
- Per-user rate limits
- Per-API key rate limits
- Custom rate limit policies
- Automatic enforcement

### Analytics
- Request volume trends
- Cost trends
- Latency percentiles
- Error rates

Visit [Helicone Pricing](https://helicone.ai/pricing) for details.

## Additional Resources

- [Helicone Official Documentation](https://docs.helicone.ai)
- [Helicone Dashboard](https://helicone.ai)
- [Helicone GitHub](https://github.com/Helicone/helicone)
- [API Reference](https://docs.helicone.ai/rest/ai-gateway/post-v1-chat-completions)

