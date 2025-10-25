# Gatewayz

LiteLLM supports using Gatewayz as an LLM provider with OpenAI-compatible chat completions.

## Quick Start

```python
import litellm
import os

os.environ["GATEWAYZ_API_KEY"] = "your-api-key"

response = litellm.completion(
    model="gatewayz/your-model-name",
    messages=[{"role": "user", "content": "Hello, how are you?"}]
)
print(response)
```

## Environment Variables

Set the following environment variables:

```bash
export GATEWAYZ_API_KEY="your-api-key"  # Required
export GATEWAYZ_API_BASE="https://api.gatewayz.com"  # Optional, defaults to https://api.gatewayz.com
```

## Usage Examples

### Basic Chat Completion

```python
from litellm import completion

response = completion(
    model="gatewayz/your-model",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ]
)
print(response.choices[0].message.content)
```

### Streaming

```python
from litellm import completion

response = completion(
    model="gatewayz/your-model",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### Async Completion

```python
import asyncio
from litellm import acompletion

async def main():
    response = await acompletion(
        model="gatewayz/your-model",
        messages=[{"role": "user", "content": "Hello!"}]
    )
    print(response)

asyncio.run(main())
```

### Custom API Base

If you're using a custom Gatewayz deployment or endpoint:

```python
from litellm import completion

response = completion(
    model="gatewayz/your-model",
    messages=[{"role": "user", "content": "Hello!"}],
    api_base="https://custom.gatewayz.com"
)
```

## Supported Parameters

Gatewayz supports the following OpenAI-compatible parameters:

- `stream` - Enable streaming responses
- `temperature` - Controls randomness (0.0-2.0)
- `max_tokens` - Maximum tokens to generate
- `top_p` - Nucleus sampling parameter
- `frequency_penalty` - Reduces repetition of tokens
- `presence_penalty` - Reduces repetition of topics
- `stop` - Sequences where generation should stop
- `n` - Number of completions to generate

Example with parameters:

```python
response = completion(
    model="gatewayz/your-model",
    messages=[{"role": "user", "content": "Hello!"}],
    temperature=0.7,
    max_tokens=150,
    top_p=0.9,
    frequency_penalty=0.5,
    presence_penalty=0.5,
    stop=["END"]
)
```

## Response Format

Responses follow the standard LiteLLM format:

```python
{
    "id": "chatcmpl-123",
    "object": "chat.completion",
    "created": 1677652288,
    "model": "gatewayz/your-model",
    "choices": [{
        "index": 0,
        "message": {
            "role": "assistant",
            "content": "Hello! How can I help you today?"
        },
        "finish_reason": "stop"
    }],
    "usage": {
        "prompt_tokens": 9,
        "completion_tokens": 10,
        "total_tokens": 19
    }
}
```

## Error Handling

The provider implements standard error handling:

```python
from litellm import completion
from litellm.exceptions import (
    AuthenticationError,
    RateLimitError,
    APIError
)

try:
    response = completion(
        model="gatewayz/your-model",
        messages=[{"role": "user", "content": "Hello!"}]
    )
except AuthenticationError as e:
    print(f"Authentication failed: {e}")
except RateLimitError as e:
    print(f"Rate limit exceeded: {e}")
except APIError as e:
    print(f"API error: {e}")
```

## Token Accounting

Gatewayz returns usage information including:
- `prompt_tokens` - Number of tokens in the prompt
- `completion_tokens` - Number of tokens in the completion
- `total_tokens` - Total tokens used

Access usage information:

```python
response = completion(
    model="gatewayz/your-model",
    messages=[{"role": "user", "content": "Hello!"}]
)

print(f"Prompt tokens: {response.usage.prompt_tokens}")
print(f"Completion tokens: {response.usage.completion_tokens}")
print(f"Total tokens: {response.usage.total_tokens}")
```

## Notes

- Gatewayz uses an OpenAI-compatible API format
- The provider prefix `gatewayz/` is required when specifying models
- All requests require a valid API key set via `GATEWAYZ_API_KEY` environment variable or passed as `api_key` parameter
- Custom API base URLs are supported via `GATEWAYZ_API_BASE` environment variable or `api_base` parameter

## Support

For Gatewayz-specific questions or issues:
- Contact Gatewayz support
- Check the Gatewayz documentation

For LiteLLM integration issues:
- Open an issue on [GitHub](https://github.com/BerriAI/litellm/issues)
