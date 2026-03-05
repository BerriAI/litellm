import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Token Counting

## Overview

LiteLLM provides exact token counting by calling provider-specific token counting APIs. This gives you accurate token counts before sending requests, helping with cost estimation and context window management.

| Feature | Details |
|---------|---------|
| SDK Method | `litellm.acount_tokens()` |
| Proxy Endpoints | `/v1/messages/count_tokens` (Anthropic format), `/v1/responses/input_tokens` (OpenAI format) |
| Fallback | Local tiktoken-based counting for unsupported providers |

## Supported Providers

| Provider | Token Counting API | Format |
|----------|-------------------|--------|
| OpenAI | [Responses API `/input_tokens`](https://platform.openai.com/docs/api-reference/responses/input-tokens) | OpenAI Responses |
| Anthropic | [Messages `/count_tokens`](https://docs.anthropic.com/en/docs/build-with-claude/token-counting) | Anthropic Messages |
| Vertex AI (Claude) | Vertex AI Partner Models Token Counter | Anthropic Messages |
| Bedrock (Claude) | AWS Bedrock CountTokens API | Anthropic Messages |
| Gemini | Google AI Studio countTokens API | Anthropic Messages |
| Vertex AI (Gemini) | Vertex AI countTokens API | Anthropic Messages |
| Other providers | Local tiktoken fallback | N/A |

## SDK Usage

### Basic Usage

```python
import asyncio
import litellm

async def main():
    # OpenAI
    result = await litellm.acount_tokens(
        model="openai/gpt-4o",
        messages=[{"role": "user", "content": "Hello, how are you?"}],
    )
    print(f"Token count: {result.total_tokens}")
    print(f"Tokenizer: {result.tokenizer_type}")  # "openai_api"

    # Anthropic
    result = await litellm.acount_tokens(
        model="anthropic/claude-3-5-sonnet-20241022",
        messages=[{"role": "user", "content": "Hello, how are you?"}],
    )
    print(f"Token count: {result.total_tokens}")
    print(f"Tokenizer: {result.tokenizer_type}")  # "anthropic_api"

asyncio.run(main())
```

### With Tools and System Message

```python
import asyncio
import litellm

async def main():
    result = await litellm.acount_tokens(
        model="openai/gpt-4o",
        messages=[{"role": "user", "content": "What's the weather in Paris?"}],
        tools=[{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            },
        }],
        system="You are a helpful weather assistant.",
    )
    print(f"Token count (with tools): {result.total_tokens}")

asyncio.run(main())
```

### Response Format

`litellm.acount_tokens()` returns a `TokenCountResponse`:

```python
TokenCountResponse(
    total_tokens=15,           # Token count
    request_model="openai/gpt-4o",  # Model requested
    model_used="gpt-4o",      # Model used for counting
    tokenizer_type="openai_api",    # "openai_api", "anthropic_api", "local_tokenizer"
    original_response={"input_tokens": 15},  # Raw API response
    error=False,               # True if counting failed
    error_message=None,        # Error details if failed
)
```

### Fallback Behavior

If a provider doesn't support a token counting API, or if the API key is missing, `acount_tokens()` automatically falls back to local tiktoken-based counting:

```python
# Unsupported provider → automatic fallback
result = await litellm.acount_tokens(
    model="together_ai/meta-llama/Llama-3-8b-chat-hf",
    messages=[{"role": "user", "content": "Hello"}],
)
print(result.tokenizer_type)  # "local_tokenizer"
```

## Proxy Usage

### OpenAI Format — `/v1/responses/input_tokens`

<Tabs>
<TabItem value="curl" label="curl">

```bash
curl -X POST "http://localhost:4000/v1/responses/input_tokens" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -d '{
    "model": "gpt-4o",
    "input": "Hello, how are you?"
  }'
```

</TabItem>
<TabItem value="python" label="Python (httpx)">

```python
import httpx

response = httpx.post(
    "http://localhost:4000/v1/responses/input_tokens",
    headers={
        "Content-Type": "application/json",
        "Authorization": "Bearer sk-1234"
    },
    json={
        "model": "gpt-4o",
        "input": "Hello, how are you?"
    }
)

print(response.json())
# {"input_tokens": 7}
```

</TabItem>
</Tabs>

**Response:**
```json
{"input_tokens": 7}
```

### Anthropic Format — `/v1/messages/count_tokens`

See [Anthropic Token Counting](./anthropic_count_tokens.md) for full documentation.

```bash
curl -X POST "http://localhost:4000/v1/messages/count_tokens" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -d '{
    "model": "claude-3-5-sonnet-20241022",
    "messages": [
      {"role": "user", "content": "Hello, how are you?"}
    ]
  }'
```

## Proxy Configuration

```yaml
model_list:
  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o
      api_key: os.environ/OPENAI_API_KEY

  - model_name: claude-3-5-sonnet
    litellm_params:
      model: anthropic/claude-3-5-sonnet-20241022
      api_key: os.environ/ANTHROPIC_API_KEY
```
