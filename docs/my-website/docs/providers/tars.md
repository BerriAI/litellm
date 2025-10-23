# TARS (Tetrate Agent Router Service)

https://router.tetrate.ai

TARS is an AI Gateway-as-a-Service from Tetrate that provides intelligent routing for GenAI applications. It's OpenAI-compatible and routes to multiple LLM providers.

## Quick Start

```python
import litellm
import os

# Set your TARS API key
os.environ["TARS_API_KEY"] = "your-tars-api-key"

# Chat Completions
response = litellm.completion(
    model="tars/claude-sonnet-4-20250514",
    messages=[{"role": "user", "content": "Hello, how are you?"}]
)
print(response.choices[0].message.content)

# Embeddings
response = litellm.embedding(
    model="tars/text-embedding-3-small",
    input=["Hello world"]
)
print(response.data[0].embedding)
```

## Features

TARS supports:

- ✅ Chat Completions
- ✅ Embeddings
- ✅ Streaming
- ✅ Async calls
- ✅ Function/Tool calling
- ✅ Vision models

## API Configuration

### Required Environment Variables

```bash
export TARS_API_KEY="your-tars-api-key"
```

### Optional Configuration

```bash
# Override the default API base URL
export TARS_API_BASE="https://api.router.tetrate.ai/v1"
```

## Supported Models

TARS provides access to models from multiple providers including:

- OpenAI (GPT-4o, GPT-4.1, GPT-5, O1, O3, etc.)
- Anthropic (Claude 4, Claude 3.7 Sonnet, Claude 3.5 Haiku, etc.)
- xAI (Grok 4, Grok 3, etc.)
- Google (Gemini 2.5 Pro, Gemini 2.0 Flash, etc.)
- DeepSeek, Qwen, and many more

To see the full list of available models, visit: https://api.router.tetrate.ai/v1/models

## Usage Examples

### Chat Completions

```python
import litellm

response = litellm.completion(
    model="tars/claude-sonnet-4-20250514",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum computing"}
    ],
    temperature=0.7,
    max_tokens=500
)
print(response.choices[0].message.content)
```

### Streaming

```python
import litellm

response = litellm.completion(
    model="tars/gpt-4o",
    messages=[{"role": "user", "content": "Write a short poem"}],
    stream=True
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### Async Chat Completions

```python
import litellm
import asyncio

async def test_async():
    response = await litellm.acompletion(
        model="tars/claude-sonnet-4-20250514",
        messages=[{"role": "user", "content": "Hello!"}]
    )
    print(response.choices[0].message.content)

asyncio.run(test_async())
```

### Function/Tool Calling

```python
import litellm

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"]
            }
        }
    }
]

response = litellm.completion(
    model="tars/gpt-4o",
    messages=[{"role": "user", "content": "What's the weather in SF?"}],
    tools=tools
)
print(response.choices[0].message.tool_calls)
```

### Embeddings

```python
import litellm

response = litellm.embedding(
    model="tars/text-embedding-3-large",
    input=["Hello world", "Goodbye world"]
)

for embedding in response.data:
    print(f"Embedding {embedding.index}: {len(embedding.embedding)} dimensions")
```

### Vision Models

```python
import litellm

response = litellm.completion(
    model="tars/gpt-4o",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/image.jpg"}
                }
            ]
        }
    ]
)
print(response.choices[0].message.content)
```

## Platform Features

TARS provides intelligent routing and platform features that enable:

- Intelligent routing and load balancing
- Automatic fallback to alternative models
- Cost optimization features
- Performance monitoring
- Unified API access

For detailed pricing and platform information, see: https://router.tetrate.ai/models

## Getting Your API Key

1. Sign up at https://router.tetrate.ai
2. Get $5 free credit with a business email
3. Generate your API key from the dashboard
4. Set the `TARS_API_KEY` environment variable

## LiteLLM Proxy Configuration

Add TARS to your `litellm_config.yaml`:

```yaml
model_list:
  - model_name: claude-4
    litellm_params:
      model: tars/claude-sonnet-4-20250514
      api_key: os.environ/TARS_API_KEY

  - model_name: gpt-4o
    litellm_params:
      model: tars/gpt-4o
      api_key: os.environ/TARS_API_KEY

  - model_name: embeddings
    litellm_params:
      model: tars/text-embedding-3-large
      api_key: os.environ/TARS_API_KEY
```

## Advanced Features

### Cost Optimization

TARS automatically routes requests to optimize for cost while maintaining performance:

```python
# TARS can automatically switch to cheaper models
response = litellm.completion(
    model="tars/gpt-4o-mini",  # Use cost-effective model
    messages=[{"role": "user", "content": "Simple question"}]
)
```

### Automatic Fallback

TARS provides automatic fallback to alternative models when primary models are unavailable:

```python
# If the specified model is down, TARS routes to alternatives
response = litellm.completion(
    model="tars/claude-sonnet-4-20250514",
    messages=[{"role": "user", "content": "Hello"}],
    # TARS handles fallback automatically
)
```

## Support

- Dashboard: https://router.tetrate.ai
- Documentation: https://docs.tetrate.io
- Support: Contact through the TARS dashboard

## Troubleshooting

### Authentication Errors

If you get authentication errors:

1. Verify your API key is set correctly: `echo $TARS_API_KEY`
2. Check your key hasn't expired in the dashboard
3. Ensure you have sufficient credits

### Rate Limits

TARS respects the rate limits of underlying providers. If you hit rate limits:

1. Check your usage in the dashboard
2. Consider upgrading to Enterprise plan for higher limits
3. Implement exponential backoff in your code

### Model Not Found

If a model isn't available:

1. Check the latest model list: https://api.router.tetrate.ai/v1/models
2. Verify the model ID is correct (e.g., `tars/claude-sonnet-4-20250514`)
3. Some models may require specific account permissions
