
# 📊 Neatlogs - Observability Platform

[Neatlogs](https://neatlogs.com/) is a comprehensive observability platform that provides detailed logging, monitoring, and analytics for LLM applications in production environments.

## Using Neatlogs with LiteLLM

LiteLLM provides `success_callbacks` and `failure_callbacks`, allowing you to easily integrate Neatlogs for comprehensive tracing and monitoring of your LLM operations.

### Integration

Use just a few lines of code to instantly log your LLM responses **across all providers** with Neatlogs:

```python
import litellm

# Configure LiteLLM to use Neatlogs
litellm.success_callback = ["neatlogs"]

# Make your LLM calls as usual
response = litellm.completion(
    model="gpt-4o-minio-mini",
    messages=[{"role": "user", "content": "Hello, how are you?"}],
)
```

### Complete Code Example

```python
import os
from litellm import completion

# Set environment variables
os.environ["OPENAI_API_KEY"] = "your-openai-key"
os.environ["NEATLOGS_API_KEY"] = "your-neatlogs-api-key"

# Configure LiteLLM to use Neatlogs
litellm.success_callback = ["neatlogs"]

# Make LLM call
response = completion(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hi 👋 - I'm using Neatlogs!"}],
)

print(response)
```

## Configuration Options

The Neatlogs integration can be configured through environment variables:

- `NEATLOGS_API_KEY` (str, required): Your Neatlogs API key


## Advanced Usage


### Async Support

Neatlogs fully supports async operations:

```python
import asyncio
import litellm

litellm.success_callback = ["neatlogs"]

async def main():
    response = await litellm.acompletion(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello async world!"}],
    )
    print(response)

asyncio.run(main())
```

### Streaming Support

Neatlogs automatically handles streaming responses:

```python
import litellm

litellm.success_callback = ["neatlogs"]

response = litellm.completion(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True
)

for chunk in response:
    print(chunk.choices[0].delta.content, end="")
```

### Error Tracking

Neatlogs also tracks failed requests:

```python
import litellm

# Enable both success and failure callbacks
litellm.success_callback = ["neatlogs"]
litellm.failure_callback = ["neatlogs"]

try:
    response = litellm.completion(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello!"}],
        # This might fail due to invalid parameters
        temperature=3.0  # Invalid temperature
    )
except Exception as e:
    print(f"Error: {e}")
    # Error will be logged to Neatlogs automatically
```

## Data Tracked

Neatlogs captures comprehensive data for each LLM request:

- **Request Details**: Model, provider, messages, parameters
- **Response Data**: Completion text, token usage, cost
- **Error Information**: Failure reasons and stack traces
- **Metadata**:session IDs

## Supported Models and Providers

Neatlogs works with all LiteLLM-supported providers:

- OpenAI (GPT-3.5, GPT-4O-MINI, etc.)
- Anthropic (Claude)
- Google (Gemini)
- Azure OpenAI
- AWS Bedrock
- And 100+ more providers

## Support

For issues or questions, please refer to:

- [Neatlogs Documentation](https://docs.neatlogs.com)
- [Github](https://github.com/Neatlogs/neatlogs)
