# Adding an OpenAI-Compatible Provider

For providers that are fully OpenAI-compatible (like Hyperbolic, Nscale, etc.), you can add support with a simple JSON configuration.

## Quick Start

### 1. Add Provider to JSON Config

Edit `litellm/llms/openai_like/providers.json`:

```json
{
  "your_provider": {
    "base_url": "https://api.yourprovider.com/v1",
    "api_key_env": "YOUR_PROVIDER_API_KEY"
  }
}
```

### 2. Add to Provider Enum

Edit `litellm/types/utils.py` and add to `LlmProviders` enum:

```python
class LlmProviders(str, Enum):
    # ... existing providers ...
    YOUR_PROVIDER = "your_provider"
```

### 3. Test It

```python
import litellm
import os

os.environ["YOUR_PROVIDER_API_KEY"] = "your-key"

response = litellm.completion(
    model="your_provider/model-name",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

That's it! Your provider is now integrated.

## Optional Configuration

### Parameter Mapping

If your provider uses different parameter names than OpenAI:

```json
{
  "your_provider": {
    "base_url": "https://api.yourprovider.com/v1",
    "api_key_env": "YOUR_PROVIDER_API_KEY",
    "param_mappings": {
      "max_completion_tokens": "max_tokens"
    }
  }
}
```

### Environment Variable Override

Allow users to override the base URL via environment variable:

```json
{
  "your_provider": {
    "base_url": "https://api.yourprovider.com/v1",
    "api_key_env": "YOUR_PROVIDER_API_KEY",
    "api_base_env": "YOUR_PROVIDER_API_BASE"
  }
}
```

### Base Class Selection

Choose between `openai_gpt` (default) or `openai_like`:

```json
{
  "your_provider": {
    "base_url": "https://api.yourprovider.com/v1",
    "api_key_env": "YOUR_PROVIDER_API_KEY",
    "base_class": "openai_like"
  }
}
```

### Parameter Constraints

Apply constraints to parameters (e.g., temperature limits):

```json
{
  "your_provider": {
    "base_url": "https://api.yourprovider.com/v1",
    "api_key_env": "YOUR_PROVIDER_API_KEY",
    "constraints": {
      "temperature_max": 1.0,
      "temperature_min": 0.0
    }
  }
}
```

### Special Handling

Enable special behaviors:

```json
{
  "your_provider": {
    "base_url": "https://api.yourprovider.com/v1",
    "api_key_env": "YOUR_PROVIDER_API_KEY",
    "special_handling": {
      "convert_content_list_to_string": true
    }
  }
}
```

## Complete Example

Here's PublicAI's full configuration:

```json
{
  "publicai": {
    "base_url": "https://api.publicai.co/v1",
    "api_key_env": "PUBLICAI_API_KEY",
    "api_base_env": "PUBLICAI_API_BASE",
    "base_class": "openai_gpt",
    "param_mappings": {
      "max_completion_tokens": "max_tokens"
    },
    "special_handling": {
      "convert_content_list_to_string": true
    }
  }
}
```

## When to Use Python Instead

Use a Python config class if your provider needs:
- Custom authentication (OAuth, rotating tokens)
- Complex request/response transformations
- Provider-specific streaming logic
- Advanced tool calling transformations

If you need more than 5 optional JSON fields, consider using Python.

## Testing

Test your provider with the integration test template:

```python
import os
import litellm

os.environ["YOUR_PROVIDER_API_KEY"] = "test-key"

# Basic test
response = litellm.completion(
    model="your_provider/model-name",
    messages=[{"role": "user", "content": "Test"}],
    max_tokens=10,
)
assert response.choices[0].message.content

# Streaming test
response = litellm.completion(
    model="your_provider/model-name",
    messages=[{"role": "user", "content": "Test"}],
    max_tokens=10,
    stream=True,
)
for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

## Submitting a PR

1. Add your provider to `providers.json`
2. Add enum entry to `LlmProviders`
3. Test basic completion and streaming
4. Submit PR with test results

That's all you need to add a new OpenAI-compatible provider!
