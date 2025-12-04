# Adding OpenAI-Compatible Providers via JSON

For simple OpenAI-compatible providers, you can add support by editing a single JSON file - no Python code needed.

## Quick Start

### 1. Add Provider to JSON

Edit `litellm/llms/openai_like/providers.json`:

```json
{
  "your_provider": {
    "base_url": "https://api.yourprovider.com/v1",
    "api_key_env": "YOUR_PROVIDER_API_KEY"
  }
}
```

### 2. Add to LlmProviders Enum

Edit `litellm/types/utils.py`:

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
    messages=[{"role": "user", "content": "Hello"}],
)
print(response.choices[0].message.content)
```

That's it! Your provider is now integrated.

## Optional Configuration

### Parameter Mappings

If the provider uses different parameter names than OpenAI:

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

### API Base Override

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

### Temperature Constraints

If the provider has different temperature limits:

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

### Content Format Conversion

If the provider doesn't support content as a list:

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

PublicAI provider configuration:

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

Use a Python config class (traditional approach) if you need:

- Custom authentication (OAuth, JWT, rotating tokens)
- Complex request/response transformations
- Provider-specific streaming logic
- Advanced tool calling transformations
- Request signing or encryption

If your provider is OpenAI-compatible and only needs basic configuration, use JSON.

## Testing Your Provider

```python
import litellm
import os

os.environ["YOUR_PROVIDER_API_KEY"] = "test-key"

# Test basic completion
response = litellm.completion(
    model="your_provider/model-name",
    messages=[{"role": "user", "content": "test"}],
    max_tokens=10,
)
assert response.choices[0].message.content

# Test streaming
response = litellm.completion(
    model="your_provider/model-name",
    messages=[{"role": "user", "content": "test"}],
    max_tokens=10,
    stream=True,
)
for chunk in response:
    assert chunk is not None

print("✅ All tests passed!")
```

## Submitting Your Provider

1. Add configuration to `litellm/llms/openai_like/providers.json`
2. Add to `LlmProviders` enum in `litellm/types/utils.py`
3. Test basic completion and streaming
4. Submit PR with your changes

Your PR will be much easier to review since it's just JSON configuration!
