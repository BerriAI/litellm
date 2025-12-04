# Adding an OpenAI-Compatible Provider

For providers with OpenAI-compatible APIs, you can add support via JSON configuration - no Python code required.

## Quick Start

1. **Add provider to JSON config**

Edit `litellm/llms/openai_like/providers.json`:

```json
{
  "your_provider": {
    "base_url": "https://api.yourprovider.com/v1",
    "api_key_env": "YOUR_PROVIDER_API_KEY"
  }
}
```

2. **Add to provider enum**

Edit `litellm/types/utils.py` and add to `LlmProviders` enum:

```python
class LlmProviders(str, Enum):
    # ... existing providers ...
    YOUR_PROVIDER = "your_provider"
```

3. **Test it**

```python
import litellm
import os

os.environ["YOUR_PROVIDER_API_KEY"] = "your-key"

response = litellm.completion(
    model="your_provider/model-name",
    messages=[{"role": "user", "content": "Hello"}]
)
```

That's it! ✅

## Optional Configuration

### Parameter Mappings

If your provider uses different parameter names:

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

Allow users to override the base URL:

```json
{
  "your_provider": {
    "base_url": "https://api.yourprovider.com/v1",
    "api_key_env": "YOUR_PROVIDER_API_KEY",
    "api_base_env": "YOUR_PROVIDER_API_BASE"
  }
}
```

### Content Format Conversion

If your provider doesn't support content as a list:

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

### Parameter Constraints

For providers with stricter limits than OpenAI:

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

## Complete Example

PublicAI provider configuration:

```json
{
  "publicai": {
    "base_url": "https://api.publicai.co/v1",
    "api_key_env": "PUBLICAI_API_KEY",
    "api_base_env": "PUBLICAI_API_BASE",
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

For these cases, follow the existing provider patterns in `litellm/llms/`.

## Configuration Reference

| Field | Required | Description |
|-------|----------|-------------|
| `base_url` | ✅ | API endpoint base URL |
| `api_key_env` | ✅ | Environment variable for API key |
| `api_base_env` | ❌ | Environment variable to override base_url |
| `base_class` | ❌ | `"openai_gpt"` or `"openai_like"` (default: `"openai_gpt"`) |
| `param_mappings` | ❌ | Map OpenAI params to provider params |
| `constraints` | ❌ | Parameter constraints (temperature limits, etc) |
| `special_handling` | ❌ | Special behavior flags |
