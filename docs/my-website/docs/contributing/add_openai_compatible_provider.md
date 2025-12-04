---
id: add_openai_compatible_provider
title: Add OpenAI-Compatible Provider
sidebar_label: Add OpenAI-Compatible Provider
---

# Add an OpenAI-Compatible Provider

For providers that follow OpenAI's API format, you can add support by editing a single JSON file.

## Quick Start

1. **Edit the JSON file**: `litellm/llms/openai_like/providers.json`

2. **Add your provider**:
```json
{
  "your_provider": {
    "base_url": "https://api.yourprovider.com/v1",
    "api_key_env": "YOUR_PROVIDER_API_KEY"
  }
}
```

3. **Add to provider enum**: Edit `litellm/types/utils.py`
```python
class LlmProviders(str, Enum):
    # ... existing providers ...
    YOUR_PROVIDER = "your_provider"
```

4. **Test it**:
```python
import litellm
import os

os.environ["YOUR_PROVIDER_API_KEY"] = "your-key"

response = litellm.completion(
    model="your_provider/model-name",
    messages=[{"role": "user", "content": "Hello"}]
)
```

That's it! 🎉

## Optional Configuration

### Override Base URL
```json
{
  "your_provider": {
    "base_url": "https://api.yourprovider.com/v1",
    "api_key_env": "YOUR_PROVIDER_API_KEY",
    "api_base_env": "YOUR_PROVIDER_API_BASE"  // Allow users to override base_url
  }
}
```

### Parameter Mapping
If the provider uses different parameter names:
```json
{
  "your_provider": {
    "base_url": "https://api.yourprovider.com/v1",
    "api_key_env": "YOUR_PROVIDER_API_KEY",
    "param_mappings": {
      "max_completion_tokens": "max_tokens"  // Map OpenAI param to provider param
    }
  }
}
```

### Base Class Selection
```json
{
  "your_provider": {
    "base_url": "https://api.yourprovider.com/v1",
    "api_key_env": "YOUR_PROVIDER_API_KEY",
    "base_class": "openai_gpt"  // or "openai_like" (default: "openai_gpt")
  }
}
```

### Temperature Constraints
```json
{
  "your_provider": {
    "base_url": "https://api.yourprovider.com/v1",
    "api_key_env": "YOUR_PROVIDER_API_KEY",
    "constraints": {
      "temperature_max": 1.0,              // Clamp max temperature
      "temperature_min": 0.0,              // Clamp min temperature
      "temperature_min_with_n_gt_1": 0.3  // Min temp when n > 1
    }
  }
}
```

### Content Format Conversion
```json
{
  "your_provider": {
    "base_url": "https://api.yourprovider.com/v1",
    "api_key_env": "YOUR_PROVIDER_API_KEY",
    "special_handling": {
      "convert_content_list_to_string": true  // Convert content arrays to strings
    }
  }
}
```

## Complete Example

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

Use a Python config class if you need:
- Custom authentication (OAuth, rotating tokens)
- Complex request/response transformations
- Provider-specific streaming logic
- Advanced tool calling modifications

For these cases, see existing providers in `litellm/llms/` for examples.

## Adding to Constants

Add your provider to `litellm/constants.py`:

```python
openai_compatible_endpoints: List = [
    # ... existing endpoints ...
    "https://api.yourprovider.com/v1",
]

openai_compatible_providers: List = [
    # ... existing providers ...
    "your_provider",
]
```

## Testing

Run the test suite:
```bash
pytest tests/test_litellm/llms/openai_like/test_json_providers.py -v
```

Or test manually:
```python
import litellm
import os

os.environ["YOUR_PROVIDER_API_KEY"] = "test-key"

# Basic test
response = litellm.completion(
    model="your_provider/model-name",
    messages=[{"role": "user", "content": "test"}],
    max_tokens=5
)
print(response.choices[0].message.content)

# Streaming test
response = litellm.completion(
    model="your_provider/model-name",
    messages=[{"role": "user", "content": "test"}],
    max_tokens=5,
    stream=True
)
for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

## Submit PR

1. Add your provider to `providers.json`
2. Add to `LlmProviders` enum
3. Add to constants lists
4. Test with real API
5. Submit pull request

Your PR will be reviewed quickly since it's just a JSON change!
