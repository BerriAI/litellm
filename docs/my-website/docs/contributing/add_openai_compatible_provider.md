---
id: add_openai_compatible_provider
title: Adding OpenAI-Compatible Providers
sidebar_label: OpenAI-Compatible Providers
---

# Adding OpenAI-Compatible Providers

For providers that use OpenAI's `/v1/chat/completions` API format, you can add support by editing a single JSON file.

## Quick Start

1. Edit `litellm/llms/openai_like/providers.json`
2. Add your provider configuration
3. Test with `litellm.completion()`

## Basic Example

```json
{
  "your_provider": {
    "base_url": "https://api.yourprovider.com/v1",
    "api_key_env": "YOUR_PROVIDER_API_KEY"
  }
}
```

That's it! Your provider is now available.

## Configuration Options

### Required Fields

- **`base_url`**: The API endpoint (e.g., `https://api.provider.com/v1`)
- **`api_key_env`**: Environment variable name for the API key (e.g., `PROVIDER_API_KEY`)

### Optional Fields

- **`api_base_env`**: Environment variable to override `base_url` (useful for custom deployments)
- **`base_class`**: Either `"openai_gpt"` (default) or `"openai_like"`
- **`param_mappings`**: Map OpenAI parameter names to provider-specific names
- **`constraints`**: Apply parameter constraints (temperature limits, etc.)
- **`special_handling`**: Enable special behaviors

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

## Parameter Mappings

Some providers use different parameter names than OpenAI. Use `param_mappings` to translate:

```json
{
  "param_mappings": {
    "max_completion_tokens": "max_tokens",
    "stop_sequences": "stop"
  }
}
```

## Parameter Constraints

Apply provider-specific limits:

```json
{
  "constraints": {
    "temperature_min": 0.0,
    "temperature_max": 1.0,
    "temperature_min_with_n_gt_1": 0.3
  }
}
```

## Special Handling

Enable special behaviors:

```json
{
  "special_handling": {
    "convert_content_list_to_string": true
  }
}
```

**Available flags**:
- `convert_content_list_to_string`: Convert message content from list format to string

## Usage

```python
import litellm
import os

# Set API key
os.environ["YOUR_PROVIDER_API_KEY"] = "your-key"

# Use the provider
response = litellm.completion(
    model="your_provider/model-name",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

## Testing

```python
# Basic test
response = litellm.completion(
    model="your_provider/model-name",
    messages=[{"role": "user", "content": "test"}],
    max_tokens=10
)
print(response.choices[0].message.content)

# Streaming test
response = litellm.completion(
    model="your_provider/model-name",
    messages=[{"role": "user", "content": "test"}],
    stream=True
)
for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

## When to Use Python Instead

Use a Python config class if you need:
- Custom authentication (OAuth, JWT, rotating tokens)
- Complex request/response transformations
- Provider-specific streaming logic
- Advanced tool calling transformations

See [Adding Custom Providers](./custom_provider.md) for Python-based implementations.

## Additional Setup

After adding your provider to `providers.json`, you also need to:

1. Add to `litellm/types/utils.py` - `LlmProviders` enum:
```python
YOUR_PROVIDER = "your_provider"
```

2. Add to `litellm/constants.py` - `openai_compatible_providers` list:
```python
"your_provider",
```

## Examples

### Simple Provider (Fully OpenAI-compatible)
```json
{
  "hyperbolic": {
    "base_url": "https://api.hyperbolic.xyz/v1",
    "api_key_env": "HYPERBOLIC_API_KEY"
  }
}
```

### Provider with Parameter Mapping
```json
{
  "moonshot": {
    "base_url": "https://api.moonshot.ai/v1",
    "api_key_env": "MOONSHOT_API_KEY",
    "param_mappings": {
      "max_completion_tokens": "max_tokens"
    }
  }
}
```

### Provider with Constraints
```json
{
  "provider_with_limits": {
    "base_url": "https://api.provider.com/v1",
    "api_key_env": "PROVIDER_API_KEY",
    "constraints": {
      "temperature_max": 1.0,
      "temperature_min_with_n_gt_1": 0.3
    }
  }
}
```

## Benefits

- ✅ **Simple**: Just edit one JSON file
- ✅ **Fast**: Add a provider in 5 minutes
- ✅ **Safe**: No Python code to debug
- ✅ **Consistent**: All providers follow the same pattern
- ✅ **Self-documenting**: Configuration is the documentation

## Support

If you encounter issues:
1. Check the [provider's API documentation](https://docs.litellm.ai/docs/providers)
2. Test with `curl` to verify the API endpoint works
3. Open an issue on [GitHub](https://github.com/BerriAI/litellm/issues)
