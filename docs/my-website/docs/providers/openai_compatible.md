# Adding OpenAI-Compatible Providers

## Quick Start

For providers with OpenAI-compatible APIs, you can add support by editing a single JSON file.

## Step-by-Step Guide

### 1. Add to JSON Configuration

Edit `litellm/llms/openai_like/providers.json`:

```json
{
  "your_provider": {
    "base_url": "https://api.yourprovider.com/v1",
    "api_key_env": "YOUR_PROVIDER_API_KEY"
  }
}
```

That's it! Your provider is now available.

### 2. Add to Provider Enum

Edit `litellm/types/utils.py` and add to the `LlmProviders` enum:

```python
class LlmProviders(str, Enum):
    # ... existing providers ...
    YOUR_PROVIDER = "your_provider"
```

### 3. Test It

```python
import litellm
import os

os.environ["YOUR_PROVIDER_API_KEY"] = "your-api-key"

response = litellm.completion(
    model="your_provider/model-name",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

## Configuration Options

### Required Fields

| Field | Description | Example |
|-------|-------------|---------|
| `base_url` | API endpoint base URL | `"https://api.provider.com/v1"` |
| `api_key_env` | Environment variable for API key | `"PROVIDER_API_KEY"` |

### Optional Fields

| Field | Description | Example |
|-------|-------------|---------|
| `api_base_env` | Override base_url via environment variable | `"PROVIDER_API_BASE"` |
| `base_class` | Base config class to use | `"openai_gpt"` (default) or `"openai_like"` |
| `param_mappings` | Map OpenAI params to provider params | `{"max_completion_tokens": "max_tokens"}` |
| `constraints` | Parameter constraints | `{"temperature_max": 1.0}` |
| `special_handling` | Special behavior flags | `{"convert_content_list_to_string": true}` |

## Examples

### Simple Provider (No Special Handling)

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
  "publicai": {
    "base_url": "https://api.publicai.co/v1",
    "api_key_env": "PUBLICAI_API_KEY",
    "param_mappings": {
      "max_completion_tokens": "max_tokens"
    }
  }
}
```

### Provider with Constraints

```json
{
  "moonshot": {
    "base_url": "https://api.moonshot.ai/v1",
    "api_key_env": "MOONSHOT_API_KEY",
    "param_mappings": {
      "max_completion_tokens": "max_tokens"
    },
    "constraints": {
      "temperature_max": 1.0,
      "temperature_min_with_n_gt_1": 0.3
    }
  }
}
```

## Supported Constraints

| Constraint | Description |
|------------|-------------|
| `temperature_max` | Maximum temperature value (auto-clamped) |
| `temperature_min` | Minimum temperature value (auto-clamped) |
| `temperature_min_with_n_gt_1` | Minimum temperature when n > 1 (auto-adjusted) |

## Special Handling Flags

| Flag | Description |
|------|-------------|
| `convert_content_list_to_string` | Convert message content from list format to string |

## When to Use Python Instead

Use a Python config class if you need:
- Custom authentication (OAuth, rotating tokens)
- Complex request/response transformations
- Provider-specific streaming logic
- Advanced tool calling transformations

For simple OpenAI-compatible providers, JSON configuration is recommended.

## Testing Your Provider

1. Set the API key:
```bash
export YOUR_PROVIDER_API_KEY="your-api-key"
```

2. Test completion:
```python
import litellm

response = litellm.completion(
    model="your_provider/model-name",
    messages=[{"role": "user", "content": "Test"}],
    max_tokens=10
)
print(response.choices[0].message.content)
```

3. Test streaming:
```python
response = litellm.completion(
    model="your_provider/model-name",
    messages=[{"role": "user", "content": "Count to 3"}],
    max_tokens=20,
    stream=True
)
for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

## Troubleshooting

### Provider not found
- Verify the provider is in `providers.json`
- Check the provider is added to `LlmProviders` enum
- Restart your Python session to reload the JSON

### API key not working
- Verify the environment variable name matches `api_key_env`
- Check the API key is correctly set: `echo $YOUR_PROVIDER_API_KEY`

### Wrong endpoint URL
- Use `api_base_env` to override via environment variable
- Or pass `api_base` directly in the completion call

## Contributing

When adding a new provider:
1. Add to `providers.json`
2. Add to `LlmProviders` enum
3. Test with real API
4. Submit PR with test results

That's it!
