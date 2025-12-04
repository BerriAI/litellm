# Adding OpenAI-Compatible Providers

For providers that use OpenAI's `/chat/completions` API format, you can add support by editing a single JSON file.

## Quick Start

1. Edit `litellm/llms/openai_like/providers.json`
2. Add your provider configuration
3. Test with `litellm.completion(model="your_provider/model-name", ...)`

## Basic Configuration

For a fully OpenAI-compatible provider, you only need 2 fields:

```json
{
  "your_provider": {
    "base_url": "https://api.yourprovider.com/v1",
    "api_key_env": "YOUR_PROVIDER_API_KEY"
  }
}
```

## Example: PublicAI

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

## Configuration Options

| Field | Required | Description | Default |
|-------|----------|-------------|---------|
| `base_url` | ✅ | API endpoint base URL | - |
| `api_key_env` | ✅ | Environment variable for API key | - |
| `api_base_env` | ❌ | Environment variable to override base_url | - |
| `base_class` | ❌ | "openai_gpt" or "openai_like" | "openai_gpt" |
| `param_mappings` | ❌ | Map parameter names (e.g., `max_completion_tokens` → `max_tokens`) | `{}` |
| `constraints` | ❌ | Parameter constraints (temperature limits, etc) | `{}` |
| `special_handling` | ❌ | Special behavior flags | `{}` |

### Parameter Mappings

Use when your provider uses different parameter names:

```json
{
  "param_mappings": {
    "max_completion_tokens": "max_tokens",
    "top_p": "topP"
  }
}
```

### Constraints

Use for provider-specific parameter limits:

```json
{
  "constraints": {
    "temperature_max": 1.0,
    "temperature_min": 0.0,
    "temperature_min_with_n_gt_1": 0.3
  }
}
```

### Special Handling

Current supported flags:

```json
{
  "special_handling": {
    "convert_content_list_to_string": true
  }
}
```

## Usage

```python
import os
import litellm

os.environ["YOUR_PROVIDER_API_KEY"] = "your-api-key"

response = litellm.completion(
    model="your_provider/model-name",
    messages=[{"role": "user", "content": "Hello"}],
)
```

## When to Use Python Instead

Use a Python config class (instead of JSON) if you need:
- Custom authentication (OAuth, JWT, rotating tokens)
- Complex request/response transformations
- Custom streaming logic
- Advanced tool calling transformations

See existing providers in `litellm/llms/` for examples.

## Testing

Test your provider with:

```python
import litellm

response = litellm.completion(
    model="your_provider/model-name",
    messages=[{"role": "user", "content": "test"}],
    max_tokens=10,
)
print(response.choices[0].message.content)
```

## Submitting

1. Add your configuration to `providers.json`
2. Test basic completion and streaming
3. Update `litellm/constants.py` - add provider to `openai_compatible_providers` list
4. Add provider to `LlmProviders` enum in `litellm/types/utils.py`
5. Submit PR with your changes
