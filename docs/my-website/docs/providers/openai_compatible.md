# Adding OpenAI-Compatible Providers

## Quick Start

For providers with OpenAI-compatible APIs, add them via JSON configuration - no Python code required.

## Steps

### 1. Add to JSON Config

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

Add one line to `litellm/types/utils.py`:

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
```

## Optional Configuration

### Parameter Mapping

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

### Base URL Override

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

### Content Conversion

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
- Custom authentication (OAuth, JWT, etc.)
- Complex request/response transformations
- Provider-specific streaming logic
- Advanced parameter validation

## Supported Fields

| Field | Required | Description |
|-------|----------|-------------|
| `base_url` | ✅ | API endpoint base URL |
| `api_key_env` | ✅ | Environment variable for API key |
| `api_base_env` | ❌ | Environment variable to override base_url |
| `base_class` | ❌ | "openai_gpt" (default) or "openai_like" |
| `param_mappings` | ❌ | Map OpenAI params to provider params |
| `special_handling` | ❌ | Provider-specific transformations |

## Testing

```bash
# Set your API key
export YOUR_PROVIDER_API_KEY="your-key"

# Test basic completion
python3 -c "
import litellm
response = litellm.completion(
    model='your_provider/model-name',
    messages=[{'role': 'user', 'content': 'test'}]
)
print(response.choices[0].message.content)
"
```

## Contributing

Submit a PR with:
1. Your provider added to `providers.json`
2. Provider enum added to `LlmProviders`
3. Test showing it works

That's it!
