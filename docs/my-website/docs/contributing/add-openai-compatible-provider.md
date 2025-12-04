---
sidebar_position: 5
---

# Adding OpenAI-Compatible Providers via JSON

For simple OpenAI-compatible providers, add them via JSON configuration instead of writing Python code.

## Quick Start

### 1. Edit the JSON config

Add your provider to `litellm/llms/openai_like/providers.json`:

```json
{
  "your_provider": {
    "base_url": "https://api.yourprovider.com/v1",
    "api_key_env": "YOUR_PROVIDER_API_KEY"
  }
}
```

### 2. Add to provider enum

Add your provider to the `LlmProviders` enum in `litellm/types/utils.py`:

```python
class LlmProviders(str, Enum):
    # ... existing providers ...
    YOUR_PROVIDER = "your_provider"
```

### 3. Add to constants list

Add to `openai_compatible_providers` in `litellm/constants.py`:

```python
openai_compatible_providers: List = [
    # ... existing providers ...
    "your_provider",
]
```

### 4. Test it

```python
import litellm
import os

os.environ["YOUR_PROVIDER_API_KEY"] = "sk-..."

response = litellm.completion(
    model="your_provider/model-name",
    messages=[{"role": "user", "content": "Hello"}]
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

### Override Base URL via Environment

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

### Temperature Constraints

If your provider has different temperature limits:

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

## All Configuration Options

| Field | Required | Description |
|-------|----------|-------------|
| `base_url` | ✅ Yes | API endpoint base URL |
| `api_key_env` | ✅ Yes | Environment variable name for API key |
| `api_base_env` | ❌ No | Environment variable to override `base_url` |
| `base_class` | ❌ No | `"openai_gpt"` (default) or `"openai_like"` |
| `param_mappings` | ❌ No | Map OpenAI parameter names to provider names |
| `constraints` | ❌ No | Parameter value constraints |
| `special_handling` | ❌ No | Special behavior flags |

## When to Use Python Instead

Use a Python config class (instead of JSON) if you need:

- Custom authentication (OAuth, rotating tokens, etc.)
- Complex request/response transformations
- Provider-specific streaming logic
- Advanced tool calling transformations
- Custom retry logic

See existing providers in `litellm/llms/` for examples.
