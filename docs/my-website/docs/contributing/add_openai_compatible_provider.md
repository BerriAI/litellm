# Adding OpenAI-Compatible Providers via JSON

## Overview

For providers that are OpenAI-compatible (same API format as OpenAI), you can add support by simply editing a JSON file - no Python code required.

## Quick Start

### 1. Edit the JSON Configuration

Edit `litellm/llms/openai_like/providers.json` and add your provider:

```json
{
  "your_provider": {
    "base_url": "https://api.yourprovider.com/v1",
    "api_key_env": "YOUR_PROVIDER_API_KEY"
  }
}
```

### 2. Add to Provider Enum

Add your provider to `litellm/types/utils.py`:

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
print(response.choices[0].message.content)
```

That's it! ✅

## Optional Configuration

### Parameter Mapping

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
- Custom tool calling formats

See existing providers in `litellm/llms/` for Python implementation examples.

## Testing Your Provider

Run the test to verify everything works:

```bash
python3 -c "
import os
import litellm

os.environ['YOUR_PROVIDER_API_KEY'] = 'your-key'

response = litellm.completion(
    model='your_provider/model-name',
    messages=[{'role': 'user', 'content': 'test'}]
)
print('✓ Success:', response.choices[0].message.content)
"
```

## Submit Your PR

1. Add your provider to `providers.json`
2. Add to `LlmProviders` enum
3. Test with real API
4. Submit PR with test results

That's all you need! 🚀
