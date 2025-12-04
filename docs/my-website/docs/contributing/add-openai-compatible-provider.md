# Adding OpenAI-Compatible Providers

For providers with OpenAI-compatible APIs, you can add support via JSON configuration without writing Python code.

## Quick Start

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

### 2. Add to Provider List

Add to `litellm/constants.py`:

```python
openai_compatible_providers: List = [
    # ... existing providers ...
    "your_provider",
]
```

### 3. Add to Provider Enum

Add to `litellm/types/utils.py`:

```python
class LlmProviders(str, Enum):
    # ... existing providers ...
    YOUR_PROVIDER = "your_provider"
```

### 4. Test

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

### Base Class Selection

Choose between two base classes:
- `openai_gpt` (default) - For most providers
- `openai_like` - For providers needing OpenAI-like handling

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

If your provider has stricter limits than OpenAI:

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

### Content Transformations

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

Use a Python config class if your provider needs:
- Custom authentication (OAuth, token rotation)
- Complex request/response transformations
- Provider-specific streaming logic
- Advanced tool calling transformations

## Testing

Run the integration test:

```bash
python3 -c "
import litellm
import os

os.environ['YOUR_PROVIDER_API_KEY'] = 'test-key'

response = litellm.completion(
    model='your_provider/test-model',
    messages=[{'role': 'user', 'content': 'Hello'}],
)
print(response.choices[0].message.content)
"
```

## Reference

See `litellm/llms/openai_like/providers.json` for more examples.
