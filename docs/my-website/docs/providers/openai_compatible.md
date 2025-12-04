# Adding OpenAI-Compatible Providers

## Quick Start

For providers with OpenAI-compatible APIs, add them via JSON configuration instead of writing Python code.

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
    messages=[{"role": "user", "content": "Hello"}]
)
```

That's it! Your provider is now integrated.

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

### API Base Override

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

## Complete Example

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

Use a Python config class if you need:
- Custom authentication (OAuth, JWT, etc.)
- Complex request/response transformations
- Provider-specific streaming logic
- Advanced tool calling transformations

See existing providers in `litellm/llms/` for Python examples.

## Supported Features

All JSON-configured providers automatically support:
- ✅ Streaming
- ✅ Async completion
- ✅ Standard OpenAI parameters
- ✅ Tool calling (if provider supports it)
- ✅ Function calling
- ✅ Response format (JSON mode)

## Testing

Test your provider with:

```bash
# Set your API key
export YOUR_PROVIDER_API_KEY="your-key"

# Test basic completion
python -c "
import litellm
response = litellm.completion(
    model='your_provider/model-name',
    messages=[{'role': 'user', 'content': 'test'}]
)
print(response.choices[0].message.content)
"
```

## Submit Your Provider

Once tested:
1. Add your provider to `providers.json`
2. Add to `LlmProviders` enum
3. Test with the command above
4. Submit a PR

Your PR will be reviewed quickly since it only changes 2 files!
