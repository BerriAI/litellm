# Adding an OpenAI-Compatible Provider

For simple OpenAI-compatible providers (like Hyperbolic, Nscale, etc.), you can add support by editing a single JSON file.

## Quick Start

### 1. Add Provider Configuration

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

os.environ["YOUR_PROVIDER_API_KEY"] = "sk-..."

response = litellm.completion(
    model="your_provider/model-name",
    messages=[{"role": "user", "content": "Hello"}]
)
print(response.choices[0].message.content)
```

That's it! No Python code needed.

## Optional Configuration

### Parameter Mappings

If the provider uses different parameter names:

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

### Parameter Constraints

For providers with parameter limits:

```json
{
  "your_provider": {
    "base_url": "https://api.yourprovider.com/v1",
    "api_key_env": "YOUR_PROVIDER_API_KEY",
    "constraints": {
      "temperature_max": 1.0,
      "temperature_min": 0.0,
      "temperature_min_with_n_gt_1": 0.3
    }
  }
}
```

## Complete Example: PublicAI

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

Use a full Python implementation if you need:

- Custom authentication (OAuth, rotating tokens, etc.)
- Complex request/response transformations
- Provider-specific streaming logic
- Advanced tool calling transformations

For those cases, follow the standard provider implementation pattern in `litellm/llms/`.

## Testing Your Provider

Create a simple test:

```python
import os
import litellm

os.environ["YOUR_PROVIDER_API_KEY"] = "your-test-key"

# Test basic completion
response = litellm.completion(
    model="your_provider/test-model",
    messages=[{"role": "user", "content": "Say hello"}],
    max_tokens=10
)
assert response.choices[0].message.content

# Test streaming
response = litellm.completion(
    model="your_provider/test-model",
    messages=[{"role": "user", "content": "Count to 3"}],
    max_tokens=20,
    stream=True
)
for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

## Submitting Your PR

1. Add your provider to `providers.json`
2. Add to `LlmProviders` enum
3. Test with real API
4. Submit PR with:
   - Provider configuration
   - Basic test showing it works
   - Model pricing info (if available)

That's it! The JSON system handles everything else automatically.
