# Adding an OpenAI-Compatible Provider

For simple OpenAI-compatible providers, you can add support by editing a single JSON file.

## Quick Start

1. **Add your provider to `litellm/llms/openai_like/providers.json`**:

```json
{
  "your_provider": {
    "base_url": "https://api.yourprovider.com/v1",
    "api_key_env": "YOUR_PROVIDER_API_KEY"
  }
}
```

2. **Add provider to enum in `litellm/types/utils.py`**:

```python
class LlmProviders(str, Enum):
    # ... existing providers
    YOUR_PROVIDER = "your_provider"
```

3. **Test it**:

```python
import litellm
import os

os.environ["YOUR_PROVIDER_API_KEY"] = "your-key"

response = litellm.completion(
    model="your_provider/model-name",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

That's it!

## Configuration Options

### Basic (Required)

```json
{
  "provider": {
    "base_url": "https://api.provider.com/v1",
    "api_key_env": "PROVIDER_API_KEY"
  }
}
```

### With Optional Fields

```json
{
  "provider": {
    "base_url": "https://api.provider.com/v1",
    "api_key_env": "PROVIDER_API_KEY",
    
    // Override base URL via environment variable
    "api_base_env": "PROVIDER_API_BASE",
    
    // Base class: "openai_gpt" (default) or "openai_like"
    "base_class": "openai_gpt",
    
    // Map OpenAI params to provider-specific names
    "param_mappings": {
      "max_completion_tokens": "max_tokens"
    },
    
    // Parameter constraints
    "constraints": {
      "temperature_max": 1.0,
      "temperature_min_with_n_gt_1": 0.3
    },
    
    // Special handling flags
    "special_handling": {
      "convert_content_list_to_string": true
    }
  }
}
```

## Field Reference

| Field | Required | Description | Example |
|-------|----------|-------------|---------|
| `base_url` | ✅ | API endpoint base URL | `"https://api.provider.com/v1"` |
| `api_key_env` | ✅ | Environment variable for API key | `"PROVIDER_API_KEY"` |
| `api_base_env` | ❌ | Env var to override base_url | `"PROVIDER_API_BASE"` |
| `base_class` | ❌ | Base config class (default: `openai_gpt`) | `"openai_gpt"` or `"openai_like"` |
| `param_mappings` | ❌ | Parameter name translations | `{"max_completion_tokens": "max_tokens"}` |
| `constraints` | ❌ | Parameter limits | `{"temperature_max": 1.0}` |
| `special_handling` | ❌ | Special behavior flags | `{"convert_content_list_to_string": true}` |

## Parameter Mappings

Use `param_mappings` when the provider uses different parameter names than OpenAI:

```json
{
  "param_mappings": {
    "max_completion_tokens": "max_tokens",
    "frequency_penalty": "repetition_penalty"
  }
}
```

## Constraints

### Temperature Constraints

```json
{
  "constraints": {
    "temperature_min": 0.0,
    "temperature_max": 1.0,
    "temperature_clamp": true
  }
}
```

### Conditional Constraints

```json
{
  "constraints": {
    // When n > 1, ensure temperature >= 0.3
    "temperature_min_with_n_gt_1": 0.3
  }
}
```

## Special Handling

### Convert Content List to String

Some providers don't support content as a list of objects:

```json
{
  "special_handling": {
    "convert_content_list_to_string": true
  }
}
```

This converts:
```python
{"role": "user", "content": [{"type": "text", "text": "Hello"}]}
```

To:
```python
{"role": "user", "content": "Hello"}
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

Usage:

```python
import litellm
import os

os.environ["PUBLICAI_API_KEY"] = "your-key"

# Basic completion
response = litellm.completion(
    model="publicai/swiss-ai/apertus-8b-instruct",
    messages=[{"role": "user", "content": "Hello!"}],
)

# Streaming
response = litellm.completion(
    model="publicai/swiss-ai/apertus-8b-instruct",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True,
)
for chunk in response:
    print(chunk.choices[0].delta.content, end="")
```

## When to Use Python Instead

Use a Python config class if you need:
- Custom authentication (OAuth, token rotation)
- Complex request/response transformations
- Provider-specific streaming logic
- Advanced tool calling transformations

For these cases, create a provider in `litellm/llms/your_provider/` following existing patterns.

## Testing

After adding your provider, test with:

```python
import litellm
import os

os.environ["YOUR_PROVIDER_API_KEY"] = "test-key"

# Test basic completion
response = litellm.completion(
    model="your_provider/model-name",
    messages=[{"role": "user", "content": "test"}],
    max_tokens=10,
)
assert response.choices[0].message.content

# Test streaming
response = litellm.completion(
    model="your_provider/model-name",
    messages=[{"role": "user", "content": "test"}],
    stream=True,
)
for chunk in response:
    assert chunk is not None
```

## Troubleshooting

### Provider not found

Ensure:
1. Provider is in `litellm/llms/openai_like/providers.json`
2. Provider is added to `LlmProviders` enum in `litellm/types/utils.py`
3. JSON is valid (use a JSON validator)

### API errors

Check:
1. `base_url` is correct
2. API key environment variable is set
3. Model name is correct for the provider
4. Provider endpoint is `/chat/completions` compatible

### Parameter errors

If parameters aren't being accepted:
1. Check if parameter names need mapping (`param_mappings`)
2. Verify parameter constraints (`constraints`)
3. Check provider documentation for supported parameters
