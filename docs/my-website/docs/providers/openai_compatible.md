# Adding OpenAI-Compatible Providers

For providers that follow the OpenAI API format, you can add support via a simple JSON configuration.

## Quick Start

Edit `litellm/llms/openai_like/providers.json` and add your provider:

```json
{
  "your_provider": {
    "base_url": "https://api.yourprovider.com/v1",
    "api_key_env": "YOUR_PROVIDER_API_KEY"
  }
}
```

That's it! The provider is now available.

## Usage

```python
import litellm
import os

os.environ["YOUR_PROVIDER_API_KEY"] = "your-api-key"

response = litellm.completion(
    model="your_provider/model-name",
    messages=[{"role": "user", "content": "Hello!"}],
)
```

## Configuration Options

### Required Fields

- **`base_url`**: API endpoint (e.g., `https://api.provider.com/v1`)
- **`api_key_env`**: Environment variable name for the API key

### Optional Fields

```json
{
  "your_provider": {
    "base_url": "https://api.yourprovider.com/v1",
    "api_key_env": "YOUR_PROVIDER_API_KEY",
    
    // Override base_url via environment variable
    "api_base_env": "YOUR_PROVIDER_API_BASE",
    
    // Base class: "openai_gpt" (default) or "openai_like"
    "base_class": "openai_gpt",
    
    // Map parameter names
    "param_mappings": {
      "max_completion_tokens": "max_tokens"
    },
    
    // Parameter constraints
    "constraints": {
      "temperature_max": 1.0,
      "temperature_min": 0.0
    },
    
    // Special handling flags
    "special_handling": {
      "convert_content_list_to_string": true
    }
  }
}
```

## Examples

### Simple Provider (Fully OpenAI-Compatible)

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

### Provider with Temperature Constraints

```json
{
  "custom_provider": {
    "base_url": "https://api.custom.com/v1",
    "api_key_env": "CUSTOM_API_KEY",
    "constraints": {
      "temperature_max": 1.0,
      "temperature_min": 0.1
    }
  }
}
```

## When to Use Python Instead

Use a Python config class if you need:
- Custom authentication (OAuth, token rotation)
- Complex request/response transformations
- Provider-specific streaming logic
- Advanced tool calling transformations

For simple OpenAI-compatible providers, JSON is recommended.

## Testing Your Provider

```python
import litellm
import os

# Set API key
os.environ["YOUR_PROVIDER_API_KEY"] = "your-key"

# Test basic completion
response = litellm.completion(
    model="your_provider/model-name",
    messages=[{"role": "user", "content": "test"}],
    max_tokens=10,
)
print(response.choices[0].message.content)

# Test streaming
response = litellm.completion(
    model="your_provider/model-name",
    messages=[{"role": "user", "content": "test"}],
    stream=True,
)
for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

## Supported Features

JSON-configured providers automatically support:
- ✅ Basic completions
- ✅ Streaming
- ✅ Async operations
- ✅ Parameter mapping
- ✅ Environment variable overrides
- ✅ Temperature constraints
- ✅ Content format conversions

## File Location

**Config file**: `litellm/llms/openai_like/providers.json`

**Add provider**: Edit the JSON file and add your configuration under a new key.
