# Adding OpenAI-Compatible Providers

For simple OpenAI-compatible providers (like Hyperbolic, Nscale, etc.), you can add support by editing a single JSON file.

## Quick Start

### Option 1: Edit Local JSON File
1. Edit `litellm/llms/openai_like/providers.json`
2. Add your provider configuration
3. Test with: `litellm.completion(model="your_provider/model-name", ...)`

### Option 2: Load from Environment Variable (JSON String)
1. Set the `LITELLM_CUSTOM_PROVIDERS` environment variable with your JSON configuration
2. Start LiteLLM - it will automatically load and merge your custom providers

```bash
export LITELLM_CUSTOM_PROVIDERS='{"my_provider": {"base_url": "https://api.myprovider.com/v1", "api_key_env": "MY_PROVIDER_KEY"}}'
```

This is useful for containerized deployments, CI/CD, or when you want inline configuration.

### Option 3: Load from Custom URL
1. Create a custom JSON file with your provider configurations
2. Host it at a URL (can be a local file server, cloud storage, or any HTTP endpoint)
3. Set the environment variable: `LITELLM_CUSTOM_PROVIDERS_URL=https://example.com/my-providers.json`
4. Start LiteLLM - it will automatically load and merge your custom providers

This is useful when you want to define custom providers without modifying LiteLLM's source code or waiting for PR approval.

## Basic Configuration

For a fully OpenAI-compatible provider:

```json
{
  "your_provider": {
    "base_url": "https://api.yourprovider.com/v1",
    "api_key_env": "YOUR_PROVIDER_API_KEY"
  }
}
```

That's it! The provider is now available.

## Configuration Options

### Required Fields

- `base_url` - API endpoint (e.g., `https://api.provider.com/v1`)
- `api_key_env` - Environment variable name for API key (e.g., `PROVIDER_API_KEY`)

### Optional Fields

- `api_base_env` - Environment variable to override `base_url`
- `base_class` - Use `"openai_gpt"` (default) or `"openai_like"`
- `param_mappings` - Map OpenAI parameter names to provider-specific names
- `constraints` - Parameter value constraints (min/max)
- `special_handling` - Special behaviors like content format conversion

## Examples

### Simple Provider (Fully Compatible)

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
  "custom_provider": {
    "base_url": "https://api.custom.com/v1",
    "api_key_env": "CUSTOM_API_KEY",
    "constraints": {
      "temperature_max": 1.0,
      "temperature_min": 0.0
    }
  }
}
```

## Usage

### Using Custom Providers from JSON String

```python
import litellm
import os
import json

# Define providers inline
custom_providers = {
    "my_provider": {
        "base_url": "https://api.myprovider.com/v1",
        "api_key_env": "MY_PROVIDER_API_KEY"
    }
}

# Set as environment variable
os.environ["LITELLM_CUSTOM_PROVIDERS"] = json.dumps(custom_providers)

# Set your API key
os.environ["MY_PROVIDER_API_KEY"] = "your-key-here"

# Use the provider
response = litellm.completion(
    model="my_provider/model-name",
    messages=[{"role": "user", "content": "Hello"}],
)
```

### Using Custom Providers from URL

```python
import litellm
import os

# Set the URL to your custom providers JSON
os.environ["LITELLM_CUSTOM_PROVIDERS_URL"] = "https://example.com/my-providers.json"

# Set your API key
os.environ["YOUR_PROVIDER_API_KEY"] = "your-key-here"

# Use the provider
response = litellm.completion(
    model="your_provider/model-name",
    messages=[{"role": "user", "content": "Hello"}],
)
```

### Using Built-in Providers

```python
import litellm
import os

# Set your API key
os.environ["YOUR_PROVIDER_API_KEY"] = "your-key-here"

# Use the provider
response = litellm.completion(
    model="your_provider/model-name",
    messages=[{"role": "user", "content": "Hello"}],
)
```

## Custom Provider Loading Behavior

**Loading order:**
1. Local providers from `providers.json` are loaded first
2. If `LITELLM_CUSTOM_PROVIDERS` is set, providers from the JSON string are merged
3. If `LITELLM_CUSTOM_PROVIDERS_URL` is set, providers from the URL are merged

**Key behaviors:**
- Custom providers can overwrite local providers with the same name
- Both environment variables can be used together
- If JSON string is invalid, LiteLLM logs a warning and continues
- If URL is unreachable or returns invalid JSON, LiteLLM logs a warning and continues with local providers only
- The fetch/parse happens once at startup (providers are cached)

## When to Use Python Instead

Use a Python config class if you need:

- Custom authentication flows (OAuth, JWT, etc.)
- Complex request/response transformations
- Provider-specific streaming logic
- Advanced tool calling modifications

For these cases, create a config class in `litellm/llms/your_provider/chat/transformation.py` that inherits from `OpenAIGPTConfig` or `OpenAILikeChatConfig`.

## Testing

Test your provider:

```bash
# Quick test
python -c "
import litellm
import os
os.environ['PROVIDER_API_KEY'] = 'your-key'
response = litellm.completion(
    model='provider/model-name',
    messages=[{'role': 'user', 'content': 'test'}]
)
print(response.choices[0].message.content)
"
```

## Reference

See existing providers in `litellm/llms/openai_like/providers.json` for examples.
