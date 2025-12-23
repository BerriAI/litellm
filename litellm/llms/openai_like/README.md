# JSON-Based OpenAI-Compatible Provider Configuration

This directory contains the new JSON-based configuration system for OpenAI-compatible providers.

## Overview

Instead of creating a full Python module for simple OpenAI-compatible providers, you can now define them in a single JSON file. You can also load custom providers from a URL without modifying LiteLLM's source code.

## Files

- `providers.json` - Configuration file for all JSON-based providers
- `json_loader.py` - Loads and parses the JSON configuration (local and from URL)
- `dynamic_config.py` - Generates Python config classes from JSON
- `chat/` - Existing OpenAI-like chat completion handlers

## Adding a New Provider

### Option 1: Edit Local JSON File (For Contributors)

Edit `providers.json` and add your provider:

```json
{
  "your_provider": {
    "base_url": "https://api.yourprovider.com/v1",
    "api_key_env": "YOUR_PROVIDER_API_KEY"
  }
}
```

That's it! The provider will be automatically loaded and available.

### Option 2: Load from Custom URL (For Users)

Create a custom JSON file with your provider configurations and set the environment variable:

```bash
export LITELLM_CUSTOM_PROVIDERS_URL=https://example.com/my-providers.json
```

Or in Python:

```python
import os
os.environ["LITELLM_CUSTOM_PROVIDERS_URL"] = "https://example.com/my-providers.json"
```

Your custom providers will be automatically loaded and merged with the built-in providers. This is useful when:
- You want to define custom providers without modifying LiteLLM's source code
- You're waiting for a PR to be merged
- You want to keep your provider configurations separate
- You need different provider configurations for different environments

### Optional Configuration Fields

```json
{
  "your_provider": {
    "base_url": "https://api.yourprovider.com/v1",
    "api_key_env": "YOUR_PROVIDER_API_KEY",
    
    // Optional: Override base_url via environment variable
    "api_base_env": "YOUR_PROVIDER_API_BASE",
    
    // Optional: Which base class to use (default: "openai_gpt")
    "base_class": "openai_gpt",  // or "openai_like"
    
    // Optional: Parameter name mappings
    "param_mappings": {
      "max_completion_tokens": "max_tokens"
    },
    
    // Optional: Parameter constraints
    "constraints": {
      "temperature_max": 1.0,
      "temperature_min": 0.0,
      "temperature_min_with_n_gt_1": 0.3
    },
    
    // Optional: Special handling flags
    "special_handling": {
      "convert_content_list_to_string": true
    }
  }
}
```

## Example: PublicAI

The first JSON-configured provider:

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

## Usage

### Using Custom Providers from URL

```python
import litellm
import os

# Set the URL to your custom providers JSON
os.environ["LITELLM_CUSTOM_PROVIDERS_URL"] = "https://example.com/my-providers.json"

# Use your custom provider
response = litellm.completion(
    model="your_provider/model-name",
    messages=[{"role": "user", "content": "Hello"}],
)
```

### Using Built-in Providers

```python
import litellm

response = litellm.completion(
    model="publicai/swiss-ai/apertus-8b-instruct",
    messages=[{"role": "user", "content": "Hello"}],
)
```

## Custom Provider URL Behavior

When `LITELLM_CUSTOM_PROVIDERS_URL` is set:
- LiteLLM loads local providers from `providers.json` first
- Then fetches and merges providers from the specified URL
- Custom providers from the URL can overwrite local providers with the same name
- If the URL is unreachable or returns invalid JSON, LiteLLM logs a warning and continues with local providers only
- The fetch happens once at startup (providers are cached)
- Timeout is set to 10 seconds for the HTTP request

## Benefits

- **Simple**: 2-5 lines of JSON vs 100+ lines of Python
- **Fast**: Add a provider in 5 minutes
- **Safe**: No Python code to mess up
- **Consistent**: All providers follow the same pattern
- **Maintainable**: Centralized configuration

## When to Use Python Instead

Use a Python config class if you need:
- Custom authentication (OAuth, rotating tokens, etc.)
- Complex request/response transformations
- Provider-specific streaming logic
- Advanced tool calling transformations

## Implementation Details

### How It Works

1. `json_loader.py` loads `providers.json` on import
2. If `LITELLM_CUSTOM_PROVIDERS_URL` is set, fetches and merges providers from that URL
3. `dynamic_config.py` generates config classes on-demand
4. Provider resolution checks JSON registry first
5. ProviderConfigManager returns JSON-based configs

### Integration Points

The JSON system is integrated at:
- `litellm/litellm_core_utils/get_llm_provider_logic.py` - Provider resolution
- `litellm/utils.py` - ProviderConfigManager
- `litellm/constants.py` - openai_compatible_providers list

### URL Loading

The URL loading feature uses:
- `httpx.Client` with 10-second timeout
- Graceful error handling (logs warnings, doesn't crash)
- Merge behavior: custom providers can overwrite built-in ones
- Single fetch at startup (no repeated requests)
