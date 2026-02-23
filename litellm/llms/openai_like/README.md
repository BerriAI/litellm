# JSON-Based OpenAI-Compatible Provider Configuration

This directory contains the new JSON-based configuration system for OpenAI-compatible providers.

## Overview

Instead of creating a full Python module for simple OpenAI-compatible providers, you can now define them in a single JSON file.

## Files

- `providers.json` - Configuration file for all JSON-based providers
- `json_loader.py` - Loads and parses the JSON configuration
- `dynamic_config.py` - Generates Python config classes from JSON (chat + responses)
- `chat/` - OpenAI-like chat completion handlers
- `responses/` - OpenAI-like Responses API handlers

## Adding a New Provider

### For Simple OpenAI-Compatible Providers

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

```python
import litellm

response = litellm.completion(
    model="publicai/swiss-ai/apertus-8b-instruct",
    messages=[{"role": "user", "content": "Hello"}],
)
```

## Responses API Support

Providers that support the OpenAI Responses API (`/v1/responses`) can declare it via `supported_endpoints`:

```json
{
  "your_provider": {
    "base_url": "https://api.yourprovider.com/v1",
    "api_key_env": "YOUR_PROVIDER_API_KEY",
    "supported_endpoints": ["/v1/chat/completions", "/v1/responses"]
  }
}
```

This enables `litellm.responses(model="your_provider/model-name", ...)` with zero Python code.
The provider inherits all request/response handling from OpenAI's Responses API config.

If `supported_endpoints` is omitted, it defaults to `[]` (only chat completions, which is always enabled for JSON providers).

### How It Works

1. `json_loader.py` checks `supported_endpoints` for `/v1/responses`
2. `dynamic_config.py` generates a responses config class (inherits from `OpenAIResponsesAPIConfig`)
3. `ProviderConfigManager.get_provider_responses_api_config()` returns the generated config
4. Request/response transformation is inherited from OpenAI — no custom code needed

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

For providers that are *mostly* OpenAI-compatible but need small overrides (e.g. preset model handling),
you can inherit from `OpenAIResponsesAPIConfig` and override only what's needed — see
`litellm/llms/perplexity/responses/transformation.py` for a minimal example (~40 lines).

## Implementation Details

### How It Works

1. `json_loader.py` loads `providers.json` on import
2. `dynamic_config.py` generates config classes on-demand
3. Provider resolution checks JSON registry first
4. ProviderConfigManager returns JSON-based configs

### Integration Points

The JSON system is integrated at:
- `litellm/litellm_core_utils/get_llm_provider_logic.py` - Provider resolution
- `litellm/utils.py` - ProviderConfigManager (chat + responses)
- `litellm/responses/main.py` - Responses API routing
- `litellm/constants.py` - openai_compatible_providers list
