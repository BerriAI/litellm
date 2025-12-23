# JSON-Based OpenAI-Compatible Provider Configuration

This directory contains the new JSON-based configuration system for OpenAI-compatible providers.

## Overview

Instead of creating a full Python module for simple OpenAI-compatible providers, you can now define them in a single JSON file.

## Files

- `providers.json` - Configuration file for all JSON-based providers
- `json_loader.py` - Loads and parses the JSON configuration
- `dynamic_config.py` - Generates Python config classes from JSON
- `chat/` - Existing OpenAI-like chat completion handlers

## Adding a New Provider

### For Simple OpenAI-Compatible Providers

Adding a new JSON-configured provider requires updates in multiple locations:

#### 1. Add to `providers.json` (this directory)

```json
{
  "your_provider": {
    "base_url": "https://api.yourprovider.com/v1",
    "api_key_env": "YOUR_PROVIDER_API_KEY"
  }
}
```

#### 2. Add enum to `litellm/types/utils.py`

Add your provider to the `LlmProviders` enum:

```python
class LlmProviders(str, Enum):
    # ... existing providers ...
    YOUR_PROVIDER = "your_provider"
```

#### 3. Add endpoint mapping to `litellm/litellm_core_utils/get_llm_provider_logic.py`

In the `get_llm_provider()` function, add an endpoint recognition condition:

```python
elif endpoint == "https://api.yourprovider.com/v1":
    custom_llm_provider = "your_provider"
    dynamic_api_key = get_secret_str("YOUR_PROVIDER_API_KEY")
```

#### 4. Add to `litellm/constants.py` lists

Add your provider to three lists in `constants.py`:

- `openai_compatible_endpoints` - Add the base URL
- `openai_compatible_providers` - Add the provider name
- `openai_text_completion_compatible_providers` - Add the provider name (if it supports text completions)

#### 5. Add to `provider_endpoints_support.json` (root directory)

Add an entry documenting which endpoints your provider supports:

```json
{
  "your_provider": {
    "display_name": "Your Provider (`your_provider`)",
    "endpoints": {
      "chat_completions": true,
      "messages": false,
      "responses": false,
      "embeddings": true,
      "image_generations": false,
      "audio_transcriptions": false,
      "audio_speech": false,
      "moderations": false,
      "batches": false,
      "rerank": false,
      "a2a": false
    }
  }
}
```

#### Summary Checklist

- [ ] `providers.json` - Add provider configuration
- [ ] `litellm/types/utils.py` - Add enum entry
- [ ] `litellm/litellm_core_utils/get_llm_provider_logic.py` - Add endpoint mapping
- [ ] `litellm/constants.py` - Add to all three lists
- [ ] `provider_endpoints_support.json` - Add endpoint support documentation

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
2. `dynamic_config.py` generates config classes on-demand
3. Provider resolution checks JSON registry first
4. ProviderConfigManager returns JSON-based configs

### Integration Points

The JSON system is integrated at:
- `litellm/litellm_core_utils/get_llm_provider_logic.py` - Provider resolution
- `litellm/utils.py` - ProviderConfigManager
- `litellm/constants.py` - openai_compatible_providers list
