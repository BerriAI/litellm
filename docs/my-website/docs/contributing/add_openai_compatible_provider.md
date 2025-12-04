---
id: add_openai_compatible_provider
title: Add OpenAI-Compatible Provider
sidebar_label: Add OpenAI-Compatible Provider
---

# Add OpenAI-Compatible Provider

Quick guide to add a new OpenAI-compatible provider to LiteLLM.

## Simple Providers (via JSON)

For providers that are fully OpenAI-compatible, just edit one JSON file.

### Steps

1. **Edit the JSON config file**

   Open `litellm/llms/openai_like/providers.json` and add your provider:

   ```json
   {
     "your_provider": {
       "base_url": "https://api.yourprovider.com/v1",
       "api_key_env": "YOUR_PROVIDER_API_KEY"
     }
   }
   ```

2. **Add to LlmProviders enum**

   Add your provider to `litellm/types/utils.py`:

   ```python
   class LlmProviders(str, Enum):
       # ... existing providers
       YOUR_PROVIDER = "your_provider"
   ```

3. **Test it**

   ```python
   import litellm
   import os

   os.environ["YOUR_PROVIDER_API_KEY"] = "..."

   response = litellm.completion(
       model="your_provider/model-name",
       messages=[{"role": "user", "content": "Hello"}]
   )
   ```

That's it! Your provider is now integrated.

## Optional JSON Fields

```json
{
  "your_provider": {
    "base_url": "https://api.yourprovider.com/v1",
    "api_key_env": "YOUR_PROVIDER_API_KEY",
    
    // Override base_url via environment variable
    "api_base_env": "YOUR_PROVIDER_API_BASE",
    
    // Parameter name mappings
    "param_mappings": {
      "max_completion_tokens": "max_tokens"
    },
    
    // Parameter constraints
    "constraints": {
      "temperature_max": 1.0,
      "temperature_min": 0.0
    },
    
    // Special handling
    "special_handling": {
      "convert_content_list_to_string": true
    }
  }
}
```

## When to Use Python Instead

Use a full Python implementation if you need:
- Custom authentication (OAuth, API key rotation)
- Complex request/response transformations
- Provider-specific streaming logic
- Non-standard endpoints

For Python implementations, see [existing providers](https://github.com/BerriAI/litellm/tree/main/litellm/llms).

## Example: PublicAI

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
