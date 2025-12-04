# Adding an OpenAI-Compatible Provider

For providers that are OpenAI-compatible (use the same API format), you can add support by editing a single JSON file.

## Quick Start

1. Edit `litellm/llms/openai_like/providers.json`
2. Add your provider configuration
3. Test it works
4. Submit a PR

## Minimal Configuration

For a basic OpenAI-compatible provider, you only need 2 fields:

```json
{
  "your_provider": {
    "base_url": "https://api.yourprovider.com/v1",
    "api_key_env": "YOUR_PROVIDER_API_KEY"
  }
}
```

## Example: Adding a Provider

Let's add a fictional provider called "FastAI":

**Step 1:** Edit `litellm/llms/openai_like/providers.json`:

```json
{
  "publicai": {
    "base_url": "https://api.publicai.co/v1",
    "api_key_env": "PUBLICAI_API_KEY"
  },
  "fastai": {
    "base_url": "https://api.fastai.com/v1",
    "api_key_env": "FASTAI_API_KEY"
  }
}
```

**Step 2:** Add provider to enum in `litellm/types/utils.py`:

```python
class LlmProviders(str, Enum):
    # ... existing providers ...
    FASTAI = "fastai"
```

**Step 3:** Test it:

```python
import litellm
import os

os.environ["FASTAI_API_KEY"] = "your-api-key"

response = litellm.completion(
    model="fastai/gpt-4",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

That's it! Your provider is now integrated.

## Optional Configuration

If your provider has specific requirements, you can add optional fields:

```json
{
  "your_provider": {
    "base_url": "https://api.yourprovider.com/v1",
    "api_key_env": "YOUR_PROVIDER_API_KEY",
    
    // Allow base URL override via environment variable
    "api_base_env": "YOUR_PROVIDER_API_BASE",
    
    // Base class: "openai_gpt" (default) or "openai_like"
    "base_class": "openai_gpt",
    
    // Map OpenAI parameter names to your provider's names
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

## Available Options

### `base_url` (required)
The API endpoint URL. Should end with `/v1` typically.

### `api_key_env` (required)
Environment variable name for the API key.

### `api_base_env` (optional)
Environment variable to override `base_url` at runtime.

### `base_class` (optional)
- `"openai_gpt"` (default) - For standard OpenAI-compatible APIs
- `"openai_like"` - For APIs with slight differences

### `param_mappings` (optional)
Map OpenAI parameter names to provider-specific names:
```json
{
  "max_completion_tokens": "max_tokens",
  "stop": "stop_sequences"
}
```

### `constraints` (optional)
Apply parameter constraints:
- `temperature_max` - Maximum temperature value
- `temperature_min` - Minimum temperature value
- `temperature_min_with_n_gt_1` - Minimum temp when n > 1

### `special_handling` (optional)
- `convert_content_list_to_string`: true - Convert message content from list to string format

## When to Use Python Instead

Use a Python config class (instead of JSON) if you need:
- Custom authentication (OAuth, rotating tokens)
- Complex request/response transformations
- Provider-specific streaming logic
- Advanced parameter validation

For these cases, create a provider directory under `litellm/llms/your_provider/` following existing patterns.

## Testing Your Provider

```python
import litellm
import os

os.environ["YOUR_PROVIDER_API_KEY"] = "test-key"

# Test basic completion
response = litellm.completion(
    model="your_provider/model-name",
    messages=[{"role": "user", "content": "Hello"}]
)
print(response.choices[0].message.content)

# Test streaming
response = litellm.completion(
    model="your_provider/model-name",
    messages=[{"role": "user", "content": "Hello"}],
    stream=True
)
for chunk in response:
    print(chunk.choices[0].delta.content, end="")
```

## Pull Request Checklist

- [ ] Added provider to `litellm/llms/openai_like/providers.json`
- [ ] Added enum entry to `litellm/types/utils.py`
- [ ] Tested with real API key
- [ ] Verified streaming works
- [ ] Added to `openai_compatible_providers` list in `litellm/constants.py` (if not already there)

## Real Example: PublicAI

Here's how PublicAI was added:

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

**Usage:**
```python
import litellm
import os

os.environ["PUBLICAI_API_KEY"] = "your-key"

response = litellm.completion(
    model="publicai/swiss-ai/apertus-8b-instruct",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

That's it! Simple, fast, and maintainable.
