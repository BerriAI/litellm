# Troubleshooting & Error Handling

## Differentiating LiteLLM vs OpenAI Errors

LiteLLM maps exceptions across all providers to their OpenAI counterparts, but adds additional context to help you identify the source and debug issues effectively.

### Key Identifiers

#### 1. **Error Message Prefix**
LiteLLM errors always include a `litellm.` prefix in the error message:

```python
# LiteLLM Error
litellm.AuthenticationError: Invalid API key provided

# Original OpenAI Error  
openai.AuthenticationError: Invalid API key provided
```

#### 2. **LLM Provider Attribute**
All LiteLLM exceptions include an `llm_provider` attribute that tells you which provider caused the error:

```python
try:
    response = litellm.completion(
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello"}]
    )
except Exception as e:
    print(f"Provider: {getattr(e, 'llm_provider', 'unknown')}")
    print(f"Model: {getattr(e, 'model', 'unknown')}")
    print(f"Error: {e}")
```

#### 3. **Additional LiteLLM Attributes**
LiteLLM errors include extra debugging information:

```python
try:
    response = litellm.completion(model="gpt-4", messages=messages)
except litellm.RateLimitError as e:
    print(f"LLM Provider: {e.llm_provider}")  # e.g., "openai", "anthropic"
    print(f"Model: {e.model}")               # e.g., "gpt-4"
    print(f"Status Code: {e.status_code}")   # e.g., 429
    print(f"Max Retries: {e.max_retries}")   # LiteLLM specific
    print(f"Num Retries: {e.num_retries}")   # LiteLLM specific
    print(f"Debug Info: {e.litellm_debug_info}")  # LiteLLM specific
```

### Common Error Scenarios

#### Authentication Errors

**LiteLLM Error:**
```
litellm.AuthenticationError: InvalidAPIKey
Provider: openai
Model: gpt-4
```

**Solutions:**
- Check your API key configuration
- Verify the key is set correctly: `export OPENAI_API_KEY="your-key"`
- Ensure the key has proper permissions

#### Rate Limit Errors

**LiteLLM Error:**
```
litellm.RateLimitError: Rate limit exceeded for model gpt-4
Provider: openai  
Model: gpt-4
LiteLLM Retried: 3 times, LiteLLM Max Retries: 3
```

**Solutions:**
- Implement exponential backoff
- Use LiteLLM's built-in retry logic
- Consider upgrading your API plan

#### Context Window Exceeded

**LiteLLM Error:**
```
litellm.ContextWindowExceededError: This model's maximum context length is 4096 tokens
Provider: openai
Model: gpt-3.5-turbo
```

**Solutions:**
- Truncate your input messages
- Use LiteLLM's fallback logic to automatically switch to larger context models
- Split large inputs into smaller chunks

#### Model Not Found

**LiteLLM Error:**
```
litellm.NotFoundError: The model gpt-8 does not exist
Provider: openai
Model: gpt-8
```

**Solutions:**
- Check the model name spelling
- Verify the model is available in your region
- Use `litellm.model_list` to see available models

### Debugging Techniques

#### 1. **Enable Debug Mode**
```python
import litellm
litellm.set_verbose = True  # Enable detailed logging

# Or for more granular control
litellm._turn_on_debug()  # ⚠️ Don't use in production - logs API keys
```

#### 2. **Custom Error Handling**
```python
import litellm
from litellm import AuthenticationError, RateLimitError, ContextWindowExceededError

def handle_litellm_error(e):
    """Custom error handler that provides specific guidance"""
    
    if isinstance(e, AuthenticationError):
        return {
            "error_type": "Authentication",
            "provider": e.llm_provider,
            "solution": "Check your API key configuration",
            "debug_info": e.litellm_debug_info
        }
    
    elif isinstance(e, RateLimitError):
        return {
            "error_type": "Rate Limit", 
            "provider": e.llm_provider,
            "retries": f"{e.num_retries}/{e.max_retries}",
            "solution": "Implement backoff or upgrade plan"
        }
    
    elif isinstance(e, ContextWindowExceededError):
        return {
            "error_type": "Context Window",
            "provider": e.llm_provider,
            "model": e.model,
            "solution": "Reduce input size or use larger context model"
        }
    
    return {
        "error_type": "Unknown",
        "provider": getattr(e, 'llm_provider', 'unknown'),
        "message": str(e)
    }

# Usage
try:
    response = litellm.completion(model="gpt-4", messages=messages)
except Exception as e:
    error_info = handle_litellm_error(e)
    print(f"Error Type: {error_info['error_type']}")
    print(f"Provider: {error_info['provider']}")
    print(f"Solution: {error_info['solution']}")
```

#### 3. **Provider-Specific Error Handling**
```python
try:
    response = litellm.completion(model="gpt-4", messages=messages)
except Exception as e:
    provider = getattr(e, 'llm_provider', 'unknown')
    
    if provider == "openai":
        # OpenAI-specific error handling
        print("OpenAI error occurred")
    elif provider == "anthropic":
        # Anthropic-specific error handling  
        print("Anthropic error occurred")
    elif provider == "vertex_ai":
        # Google Vertex AI-specific error handling
        print("Vertex AI error occurred")
    
    print(f"Full error: {e}")
```

### Error Response Structure

LiteLLM errors maintain OpenAI compatibility while adding helpful context:

```python
{
    "error": {
        "message": "litellm.RateLimitError: Rate limit exceeded",
        "type": "rate_limit_error", 
        "code": 429,
        "llm_provider": "openai",      # LiteLLM addition
        "model": "gpt-4",              # LiteLLM addition  
        "litellm_debug_info": "...",   # LiteLLM addition
        "max_retries": 3,              # LiteLLM addition
        "num_retries": 2               # LiteLLM addition
    }
}
```

### Quick Reference: Error Types

| Error Type | LiteLLM Class | Common Causes | Key Solutions |
|------------|---------------|---------------|---------------|
| 400 | `BadRequestError` | Invalid parameters | Check request format |
| 401 | `AuthenticationError` | Invalid API key | Verify credentials |
| 403 | `PermissionDeniedError` | Insufficient permissions | Check account access |
| 404 | `NotFoundError` | Model not found | Verify model name |
| 429 | `RateLimitError` | Rate limit exceeded | Implement backoff |
| 500 | `InternalServerError` | Server error | Retry request |
| 503 | `ServiceUnavailableError` | Service overloaded | Wait and retry |

### When to Contact Support

Contact LiteLLM support when you see:
- Persistent errors across multiple providers
- Unexpected behavior in error mapping
- Missing error context that would help debugging
- Provider-specific errors that aren't well documented

## Support & Talk with founders
[Schedule Demo 👋](https://calendly.com/d/4mp-gd3-k5k/berriai-1-1-onboarding-litellm-hosted-version)

[Community Discord 💭](https://discord.gg/wuPM9dRgDw)

Our numbers 📞 +1 (770) 8783-106 / ‭+1 (412) 618-6238‬

Our emails ✉️ ishaan@berri.ai / krrish@berri.ai

[![Chat on WhatsApp](https://img.shields.io/static/v1?label=Chat%20on&message=WhatsApp&color=success&logo=WhatsApp&style=flat-square)](https://wa.link/huol9n) [![Chat on Discord](https://img.shields.io/static/v1?label=Chat%20on&message=Discord&color=blue&logo=Discord&style=flat-square)](https://discord.gg/wuPM9dRgDw) 

