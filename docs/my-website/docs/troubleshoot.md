# Troubleshooting & Error Handling Guide

## LiteLLM Exception Types (from `exceptions.py`)

LiteLLM provides OpenAI-compatible exceptions with additional debugging context. All exceptions inherit from their OpenAI counterparts but include extra attributes for better debugging.

### Core Exception Attributes

Every LiteLLM exception includes these debugging attributes:

```python
# Available on ALL LiteLLM exceptions
e.llm_provider          # "openai", "anthropic", "vertex_ai", etc.
e.model                 # "gpt-4", "claude-3", etc.
e.message               # Error message with "litellm." prefix
e.status_code           # HTTP status code
e.litellm_debug_info    # Additional debug context
e.max_retries           # LiteLLM retry configuration
e.num_retries           # Actual number of retries attempted
```

### Complete Exception Reference

```python
# Authentication & Authorization (4xx)
litellm.AuthenticationError        # 401 - Invalid API key/credentials
litellm.PermissionDeniedError      # 403 - Insufficient permissions  
litellm.NotFoundError              # 404 - Model/resource not found

# Request Issues (4xx)  
litellm.BadRequestError            # 400 - Invalid request format
litellm.UnprocessableEntityError   # 422 - Valid format, invalid data
litellm.UnsupportedParamsError     # 400 - Unsupported parameters
litellm.ContextWindowExceededError # 400 - Input too long (subclass of BadRequestError)
litellm.ContentPolicyViolationError # 400 - Content against policies
litellm.RejectedRequestError       # 400 - Rejected by guardrails

# Rate Limiting & Capacity (4xx)
litellm.RateLimitError             # 429 - Rate limit exceeded
litellm.Timeout                    # 408 - Request timeout

# Server Issues (5xx)
litellm.InternalServerError        # 500 - Server error
litellm.ServiceUnavailableError    # 503 - Service overloaded
litellm.APIConnectionError         # 500 - Connection failed
litellm.APIError                   # 500 - Generic API error
litellm.APIResponseValidationError # 500 - Invalid response format

# LiteLLM Specific
litellm.BudgetExceededError        # Budget/spend limits exceeded
litellm.JSONSchemaValidationError  # Response doesn't match schema
litellm.LiteLLMUnknownProvider     # Invalid provider specified
```

## Debugging in Python SDK vs LiteLLM Proxy

### Python SDK Debugging

When using LiteLLM Python SDK directly:

```python
import litellm
from litellm import AuthenticationError, RateLimitError, ContextWindowExceededError

# Enable debugging
litellm.set_verbose = True  # Basic logging
# OR
litellm._turn_on_debug()    # Detailed logging (⚠️ Don't use in production)

def debug_litellm_exception(e):
    """Extract all available debugging info from LiteLLM exception"""
    return {
        "error_type": type(e).__name__,
        "error_source": "LLM API",  # All SDK errors are from LLM APIs
        "llm_provider": getattr(e, 'llm_provider', 'unknown'),
        "model": getattr(e, 'model', 'unknown'),
        "status_code": getattr(e, 'status_code', None),
        "message": getattr(e, 'message', str(e)),
        "debug_info": getattr(e, 'litellm_debug_info', None),
        "retries": f"{getattr(e, 'num_retries', 0)}/{getattr(e, 'max_retries', 0)}",
        "headers": getattr(e, 'headers', {}),
    }

# Usage example
try:
    response = litellm.completion(
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello"}]
    )
except Exception as e:
    debug_info = debug_litellm_exception(e)
    print(f"🐛 Debug Info: {debug_info}")
    
    # Check specific error types
    if isinstance(e, AuthenticationError):
        print("❌ Authentication failed - check your API key")
    elif isinstance(e, RateLimitError):
        print(f"⏰ Rate limited - retried {e.num_retries} times")
    elif isinstance(e, ContextWindowExceededError):
        print(f"📝 Context too long for model {e.model}")
```

### LiteLLM Proxy Debugging

When using LiteLLM Proxy, there are TWO types of errors:

#### 1. **LLM API Errors** (from upstream providers)
These are the same LiteLLM exceptions but wrapped in `ProxyException`:

```python
# Proxy wraps LLM API errors like this:
{
    "error": {
        "message": "litellm.AuthenticationError: Invalid API key provided",
        "type": "auth_error", 
        "code": "401",
        "param": null
    }
}
```

#### 2. **Proxy-Only Errors** (from LiteLLM Proxy itself)
These originate from the proxy layer:

```python
# Proxy authentication errors:
{
    "error": {
        "message": "Authentication Error, Invalid API key provided",
        "type": "auth_error",
        "code": "401", 
        "param": null
    }
}

# Proxy configuration errors:
{
    "error": {
        "message": "Budget exceeded for key",
        "type": "budget_exceeded",
        "code": "400",
        "param": null  
    }
}
```

### How to Identify Error Source in Proxy

```python
import requests
import json

def debug_proxy_error(response):
    """Determine if error is from LLM API vs LiteLLM Proxy"""
    try:
        error_data = response.json()["error"]
        message = error_data.get("message", "")
        
        # Check if it's an LLM API error wrapped by proxy
        if message.startswith("litellm."):
            return {
                "source": "LLM_API",
                "provider": extract_provider_from_message(message),
                "original_error": message.replace("litellm.", "").split(":")[0],
                "full_message": message,
                "proxy_wrapped": True
            }
        
        # Check for proxy-specific error types
        elif any(keyword in message.lower() for keyword in [
            "authentication error", "budget exceeded", "rate limit", 
            "invalid api key", "no healthy deployment", "no deployments available"
        ]):
            return {
                "source": "LITELLM_PROXY", 
                "error_category": categorize_proxy_error(message),
                "full_message": message,
                "proxy_wrapped": False
            }
        
        return {
            "source": "UNKNOWN",
            "full_message": message,
            "requires_investigation": True
        }
        
    except Exception as e:
        return {"source": "PARSE_ERROR", "error": str(e)}

def extract_provider_from_message(message):
    """Extract provider from litellm error message"""
    # Look for provider hints in the message
    providers = ["openai", "anthropic", "vertex_ai", "azure", "cohere", "bedrock"]
    for provider in providers:
        if provider in message.lower():
            return provider
    return "unknown"

def categorize_proxy_error(message):
    """Categorize proxy-specific errors"""
    message_lower = message.lower()
    if "authentication" in message_lower or "api key" in message_lower:
        return "authentication"
    elif "budget" in message_lower:
        return "budget_management"
    elif "rate limit" in message_lower:
        return "rate_limiting"
    elif "deployment" in message_lower:
        return "routing/load_balancing"
    return "other_proxy_error"

# Usage with requests
try:
    response = requests.post(
        "http://localhost:4000/v1/chat/completions",
        headers={"Authorization": "Bearer your-key"},
        json={
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}]
        }
    )
    response.raise_for_status()
except requests.exceptions.HTTPError as e:
    debug_info = debug_proxy_error(e.response)
    print(f"🔍 Error Analysis: {json.dumps(debug_info, indent=2)}")
    
    if debug_info["source"] == "LLM_API":
        print(f"🌐 This is an {debug_info['provider']} API error, not a proxy issue")
        print(f"📋 Original error: {debug_info['original_error']}")
    elif debug_info["source"] == "LITELLM_PROXY":
        print(f"🔧 This is a LiteLLM Proxy configuration issue")
        print(f"📂 Category: {debug_info['error_category']}")
```

### Advanced Debugging Techniques

#### 1. **Enable Proxy Debug Mode**
```bash
# Run proxy with debug logging
litellm --config config.yaml --debug

# Or set in config.yaml:
general_settings:
  set_verbose: true
```

#### 2. **Custom Error Handler for Proxy**
```python
def comprehensive_error_handler(response):
    """Complete error analysis for proxy responses"""
    
    if response.status_code >= 400:
        try:
            error_data = response.json()["error"]
            
            analysis = {
                "status_code": response.status_code,
                "error_type": error_data.get("type"),
                "error_code": error_data.get("code"),
                "message": error_data.get("message"),
                "timestamp": response.headers.get("date"),
                "request_id": response.headers.get("x-litellm-call-id"),
            }
            
            # Determine error source
            if error_data["message"].startswith("litellm."):
                analysis["source"] = "LLM_API"
                analysis["wrapped_by_proxy"] = True
                
                # Extract LiteLLM exception details
                exception_name = error_data["message"].split(":")[0].replace("litellm.", "")
                analysis["litellm_exception"] = exception_name
                
            else:
                analysis["source"] = "PROXY"
                analysis["proxy_component"] = identify_proxy_component(error_data)
            
            # Add resolution suggestions
            analysis["suggested_actions"] = get_resolution_suggestions(analysis)
            
            return analysis
            
        except Exception as parse_error:
            return {"parse_error": str(parse_error), "raw_response": response.text}
    
    return {"status": "success"}

def identify_proxy_component(error_data):
    """Identify which proxy component caused the error"""
    message = error_data.get("message", "").lower()
    
    if "authentication" in message:
        return "auth_layer"
    elif "budget" in message or "spend" in message:
        return "budget_manager" 
    elif "rate limit" in message:
        return "rate_limiter"
    elif "deployment" in message:
        return "router"
    elif "cache" in message:
        return "cache_layer"
    elif "database" in message or "db" in message:
        return "database"
    
    return "unknown_component"

def get_resolution_suggestions(analysis):
    """Provide specific resolution steps based on error analysis"""
    suggestions = []
    
    if analysis["source"] == "LLM_API":
        suggestions.append("This is an upstream API error - check provider status")
        if "AuthenticationError" in analysis.get("litellm_exception", ""):
            suggestions.append("Verify your provider API key in proxy config")
        elif "RateLimitError" in analysis.get("litellm_exception", ""):
            suggestions.append("Check your provider rate limits and consider load balancing")
            
    elif analysis["source"] == "PROXY":
        component = analysis.get("proxy_component")
        if component == "auth_layer":
            suggestions.append("Check your LiteLLM proxy API key")
        elif component == "budget_manager":
            suggestions.append("Check budget settings in config or increase limits")
        elif component == "router":
            suggestions.append("Check model deployments and routing configuration")
    
    return suggestions
```

#### 3. **Monitoring Error Patterns**
```python
import time
from collections import defaultdict

class ProxyErrorMonitor:
    def __init__(self):
        self.error_counts = defaultdict(int)
        self.error_history = []
    
    def track_error(self, error_analysis):
        """Track errors for pattern detection"""
        timestamp = time.time()
        
        # Count by source and type
        key = f"{error_analysis['source']}:{error_analysis.get('error_type', 'unknown')}"
        self.error_counts[key] += 1
        
        # Store for pattern analysis
        self.error_history.append({
            "timestamp": timestamp,
            "analysis": error_analysis
        })
        
        # Keep only last 1000 errors
        if len(self.error_history) > 1000:
            self.error_history = self.error_history[-1000:]
    
    def get_error_summary(self, last_minutes=60):
        """Get error summary for the last N minutes"""
        cutoff = time.time() - (last_minutes * 60)
        recent_errors = [e for e in self.error_history if e["timestamp"] > cutoff]
        
        summary = {
            "total_errors": len(recent_errors),
            "llm_api_errors": len([e for e in recent_errors if e["analysis"]["source"] == "LLM_API"]),
            "proxy_errors": len([e for e in recent_errors if e["analysis"]["source"] == "PROXY"]),
            "most_common": defaultdict(int)
        }
        
        for error in recent_errors:
            error_type = error["analysis"].get("error_type", "unknown")
            summary["most_common"][error_type] += 1
        
        return summary

# Usage
monitor = ProxyErrorMonitor()

# In your error handling:
error_analysis = comprehensive_error_handler(response)
monitor.track_error(error_analysis)

# Get insights
summary = monitor.get_error_summary(last_minutes=30)
print(f"Error Summary (last 30 min): {summary}")
```

## Quick Debugging Checklist

### For Python SDK:
1. ✅ **Enable debug mode**: `litellm.set_verbose = True`
2. ✅ **Check exception type**: `isinstance(e, litellm.AuthenticationError)`
3. ✅ **Inspect provider**: `e.llm_provider` 
4. ✅ **Check retry info**: `e.num_retries` / `e.max_retries`
5. ✅ **Review debug info**: `e.litellm_debug_info`

### For LiteLLM Proxy:
1. ✅ **Check error source**: Does message start with `"litellm."`?
2. ✅ **Verify proxy auth**: Is this a proxy authentication error?
3. ✅ **Check request ID**: Look for `x-litellm-call-id` header
4. ✅ **Review proxy logs**: Enable `set_verbose: true` in config
5. ✅ **Test direct API**: Try same request directly to provider

### Common Error Patterns:

| Error Pattern | Source | Common Cause | Quick Fix |
|---------------|--------|--------------|-----------|
| `litellm.AuthenticationError` | LLM API | Invalid provider API key | Check API key in config |
| `Authentication Error, Invalid API key` | Proxy | Invalid proxy API key | Check your proxy auth token |
| `Budget exceeded` | Proxy | Spending limits hit | Increase budget or check usage |
| `No healthy deployment available` | Proxy | All models down/overloaded | Check model status/add fallbacks |
| `litellm.RateLimitError` | LLM API | Provider rate limits | Add retry logic or load balancing |

## Support & Contact

When contacting support, include:
- Error source (LLM API vs Proxy)
- Complete error message 
- `llm_provider` and `model` if available
- Request ID (`x-litellm-call-id`)
- Relevant configuration

[Schedule Demo 👋](https://calendly.com/d/4mp-gd3-k5k/berriai-1-1-onboarding-litellm-hosted-version)

[Community Discord 💭](https://discord.gg/wuPM9dRgDw)

Our numbers 📞 +1 (770) 8783-106 / ‭+1 (412) 618-6238‬

Our emails ✉️ ishaan@berri.ai / krrish@berri.ai

[![Chat on WhatsApp](https://img.shields.io/static/v1?label=Chat%20on&message=WhatsApp&color=success&logo=WhatsApp&style=flat-square)](https://wa.link/huol9n) [![Chat on Discord](https://img.shields.io/static/v1?label=Chat%20on&message=Discord&color=blue&logo=Discord&style=flat-square)](https://discord.gg/wuPM9dRgDw)

