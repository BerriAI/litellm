# SDK Header Support

LiteLLM SDK provides comprehensive support for passing additional headers with API requests. This is essential for enterprise environments using API gateways, service meshes, and multi-tenant architectures.

## Overview

Headers can be passed to LiteLLM in three ways, with the following priority order:
1. **Request-specific headers** (highest priority)
2. **extra_headers parameter**
3. **Global litellm.headers** (lowest priority)

When the same header key is specified in multiple places, the higher priority value will be used.

## Usage Methods

### 1. Global Headers (litellm.headers)

Set headers that will be included in all API requests:

```python
import litellm

# Set global headers for all requests
litellm.headers = {
    "X-API-Gateway-Key": "your-gateway-key",
    "X-Company-ID": "acme-corp",
    "X-Environment": "production"
}

# Now all completion calls will include these headers
response = litellm.completion(
    model="claude-3-5-sonnet-latest",
    messages=[{"role": "user", "content": "Hello"}]
)
```

### 2. Per-Request Headers (extra_headers)

Pass headers for specific requests using the `extra_headers` parameter:

```python
import litellm

response = litellm.completion(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello"}],
    extra_headers={
        "X-Request-ID": "req-12345",
        "X-Tenant-ID": "tenant-abc",
        "X-Custom-Auth": "bearer-token-xyz"
    }
)
```

### 3. Request Headers (headers parameter)

Use the `headers` parameter for the highest priority header control:

```python
import litellm

response = litellm.completion(
    model="claude-3-5-sonnet-latest",
    messages=[{"role": "user", "content": "Hello"}],
    headers={
        "X-Priority-Header": "high-priority-value",
        "Authorization": "Bearer custom-token"
    }
)
```

### 4. Combining All Methods

You can combine all three methods. Headers will be merged with the priority order:

```python
import litellm

# Global headers (lowest priority)
litellm.headers = {
    "X-Company-ID": "acme-corp",
    "X-Shared-Header": "global-value"
}

response = litellm.completion(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello"}],
    extra_headers={
        "X-Request-ID": "req-12345",
        "X-Shared-Header": "extra-value"  # Overrides global
    },
    headers={
        "X-Priority-Header": "important",
        "X-Shared-Header": "request-value"  # Overrides both global and extra
    }
)

# Final headers sent to API:
# {
#     "X-Company-ID": "acme-corp",           # From global
#     "X-Request-ID": "req-12345",           # From extra_headers
#     "X-Priority-Header": "important",      # From headers
#     "X-Shared-Header": "request-value"     # From headers (highest priority)
# }
```

## Enterprise Use Cases

### API Gateway Integration (Apigee, Kong, AWS API Gateway)

```python
import litellm

# Set up headers for API gateway routing and authentication
litellm.headers = {
    "X-API-Gateway-Key": "your-gateway-key",
    "X-Route-Version": "v2"
}

# Per-tenant requests
response = litellm.completion(
    model="claude-3-5-sonnet-latest",
    messages=[{"role": "user", "content": "Analyze this data"}],
    extra_headers={
        "X-Tenant-ID": "tenant-123",
        "X-Department": "engineering"
    }
)
```

### Service Mesh (Istio, Linkerd)

```python
import litellm

response = litellm.completion(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello"}],
    extra_headers={
        "X-Trace-ID": "trace-abc-123",
        "X-Service-Name": "ai-service",
        "X-Version": "1.2.3"
    }
)
```

### Multi-Tenant SaaS Applications

```python
import litellm

def make_ai_request(user_id, tenant_id, content):
    return litellm.completion(
        model="claude-3-5-sonnet-latest",
        messages=[{"role": "user", "content": content}],
        extra_headers={
            "X-User-ID": user_id,
            "X-Tenant-ID": tenant_id,
            "X-Request-Time": str(int(time.time()))
        }
    )

# Usage
response = make_ai_request("user-456", "tenant-org-1", "Help me write code")
```

### Request Tracing and Debugging

```python
import litellm
import uuid

def traced_completion(model, messages, **kwargs):
    trace_id = str(uuid.uuid4())

    return litellm.completion(
        model=model,
        messages=messages,
        extra_headers={
            "X-Trace-ID": trace_id,
            "X-Debug-Mode": "true",
            "X-Source-Service": "my-app"
        },
        **kwargs
    )

# Usage
response = traced_completion(
    model="gpt-4",
    messages=[{"role": "user", "content": "Debug this issue"}]
)
```

### Custom Authentication

```python
import litellm

def get_custom_auth_token():
    # Your custom authentication logic
    return "custom-auth-token"

response = litellm.completion(
    model="claude-3-5-sonnet-latest",
    messages=[{"role": "user", "content": "Hello"}],
    headers={
        "X-Custom-Auth": get_custom_auth_token(),
        "X-Auth-Type": "custom"
    }
)
```

## Provider Support

Headers are supported across all LiteLLM providers including:

- **OpenAI** (GPT models)
- **Anthropic** (Claude models)
- **Cohere**
- **Hugging Face**
- **Custom providers**
- **Azure OpenAI**
- **AWS Bedrock**
- **Google Vertex AI**

Each provider will receive your custom headers along with their required authentication and API-specific headers.

## Best Practices

### 1. Use Meaningful Header Names
```python
# Good
extra_headers = {
    "X-Request-ID": "req-12345",
    "X-Tenant-ID": "org-456"
}

# Avoid
extra_headers = {
    "custom1": "value1",
    "h2": "value2"
}
```

### 2. Include Tracing Information
```python
extra_headers = {
    "X-Trace-ID": trace_id,
    "X-Span-ID": span_id,
    "X-Service-Name": "ai-service"
}
```

### 3. Handle Sensitive Information Carefully
```python
# Don't log sensitive headers
import os

if os.getenv("ENVIRONMENT") != "production":
    extra_headers["X-Debug-User"] = user_id
```

### 4. Use Environment-Specific Headers
```python
import os

environment = os.getenv("ENVIRONMENT", "development")

litellm.headers = {
    "X-Environment": environment,
    "X-Service-Version": os.getenv("SERVICE_VERSION", "unknown")
}
```

## Troubleshooting

### Headers Not Being Passed

If your headers aren't reaching the API:

1. **Check Header Names**: Ensure header names don't conflict with provider-specific headers
2. **Verify Priority**: Remember that `headers` > `extra_headers` > `litellm.headers`
3. **Test with Logging**: Enable verbose logging to see what headers are being sent

```python
import litellm

# Enable debug logging
litellm.set_verbose = True

response = litellm.completion(
    model="gpt-4",
    messages=[{"role": "user", "content": "test"}],
    extra_headers={"X-Debug": "test"}
)
```

### Gateway or Proxy Issues

If using API gateways or proxies:

1. **Check Gateway Requirements**: Verify required headers for your gateway
2. **Test Direct vs Gateway**: Compare direct API calls vs gateway calls
3. **Validate Header Format**: Some gateways have header format requirements

## Security Considerations

1. **Don't Log Sensitive Headers**: Avoid logging authentication tokens or personal data
2. **Use HTTPS**: Always use secure connections when passing sensitive headers
3. **Validate Header Values**: Sanitize user-provided header values
4. **Rotate Keys**: Regularly rotate any API keys passed in headers

```python
import litellm
import re

def safe_header_value(value):
    # Remove potentially dangerous characters
    return re.sub(r'[^\w\-.]', '', str(value))

response = litellm.completion(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello"}],
    extra_headers={
        "X-User-ID": safe_header_value(user_id)
    }
)
```