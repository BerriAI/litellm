# JSON Endpoint Configuration Guide

## Overview

This guide explains how to add new pass-through endpoints to LiteLLM using JSON configuration instead of writing Python code. This approach makes it **10X easier** to add new endpoints.

## Quick Start

### Adding a New Endpoint

1. Open `endpoints_config.json`
2. Add your endpoint configuration:

```json
{
  "your_provider": {
    "route_prefix": "/your_provider/{endpoint:path}",
    "target_base_url": "https://api.your-provider.com",
    "target_base_url_env": "YOUR_PROVIDER_API_BASE",
    "auth": {
      "type": "bearer_token",
      "env_var": "YOUR_PROVIDER_API_KEY"
    },
    "streaming": {
      "detection_method": "request_body_field",
      "field_name": "stream"
    },
    "features": {
      "require_litellm_auth": true,
      "subpath_routing": true
    },
    "tags": ["Your Provider Pass-through", "pass-through"],
    "docs_url": "https://docs.litellm.ai/docs/pass_through/your_provider"
  }
}
```

3. Set the environment variable: `YOUR_PROVIDER_API_KEY=your_key`
4. Restart LiteLLM proxy
5. Done! Your endpoint is now available at `/your_provider/*`

## Configuration Schema

### Top-Level Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `provider_slug` | string | Yes (auto) | Unique identifier (key in JSON object) |
| `route_prefix` | string | Yes | FastAPI route pattern |
| `target_base_url` | string | Conditional | Static base URL |
| `target_base_url_template` | string | Conditional | Dynamic base URL template |
| `target_base_url_env` | string | No | Environment variable for URL override |
| `auth` | object | Yes | Authentication configuration |
| `streaming` | object | Yes | Streaming detection configuration |
| `features` | object | No | Feature flags |
| `tags` | array | No | OpenAPI documentation tags |
| `docs_url` | string | No | Link to provider documentation |

### Authentication Configuration (`auth`)

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `type` | string | Yes | `bearer_token`, `custom_header`, `query_param`, or `custom_handler` |
| `env_var` | string | Yes | Environment variable containing API key |
| `header_name` | string | Conditional | Header name (for `custom_header` type) |
| `header_format` | string | No | Format string for header value (default: `Bearer {api_key}`) |
| `param_name` | string | Conditional | Query param name (for `query_param` type) |

### Streaming Configuration (`streaming`)

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `detection_method` | string | Yes | `request_body_field`, `url_contains`, `header`, or `none` |
| `field_name` | string | Conditional | Request body field to check (for `request_body_field`) |
| `pattern` | string | Conditional | Pattern to match in URL (for `url_contains`) |
| `query_param_suffix` | string | No | Query param to append for streaming (e.g., `?alt=sse`) |
| `response_content_type` | string | No | Content-Type for streaming responses (default: `text/event-stream`) |

### Features Configuration (`features`)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `forward_headers` | boolean | `false` | Forward incoming request headers |
| `merge_query_params` | boolean | `false` | Merge query params from request |
| `require_litellm_auth` | boolean | `true` | Require LiteLLM API key authentication |
| `subpath_routing` | boolean | `true` | Support wildcard subpath routing |
| `custom_auth_handler` | boolean | `false` | Use custom authentication handler |
| `dynamic_base_url` | boolean | `false` | Base URL is dynamically constructed |
| `custom_query_params` | boolean | `false` | Custom query param handling |

## Authentication Types

### Bearer Token (Most Common)

Standard `Authorization: Bearer <token>` header:

```json
{
  "auth": {
    "type": "bearer_token",
    "env_var": "PROVIDER_API_KEY"
  }
}
```

### Custom Header

Custom header name and format:

```json
{
  "auth": {
    "type": "custom_header",
    "env_var": "PROVIDER_API_KEY",
    "header_name": "x-api-key",
    "header_format": "{api_key}"
  }
}
```

### Query Parameter

API key passed as query parameter:

```json
{
  "auth": {
    "type": "query_param",
    "env_var": "PROVIDER_API_KEY",
    "param_name": "key"
  }
}
```

### Custom Handler

For complex authentication (OAuth, SigV4, etc.):

```json
{
  "auth": {
    "type": "custom_handler",
    "env_var": "PROVIDER_CREDENTIALS",
    "handler_function": "custom_auth_handler_name"
  },
  "features": {
    "custom_auth_handler": true
  }
}
```

Note: Custom handlers require Python code for complex auth logic.

## Streaming Detection Methods

### Request Body Field (Recommended)

Check a field in the request body:

```json
{
  "streaming": {
    "detection_method": "request_body_field",
    "field_name": "stream"
  }
}
```

Checks if `request.body.stream == true`.

### URL Pattern

Check if URL contains a pattern:

```json
{
  "streaming": {
    "detection_method": "url_contains",
    "pattern": "stream"
  }
}
```

Checks if `"stream"` appears anywhere in the endpoint path.

### Header

Check the Accept header:

```json
{
  "streaming": {
    "detection_method": "header"
  }
}
```

Checks if `Accept: text/event-stream` header is present.

### None

No streaming support:

```json
{
  "streaming": {
    "detection_method": "none"
  }
}
```

## Examples

### Example 1: Simple OpenAI-Compatible API

```json
{
  "my_provider": {
    "route_prefix": "/my_provider/{endpoint:path}",
    "target_base_url": "https://api.myprovider.com/v1",
    "target_base_url_env": "MY_PROVIDER_API_BASE",
    "auth": {
      "type": "bearer_token",
      "env_var": "MY_PROVIDER_API_KEY"
    },
    "streaming": {
      "detection_method": "request_body_field",
      "field_name": "stream"
    },
    "features": {
      "require_litellm_auth": true,
      "subpath_routing": true
    },
    "tags": ["My Provider Pass-through", "pass-through"]
  }
}
```

**Usage:**
```bash
export MY_PROVIDER_API_KEY="your-key"
curl http://localhost:4000/my_provider/chat/completions \
  -H "Authorization: Bearer YOUR_LITELLM_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "Hello"}]}'
```

### Example 2: Custom Header Authentication

```json
{
  "anthropic": {
    "route_prefix": "/anthropic/{endpoint:path}",
    "target_base_url": "https://api.anthropic.com",
    "auth": {
      "type": "custom_header",
      "env_var": "ANTHROPIC_API_KEY",
      "header_name": "x-api-key",
      "header_format": "{api_key}"
    },
    "streaming": {
      "detection_method": "request_body_field",
      "field_name": "stream"
    },
    "features": {
      "forward_headers": true,
      "require_litellm_auth": true,
      "subpath_routing": true
    },
    "tags": ["Anthropic Pass-through", "pass-through"]
  }
}
```

### Example 3: Query Parameter Authentication

```json
{
  "gemini": {
    "route_prefix": "/gemini/{endpoint:path}",
    "target_base_url": "https://generativelanguage.googleapis.com",
    "auth": {
      "type": "query_param",
      "env_var": "GEMINI_API_KEY",
      "param_name": "key"
    },
    "streaming": {
      "detection_method": "url_contains",
      "pattern": "stream"
    },
    "features": {
      "require_litellm_auth": true,
      "custom_query_params": true
    },
    "tags": ["Google AI Studio Pass-through", "pass-through"]
  }
}
```

### Example 4: With Streaming Query Param

```json
{
  "vertex_ai": {
    "route_prefix": "/vertex_ai/{endpoint:path}",
    "target_base_url_template": "https://{location}-aiplatform.googleapis.com/",
    "auth": {
      "type": "custom_handler",
      "env_var": "VERTEX_CREDENTIALS"
    },
    "streaming": {
      "detection_method": "url_contains",
      "pattern": "stream",
      "query_param_suffix": "?alt=sse"
    },
    "features": {
      "require_litellm_auth": true,
      "dynamic_base_url": true
    },
    "tags": ["Vertex AI Pass-through", "pass-through"]
  }
}
```

## Testing Your Endpoint

### 1. Validate Configuration

```python
from litellm.proxy.pass_through_endpoints.endpoint_config_registry import EndpointConfigRegistry

# Load and validate
EndpointConfigRegistry.reload()
config = EndpointConfigRegistry.get("your_provider")
print(f"Loaded config: {config}")
```

### 2. Test Endpoint

```bash
# Start proxy
litellm --config config.yaml

# Test endpoint
curl http://localhost:4000/your_provider/test \
  -H "Authorization: Bearer YOUR_LITELLM_KEY" \
  -H "Content-Type: application/json" \
  -d '{"test": "data"}'
```

### 3. Check Logs

Look for:
- `Registering JSON-configured endpoint: your_provider`
- `Successfully registered your_provider endpoint`

## Migration from Python to JSON

### Before (Python):

```python
@router.api_route(
    "/cohere/{endpoint:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    tags=["Cohere Pass-through", "pass-through"],
)
async def cohere_proxy_route(
    endpoint: str,
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    base_target_url = os.getenv("COHERE_API_BASE") or "https://api.cohere.com"
    encoded_endpoint = httpx.URL(endpoint).path
    
    if not encoded_endpoint.startswith("/"):
        encoded_endpoint = "/" + encoded_endpoint
    
    base_url = httpx.URL(base_target_url)
    updated_url = base_url.copy_with(path=encoded_endpoint)
    
    cohere_api_key = passthrough_endpoint_router.get_credentials(
        custom_llm_provider="cohere",
        region_name=None,
    )
    
    is_streaming_request = False
    if "stream" in str(updated_url):
        is_streaming_request = True
    
    endpoint_func = create_pass_through_route(
        endpoint=endpoint,
        target=str(updated_url),
        custom_headers={"Authorization": "Bearer {}".format(cohere_api_key)},
        is_streaming_request=is_streaming_request,
    )
    
    return await endpoint_func(request, fastapi_response, user_api_key_dict)
```

### After (JSON):

```json
{
  "cohere": {
    "route_prefix": "/cohere/{endpoint:path}",
    "target_base_url": "https://api.cohere.com",
    "target_base_url_env": "COHERE_API_BASE",
    "auth": {
      "type": "bearer_token",
      "env_var": "COHERE_API_KEY"
    },
    "streaming": {
      "detection_method": "url_contains",
      "pattern": "stream"
    },
    "features": {
      "require_litellm_auth": true,
      "subpath_routing": true
    },
    "tags": ["Cohere Pass-through", "pass-through"]
  }
}
```

**Result:** 50+ lines of Python → 14 lines of JSON (73% reduction!)

## Advanced Features

### Environment Variable Override

Users can override the base URL:

```bash
export PROVIDER_API_BASE="https://custom.api.com"
```

The `target_base_url_env` field enables this.

### Subpath Routing

With `"subpath_routing": true`, requests to:
- `/provider/v1/chat/completions`
- `/provider/v1/embeddings`
- `/provider/any/nested/path`

All work automatically.

### Header Forwarding

With `"forward_headers": true`, incoming headers are forwarded to the target API. Useful for:
- Custom authentication headers
- Request IDs
- User agents
- etc.

### Query Param Merging

With `"merge_query_params": true`, query params from the incoming request are merged with the target URL.

## Troubleshooting

### Endpoint Not Registered

**Symptom:** `404 Not Found` when calling endpoint

**Solutions:**
1. Check JSON syntax: `python -m json.tool endpoints_config.json`
2. Check logs for registration errors
3. Verify file is named `endpoints_config.json`
4. Restart proxy

### Authentication Errors

**Symptom:** `401 Unauthorized` or missing API key errors

**Solutions:**
1. Verify environment variable is set: `echo $YOUR_API_KEY_ENV`
2. Check `env_var` field matches your environment variable name
3. Ensure auth type matches provider requirements

### Streaming Not Working

**Symptom:** Streaming responses don't stream

**Solutions:**
1. Check `detection_method` is correct for your provider
2. Verify `field_name` matches the request body field
3. Check if provider requires query param suffix

### URL Construction Issues

**Symptom:** Incorrect target URLs in logs

**Solutions:**
1. Check `target_base_url` includes trailing slash if needed
2. Verify environment variable override isn't set incorrectly
3. Check endpoint path construction in logs

## Best Practices

### 1. Use Environment Variables

Always allow environment variable overrides:

```json
{
  "target_base_url": "https://api.provider.com",
  "target_base_url_env": "PROVIDER_API_BASE"
}
```

### 2. Document Your Endpoints

Include tags and docs_url:

```json
{
  "tags": ["Provider Pass-through", "pass-through"],
  "docs_url": "https://docs.litellm.ai/docs/pass_through/provider"
}
```

### 3. Test All Authentication Types

Test with:
- Valid API key
- Invalid API key
- Missing API key

### 4. Test Streaming

Test both:
- Streaming requests
- Non-streaming requests

### 5. Version Your Config

Use git to version control `endpoints_config.json`.

## Contributing

### Adding Your Endpoint

1. Add configuration to `endpoints_config.json`
2. Test thoroughly
3. Add to `endpoints_config_examples.json`
4. Update documentation
5. Submit PR!

### Validation Checklist

- [ ] JSON syntax is valid
- [ ] Required fields are present
- [ ] Auth type is correct
- [ ] Streaming detection works
- [ ] Endpoint responds correctly
- [ ] Error handling works
- [ ] Documentation is complete

## FAQ

**Q: Can I use this for non-OpenAI-compatible APIs?**  
A: Yes! Configure auth and streaming detection to match your API.

**Q: What about complex authentication like OAuth?**  
A: Use `"type": "custom_handler"` and implement auth logic in Python.

**Q: Can I migrate existing Python endpoints?**  
A: Yes! Most endpoints can be migrated to JSON configuration.

**Q: What's the performance impact?**  
A: Negligible. Config is loaded once on startup.

**Q: Can I hot-reload config changes?**  
A: Not yet, but it's planned. Restart the proxy for now.

**Q: Do I still need Python for complex endpoints?**  
A: Only for very complex auth or transformations. 80%+ of endpoints work with just JSON.

## Support

- Documentation: https://docs.litellm.ai
- Discord: https://discord.gg/wuPM9dRgDw
- GitHub Issues: https://github.com/BerriAI/litellm/issues

## License

This feature is part of LiteLLM and follows the same license.
