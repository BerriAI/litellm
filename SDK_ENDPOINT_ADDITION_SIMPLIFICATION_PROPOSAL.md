# SDK Endpoint Addition Simplification Proposal

## Executive Summary

This proposal outlines a strategy to make adding new SDK endpoints to LiteLLM **10X easier** by introducing a declarative JSON-based configuration system. Instead of writing 50-100 lines of Python code per endpoint, developers will simply add a JSON object with the endpoint configuration.

**Current State:** Adding a new pass-through endpoint requires:
- Writing custom Python functions (50-100+ lines)
- Understanding FastAPI routing
- Managing authentication, headers, streaming detection
- Handling provider-specific quirks manually
- Registering routes programmatically

**Future State:** Adding a new endpoint requires:
- Adding a single JSON object to a configuration file
- Zero Python code for standard endpoints
- Automatic handling of auth, streaming, headers
- Provider-specific customization via simple config flags

---

## Current Implementation Analysis

### Pain Points Identified

#### 1. **Hardcoded Route Definitions**
Each provider endpoint requires a dedicated `@router.api_route()` decorator with custom logic:

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
    # ... 30+ more lines of boilerplate
```

This is repeated 15+ times in `llm_passthrough_endpoints.py`!

#### 2. **Inconsistent Patterns**
Different providers use different approaches:
- **Gemini**: Query param-based API key
- **Anthropic**: `x-api-key` header
- **OpenAI**: `Authorization: Bearer` header
- **Bedrock**: AWS SigV4 signing
- **Vertex AI**: OAuth2 token exchange

#### 3. **Duplicate Logic**
Each endpoint reimplements:
- Base URL construction
- API key retrieval from environment
- Streaming detection
- Header management
- Error handling

#### 4. **Limited Reusability**
The `llm_passthrough_factory_proxy_route()` function exists but is underutilized. Most endpoints have custom implementations.

#### 5. **No Schema Validation**
Endpoint configurations are scattered across Python code with no centralized schema or validation.

---

## Proposed Solution: Declarative Endpoint Configuration

### Core Concept

Create a **JSON-based endpoint registry** similar to the existing `providers.json` for OpenAI-compatible providers, but extended to support **all pass-through endpoint types**.

### Architecture Overview

```
┌─────────────────────────────────────┐
│  endpoints_config.json              │
│  (Declarative endpoint definitions) │
└──────────────┬──────────────────────┘
               │
               │ Load on startup
               ▼
┌─────────────────────────────────────┐
│  EndpointConfigRegistry             │
│  (Validates & stores configs)       │
└──────────────┬──────────────────────┘
               │
               │ Auto-register routes
               ▼
┌─────────────────────────────────────┐
│  PassthroughEndpointFactory         │
│  (Generates route handlers)         │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  FastAPI App Routes                 │
└─────────────────────────────────────┘
```

---

## JSON Schema Design

### Basic Endpoint Definition

```json
{
  "provider_slug": {
    "route_prefix": "/provider/{endpoint:path}",
    "target_base_url": "https://api.provider.com",
    "target_base_url_env": "PROVIDER_API_BASE",
    "auth": {
      "type": "bearer_token",
      "env_var": "PROVIDER_API_KEY",
      "header_name": "Authorization",
      "header_format": "Bearer {api_key}"
    },
    "streaming": {
      "detection_method": "request_body_field",
      "field_name": "stream",
      "response_content_type": "text/event-stream"
    },
    "features": {
      "forward_headers": false,
      "merge_query_params": false,
      "require_litellm_auth": true,
      "subpath_routing": true
    },
    "tags": ["Provider Pass-through", "pass-through"],
    "docs_url": "https://docs.litellm.ai/docs/pass_through/provider"
  }
}
```

### Extended Schema with Advanced Features

```json
{
  "anthropic": {
    "route_prefix": "/anthropic/{endpoint:path}",
    "target_base_url": "https://api.anthropic.com",
    "target_base_url_env": "ANTHROPIC_API_BASE",
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
    "custom_transformations": {
      "request": null,
      "response": null
    },
    "tags": ["Anthropic Pass-through", "pass-through"],
    "docs_url": "https://docs.litellm.ai/docs/pass_through/anthropic_completion"
  },

  "gemini": {
    "route_prefix": "/gemini/{endpoint:path}",
    "target_base_url": "https://generativelanguage.googleapis.com",
    "target_base_url_env": "GEMINI_API_BASE",
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
      "subpath_routing": true,
      "custom_query_params": true
    },
    "auth_extraction": {
      "from_query_param": "key",
      "from_header": "x-goog-api-key"
    },
    "tags": ["Google AI Studio Pass-through", "pass-through"],
    "docs_url": "https://docs.litellm.ai/docs/pass_through/google_ai_studio"
  },

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
    "tags": ["Cohere Pass-through", "pass-through"],
    "docs_url": "https://docs.litellm.ai/docs/pass_through/cohere"
  },

  "mistral": {
    "route_prefix": "/mistral/{endpoint:path}",
    "target_base_url": "https://api.mistral.ai",
    "target_base_url_env": "MISTRAL_API_BASE",
    "auth": {
      "type": "bearer_token",
      "env_var": "MISTRAL_API_KEY"
    },
    "streaming": {
      "detection_method": "request_body_field",
      "field_name": "stream"
    },
    "features": {
      "require_litellm_auth": true,
      "subpath_routing": true
    },
    "tags": ["Mistral Pass-through", "pass-through"],
    "docs_url": "https://docs.litellm.ai/docs/pass_through/mistral"
  }
}
```

### Schema for Complex Endpoints (e.g., Vertex AI)

```json
{
  "vertex_ai": {
    "route_prefix": "/vertex_ai/{endpoint:path}",
    "target_base_url_template": "https://{location}-aiplatform.googleapis.com/",
    "auth": {
      "type": "custom_handler",
      "handler_function": "vertex_ai_auth_handler",
      "requires": ["vertex_project", "vertex_location", "vertex_credentials"]
    },
    "url_transformation": {
      "extract_params": {
        "vertex_project": {
          "method": "url_regex",
          "pattern": "/projects/([^/]+)/"
        },
        "vertex_location": {
          "method": "url_regex",
          "pattern": "locations/([^/]+)/"
        }
      },
      "inject_params_to_url": true
    },
    "streaming": {
      "detection_method": "url_contains",
      "pattern": "stream",
      "query_param_suffix": "?alt=sse"
    },
    "features": {
      "require_litellm_auth": true,
      "custom_auth_handler": true,
      "dynamic_base_url": true
    },
    "tags": ["Vertex AI Pass-through", "pass-through"],
    "docs_url": "https://docs.litellm.ai/docs/pass_through/vertex_ai"
  }
}
```

---

## Implementation Plan

### Phase 1: Core Infrastructure (Week 1)

#### 1.1 Create Endpoint Config Registry

**File:** `litellm/proxy/pass_through_endpoints/endpoint_config_registry.py`

```python
"""
Centralized registry for endpoint configurations loaded from JSON.
"""
import json
from pathlib import Path
from typing import Dict, Optional, Any
from pydantic import BaseModel, Field

class AuthConfig(BaseModel):
    """Authentication configuration"""
    type: str = Field(..., description="Auth type: bearer_token, custom_header, query_param, custom_handler")
    env_var: str = Field(..., description="Environment variable for API key")
    header_name: Optional[str] = Field(None, description="Header name for auth")
    header_format: Optional[str] = Field(None, description="Format string for header value")
    param_name: Optional[str] = Field(None, description="Query param name for API key")
    handler_function: Optional[str] = Field(None, description="Custom auth handler function name")

class StreamingConfig(BaseModel):
    """Streaming detection configuration"""
    detection_method: str = Field(..., description="Method: request_body_field, url_contains, header")
    field_name: Optional[str] = Field(None, description="Request body field to check")
    pattern: Optional[str] = Field(None, description="Pattern to match in URL")
    query_param_suffix: Optional[str] = Field(None, description="Query param to append for streaming")

class FeaturesConfig(BaseModel):
    """Feature flags for endpoint"""
    forward_headers: bool = Field(default=False)
    merge_query_params: bool = Field(default=False)
    require_litellm_auth: bool = Field(default=True)
    subpath_routing: bool = Field(default=True)
    custom_auth_handler: bool = Field(default=False)
    dynamic_base_url: bool = Field(default=False)

class EndpointConfig(BaseModel):
    """Complete endpoint configuration"""
    provider_slug: str
    route_prefix: str
    target_base_url: Optional[str] = None
    target_base_url_template: Optional[str] = None
    target_base_url_env: Optional[str] = None
    auth: AuthConfig
    streaming: StreamingConfig
    features: FeaturesConfig
    tags: list[str] = Field(default_factory=lambda: ["pass-through"])
    docs_url: Optional[str] = None
    custom_transformations: Optional[Dict[str, Any]] = None

class EndpointConfigRegistry:
    """Registry for endpoint configurations"""
    _configs: Dict[str, EndpointConfig] = {}
    _loaded = False

    @classmethod
    def load(cls, config_path: Optional[Path] = None):
        """Load endpoint configurations from JSON"""
        if cls._loaded:
            return
        
        if config_path is None:
            config_path = Path(__file__).parent / "endpoints_config.json"
        
        if not config_path.exists():
            cls._loaded = True
            return
        
        try:
            with open(config_path) as f:
                data = json.load(f)
            
            for slug, config_data in data.items():
                config_data["provider_slug"] = slug
                cls._configs[slug] = EndpointConfig(**config_data)
            
            cls._loaded = True
        except Exception as e:
            from litellm._logging import verbose_logger
            verbose_logger.error(f"Failed to load endpoint configs: {e}")
            cls._loaded = True
    
    @classmethod
    def get(cls, slug: str) -> Optional[EndpointConfig]:
        """Get endpoint configuration by provider slug"""
        return cls._configs.get(slug)
    
    @classmethod
    def list_providers(cls) -> list[str]:
        """List all registered provider slugs"""
        return list(cls._configs.keys())

# Load on import
EndpointConfigRegistry.load()
```

#### 1.2 Create Endpoint Factory

**File:** `litellm/proxy/pass_through_endpoints/endpoint_factory.py`

```python
"""
Factory for generating pass-through endpoint handlers from JSON config.
"""
import os
from typing import Optional, Callable, Any
from fastapi import Request, Response, Depends
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from .endpoint_config_registry import EndpointConfig, EndpointConfigRegistry
from .pass_through_endpoints import create_pass_through_route
import httpx

class PassthroughEndpointFactory:
    """Factory for creating pass-through endpoint handlers"""
    
    @staticmethod
    def create_auth_headers(config: EndpointConfig, api_key: str) -> dict:
        """Generate authentication headers based on config"""
        if config.auth.type == "bearer_token":
            return {"Authorization": f"Bearer {api_key}"}
        elif config.auth.type == "custom_header":
            header_name = config.auth.header_name or "Authorization"
            header_format = config.auth.header_format or "{api_key}"
            return {header_name: header_format.format(api_key=api_key)}
        elif config.auth.type == "query_param":
            # Handle query params separately
            return {}
        elif config.auth.type == "custom_handler":
            # Handled by custom logic
            return {}
        return {}
    
    @staticmethod
    def get_api_key_from_env(config: EndpointConfig) -> Optional[str]:
        """Get API key from environment variable"""
        from litellm.secret_managers.main import get_secret_str
        return get_secret_str(config.auth.env_var)
    
    @staticmethod
    def get_target_url(config: EndpointConfig) -> str:
        """Get target base URL with environment variable override"""
        if config.target_base_url_env:
            env_url = os.getenv(config.target_base_url_env)
            if env_url:
                return env_url
        
        if config.target_base_url:
            return config.target_base_url
        
        if config.target_base_url_template:
            return config.target_base_url_template
        
        raise ValueError(f"No target URL configured for {config.provider_slug}")
    
    @staticmethod
    def detect_streaming(config: EndpointConfig, request: Request, endpoint: str) -> bool:
        """Detect if request is streaming based on config"""
        if config.streaming.detection_method == "request_body_field":
            # Check request body for streaming field
            # This would need async handling in real implementation
            return False  # Placeholder
        elif config.streaming.detection_method == "url_contains":
            pattern = config.streaming.pattern or "stream"
            return pattern in endpoint
        elif config.streaming.detection_method == "header":
            return request.headers.get("accept") == "text/event-stream"
        return False
    
    @classmethod
    def create_endpoint_handler(cls, config: EndpointConfig) -> Callable:
        """Create an endpoint handler function from config"""
        
        async def handler(
            endpoint: str,
            request: Request,
            fastapi_response: Response,
            user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
        ):
            # Get base URL
            base_target_url = cls.get_target_url(config)
            
            # Build full URL
            encoded_endpoint = httpx.URL(endpoint).path
            if not encoded_endpoint.startswith("/"):
                encoded_endpoint = "/" + encoded_endpoint
            
            base_url = httpx.URL(base_target_url)
            updated_url = base_url.copy_with(path=encoded_endpoint)
            
            # Get API key and create headers
            api_key = cls.get_api_key_from_env(config)
            if api_key is None and not config.features.custom_auth_handler:
                raise Exception(
                    f"Required '{config.auth.env_var}' in environment for {config.provider_slug}"
                )
            
            custom_headers = cls.create_auth_headers(config, api_key or "")
            
            # Handle query params for auth
            query_params = None
            if config.auth.type == "query_param":
                query_params = dict(request.query_params)
                query_params[config.auth.param_name or "key"] = api_key
            
            # Detect streaming
            is_streaming = cls.detect_streaming(config, request, endpoint)
            
            # Add streaming query param if needed
            target = str(updated_url)
            if is_streaming and config.streaming.query_param_suffix:
                target += config.streaming.query_param_suffix
            
            # Create pass-through
            endpoint_func = create_pass_through_route(
                endpoint=endpoint,
                target=target,
                custom_headers=custom_headers,
                _forward_headers=config.features.forward_headers,
                merge_query_params=config.features.merge_query_params,
                is_streaming_request=is_streaming,
                query_params=query_params,
            )
            
            return await endpoint_func(request, fastapi_response, user_api_key_dict)
        
        return handler
    
    @classmethod
    def register_endpoint_from_config(cls, app, config: EndpointConfig):
        """Register endpoint route on FastAPI app from config"""
        handler = cls.create_endpoint_handler(config)
        
        dependencies = []
        if config.features.require_litellm_auth:
            dependencies = [Depends(user_api_key_auth)]
        
        app.add_api_route(
            path=config.route_prefix,
            endpoint=handler,
            methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
            tags=config.tags,
            dependencies=dependencies,
        )
    
    @classmethod
    def register_all_endpoints(cls, app):
        """Register all endpoints from the registry"""
        for provider_slug in EndpointConfigRegistry.list_providers():
            config = EndpointConfigRegistry.get(provider_slug)
            if config:
                try:
                    cls.register_endpoint_from_config(app, config)
                    from litellm._logging import verbose_logger
                    verbose_logger.debug(f"Registered endpoint: {provider_slug}")
                except Exception as e:
                    from litellm._logging import verbose_logger
                    verbose_logger.error(f"Failed to register {provider_slug}: {e}")
```

### Phase 2: JSON Configuration File (Week 1)

Create `litellm/proxy/pass_through_endpoints/endpoints_config.json` with all standard endpoints.

### Phase 3: Integration (Week 2)

#### 3.1 Update `proxy_server.py`

Add endpoint registration on startup:

```python
# In proxy_server.py startup
from litellm.proxy.pass_through_endpoints.endpoint_factory import PassthroughEndpointFactory

@app.on_event("startup")
async def startup_event():
    # ... existing startup code ...
    
    # Register JSON-configured endpoints
    PassthroughEndpointFactory.register_all_endpoints(app)
```

#### 3.2 Backward Compatibility

Keep existing hardcoded endpoints as fallback, gradually migrate them to JSON.

### Phase 4: Migration & Testing (Week 2-3)

1. Migrate simple endpoints (Cohere, Mistral, etc.) to JSON
2. Test each migrated endpoint
3. Migrate complex endpoints (Vertex AI, Bedrock) with custom handlers
4. Update documentation

### Phase 5: Developer Experience (Week 3-4)

1. Create CLI tool for endpoint generation: `litellm endpoint add <provider>`
2. Add validation schemas
3. Create testing framework for endpoint configs
4. Update contribution guidelines

---

## Benefits

### For Contributors
- **90% less code**: Add endpoint in ~10 lines of JSON vs 50-100 lines of Python
- **No FastAPI knowledge required**: Just configure, don't code
- **Instant validation**: Schema validation catches errors early
- **Easy testing**: Test configs without deploying

### For Maintainers
- **Centralized config**: All endpoints in one place
- **Consistent patterns**: Enforced by schema
- **Easy auditing**: See all endpoints at a glance
- **Reduced bugs**: Less custom code = fewer bugs

### For Users
- **Faster feature delivery**: New endpoints ship faster
- **Better documentation**: Auto-generated from configs
- **More providers**: Lower barrier = more integrations

---

## Example: Adding a New Endpoint

### Before (Current Approach)

```python
# Add 50-100 lines to llm_passthrough_endpoints.py
@router.api_route(
    "/newprovider/{endpoint:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    tags=["NewProvider Pass-through", "pass-through"],
)
async def newprovider_proxy_route(
    endpoint: str,
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    base_target_url = os.getenv("NEWPROVIDER_API_BASE") or "https://api.newprovider.com"
    encoded_endpoint = httpx.URL(endpoint).path
    
    if not encoded_endpoint.startswith("/"):
        encoded_endpoint = "/" + encoded_endpoint
    
    base_url = httpx.URL(base_target_url)
    updated_url = base_url.copy_with(path=encoded_endpoint)
    
    api_key = passthrough_endpoint_router.get_credentials(
        custom_llm_provider="newprovider",
        region_name=None,
    )
    
    is_streaming_request = False
    if request.method == "POST":
        _request_body = await request.json()
        if _request_body.get("stream"):
            is_streaming_request = True
    
    endpoint_func = create_pass_through_route(
        endpoint=endpoint,
        target=str(updated_url),
        custom_headers={"Authorization": f"Bearer {api_key}"},
        is_streaming_request=is_streaming_request,
    )
    
    return await endpoint_func(request, fastapi_response, user_api_key_dict)
```

### After (JSON Approach)

Add to `endpoints_config.json`:

```json
{
  "newprovider": {
    "route_prefix": "/newprovider/{endpoint:path}",
    "target_base_url": "https://api.newprovider.com",
    "target_base_url_env": "NEWPROVIDER_API_BASE",
    "auth": {
      "type": "bearer_token",
      "env_var": "NEWPROVIDER_API_KEY"
    },
    "streaming": {
      "detection_method": "request_body_field",
      "field_name": "stream"
    },
    "features": {
      "require_litellm_auth": true,
      "subpath_routing": true
    },
    "tags": ["NewProvider Pass-through", "pass-through"],
    "docs_url": "https://docs.litellm.ai/docs/pass_through/newprovider"
  }
}
```

**That's it! No Python code needed.**

---

## CLI Tool Design

```bash
# Interactive endpoint creation
$ litellm endpoint add

? Provider slug: newprovider
? Base URL: https://api.newprovider.com
? API Key env var: NEWPROVIDER_API_KEY
? Auth type: bearer_token
? Streaming support: yes
? Streaming detection: request_body_field
? Field name: stream

✓ Created endpoint configuration!
✓ Added to endpoints_config.json
✓ Run tests: litellm endpoint test newprovider

# From CLI flags
$ litellm endpoint add \
  --slug newprovider \
  --base-url https://api.newprovider.com \
  --api-key-env NEWPROVIDER_API_KEY \
  --auth bearer_token \
  --streaming request_body_field:stream

# Validate config
$ litellm endpoint validate
✓ All 23 endpoints validated successfully

# Test endpoint
$ litellm endpoint test newprovider
Running tests for newprovider...
✓ Authentication works
✓ Streaming detection works
✓ Error handling works
```

---

## Success Metrics

### Adoption
- ✅ 50%+ of existing endpoints migrated to JSON within 6 months
- ✅ 90%+ of new endpoints use JSON config
- ✅ Zero Python code required for 80% of endpoints

### Developer Experience
- ✅ Time to add endpoint: 50+ min → 5 min (10X improvement)
- ✅ Lines of code per endpoint: 50-100 → 5-10 (10X reduction)
- ✅ Bug rate: 50% reduction due to standardization

### Maintenance
- ✅ All endpoints documented in one place
- ✅ Easy to audit and update
- ✅ Automated testing coverage

---

## Migration Strategy

### Phase 1: Simple Endpoints (Week 1-2)
- ✅ Cohere
- ✅ Mistral
- ✅ OpenAI
- ✅ Anthropic (simple auth)

### Phase 2: Medium Complexity (Week 3-4)
- ✅ Gemini (query param auth)
- ✅ AssemblyAI (regional endpoints)
- ✅ Azure (model routing)

### Phase 3: Complex Endpoints (Week 5-8)
- ⚠️ Vertex AI (OAuth, dynamic URLs)
- ⚠️ Bedrock (SigV4, model extraction)
- ⚠️ Custom endpoints requiring special logic

### Phase 4: Deprecation (Month 3-6)
- Mark old Python endpoints as deprecated
- Add migration warnings
- Eventually remove old code

---

## FAQ

### Q: What about complex endpoints like Bedrock?
**A:** Use `"custom_handler": true` flag and reference a Python function for complex auth. The framework handles routing/headers, custom handler only does the complex part.

### Q: How do we handle provider-specific transformations?
**A:** Add `"custom_transformations"` field pointing to transformation functions. Most endpoints won't need this.

### Q: What about backward compatibility?
**A:** Both systems run side-by-side during migration. Old endpoints continue to work.

### Q: Performance impact?
**A:** Minimal. Config loaded once on startup, routing is identical to current implementation.

### Q: What about WebSocket endpoints?
**A:** Extend the schema to support WebSocket configurations with similar patterns.

---

## Next Steps

1. **Review & Approve Proposal** (Week 0)
2. **Implement Core Registry & Factory** (Week 1)
3. **Create Initial JSON Config** (Week 1)
4. **Migrate 3-5 Simple Endpoints** (Week 2)
5. **Test & Refine** (Week 2)
6. **Roll Out to All Endpoints** (Week 3-8)
7. **Create CLI Tool** (Week 4-5)
8. **Update Documentation** (Week 6)

---

## Conclusion

This proposal provides a **clear path to 10X simplification** of SDK endpoint addition. By leveraging declarative configuration, we reduce complexity, improve maintainability, and lower the barrier for contributions.

**Key Takeaway:** From 50+ lines of boilerplate Python to a simple JSON object. That's the 10X we're aiming for.
