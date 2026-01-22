# Pass-Through Endpoints Architecture

## Overview

Pass-through endpoints in LiteLLM allow you to use provider-native SDKs (Vertex AI SDK, Anthropic SDK, etc.) while still benefiting from LiteLLM's proxy features like authentication, rate limiting, spend tracking, and observability.

**The Core Question:** *"Why does a pass-through touch anything in the first place? Shouldn't it all just pass through?"*

**The Answer:** Even the most minimal pass-through requires certain essential transformations to function. A "pure" pass-through would require users to manage their own credentials and construct provider-specific URLs, defeating the purpose of a unified gateway.

---

## What Pass-Through Endpoints Transform (and Why)

### 1. URL Construction (Essential)

Pass-through endpoints must construct the correct target URL for each provider. This is **non-trivial** for several reasons:

#### Google/Vertex AI Example
```
User sends:    POST /vertex_ai/v1/projects/my-project/locations/us-central1/publishers/google/models/gemini-pro:generateContent
Proxy builds:  https://us-central1-aiplatform.googleapis.com/v1/projects/my-project/locations/us-central1/publishers/google/models/gemini-pro:generateContent
```

**Complexity involved:**
- **Regional endpoints**: Vertex AI uses location-specific endpoints (`us-central1-aiplatform.googleapis.com` vs `europe-west1-aiplatform.googleapis.com`)
- **Global vs regional**: Some endpoints use `aiplatform.googleapis.com` (global) vs `{location}-aiplatform.googleapis.com` (regional)
- **Discovery engine**: Uses a completely different base (`discoveryengine.googleapis.com`)
- **Live API WebSockets**: Uses `wss://{location}-aiplatform.googleapis.com/ws/...`

```python
# From llm_passthrough_endpoints.py
def get_vertex_base_url(vertex_location: Optional[str]) -> str:
    if vertex_location == "global":
        return "https://aiplatform.googleapis.com/"
    return f"https://{vertex_location}-aiplatform.googleapis.com/"
```

#### Other Providers
| Provider | URL Construction Logic |
|----------|----------------------|
| **Anthropic** | Base URL from `ANTHROPIC_API_BASE` or default `https://api.anthropic.com` |
| **Gemini** | `GEMINI_API_BASE` or `https://generativelanguage.googleapis.com` |
| **Cohere** | `COHERE_API_BASE` or `https://api.cohere.com` |
| **Bedrock** | Region-specific: `https://bedrock-runtime.{region}.amazonaws.com` or `https://bedrock-agent-runtime.{region}.amazonaws.com` |
| **Azure** | User-configured `AZURE_API_BASE` with deployment-specific paths |
| **AssemblyAI** | Region-aware: `https://api.assemblyai.com` vs `https://api.eu.assemblyai.com` |

---

### 2. Authentication Header Replacement (Essential)

The proxy replaces LiteLLM virtual keys with actual provider credentials:

```
Incoming Request:
  Authorization: Bearer sk-litellm-user-key-abc123

Outgoing Request (to Anthropic):
  x-api-key: sk-ant-actual-provider-key-xyz789

Outgoing Request (to Vertex AI):
  Authorization: Bearer ya29.google-oauth-token...
```

**Why this is necessary:**
- Users authenticate to LiteLLM with virtual keys
- LiteLLM maps these to actual provider credentials
- This enables centralized credential management, rotation, and access control

```python
# From passthrough_endpoint_router.py
class PassthroughEndpointRouter:
    def get_credentials(self, custom_llm_provider: str, region_name: Optional[str]) -> Optional[str]:
        # First check in-memory credentials (set via API)
        if credential_name in self.credentials:
            return self.credentials[credential_name]
        # Fall back to environment variables
        return get_secret_str(f"{custom_llm_provider.upper()}_API_KEY")
```

#### Provider-Specific Auth Headers
| Provider | Auth Header Format |
|----------|-------------------|
| **Anthropic** | `x-api-key: {api_key}` |
| **OpenAI/Azure** | `Authorization: Bearer {api_key}` + `api-key: {api_key}` |
| **Gemini** | Query param: `?key={api_key}` |
| **Vertex AI** | `Authorization: Bearer {oauth_token}` |
| **Bedrock** | AWS SigV4 signed headers |
| **Cohere** | `Authorization: Bearer {api_key}` |

---

### 3. Query Parameter Handling

Some providers require API keys or other parameters in query strings:

```python
# Gemini example - API key goes in query params, not headers
merged_params = dict(request.query_params)
merged_params.update({"key": gemini_api_key})
```

---

## What Pass-Through Endpoints Do NOT Transform

The request/response **body** is passed through unchanged:

```
Request Body (Unchanged):
{
  "model": "claude-3-sonnet-20240229",
  "messages": [{"role": "user", "content": "Hello"}],
  "max_tokens": 1024
}
```

This means:
- Provider-specific parameters work as documented
- No format translation (unlike LiteLLM's standard `/chat/completions` endpoint)
- Provider-native features are fully accessible

---

## Architecture Components

### Request Flow

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           CLIENT REQUEST                                      │
│  POST /vertex_ai/v1/projects/proj/locations/us-central1/.../gemini:generate  │
│  Headers: Authorization: Bearer sk-litellm-key                                │
│  Body: { "contents": [...] }                                                  │
└─────────────────────────────────────┬────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                         1. AUTHENTICATION LAYER                               │
│  • Validate LiteLLM virtual key (user_api_key_auth)                          │
│  • Check rate limits, budgets, model access                                   │
│  • Extract user/team metadata for logging                                     │
└─────────────────────────────────────┬────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                       2. PRE-CALL HOOKS (Optional)                           │
│  • Guardrails (content filtering)                                            │
│  • Request modification callbacks                                             │
│  • Custom business logic                                                      │
└─────────────────────────────────────┬────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                      3. PASSTHROUGH TRANSFORMATION                            │
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐  │
│  │   URL Construction  │  │   Header Replace    │  │   Query Params      │  │
│  │   ───────────────   │  │   ──────────────    │  │   ────────────      │  │
│  │   • Parse endpoint  │  │   • Remove litellm  │  │   • Merge existing  │  │
│  │   • Get base URL    │  │     auth headers    │  │   • Add provider    │  │
│  │   • Handle regions  │  │   • Add provider    │  │     params (keys)   │  │
│  │   • Build full URL  │  │     credentials     │  │                     │  │
│  └─────────────────────┘  └─────────────────────┘  └─────────────────────┘  │
└─────────────────────────────────────┬────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                         4. HTTP REQUEST TO PROVIDER                          │
│  POST https://us-central1-aiplatform.googleapis.com/v1/projects/.../generate │
│  Headers: Authorization: Bearer ya29.oauth-token...                          │
│  Body: { "contents": [...] }  ← UNCHANGED                                    │
└─────────────────────────────────────┬────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                         5. RESPONSE HANDLING                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │  STREAMING                           │  NON-STREAMING                   │ │
│  │  • Stream chunks to client           │  • Return full response          │ │
│  │  • Collect chunks for logging        │  • Parse for logging             │ │
│  │  • Optional: inject cost in chunks   │                                  │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────┬────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                         6. LOGGING & OBSERVABILITY                           │
│  • Parse response to extract token usage (provider-specific)                  │
│  • Calculate cost using LiteLLM's pricing data                               │
│  • Send to configured callbacks (Langfuse, OpenTelemetry, etc.)              │
│  • Update spend tracking in database                                          │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Details

### PassthroughEndpointRouter

Central credential management for all pass-through endpoints:

```python
class PassthroughEndpointRouter:
    """Manages credentials for pass-through endpoints"""
    
    def __init__(self):
        self.credentials: Dict[str, str] = {}  # In-memory credential store
        self.deployment_key_to_vertex_credentials: Dict[str, VertexPassThroughCredentials] = {}
        self.default_vertex_config: Optional[VertexPassThroughCredentials] = None
    
    def get_credentials(self, custom_llm_provider: str, region_name: Optional[str]) -> Optional[str]:
        """Get API key for a provider, checking in-memory store then env vars"""
        
    def get_vertex_credentials(self, project_id: Optional[str], location: Optional[str]) -> Optional[VertexPassThroughCredentials]:
        """Get Vertex AI credentials for specific project/location"""
```

### Provider-Specific Logging Handlers

Each provider has a logging handler that parses responses into a standardized format:

```python
# Base class for all logging handlers
class BasePassthroughLoggingHandler(ABC):
    @property
    @abstractmethod
    def llm_provider_name(self) -> LlmProviders:
        pass
    
    def passthrough_chat_handler(self, httpx_response, response_body, ...):
        """Transform provider response to OpenAI format for logging"""
        model = request_body.get("model", response_body.get("model", ""))
        provider_config = self.get_provider_config(model=model)
        litellm_model_response = provider_config.transform_response(...)
        # ... calculate cost, create logging payload
```

**Available handlers:**
- `VertexPassthroughLoggingHandler` - Handles generateContent, predict, rawPredict, etc.
- `AnthropicPassthroughLoggingHandler` - Handles /messages endpoints
- `GeminiPassthroughLoggingHandler` - Handles Google AI Studio endpoints
- `CoherePassthroughLoggingHandler` - Handles /v2/chat, /v1/embed
- `OpenAIPassthroughLoggingHandler` - Handles OpenAI passthrough
- `AssemblyAIPassthroughLoggingHandler` - Handles transcription endpoints

### Streaming Handler

For streaming responses, chunks are processed while being forwarded:

```python
class PassThroughStreamingHandler:
    @staticmethod
    async def chunk_processor(response, request_body, litellm_logging_obj, ...):
        raw_bytes: List[bytes] = []
        
        async for chunk in response.aiter_bytes():
            raw_bytes.append(chunk)
            # Optional: inject cost into streaming chunks
            if include_cost_in_streaming_usage and model_name:
                chunk = _process_chunk_with_cost_injection(chunk, model_name)
            yield chunk
        
        # After streaming completes, log the full response
        asyncio.create_task(
            _route_streaming_logging_to_handler(raw_bytes, ...)
        )
```

---

## Two Types of Pass-Through Endpoints

### 1. Provider-Specific (Built-in)

Pre-configured endpoints for major providers:

| Endpoint | Provider | Features |
|----------|----------|----------|
| `/vertex_ai/{path}` | Google Vertex AI | OAuth token refresh, project/location routing |
| `/gemini/{path}` | Google AI Studio | API key in query params |
| `/anthropic/{path}` | Anthropic | x-api-key header |
| `/bedrock/{path}` | AWS Bedrock | SigV4 signing |
| `/azure/{path}` | Azure OpenAI | Deployment routing |
| `/cohere/{path}` | Cohere | Bearer token |
| `/openai/{path}` | OpenAI | Bearer token |
| `/assemblyai/{path}` | AssemblyAI | Regional routing |

### 2. Generic (User-Configured)

Custom pass-through endpoints defined in config:

```yaml
general_settings:
  pass_through_endpoints:
    - path: "/custom-api"
      target: "https://my-internal-api.company.com"
      headers:
        Authorization: "Bearer os.environ/MY_API_KEY"
      auth: true  # Require LiteLLM auth
      include_subpath: true  # Forward /custom-api/foo/bar → target/foo/bar
      guardrails:
        - my-guardrail
```

---

## Optional Features (Beyond Minimum Pass-Through)

These features add value but are not required for basic pass-through functionality:

### 1. Logging & Observability
- Parse provider responses to extract usage metrics
- Calculate costs using LiteLLM's pricing database
- Send to observability backends (Langfuse, OTEL, etc.)

### 2. Guardrails
- Content filtering on pass-through requests
- Provider-agnostic safety checks

### 3. Spend Tracking
- Track costs per user/team/key
- Enforce budgets

### 4. Rate Limiting
- Apply rate limits before forwarding to provider
- Prevent abuse

### 5. Cost Injection in Streaming
- Inject calculated costs into streaming chunks
- Enables real-time cost visibility in streaming responses

---

## Summary: Minimum Required Transformations

| Transformation | Why Required | Alternative Without Proxy |
|----------------|--------------|---------------------------|
| **URL Construction** | Providers have complex URL patterns (regional, versioned) | User manually constructs URLs |
| **Auth Header Replacement** | Users authenticate with LiteLLM keys, not provider keys | Users manage provider credentials directly |
| **Query Param Handling** | Some providers require params in query string | Users add params manually |

**Everything else is optional** but provides the value of using LiteLLM as a gateway:
- Spend tracking
- Rate limiting  
- Observability
- Guardrails
- Centralized credential management

---

## Code References

| Component | File |
|-----------|------|
| Main pass-through handler | `litellm/proxy/pass_through_endpoints/pass_through_endpoints.py` |
| Provider-specific routes | `litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py` |
| Credential router | `litellm/proxy/pass_through_endpoints/passthrough_endpoint_router.py` |
| Streaming handler | `litellm/proxy/pass_through_endpoints/streaming_handler.py` |
| Success/logging handler | `litellm/proxy/pass_through_endpoints/success_handler.py` |
| Provider logging handlers | `litellm/proxy/pass_through_endpoints/llm_provider_handlers/` |
| Base utilities | `litellm/passthrough/utils.py` |
