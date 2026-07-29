# Pass-Through Endpoints Architecture

## Why Pass-Through Endpoints Transform Requests

Even "pass-through" endpoints must perform essential transformations. The request **body** passes through unchanged, but:

```mermaid
sequenceDiagram
    participant Client
    participant Proxy as LiteLLM Proxy
    participant Provider as LLM Provider

    Client->>Proxy: POST /vertex_ai/v1/projects/.../generateContent
    Note over Client,Proxy: Headers: Authorization: Bearer sk-litellm-key
    Note over Client,Proxy: Body: { "contents": [...] }
    
    rect rgb(240, 240, 240)
        Note over Proxy: 1. URL Construction
        Note over Proxy: Build regional/provider-specific URL
    end
    
    rect rgb(240, 240, 240)
        Note over Proxy: 2. Auth Header Replacement
        Note over Proxy: LiteLLM key → provider credentials
    end

    rect rgb(240, 240, 240)
        Note over Proxy: 3. Extra Operations
        Note over Proxy: • x-pass-* headers (strip prefix, forward)
        Note over Proxy: • x-litellm-tags → metadata
        Note over Proxy: • Guardrails (opt-in)
        Note over Proxy: • Multipart form reconstruction
    end
    
    Proxy->>Provider: POST https://us-central1-aiplatform.googleapis.com/...
    Note over Proxy,Provider: Headers: Authorization: Bearer ya29.google-oauth...
    Note over Proxy,Provider: Body: { "contents": [...] } ← UNCHANGED
    
    Provider-->>Proxy: Response (streaming or non-streaming)

    rect rgb(240, 240, 240)
        Note over Proxy: 4. Response Handling (async)
        Note over Proxy: • Collect streaming chunks for logging
        Note over Proxy: • Cost injection (if enabled)
        Note over Proxy: • Parse response → calculate cost → log
    end
    
    Proxy-->>Client: Response (unchanged)
```

## Essential Transformations

- **URL Construction** - Build correct provider URL (e.g., regional endpoints for Vertex AI, Bedrock)
- **Auth Header Replacement** - Swap LiteLLM virtual key for actual provider credentials

## Extra Operations

| Operation | Description |
|-----------|-------------|
| `x-pass-*` headers | Strip prefix and forward (e.g., `x-pass-anthropic-beta` → `anthropic-beta`) |
| `x-litellm-tags` header | Extract tags and add to request metadata for logging |
| Streaming chunk collection | Collect chunks async for logging after stream completes |
| Multipart form handling | Reconstruct multipart/form-data requests for file uploads |
| Guardrails (opt-in) | Run content filtering when explicitly configured |
| Cost injection | Inject cost into streaming chunks when `include_cost_in_streaming_usage` enabled |

## What Does NOT Change

- Request body
- Response body
- Provider-specific parameters