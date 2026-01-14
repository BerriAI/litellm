# LiteLLM Architecture - LiteLLM SDK + AI Gateway

This document helps contributors understand where to make changes in LiteLLM.

---

## How It Works

The LiteLLM AI Gateway (Proxy) uses the LiteLLM SDK internally for all LLM calls:

```
OpenAI SDK (client)    ──▶  LiteLLM AI Gateway (proxy/)  ──▶  LiteLLM SDK (litellm/)  ──▶  LLM API
Anthropic SDK (client) ──▶  LiteLLMAI Gateway (proxy/)  ──▶  LiteLLM SDK (litellm/)  ──▶  LLM API
Any HTTP client        ──▶  LiteLLMAI Gateway (proxy/)  ──▶  LiteLLM SDK (litellm/)  ──▶  LLM API
```

The **AI Gateway** adds authentication, rate limiting, budgets, and routing on top of the SDK.
The **SDK** handles the actual LLM provider calls, request/response transformations, and streaming.

---

## 1. AI Gateway (Proxy) Request Flow

The AI Gateway (`litellm/proxy/`) wraps the SDK with authentication, rate limiting, and management features.

```mermaid
sequenceDiagram
    participant Client
    participant ProxyServer as proxy/proxy_server.py
    participant Auth as proxy/auth/user_api_key_auth.py
    participant Hooks as proxy/hooks/
    participant Router as router.py
    participant Main as main.py
    participant Handler as llms/custom_httpx/llm_http_handler.py
    participant Transform as llms/{provider}/chat/transformation.py
    participant Provider as LLM Provider API

    Client->>ProxyServer: POST /v1/chat/completions
    ProxyServer->>Auth: user_api_key_auth()
    ProxyServer->>Hooks: max_budget_limiter, parallel_request_limiter
    ProxyServer->>Router: route_request()
    Router->>Main: litellm.acompletion()
    Main->>Handler: BaseLLMHTTPHandler.completion()
    Handler->>Transform: ProviderConfig.transform_request()
    Handler->>Provider: HTTP Request
    Provider-->>Handler: Response
    Handler->>Transform: ProviderConfig.transform_response()
    Handler-->>Hooks: async_log_success_event()
    Handler-->>Client: ModelResponse
```

### Proxy Components

```mermaid
graph TD
    subgraph "Incoming Request"
        Client["POST /v1/chat/completions"]
    end

    subgraph "proxy/proxy_server.py"
        Endpoint["chat_completion()"]
    end

    subgraph "proxy/auth/"
        Auth["user_api_key_auth()"]
    end

    subgraph "proxy/"
        PreCall["litellm_pre_call_utils.py"]
        RouteRequest["route_llm_request.py"]
    end

    subgraph "litellm/"
        Router["router.py"]
        Main["main.py"]
    end

    Client --> Endpoint
    Endpoint --> Auth
    Auth --> PreCall
    PreCall --> RouteRequest
    RouteRequest --> Router
    Router --> Main
    Main --> Client
```

**Key proxy files:**
- `proxy/proxy_server.py` - Main API endpoints
- `proxy/auth/` - Authentication (API keys, JWT, OAuth2)
- `proxy/hooks/` - Proxy-level callbacks
- `router.py` - Load balancing, fallbacks
- `router_strategy/` - Routing algorithms (`lowest_latency.py`, `simple_shuffle.py`, etc.)

**LLM-specific proxy endpoints:**

| Endpoint | Directory | Purpose |
|----------|-----------|---------|
| `/v1/messages` | `proxy/anthropic_endpoints/` | Anthropic Messages API |
| `/vertex-ai/*` | `proxy/vertex_ai_endpoints/` | Vertex AI passthrough |
| `/gemini/*` | `proxy/google_endpoints/` | Google AI Studio passthrough |
| `/v1/images/*` | `proxy/image_endpoints/` | Image generation |
| `/v1/batches` | `proxy/batches_endpoints/` | Batch processing |
| `/v1/files` | `proxy/openai_files_endpoints/` | File uploads |
| `/v1/fine_tuning` | `proxy/fine_tuning_endpoints/` | Fine-tuning jobs |
| `/v1/rerank` | `proxy/rerank_endpoints/` | Reranking |
| `/v1/responses` | `proxy/response_api_endpoints/` | OpenAI Responses API |
| `/v1/vector_stores` | `proxy/vector_store_endpoints/` | Vector stores |
| `/*` (passthrough) | `proxy/pass_through_endpoints/` | Direct provider passthrough |

**Proxy Hooks** (`proxy/hooks/__init__.py`):

| Hook | File | Purpose |
|------|------|---------|
| `max_budget_limiter` | `proxy/hooks/max_budget_limiter.py` | Enforce budget limits |
| `parallel_request_limiter` | `proxy/hooks/parallel_request_limiter_v3.py` | Rate limiting per key/user |
| `cache_control_check` | `proxy/hooks/cache_control_check.py` | Cache validation |
| `responses_id_security` | `proxy/hooks/responses_id_security.py` | Response ID validation |
| `litellm_skills` | `proxy/hooks/skills_injection.py` | Skills injection |

To add a new proxy hook, implement `CustomLogger` and register in `PROXY_HOOKS`.

---

## 2. SDK Request Flow

The SDK (`litellm/`) provides the core LLM calling functionality used by both direct SDK users and the AI Gateway.

```mermaid
graph TD
    subgraph "SDK Entry Points"
        Completion["litellm.completion()"]
        Messages["litellm.messages()"]
    end

    subgraph "main.py"
        Main["completion()<br/>acompletion()"]
    end

    subgraph "utils.py"
        GetProvider["get_llm_provider()"]
    end

    subgraph "llms/custom_httpx/"
        Handler["llm_http_handler.py<br/>BaseLLMHTTPHandler"]
        HTTP["http_handler.py<br/>HTTPHandler / AsyncHTTPHandler"]
    end

    subgraph "llms/{provider}/chat/"
        TransformReq["transform_request()"]
        TransformResp["transform_response()"]
    end

    subgraph "litellm_core_utils/"
        Streaming["streaming_handler.py"]
    end

    subgraph "integrations/ (async, off main thread)"
        Callbacks["custom_logger.py<br/>Langfuse, Datadog, etc."]
    end

    Completion --> Main
    Messages --> Main
    Main --> GetProvider
    GetProvider --> Handler
    Handler --> TransformReq
    TransformReq --> HTTP
    HTTP --> Provider["LLM Provider API"]
    Provider --> HTTP
    HTTP --> TransformResp
    TransformResp --> Streaming
    Streaming --> Response["ModelResponse"]
    Response -.->|async| Callbacks
```

**Key SDK files:**
- `main.py` - Entry points: `completion()`, `acompletion()`, `embedding()`
- `utils.py` - `get_llm_provider()` resolves model → provider
- `llms/custom_httpx/llm_http_handler.py` - Central HTTP orchestrator
- `llms/custom_httpx/http_handler.py` - Low-level HTTP client
- `llms/{provider}/chat/transformation.py` - Provider-specific transformations
- `litellm_core_utils/streaming_handler.py` - Streaming response handling
- `integrations/` - Async callbacks (Langfuse, Datadog, etc.)

---

## 3. Translation Layer

When a request comes in, it goes through a **translation layer** that converts between API formats.
Each translation is isolated in its own file, making it easy to test and modify independently.

### Where to find translations

| Incoming API | Provider | Translation File |
|--------------|----------|------------------|
| `/v1/chat/completions` | Anthropic | `llms/anthropic/chat/transformation.py` |
| `/v1/chat/completions` | Bedrock Converse | `llms/bedrock/chat/converse_transformation.py` |
| `/v1/chat/completions` | Bedrock Invoke | `llms/bedrock/chat/invoke_transformations/anthropic_claude3_transformation.py` |
| `/v1/chat/completions` | Gemini | `llms/gemini/chat/transformation.py` |
| `/v1/chat/completions` | Vertex AI | `llms/vertex_ai/gemini/transformation.py` |
| `/v1/chat/completions` | OpenAI | `llms/openai/chat/gpt_transformation.py` |
| `/v1/messages` (passthrough) | Anthropic | `llms/anthropic/experimental_pass_through/messages/transformation.py` |
| `/v1/messages` (passthrough) | Bedrock | `llms/bedrock/messages/invoke_transformations/anthropic_claude3_transformation.py` |
| `/v1/messages` (passthrough) | Vertex AI | `llms/vertex_ai/vertex_ai_partner_models/anthropic/experimental_pass_through/transformation.py` |
| Passthrough endpoints | All | `proxy/pass_through_endpoints/llm_provider_handlers/` |

### Example: Debugging prompt caching

If `/v1/messages` → Bedrock Converse prompt caching isn't working but Bedrock Invoke works:

1. **Bedrock Converse translation**: `llms/bedrock/chat/converse_transformation.py`
2. **Bedrock Invoke translation**: `llms/bedrock/chat/invoke_transformations/anthropic_claude3_transformation.py`
3. Compare how each handles `cache_control` in `transform_request()`

### How translations work

Each provider has a `Config` class that inherits from `BaseConfig` (`llms/base_llm/chat/transformation.py`):

```python
class ProviderConfig(BaseConfig):
    def transform_request(self, model, messages, optional_params, litellm_params, headers):
        # Convert OpenAI format → Provider format
        return {"messages": transformed_messages, ...}
    
    def transform_response(self, model, raw_response, model_response, logging_obj, ...):
        # Convert Provider format → OpenAI format
        return ModelResponse(choices=[...], usage=Usage(...))
```

The `BaseLLMHTTPHandler` (`llms/custom_httpx/llm_http_handler.py`) calls these methods - you never need to modify the handler itself.

---

## 4. Adding/Modifying Providers

### To add a new provider:

1. Create `llms/{provider}/chat/transformation.py`
2. Implement `Config` class with `transform_request()` and `transform_response()`
3. Add tests in `tests/llm_translation/test_{provider}.py`

### To add a feature (e.g., prompt caching):

1. Find the translation file from the table above
2. Modify `transform_request()` to handle the new parameter
3. Add unit tests that verify the transformation

### Testing checklist

When adding a feature, verify it works across all paths:

| Test | File Pattern |
|------|--------------|
| OpenAI passthrough | `tests/llm_translation/test_openai*.py` |
| Anthropic direct | `tests/llm_translation/test_anthropic*.py` |
| Bedrock Invoke | `tests/llm_translation/test_bedrock*.py` |
| Bedrock Converse | `tests/llm_translation/test_bedrock*converse*.py` |
| Vertex AI | `tests/llm_translation/test_vertex*.py` |
| Gemini | `tests/llm_translation/test_gemini*.py` |

### Unit testing translations

Translations are designed to be unit testable without making API calls:

```python
from litellm.llms.bedrock.chat.converse_transformation import BedrockConverseConfig

def test_prompt_caching_transform():
    config = BedrockConverseConfig()
    result = config.transform_request(
        model="anthropic.claude-3-opus",
        messages=[{"role": "user", "content": "test", "cache_control": {"type": "ephemeral"}}],
        optional_params={},
        litellm_params={},
        headers={}
    )
    assert "cachePoint" in str(result)  # Verify cache_control was translated
```
