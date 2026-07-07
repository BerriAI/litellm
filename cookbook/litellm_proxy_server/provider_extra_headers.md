# Provider-Specific `extra_headers` in Proxy Config

Pass custom HTTP headers to upstream LLM providers through `litellm_params.extra_headers` in your proxy `config.yaml`.

## Basic pattern

```yaml
model_list:
  - model_name: azure-gpt-4
    litellm_params:
      model: azure/gpt-4
      api_base: https://my-resource.openai.azure.com
      api_key: os.environ/AZURE_API_KEY
      extra_headers:
        X-Custom-Deployment-Tag: production
        X-Request-Source: litellm-proxy
```

At request time, LiteLLM merges these headers into the outbound provider call.

## Provider-specific examples

### Anthropic (beta features)

```yaml
model_list:
  - model_name: claude-with-beta
    litellm_params:
      model: anthropic/claude-sonnet-4-20250514
      api_key: os.environ/ANTHROPIC_API_KEY
      extra_headers:
        anthropic-beta: context-management-2025-06-27
```

> **Note:** LiteLLM auto-injects `anthropic-beta` for the memory tool. Only set this manually when you need a different beta program.

### Vertex AI (custom metadata)

```yaml
model_list:
  - model_name: vertex-gemini
    litellm_params:
      model: vertex_ai/gemini-2.0-flash
      vertex_project: my-gcp-project
      vertex_location: us-central1
      extra_headers:
        X-Goog-User-Project: my-gcp-project
```

### OpenAI-compatible gateways

```yaml
model_list:
  - model_name: custom-gateway
    litellm_params:
      model: openai/gpt-4
      api_base: https://gateway.internal/v1
      api_key: os.environ/GATEWAY_KEY
      extra_headers:
        X-Tenant-Id: team-alpha
```

## Python SDK equivalent

```python
import litellm

response = litellm.completion(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello"}],
    extra_headers={"X-Custom-Header": "value"},
)
```

## Guardrails `extra_headers` (allowlist)

Guardrails use a different pattern: `extra_headers` is a **list of header names** to forward from the client request (not static values):

```yaml
guardrails:
  - guardrail_name: my-api-guardrail
    litellm_params:
      guardrail: generic_guardrail_api
      extra_headers:
        - X-Request-Id
        - X-User-Email
```

See [extra_headers_pattern.md](./mcp/extra_headers_pattern.md) for the MCP variant of this allowlist pattern.

## Precedence tips

1. Caller-supplied `extra_headers` in the API request override config defaults for the same header name.
2. Provider-required auth headers (`Authorization`, `x-api-key`) are set by LiteLLM—do not duplicate them unless your provider requires a second custom auth header.
3. For agents and MCP servers, see their respective `extra_headers` docs (allowlist vs static dict).