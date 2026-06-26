# Prompt Compression Plugins

Microservices that plug into the LiteLLM gateway as `pre_call` guardrails to compress incoming prompts before they reach the LLM provider. Each runs independently and speaks the [LiteLLM generic guardrail API](https://docs.litellm.ai/docs/adding_provider/generic_guardrail_api).

## How it works

LiteLLM sends the full message array to the microservice before forwarding the request to the LLM provider. The microservice compresses it and returns the compressed messages. LiteLLM then sends the smaller payload to the provider — reducing cost and latency transparently to the caller.

## Available plugins

| Plugin | Compression approach |
|--------|----------------------|
| [headroom](./headroom/) | ML-based; deduplicates JSON arrays, compresses tool outputs and prose via a trained model |

## Adding a plugin to your LiteLLM config

```yaml
guardrails:
  - guardrail_name: headroom-compression
    litellm_params:
      guardrail: generic_guardrail_api
      mode: pre_call
      api_base: http://localhost:8100
      # api_key: your-secret-key
```

The `mode: pre_call` ensures compression runs before the request reaches the provider. The plugin returns `GUARDRAIL_INTERVENED` with compressed `structured_messages` when savings are found, or `NONE` to pass through unchanged.
