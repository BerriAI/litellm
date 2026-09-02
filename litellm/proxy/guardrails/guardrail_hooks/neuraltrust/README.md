# NeuralTrust TrustGuard

Native LiteLLM guardrail. Sends chat input and output to TrustGuard `POST /v1/evaluate`.

Setup guide, verdict mapping, and the streaming caveat:
[docs.neuraltrust.ai/trustguard/integrations/litellm](https://docs.neuraltrust.ai/trustguard/integrations/litellm).

## Config

```yaml
guardrails:
  - guardrail_name: neuraltrust-trustguard
    litellm_params:
      guardrail: neuraltrust
      mode: [pre_call, post_call]
      api_key: os.environ/TRUSTGUARD_API_KEY
      api_base: os.environ/TRUSTGUARD_API_BASE  # default https://trustguard.neuraltrust.ai
      collector_key: os.environ/TRUSTGUARD_COLLECTOR_KEY  # tgcol_… ; optional if the API key is bound
      unreachable_fallback: fail_closed
      timeout: 5
      default_on: true
```

## Auth

Bearer `tgk_…` API key. Address the collector with `collector_key`, or omit it when the key is already bound to one.

## Verdicts

| TrustGuard `status` | LiteLLM |
| --- | --- |
| `block` | HTTP 400 (trace_id / request_id only; findings are not echoed) |
| `transform` | rewrite the last user message / last text from `transformed_payload` |
| `report` / `allow` | pass through (`report` is logged by trace_id) |

Unknown verdicts, malformed bodies, and `transform` without a usable payload fail closed.

## Fail-open vs fail-closed

`unreachable_fallback` applies only to transport failures: connect errors, timeouts, HTTP 502/504.

HTTP 503 entitlements, 401/403, other 4xx/5xx, and unusable TrustGuard verdicts always fail closed.

`fail_open` means the request bypasses TrustGuard entirely when the endpoint is unreachable. It is off by default.

## Streaming

LiteLLM streaming guardrails default to `block_only`. `block` still fires on streamed calls. `transform` rewrites are not applied to the streamed tokens; use non-streaming requests when DLP redaction must reach the client.

## References

- [NeuralTrust TrustGuard on LiteLLM](https://docs.neuraltrust.ai/trustguard/integrations/litellm)
- [TrustGuard Evaluate API](https://docs.neuraltrust.ai/trustguard/api/evaluate)
- [TrustGuard collectors](https://docs.neuraltrust.ai/trustguard/concepts/collectors)
- [LiteLLM Guardrails Documentation](https://docs.litellm.ai/docs/proxy/guardrails/quick_start)
