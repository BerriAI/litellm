# NeuralTrust TrustGuard

Native LiteLLM guardrail. Sends chat input and output to TrustGuard `POST /v1/evaluate`.

Setup guide, verdict mapping, and the streaming caveat:
[docs.neuraltrust.ai/integrations/litellm](https://docs.neuraltrust.ai/integrations/litellm).

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
      streaming_transform_mode: incremental_diff
      default_on: true
```

## Auth

Bearer `tgk_…` API key. Address the collector with `collector_key`, or omit it when the key is already bound to one.

## Identity

Each evaluate call carries `session_id` from the LiteLLM session and `consumer_id` from the virtual key: the key alias, else the key's user email, user id, or team alias. TrustGuard Activity and per-consumer policies group by that value.

## Verdicts

| TrustGuard `status` | LiteLLM |
| --- | --- |
| `block` | HTTP 400 (trace_id / request_id only; findings are not echoed) |
| `ask` | HTTP 400 like `block`: a proxy has no approval flow, so the response names `verdict: ask` |
| `transform` | rewrite the last user message / last text from `transformed_payload` |
| `report` / `allow` | pass through (`report` is logged by trace_id) |

Unknown verdicts, malformed bodies, and `transform` without a usable payload fail closed.

## Fail-open vs fail-closed

`unreachable_fallback` applies only to transport failures: connect errors, timeouts, HTTP 502/504. TrustGuard
does not emit 502 or 504 itself, so those mean the request never reached a verdict.

HTTP 503 entitlements, 429 rate limits, 401/403, other 4xx/5xx, and unusable TrustGuard verdicts always fail closed.

503 and 429 stay fail-closed on purpose, unlike `generic_guardrail_api`, which treats 503 as unreachable. A
TrustGuard 503 means its rate-limit entitlements could not be loaded, so TrustGuard itself declined to evaluate
rather than evaluate blind; 429 means the plan's burst or monthly quota is exhausted. Both are verdicts about
whether the traffic can be inspected, not signs that the service is down, so honouring `fail_open` for them would
pass uninspected traffic the guardrail deliberately refused. 429 logs `X-RateLimit-Reason` so a transient burst is
distinguishable from an exhausted monthly quota.

`fail_open` means the request bypasses TrustGuard entirely when the endpoint is unreachable. It is off by default.

## Streaming

`block` and `ask` end the stream in either mode. `streaming_transform_mode` decides whether a `transform` verdict reaches the client.

| Mode | What the client receives |
| --- | --- |
| `block_only` (default) | the raw model tokens, so the redaction is dropped |
| `incremental_diff` | TrustGuard's rewritten reply |

Under `incremental_diff` the reply is held until the end-of-stream evaluate returns, so the first token arrives with the last. TrustGuard re-reads the whole reply on every scan and its redaction spans move as the reply grows, so releasing tokens early would let a later scan try to rewrite text already on the wire. Holding them also means a blocking verdict ends the stream with nothing sent at all.

### Surface coverage

`incremental_diff` is implemented for OpenAI chat completions streaming only. Every other streamed surface —
including `/v1/messages` and `/v1/responses` — silently falls back to `block_only`, and the proxy logs a warning
naming the guardrail when it does.

**What that means on those surfaces:** `block` and `ask` still end the stream, so dangerous content is still
stopped. What is lost is masking: a `transform` verdict is dropped and the raw model tokens reach the client
unredacted. Treat `incremental_diff` as a chat-completions feature, and prefer non-streaming requests on
`/v1/messages` and `/v1/responses` when redaction must be enforced rather than merely attempted.

Extending the diff protocol to those surfaces means implementing the streamed text and tool-call rewrite
contract in their translation handlers, whose event shapes differ from chat completions. That work belongs in
the shared streaming layer, not in this guardrail.

Under `incremental_diff`, LiteLLM buffers tool-call deltas until TrustGuard inspects the assembled response. A blocking verdict releases no tool-call arguments or finish signal. Tool calls retain their IDs and order, with transformed arguments written into the buffered deltas before delivery

The default `block_only` mode still forwards tool-call deltas before inspection. Use `incremental_diff` or non-streaming requests when tool calls must be checked before delivery. The example selects `incremental_diff` for inspected text and tool arguments

## References

- [NeuralTrust TrustGuard on LiteLLM](https://docs.neuraltrust.ai/integrations/litellm)
- [TrustGuard Evaluate API](https://docs.neuraltrust.ai/trustguard/api/evaluate)
- [TrustGuard collectors](https://docs.neuraltrust.ai/trustguard/concepts/collectors)
- [LiteLLM Guardrails Documentation](https://docs.litellm.ai/docs/proxy/guardrails/quick_start)
