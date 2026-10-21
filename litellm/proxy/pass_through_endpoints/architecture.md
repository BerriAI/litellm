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

## Anthropic stream diagnostics

Set `LITELLM_ANTHROPIC_STREAM_DIAGNOSTICS=errors` to collect an end-of-stream
summary for incomplete or failed native Anthropic streams. `all` also includes
completed streams. The default is `off`. Remove the variable or set it to `off`
and restart the workers to disable the additional summaries. This does not
disable normal success/failure callbacks or change billing

Each native HTTP attempt can emit one `provider_read` summary. A routed stream
can also emit one `router_emit` summary. Retries have their own attempt summaries.
There are no per-token or periodic logs. At one million single-attempt routed
requests, `all` can emit approximately two million summary lines, so use
`errors` for routine troubleshooting and `all` only for a bounded investigation.
Keep the proxy logger at INFO rather than enabling detailed DEBUG logging

Correlate summaries by `call_id`, the provider's `message_id` and, when supplied,
an allowlisted `upstream_request_id` response header. The summaries
contain byte and event counts, text character counts, first-byte and first-text timings, thinking and
tool-delta counts, terminal-event counts and the end reason. Times are relative
to that observation stage starting, after the upstream headers are available.
They are not total request latency or model execution time

`provider_read.text_chars > 0` with no corresponding `router_emit.text_chars`
shows that LiteLLM received text before its router emitted it. Zero text characters
at `provider_read` means no nonempty text delta was observed there, not proof
that the provider never generated text. Thinking and ping counts help distinguish
reasoning from an entirely silent stream. `router_emit` means yielded to the
next consumer, not acknowledged by the client socket. Check proxy, ingress and
client logs for failures beyond that boundary

These summaries never include prompts, generated text, tool arguments, keys,
raw headers or raw exception messages. The diagnostic parser retains at most
64 KiB of an unfinished event. Oversized or malformed events increase
`unparsed_frames`, making content counts inconclusive for those events without
changing the forwarded bytes. A worker killed before teardown cannot emit its
final summary

An HTTP 200 with no complete `message_stop`, or an SSE provider error, is logged
as a failed stream. Partial token usage and its cost still reach the failure
callbacks. Initial `message_start.usage.output_tokens` alone does not establish
a successful response. A consumer closing an unfinished stream is recorded as
499 rather than a provider 5xx. A premature upstream EOF is recorded as 502.
These are logging outcomes, not replacements for HTTP headers already sent
