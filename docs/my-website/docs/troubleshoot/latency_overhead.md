# Latency Overhead Troubleshooting

Use this guide when you see unexpected latency overhead between LiteLLM proxy and the LLM provider.

## Quick Checklist

1. **Collect the `x-litellm-overhead-duration-ms` response header** — this tells you LiteLLM's total overhead on every request. Start here.
2. **Is DEBUG logging enabled?** This is the #1 cause of latency with large payloads.
3. **Are you sending large base64 payloads?** (images, PDFs) — see [Large Payload Overhead](#large-payload-overhead).
4. **Enable detailed timing headers** to pinpoint where time is spent.

## Diagnostic Headers

### `x-litellm-overhead-duration-ms` (always on)

Every response from LiteLLM includes this header. It shows the total latency overhead in milliseconds added by LiteLLM proxy (i.e. total response time minus the LLM API call time). Collect this on every request to understand your baseline overhead.

```bash
curl -s -D - http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer sk-..." \
  -d '{"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}' \
  2>&1 | grep x-litellm-overhead-duration-ms
```

### `x-litellm-callback-duration-ms` (always on)

Shows time spent building callback/logging payloads (ms). If this is high (>100ms), your payloads may be too large for efficient logging.

```bash
curl -s -D - http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer sk-..." \
  -d '{"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}' \
  2>&1 | grep x-litellm
```

### Detailed Timing Breakdown (opt-in)

Set `LITELLM_DETAILED_TIMING=true` to get per-phase timing in response headers:

| Header | What it measures |
|--------|-----------------|
| `x-litellm-timing-pre-processing-ms` | Auth, routing, request processing (before LLM call) |
| `x-litellm-timing-llm-api-ms` | Actual LLM API call duration |
| `x-litellm-timing-post-processing-ms` | Response processing (after LLM returns) |
| `x-litellm-timing-message-copy-ms` | Message copy time in logging layer |

```bash
# Enable detailed timing
export LITELLM_DETAILED_TIMING=true
```

## Large Payload Overhead

When sending large payloads (>1MB, e.g. base64-encoded images/PDFs), three things can add overhead:

### 1. DEBUG Logging (most common)

When `LITELLM_LOG=DEBUG` or `set_verbose=True` is enabled, every request payload is serialized with `json.dumps(indent=4)` synchronously. For a 2MB+ payload, this alone can take **2-5 seconds**.

**Fix:** Don't use DEBUG logging in production. Use `INFO` level instead:

```bash
export LITELLM_LOG=INFO
```

If you need DEBUG logging but have large payloads, you can increase the size threshold for full payload logging:

```bash
# Only fully serialize payloads under 100KB for DEBUG logs (default)
export MAX_PAYLOAD_SIZE_FOR_DEBUG_LOG=102400
```

### 2. Base64 in Logging Payloads

Callback payloads (sent to Langfuse, etc.) include message content. Large base64 strings are automatically truncated to size placeholders in logging payloads.

You can control the truncation threshold:

```bash
# Max base64 characters before truncation (default: 64)
export MAX_BASE64_LENGTH_FOR_LOGGING=64
```

## Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `LITELLM_DETAILED_TIMING` | `false` | Enable per-phase timing headers |
| `MAX_PAYLOAD_SIZE_FOR_DEBUG_LOG` | `102400` | Max payload bytes for full DEBUG serialization |
| `MAX_BASE64_LENGTH_FOR_LOGGING` | `64` | Max base64 chars before truncation in logging |
