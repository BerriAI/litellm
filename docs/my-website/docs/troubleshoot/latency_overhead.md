# Latency Overhead Troubleshooting

Use this guide when you see unexpected latency overhead between LiteLLM proxy and the LLM provider.

## The Invisible Latency Gap

LiteLLM measures latency from when its handler starts. If a request waits in uvicorn's event loop **before** the handler runs, that wait is invisible to LiteLLM's own logs.

```
T=0   Request arrives at load balancer
      [queue wait — LiteLLM never logs this]
T=10  LiteLLM handler starts → timer begins
T=20  Response sent

LiteLLM logs: 10s    User experiences: 20s
```

To measure the pre-handler wait, poll `/health/backlog` on each pod:

```bash
curl http://localhost:4000/health/backlog \
  -H "Authorization: Bearer sk-..."
# {"in_flight_requests": 47}
```

Or scrape the `litellm_in_flight_requests` Prometheus gauge at `/metrics`.

| `in_flight_requests` | ALB `TargetResponseTime` | Diagnosis |
|---|---|---|
| High | High | Pod overloaded → scale out |
| Low | High | Delay is pre-ASGI — check for sync blocking code or event loop saturation |
| High | Normal | Pod is busy but healthy, no queue buildup |

If you're on **AWS ALB**, correlate `litellm_in_flight_requests` spikes with ALB's `TargetResponseTime` CloudWatch metric. The gap between what ALB reports and what LiteLLM logs is the invisible wait.

## Quick Checklist

1. **Check `in_flight_requests` on each pod** via `/health/backlog` or the `litellm_in_flight_requests` Prometheus gauge — this tells you if requests are queuing before LiteLLM starts processing. Start here for unexplained latency.
2. **Collect the `x-litellm-overhead-duration-ms` response header** — this tells you LiteLLM's total overhead on every request.
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
