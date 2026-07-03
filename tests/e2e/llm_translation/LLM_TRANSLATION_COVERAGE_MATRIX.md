# LLM Translation Test Coverage Matrix

Scope: the proxy's two translation surfaces, end to end against a live proxy.

1. **Passthrough** - the client speaks the provider's NATIVE API (Gemini
   `generateContent`, Anthropic `/v1/messages`); the proxy forwards it and still
   logs a costed `SpendLogs` row (`call_type="pass_through_endpoint"`). Routes:
   `/gemini`, `/anthropic`, `/vertex_ai`, `/openai`, `/bedrock`, `/cohere`,
   `/mistral`, `/vllm`.
2. **Non-passthrough** - the client speaks OpenAI format
   (`/chat/completions`, `/embeddings`); litellm translates to/from the provider.

The two axes that must work in production for each: **passthrough vs
non-passthrough** and **streaming vs non-streaming**, with **cost logged** and
**tool calls** working in every cell.

Companion: live suite `test_passthrough_e2e.py` (this directory). The
non-passthrough chat/embedding cells are exercised by `../spend_tracking/`.

Levels: `live` real provider + proxy + SpendLogs row; `unit` mocked.
Status: `covered` / `partial` / `gap`.

---

## Passthrough endpoints (native provider format)

| Provider | Non-streaming | Streaming | Tool calls | Cost logged | Status |
|----------|---------------|-----------|------------|-------------|--------|
| Gemini (`/gemini/v1beta/models/{m}:generateContent` / `:streamGenerateContent`) | live | live | live | live | **covered** |
| Anthropic (`/anthropic/v1/messages`) | live | live | live | live | **covered** |
| Vertex AI (`/vertex_ai/v1/projects/{p}/locations/{loc}/.../models/{m}:generateContent`) | live | - | - | live | **partial** |
| OpenAI / Bedrock / Cohere / Mistral / VLLM | - | - | - | - | gap |

Each covered cell asserts: `call_type == "pass_through_endpoint"`, `spend > 0`,
`status == "success"`, correct `custom_llm_provider`/`model`, row correlated by the
`x-litellm-call-id` header. Gemini non-streaming also pins `request_tags`
propagation; streaming pins `chunks > 0` then a costed row; tool tests assert the
provider emitted a tool call (`functionCall` / `tool_use`) and it was costed.

Cost on passthrough is computed in the success handler by transforming the native
response to a `ModelResponse` and calling `litellm.completion_cost()`; for
streaming, chunks are buffered and costed after the stream ends. This is the path
most likely to silently break and the one a mock can't prove works.

## Non-passthrough endpoints (OpenAI-compatible translation)

| Modality | Non-streaming | Streaming | Tool calls | Cost logged | Status |
|----------|---------------|-----------|------------|-------------|--------|
| Chat | live (spend suite) | live (spend suite) | gap | live | partial |
| Embeddings | live (spend suite) | n/a | n/a | live | covered |
| Responses / image / audio / rerank / realtime | - | - | - | - | gap |

## This suite's files

| Test | Cell |
|------|------|
| `test_gemini_passthrough_nonstreaming_logs_cost` | gemini native, non-stream, cost + tags |
| `test_gemini_passthrough_streaming_logs_cost` | gemini native, stream, cost |
| `test_gemini_passthrough_tool_call_logs_cost` | gemini native, tool call, cost |
| `test_anthropic_passthrough_nonstreaming_logs_cost` | anthropic native, non-stream, cost |
| `test_anthropic_passthrough_streaming_logs_cost` | anthropic native, stream, cost |
| `test_anthropic_passthrough_tool_call_logs_cost` | anthropic native, tool call, cost |
| `test_vertex_passthrough_via_managed_model_logs_cost` | vertex_ai native, non-stream, cost |

Vertex keeps the credential on the proxy like gemini/anthropic, but the deployment is
added at runtime instead of declared in the gateway config: the test POSTs `/model/new`
with `use_in_pass_through`, so the proxy registers that deployment's service account for
the `/vertex_ai` route, then deletes it on teardown. The passthrough call sends only its
litellm virtual key (`x-litellm-api-key`), no upstream bearer, and the proxy mints the
Vertex token itself. Credentials (`VERTEXAI_PROJECT` / `VERTEXAI_CREDENTIALS`) are read
from the same env the proxy uses, so the test never mints a token.

## Gaps

- Vertex streaming / tool-call passthrough (non-streaming + cost now covered).
- OpenAI / Bedrock / Cohere passthrough (same shape; add once the provider
  credential is configured).
- Non-passthrough tool calls over `/chat/completions` end to end with cost.
- Image / audio / rerank / responses / realtime translation + cost.
- Streaming cost-injection (`include_cost_in_streaming_usage`); passthrough on
  client disconnect (partial-usage logging).

## Adding a provider/modality

Extend `PassthroughClient` with the native call (it inherits keys, cleanup, and
SpendLogs polling from `ProxyClient`), then add a test that calls it,
`require_successful_call(result)`, and `_costed_row(...)`.

## Timing

Passthrough spend is logged asynchronously after the response and lands on the
`proxy_batch_write_at` (~60s) cycle, so cost assertions poll
`/spend/logs?request_id=<x-litellm-call-id>` to a deadline. Streaming cost is only
known after the stream is fully consumed.
