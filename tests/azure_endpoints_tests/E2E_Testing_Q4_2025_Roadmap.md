# Azure OpenAI API Testing Specification

## Phase 1 - Core Provider Testing

| Provider / Host    | Model(s)                  | Status              |
|:-------------------|:--------------------------|:--------------------|
| **Azure OpenAI**   | OpenAI models             | ✅ **COMPLETED**    |
|                    |                           |                     |
| Endpoints          | Description (Regression focus) | Status         |
|:-------------------|:--------------------------|:--------------------|
| `/messages`        | Schema validity           | ✅ 6/6 tests passed |
| `/chat/completions`| Usage/cost attribution    | ✅ 13/13 tests passed |
| `/responses`       | Logging/traces            | ✅ 5/5 tests passed |
| `/batches`         | Streaming                 | ✅ 6/6 tests passed |
|                    | Error handling            | ✅ Covered          |
|                    | Batch lifecycle           | ✅ Covered          |

| Provider / Host    | Model(s)                  | Status              |
|:-------------------|:--------------------------|:--------------------|
| **Vertex Claude**  | Claude via Anthropic API  | 🔄 **PENDING**      |
|                    |                           |                     |
| Endpoints          | Description (Regression focus) | Status              |
|:-------------------|:--------------------------|:--------------------|
| `/messages`        | Schema validity           |                     |
| `/chat/completions`| Usage/cost attribution    |                     |
| `/responses`       | Logging/traces            |                     |
| `/batches`         | Streaming                 |                     |
|                    | Error handling            |                     |
|                    | Batch lifecycle           |                     |

| Provider / Host    | Model(s)                |
|:-------------------|:------------------------|
| **Vertex Gemini**  | Gemini                  |
|                   |                         |
| Endpoints         | Description (Regression focus) |
|-------------------|--------------------------|
| `/messages`       | Schema validity          |
| `/chat/completions` | Usage/cost attribution  |
| `/responses`      | Logging/traces           |
| `/batches`        | Streaming                |
|                   | Error handling           |
|                   | Batch lifecycle          |

| Provider / Host    | Model(s)                |
|:-------------------|:------------------------|
| **Deepseek**       | -                       |
|                   |                         |
| Endpoints         | Description (Regression focus) |
|-------------------|--------------------------|
| `/messages`       | Schema validity          |
| `/chat/completions` | Usage/cost attribution  |
| `/responses`      | Logging/traces           |
| `/batches`        | Streaming                |
|                   | Error handling           |
|                   | Batch lifecycle          |

| Provider / Host    | Model(s)                |
|:-------------------|:------------------------|
| **Mistral**        | -                       |
|                   |                         |
| Endpoints         | Description (Regression focus) |
|-------------------|--------------------------|
| `/messages`       | Schema validity          |
| `/chat/completions` | Usage/cost attribution  |
| `/responses`      | Logging/traces           |
| `/batches`        | Streaming                |
|                   | Error handling           |
|                   | Batch lifecycle          |

| Provider / Host    | Model(s)                |
|:-------------------|:------------------------|
| **On-Prem**        | self-hosted,            |
|                   | vLLM/Ollama             |
|                   |                         |
| Endpoints         | Description (Regression focus) |
|-------------------|--------------------------|
| `/messages`       | Schema validity          |
| `/chat/completions` | Usage/cost attribution  |
| `/responses`      | Logging/traces           |
| `/batches`        | Streaming                |
|                   | Error handling           |
|                   | Batch lifecycle          |

---

## Phase 2 (Future / Nice-to-Haves)

| Category                      | Features / Examples               |
|-------------------------------|-----------------------------------|
| Adjacent Features             | Session management                |
| Adjacent Features             | Passthrough endpoints             |
| Adjacent Features             | Transformations transparency (req + resp) |
| Adjacent Features             | Managed files                     |
| Adjacent Features             | Benchmarking overhead             |
| Adjacent Features             | SSO                               |
| OpenAI Python SDK Examples    | Responses                         |
| OpenAI Python SDK Examples    | Chat Completions                  |
| OpenAI Python SDK Examples    | Vision                            |
| OpenAI Python SDK Examples    | Audio STT/TTS                     |
| OpenAI Python SDK Examples    | Files API                         |
| OpenAI Python SDK Examples    | Embeddings                        |
| OpenAI Python SDK Examples    | Batches                           |
| OpenAI Python SDK Examples    | Retries/Pagination/Timeouts       |
| OpenAI Python SDK Examples    | AzureOpenAI client                |
| OpenAI Python SDK Examples    | Realtime WS demo                  |
| OpenAI Agents Python Examples | Basic agent                       |
| OpenAI Agents Python Examples | Agent patterns                    |
| OpenAI Agents Python Examples | Handoffs                          |
| OpenAI Agents Python Examples | Tools integration                 |
| OpenAI Agents Python Examples | MCP integration                   |
| OpenAI Agents Python Examples | Sessions & tracing                |
| OpenAI Agents Python Examples | Voice/Realtime agent              |
| OpenAI Agents Python Examples | End-to-end samples                |

## 🔧 Test Implementation Notes

### Schema Validation
- 🎯 **Exact key presence and types;** usage must be numeric
- ✅ **finish_reason** in allowed set: `{stop, length, tool_calls, content_filter}`

### Streaming
- 📊 **Verify event order** and final aggregation equals non-stream content
- 🔄 **Delta accumulation** produces identical final result

### Retries & Reliability  
- ⏰ **Bounded exponential backoff** with jitter for 429/5xx
- 🔑 **Idempotency:** Include `X-Idempotency-Key` in POST; assert dedupe

### Observability
- 📊 **Assert DB rows,** request/response transformation logs (both directions)
- 🔍 **Langfuse trace spans** (including tool steps)
- 🔒 **Redaction:** Ensure secrets removed from stored payloads

### Performance
- ⚡ **First token latency** threshold
- ⏱️ **Batch completes** within window

### Test Fixtures
- 🎯 **Deterministic prompts** (temperature:0)
- 📝 **Small golden outputs**
- 📋 **JSONL files** (valid + one bad row)
- 🖼️ **Mock images** (data URLs)
