# Agent tracing

LiteLLM accepts OpenTelemetry traces from agent frameworks (LangChain, LangGraph, Deep Agents, anything
that speaks OTEL GenAI semconv or OpenInference). It stores them in ClickHouse and joins every agent LLM
call to the LiteLLM request that served it, so one trace shows the agent tree along with the key, team,
model and cost of each call.

```
 agent app (LangChain / Deep Agents)
   │  OTLP/HTTP  POST /v1/traces  (Authorization: Bearer sk-...)
   ▼
 TraceReceiver.ingest ── decode + normalize ── stamp tenant from auth ──► ClickHouse otel_traces
   │                                                                          │ (MV)
   │                                                                          ▼
   │                                                                     agent_traces
 agent LLM calls ─► LiteLLM /chat/completions ─► `clickhouse` callback ─► spend_logs
                                                                              │
 GET /v1/traces[/{id}] ◄── ClickHouseTraceStore ── join LiteLLMRequestId = response_id
```

Code: `receiver.py` (entry point), `decode.py` (OTLP → `SpanRow`), `store.py` (writes + SQL reads),
`types.py` (every shape in this doc), `litellm/proxy/tracing_endpoints.py` (HTTP),
`litellm/integrations/clickhouse/` (client, DDL, batch writer, spend-log callback).

## Setup

Proxy:

```yaml
general_settings:
  tracing:
    store: clickhouse          # enables POST/GET /v1/traces
litellm_settings:
  callbacks: ["clickhouse"]    # optional: writes spend_logs so LLM spans get key/team/cost
```

```bash
CLICKHOUSE_URL=http://localhost:8123 CLICKHOUSE_USER=writer CLICKHOUSE_PASSWORD=... CLICKHOUSE_DATABASE=litellm
CLICKHOUSE_READER_USER=litellm_traces_reader CLICKHOUSE_READER_PASSWORD=...
```

Tables are created on startup if they don't exist. Agent side (LangSmith's built-in OTEL exporter, no LangSmith account needed):

```bash
LANGSMITH_TRACING=true
LANGSMITH_TRACING_MODE=otel
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4000          # exporter appends /v1/traces
OTEL_EXPORTER_OTLP_HEADERS="Authorization=Bearer sk-..."   # a LiteLLM key; its team owns the traces
OTEL_SERVICE_NAME=research-agent
```

Point the agent's model client at the same proxy (`base_url=http://localhost:4000`). That makes LLM spans
carry LiteLLM response ids.

## Inbound payload

OTLP/HTTP, protobuf or JSON, gzip optional. This is one real LangSmith LLM span, trimmed (`…` marks cuts):

```json
{"resourceSpans": [{
  "resource": {"attributes": [
    {"key": "service.name", "value": {"stringValue": "agent-demo"}},
    {"key": "telemetry.sdk.language", "value": {"stringValue": "python"}}]},
  "scopeSpans": [{"scope": {"name": "langsmith"}, "spans": [{
    "traceId": "hFUOSZOtjJ0Z08vMDtomjA==", "spanId": "+6NV/7MYH4g=", "parentSpanId": "XUYYPvnrVnU=",
    "name": "ChatOpenAI", "kind": "SPAN_KIND_INTERNAL",
    "startTimeUnixNano": "1790742982590574080", "endTimeUnixNano": "1790742984795666944",
    "status": {"code": "STATUS_CODE_OK"},
    "attributes": [
      {"key": "langsmith.span.kind", "value": {"stringValue": "llm"}},
      {"key": "gen_ai.operation.name", "value": {"stringValue": "chat"}},
      {"key": "gen_ai.request.model", "value": {"stringValue": "claude-sonnet-4-5"}},
      {"key": "langsmith.metadata.lc_agent_name", "value": {"stringValue": "support_triage_agent"}},
      {"key": "langsmith.metadata.langgraph_node", "value": {"stringValue": "model"}},
      {"key": "gen_ai.usage.input_tokens", "value": {"intValue": "675"}},
      {"key": "gen_ai.usage.output_tokens", "value": {"intValue": "103"}},
      {"key": "gen_ai.tool.definitions", "value": {"stringValue": "[{\"type\":\"function\",…}]"}},
      {"key": "gen_ai.prompt", "value": {"bytesValue": "eyJtZXNzYWdlcyI6W1t7ImxjIjoxLC…"}},
      {"key": "gen_ai.completion", "value": {"bytesValue": "eyJnZW5lcmF0aW9ucyI6W1t7InRleH…"}}
    ]}]}]}]}
```

`gen_ai.prompt` decodes to `{"messages": [[{"lc":1, "id":[…,"SystemMessage"], "kwargs":{"content":"You are a LiteLLM support agent…","type":"system"}}, …]]}`.
`gen_ai.completion` decodes to `{"generations": [[{"message": {"kwargs": {"content": "", "tool_calls": […], "response_metadata": {"id": "chatcmpl-3d66069a-…", …}}}}]]}`.

What we read (`decode.py`):

| Attribute | Becomes |
|---|---|
| `service.name` (resource) | `ServiceName` |
| scope `langsmith` or `langsmith.span.kind` | LangSmith normalizer (else OpenInference if `openinference.span.kind`, else GenAI semconv) |
| `langsmith.span.kind` = `llm` / `tool` | `ObservationType` llm / tool; root span or name == `lc_agent_name` → `agent`; `*.wrap_model_call`, `*.before_agent`, … → `framework`; rest → `chain` |
| `langsmith.metadata.lc_agent_name` | `AgentName` (enclosing agent) |
| `gen_ai.request.model` | `Model` |
| `gen_ai.usage.input_tokens` / `output_tokens` | `InputTokens` / `OutputTokens` |
| `gen_ai.completion` → `…message.kwargs.response_metadata.id` | `LiteLLMRequestId` |
| `gen_ai.prompt` / `gen_ai.completion` | `Input` / `Output`: llm → `[{role, content, tool_calls?}]`; tool → args / result text (LangGraph `Command` → last message content); agent → first input messages / last output message |
| GenAI semconv: `gen_ai.operation.name`, `gen_ai.agent.name`, `gen_ai.response.id`, `gen_ai.input/output.messages` | same columns |
| OpenInference: `openinference.span.kind`, `agent.name`, `llm.model_name`, `llm.token_count.*`, `input/output.value` | same columns |
| `exception` event (`exception.message`) | `StatusMessage` when `status.message` is empty |

The heavy attributes (`gen_ai.prompt`, `gen_ai.completion`, `gen_ai.tool.definitions`, `*.messages`,
`input/output.value`) move into `Input`/`Output` and are dropped from `SpanAttributes`. Everything else
is kept as a string map.

## Correlation

**Agents and subagents inside one trace.** Spans form a tree through `parent_span_id`. A span is an
`agent` when it is the root, or a chain whose name equals its `lc_agent_name`. A subagent is an agent span
nested under another agent. In Deep Agents that looks like `research_lead` → tool `task` → chain
`researcher`. Every span's `agent` field names the agent it runs inside. `AgentNode.parent_agent` walks
up from each agent span to the nearest agent span with a *different* name. All invocations of one name
collapse into one node: 200 calls to `researcher` give one `AgentNode` with `invocations: 200` and
summed `llm_calls`, `tool_calls`, `spend` and `duration_ms`.

**Agent LLM call ↔ LiteLLM request.**
1. Response id (the default). `otel_traces.LiteLLMRequestId = spend_logs.response_id`. LiteLLM returns
   its id as the chat completion `id`, and LangChain records that as `response_metadata.id`. On cache
   hits LiteLLM appends `_cache_hit<ts>` to the request id, so the callback strips that suffix before
   writing `response_id`. When several spend-log rows share a response id, the read keeps the row
   closest in time to the span.
2. W3C `traceparent` (optional). If the agent forwards `traceparent` on its LLM calls, the callback
   stores it as `spend_logs.trace_id` / `span_id`. That also covers failed calls, which have no response id.
3. `x-litellm-session-id` as a fallback: stored as `spend_logs.session_id`.

**Across services (agent-to-agent over HTTP / A2A).** Propagate W3C `traceparent` on the outbound call.
The remote service then continues the same `trace_id`, and its spans land in the same trace under the
calling span. A different `service.name` on those spans marks the hop. For fire-and-forget handoffs
that start a new trace, use OTEL span Links. Links are stored (`Links.*` columns) but the read API does
not return them yet.

## Stored row

`SpanRow` = one `otel_traces` row. The standard columns match the OTel Collector `clickhouseexporter`, so a
collector can write to the same table.

| Column | Notes |
|---|---|
| `Timestamp`, `Duration` | span start (ns), duration (ns) |
| `TraceId`, `SpanId`, `ParentSpanId`, `TraceState` | hex ids; `ParentSpanId` = `''` for roots |
| `SpanName`, `SpanKind`, `ServiceName`, `ScopeName`, `ScopeVersion` | from OTLP |
| `ResourceAttributes`, `SpanAttributes` | `Map(String, String)`, values capped at `OTLP_MAX_ATTRIBUTE_VALUE_BYTES` |
| `StatusCode`, `StatusMessage` | `STATUS_CODE_OK/ERROR/UNSET`; message falls back to the exception event |
| `TeamId`, `ApiKeyHash` | from the auth'd key, never from the payload |
| `ObservationType`, `AgentName` | `agent / llm / tool / chain / framework`, enclosing agent |
| `LiteLLMRequestId`, `Model`, `InputTokens`, `OutputTokens` | join key + usage |
| `Input`, `Output` | normalized I/O (ZSTD(3)); `InputPreview` = first 240 chars (column DEFAULT) |

Tables (`litellm/integrations/clickhouse/schema.py`):

- `otel_traces`: one row per span. MergeTree, `PARTITION BY toDate(Timestamp)`, `ORDER BY (TeamId, ServiceName, toDateTime(Timestamp), TraceId)`, bloom filters on `TraceId` and `LiteLLMRequestId`, TTL `AGENT_TRACING_RETENTION_DAYS` (30).
- `agent_traces` + `agent_traces_mv`: one row per trace (counts, tokens, models, request ids, root name). AggregatingMergeTree, `ORDER BY (TeamId, TraceId)`. It backs the list endpoint. The MV writes one partial row per insert, and reads merge them.
- `spend_logs`: one row per LiteLLM request, written by the `clickhouse` callback. ReplacingMergeTree(end_time), `PARTITION BY toYYYYMM(start_time)`, `ORDER BY (team_id, toDateTime(start_time), request_id)`, bloom filters on `response_id` and `trace_id`, TTL `AGENT_TRACING_SPEND_LOG_RETENTION_DAYS` (90).

## Read API

`GET /v1/traces/{trace_id}` → `Trace`. This is real output from a Deep Agents run (`research_lead` fans out
to 4 `researcher` subagents and a `critic`). 216 spans, trimmed to five:

```json
{
  "summary": {
    "trace_id": "e309a123963901e74c29cd2d3c86ff9e", "name": "research_lead", "service": "research-agent",
    "input_preview": "[{\"role\": \"user\", \"content\": \"Should we store OTEL agent spans in ClickHouse or Postgres at 50k spans/sec?\"}]",
    "start_time": "2026-09-30T06:43:54.291000+00:00", "duration_ms": 40198.10688, "status": "ok",
    "span_count": 216, "agent_count": 3, "llm_calls": 21, "tool_calls": 25, "error_count": 0,
    "input_tokens": 69506, "output_tokens": 2960, "spend": 0.12833025, "models": ["claude-sonnet-4-5"]
  },
  "agents": [
    {"name": "research_lead", "parent_agent": null, "invocations": 1, "llm_calls": 3, "tool_calls": 5, "spend": 0.025758, "duration_ms": 40198.10688},
    {"name": "researcher", "parent_agent": "research_lead", "invocations": 4, "llm_calls": 17, "tool_calls": 20, "spend": 0.089172, "duration_ms": 41634.618112},
    {"name": "critic", "parent_agent": "research_lead", "invocations": 1, "llm_calls": 1, "tool_calls": 0, "spend": 0.01340025, "duration_ms": 6897.236224}
  ],
  "spans": [
    {"span_id": "3586edf49d446541", "parent_span_id": null, "name": "research_lead", "type": "agent",
     "agent": "research_lead", "start_offset_ms": 0.0, "duration_ms": 40198.10688, "status": "ok",
     "input_preview": "[{\"role\": \"user\", \"content\": \"Should we store OTEL agent spans in ClickHouse or Postgres at 50k spans/sec?\"}]",
     "model": null, "input_tokens": 0, "output_tokens": 0, "litellm": null},
    {"span_id": "2c51e6ddab97ed21", "parent_span_id": "ae58781d8e2e70eb", "name": "ChatOpenAI", "type": "llm",
     "agent": "research_lead", "start_offset_ms": 5.357056, "duration_ms": 8283.9168, "status": "ok",
     "input_preview": "[{\"role\": \"system\", \"content\": \"You are a research lead. Split the question into exactly 4 narrow sub-questions…",
     "model": "claude-sonnet-4-5", "input_tokens": 3378, "output_tokens": 514,
     "litellm": {"request_id": "chatcmpl-028008eb-a34b-4132-97c2-526b707e961e", "model": "openai/claude-sonnet-4-5",
                 "model_group": "claude-sonnet-4-5", "provider": "openai", "key_alias": "research-bot",
                 "team_alias": "research-agents", "spend": 0.0087315, "prompt_tokens": 3378, "completion_tokens": 514,
                 "cache_read_tokens": 3375, "cache_write_tokens": 0, "latency_ms": 8278, "ttft_ms": 8278, "status": "success"}},
    {"span_id": "53e84f1d468bc2a6", "parent_span_id": "dcf2045cfb34c569", "name": "task", "type": "tool",
     "agent": "research_lead", "start_offset_ms": 8290.994944, "duration_ms": 6624.634112, "status": "ok",
     "input_preview": "{\"subagent_type\":\"researcher\",\"description\":\"Research and answer this narrow question: What are the write performance…",
     "model": null, "input_tokens": 0, "output_tokens": 0, "litellm": null},
    {"span_id": "8febe34dc2549d84", "parent_span_id": "53e84f1d468bc2a6", "name": "researcher", "type": "agent",
     "agent": "researcher", "start_offset_ms": 8291.454976, "duration_ms": 6624.08704, "status": "ok",
     "input_preview": "[{\"role\": \"user\", \"content\": \"Research and answer this narrow question: What are the write performance…",
     "model": null, "input_tokens": 0, "output_tokens": 0, "litellm": null},
    {"span_id": "ca8488012a54b761", "parent_span_id": "96abed5e22e54c83", "name": "search_docs", "type": "tool",
     "agent": "researcher", "start_offset_ms": 10082.61504, "duration_ms": 0.475904, "status": "ok",
     "input_preview": "{\"query\":\"Postgres write performance time-series data ingestion 50k inserts per second high-volume\"}",
     "model": null, "input_tokens": 0, "output_tokens": 0, "litellm": null}
  ]
}
```

Spans are flat and ordered by start time. Build the tree from `parent_span_id`. `litellm` is set only on
llm spans that matched a spend-log row. `status` is the root span's status, while `error_count` counts
every span with an error status (a tool that raised shows up there even when the agent recovered).

`GET /v1/traces?start_ms=&end_ms=&cursor=` → `TracePage` (newest first, default window 24h, page size
`AGENT_TRACING_LIST_PAGE_SIZE`=50):

```json
{"data": [
  {"trace_id": "f78f6df35480060fafadac887e234241", "name": "support_triage_agent", "service": "research-agent",
   "input_preview": "[{\"role\": \"user\", \"content\": \"Customer acme-404 says billing is wrong. What plan are they on?\"}]",
   "start_time": "2026-09-30T06:43:52.928000+00:00", "duration_ms": 1315.0, "status": "ok",
   "span_count": 5, "agent_count": 1, "llm_calls": 1, "tool_calls": 1, "error_count": 2,
   "input_tokens": 659, "output_tokens": 60, "spend": 0.002877, "models": ["claude-sonnet-4-5"]}
 ],
 "next_cursor": null}
```

`next_cursor` is set when the page is full. Pass it back as-is.
`GET /v1/traces/{trace_id}/spans/{span_id}` → `SpanDetail`: `{span_id, input, output, attributes}`, with the full
`Input`/`Output` and `SpanAttributes` for the span drawer.

## Scoping and privacy

- Writes: `TeamId`, `ApiKeyHash` and `litellm.team_id` / `litellm.api_key_hash` / `litellm.org_id` in
  `ResourceAttributes` always come from the authenticated key. Values the client sends for these are overwritten.
- Reads (`scope_for`): proxy admins (incl. view-only) see every team. A key with a team sees that team's
  traces. A team-less key sees only traces it sent (`TeamId = ''` and `ApiKeyHash` = its hash). A request
  with neither a team nor a key gets 403.
- Spend-log rows are scoped the same way, so a join never exposes another team's cost.
- `litellm.turn_off_message_logging = True` blanks `messages` / `response` in `spend_logs`. Span `Input` /
  `Output` come from the agent's own exporter; to keep prompts out, turn off content capture on the agent
  side (e.g. `LANGSMITH_HIDE_INPUTS=true` / `LANGSMITH_HIDE_OUTPUTS=true`).

## Limits

| Setting | Default | Behaviour |
|---|---|---|
| `OTLP_MAX_BODY_BYTES` | 8 MiB | larger bodies → 413 |
| `OTLP_MAX_ATTRIBUTE_VALUE_BYTES` | 64 KiB | attribute values, `Input`, `Output` truncated with `…[truncated N bytes]` |
| `OTLP_OFFLOAD_DECODE_BYTES` | 256 KiB | bodies above this are decoded in a worker thread |
| `CLICKHOUSE_MAX_BUFFERED_ROWS` | 200,000 | buffer full → 429 + `Retry-After: OTLP_RETRY_AFTER_SECONDS` (2); OTLP exporters retry |
| `CLICKHOUSE_BATCH_SIZE`, `CLICKHOUSE_FLUSH_INTERVAL_SECONDS` | 10,000 rows, 1s | spans are written in batches; `POST` never waits on ClickHouse |
| `CLICKHOUSE_MAX_RETRIES` | 3 | after this many failed inserts a batch is dropped and logged |


## Rust foundation

Tracing requires the compiled Rust extension from the foundation PR. Rust owns the canonical SQL schema and parameterized read transport; Python handles ingestion, batching, trace response assembly and FastAPI integration in this incremental migration

Configure `CLICKHOUSE_READER_USER` and `CLICKHOUSE_READER_PASSWORD` separately from the credentials used for setup and ingestion. Grant the reader SELECT on the configured database's `otel_traces`, `agent_traces` and `spend_logs`, with the locked profile in `litellm-rust/crates/traces/config/reader.xml`. Change the example database grants from `default` to `CLICKHOUSE_DATABASE`. Query settings alone do not restrict a privileged account

Read requests have a 10-second query limit, a 15-second HTTP timeout, a 1,000-row result limit and a 4 MiB response cap. Exceeding these limits fails the request rather than returning a silently truncated trace

Schema setup calls the Rust schema export. There is no second Python DDL definition. The initial schema creates missing objects and does not migrate incompatible existing tables
