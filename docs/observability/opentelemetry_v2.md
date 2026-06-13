import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# OpenTelemetry v2 - Full-request tracing

OpenTelemetry v2 (OTel v2) is LiteLLM Proxy's next-generation tracing. It gives you **one clean trace per request** that shows the whole story of a request — the incoming HTTP call, authentication, guardrails, the LLM call itself, and the internal database/cache work — all nested in a single tree.

It follows standard [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/), so the traces it produces are readable in any OTel backend (Grafana Tempo, Jaeger, Honeycomb, Datadog, …) and come with ready-made presets for popular LLM observability tools (Arize, Phoenix, Langfuse, Weave, Langtrace, Levo, AgentOps).

:::info Opt-in feature

OTel v2 is **off by default**. Nothing in it runs until you set `LITELLM_OTEL_V2=true`. It is separate from the existing [OpenTelemetry integration](./opentelemetry_integration) — pick one.

:::

## What you get

A single request to your proxy produces **one trace** that looks like this:

```
POST /v1/chat/completions                  ← HTTP request (server span)
├── auth /v1/chat/completions              ← authentication
│   ├── postgres get_key_object            ← DB lookups during auth
│   └── postgres get_team_membership
├── execute_guardrail presidio-pii         ← each guardrail that runs
├── chat gpt-4o                            ← the LLM call (model, tokens, cost)
└── batch_write_to_db                      ← spend/usage written to DB
```

Highlights:

- **One trace, end to end** — the HTTP request, auth, guardrails, the LLM call, and DB writes all live in the same trace, correctly nested.
- **Rich GenAI attributes** — every LLM-call span carries `gen_ai.*` attributes: model, provider, token usage, cost, finish reasons, request parameters, and more.
- **Standards-based** — built on the official OpenTelemetry GenAI semantic conventions, so it works with any OTel-compatible backend.
- **Vendor presets** — one line to ship traces to Arize, Phoenix, Langfuse, Weave, Langtrace, Levo, or AgentOps in the format each tool expects.
- **Safe by default** — prompts and responses are **not** captured unless you explicitly opt in. Noisy routes (health checks, metrics scrapes, UI assets) are excluded automatically.
- **Distributed tracing** — if your client sends a `traceparent` header, LiteLLM's spans nest inside your existing trace.

## Requirements

OTel v2 instruments the proxy's FastAPI app, so it needs the OpenTelemetry SDK plus the FastAPI instrumentation package:

```shell
pip install "litellm[proxy]" \
  opentelemetry-api \
  opentelemetry-sdk \
  opentelemetry-exporter-otlp \
  opentelemetry-instrumentation-fastapi
```

> These packages ship with the proxy Docker image. You only need to install them manually for a `pip`-based proxy.

## Getting started

### 1. Send traces to any OTLP collector

Set the feature flag plus the standard `OTEL_*` environment variables. That's it — no config change needed.

<Tabs>

<TabItem value="otlp-http" label="OTLP HTTP collector">

```shell
LITELLM_OTEL_V2=true
OTEL_EXPORTER="otlp_http"
OTEL_ENDPOINT="http://localhost:4318"
```

</TabItem>

<TabItem value="otlp-grpc" label="OTLP gRPC collector">

```shell
LITELLM_OTEL_V2=true
OTEL_EXPORTER="otlp_grpc"
OTEL_ENDPOINT="http://localhost:4317"
```

> gRPC export needs `grpcio`. Install with `pip install grpcio`.

</TabItem>

<TabItem value="console" label="Print to console (testing)">

```shell
LITELLM_OTEL_V2=true
OTEL_EXPORTER="console"
```

Spans are printed to stdout — handy for verifying everything works before pointing at a real backend.

</TabItem>

</Tabs>

Pass auth headers your backend needs via `OTEL_HEADERS`:

```shell
OTEL_HEADERS="api-key=your-key,x-tenant=acme"
```

Then start the proxy as usual:

```shell
litellm --config config.yaml
```

Make a request, and you'll see one trace per request in your backend.

### 2. Send traces to a specific tool (presets)

For LLM observability tools, use a **preset**. A preset knows the tool's endpoint and emits attributes in the schema that tool expects. To enable one, add its name to `callbacks` in your config and set the tool's credentials as env vars.

<Tabs>

<TabItem value="arize" label="Arize">

```yaml title="config.yaml"
litellm_settings:
  callbacks: ["arize"]
```

```shell
LITELLM_OTEL_V2=true
ARIZE_SPACE_ID="your-space-id"
ARIZE_API_KEY="your-api-key"
```

</TabItem>

<TabItem value="phoenix" label="Arize Phoenix">

```yaml title="config.yaml"
litellm_settings:
  callbacks: ["arize_phoenix"]
```

```shell
LITELLM_OTEL_V2=true
PHOENIX_API_KEY="your-api-key"
PHOENIX_COLLECTOR_ENDPOINT="https://app.phoenix.arize.com/v1/traces"
PHOENIX_PROJECT_NAME="my-project"   # optional
```

</TabItem>

<TabItem value="langfuse" label="Langfuse">

```yaml title="config.yaml"
litellm_settings:
  callbacks: ["langfuse_otel"]
```

```shell
LITELLM_OTEL_V2=true
LANGFUSE_PUBLIC_KEY="pk-..."
LANGFUSE_SECRET_KEY="sk-..."
LANGFUSE_HOST="https://cloud.langfuse.com"   # or your self-hosted URL
```

</TabItem>

<TabItem value="weave" label="Weave (W&B)">

```yaml title="config.yaml"
litellm_settings:
  callbacks: ["weave_otel"]
```

```shell
LITELLM_OTEL_V2=true
WANDB_API_KEY="your-api-key"
WANDB_PROJECT_ID="your-entity/your-project"
```

</TabItem>

<TabItem value="langtrace" label="Langtrace">

```yaml title="config.yaml"
litellm_settings:
  callbacks: ["langtrace"]
```

```shell
LITELLM_OTEL_V2=true
# Langtrace reads from your existing OTLP collector — point it at Langtrace:
OTEL_ENDPOINT="https://langtrace.ai/api/trace"
OTEL_HEADERS="api_key=your-langtrace-api-key"
```

</TabItem>

<TabItem value="levo" label="Levo">

```yaml title="config.yaml"
litellm_settings:
  callbacks: ["levo"]
```

```shell
LITELLM_OTEL_V2=true
LEVO_AUTH_TOKEN="your-token"
LEVO_ORG_ID="your-org"
```

</TabItem>

<TabItem value="agentops" label="AgentOps">

```yaml title="config.yaml"
litellm_settings:
  callbacks: ["agentops"]
```

```shell
LITELLM_OTEL_V2=true
AGENTOPS_API_KEY="your-api-key"
```

</TabItem>

</Tabs>

:::tip Send to several places at once

Each preset adds its destination. List more than one callback (e.g. `["arize", "langfuse_otel"]`) and your spans are shipped to all of them in parallel, each in the right format.

:::

## Capturing prompts & responses

By default, OTel v2 records **metadata only** (model, tokens, cost, timing) and **never** writes prompt or response text to your traces. This is intentional — it keeps sensitive content out of your observability backend.

To capture message content, opt in explicitly:

```shell
# no_content (default) — never capture prompts/responses
OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT="no_content"

# span_only — write prompts/responses as attributes on spans
OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT="span_only"

# span_and_event — write content to both spans and events
OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT="span_and_event"
```

The gate is enforced centrally, so it applies to **every** backend at once — a user request can never force its prompt into your backend while capture is disabled.

## Metrics

Alongside traces, OTel v2 can emit GenAI **client metrics**: histograms for call latency, token usage, and cost that your backend aggregates across requests. Like the rest of OTel v2 they stay off until you turn them on

Set the flag in the proxy environment next to `LITELLM_OTEL_V2`:

```shell
LITELLM_OTEL_V2=true
LITELLM_OTEL_INTEGRATION_ENABLE_METRICS=true
```

Metrics ship through the exporter you already configured for traces. `OTEL_EXPORTER` (`console`, `otlp_http`, `otlp_grpc`), `OTEL_ENDPOINT`, and `OTEL_HEADERS` decide where the metric stream goes exactly as they do for spans, so the collector that receives your traces receives the metrics too

### What's recorded

Each successful LLM call records the standard OpenTelemetry GenAI client metrics:

| Metric | Unit | What it measures |
|---|---|---|
| `gen_ai.client.operation.duration` | `s` | Wall-clock time for the whole LLM call |
| `gen_ai.client.token.usage` | `{token}` | Tokens consumed, split into input and output by the `gen_ai.token.type` attribute |
| `gen_ai.client.token.cost` | `USD` | LiteLLM's computed cost for the call |
| `gen_ai.client.response.time_to_first_token` | `s` | Time to the first streamed token (streaming calls) |
| `gen_ai.client.response.time_per_output_token` | `s` | Average time per output token |
| `gen_ai.client.response.duration` | `s` | Provider-side generation time |

Every sample carries the same identity attributes as the matching span (operation, provider/system, request model, framework, and selected `metadata.*` fields), so you can group the histograms by model, provider, key, or team. These are the same six metrics the [v1 OpenTelemetry integration](./opentelemetry_integration) emits, with identical names and units, so a dashboard built for one reads the other

### Control metric attribute cardinality

By default every metric sample is stamped with the full identity attribute set, which includes per-request fields such as `hidden_params` and several `metadata.*` values. Those are close to unique per request, so each one multiplies the number of time series your backend tracks (one series per distinct attribute combination). At volume this explodes metric cardinality, and some backends, for example Splunk Observability Cloud, start throttling or dropping the metrics

v2 reads the same filter v1 does, from `callback_settings.otel.attributes` in your config. Nest an `attributes` block there with either an `include_list` (allowlist; emit only the listed attributes) or an `exclude_list` (denylist; emit everything except the listed attributes). The two are mutually exclusive. The filter applies to metrics only; spans keep their full attribute set, so traces stay rich while metric cardinality stays bounded

The block sits under `callback_settings.otel` on its own. You do not add `otel` to `callbacks`; that key turns on the separate v1 integration, while v2 stays driven by `LITELLM_OTEL_V2`

Unlike v1, v2 has no per-instance `attributes` field, so this global block is the only source. v2 also resolves the filter lazily on the first metric a request records rather than at boot, so a bad config (both lists set, or a forbidden name) surfaces on that first recorded request and editing the lists takes effect only after a restart. The filter is read only on the default OTLP path (callback name `otel` or unset); preset destinations such as `arize`, `arize_phoenix`, and `langfuse_otel` emit their metrics with the full attribute set, the same as in v1

```yaml title="config.yaml"
callback_settings:
  otel:
    attributes:
      exclude_list:
        - hidden_params
        - metadata.requester_metadata
        - metadata.requester_ip_address
        - metadata.spend_logs_metadata
        - metadata.mcp_tool_call_metadata
        - metadata.vector_store_request_metadata
        - metadata.prompt_management_metadata
```

When you want the smallest, most predictable attribute set, list exactly the attributes to keep with `include_list`. Anything not listed is dropped from metrics:

```yaml title="config.yaml"
callback_settings:
  otel:
    attributes:
      include_list:
        - gen_ai.operation.name
        - gen_ai.system
        - gen_ai.request.model
        - gen_ai.framework
        - metadata.user_api_key_team_id
        - metadata.user_api_key_org_id
```

`gen_ai.token.type` is never filtered out. It is stamped on `gen_ai.client.token.usage` after the filter runs, so the input/output split survives whatever list you set, and naming it in either `include_list` or `exclude_list` is rejected

## Which routes are traced

High-frequency, non-LLM routes are **excluded by default** so they don't flood your traces: health checks (`/health*`), the Prometheus scrape (`/metrics`), and static UI/docs assets (`/ui`, `/docs`, `/redoc`, `/_next`, `/openapi.json`, favicons, …).

To change the set, use the standard OpenTelemetry env var (comma-separated paths, substring-matched):

```shell
# Trace everything, including health checks
OTEL_PYTHON_FASTAPI_EXCLUDED_URLS=""

# Exclude only your own custom paths
OTEL_PYTHON_FASTAPI_EXCLUDED_URLS="/health,/internal"
```

## Per-key / per-team destinations (multi-tenant)

Some presets (`arize`, `langfuse_otel`, `weave_otel`) support **per-request credentials**: if a request carries team- or key-scoped credentials, its spans are routed to that tenant's project automatically. This lets one proxy serve many tenants, each seeing only their own traces — no extra setup beyond configuring those credentials on the key/team.

## Distributed tracing

If the incoming request has a W3C `traceparent` header, LiteLLM continues that trace instead of starting a new one. Your LiteLLM spans then appear inline inside whatever distributed trace your application already has — so you can follow a request from your app, through the proxy, to the LLM provider, in one view.

## Configuration reference

All values are environment variables. Boolean flags accept `true`/`false`.

| Variable | Default | Purpose |
|---|---|---|
| `LITELLM_OTEL_V2` | `false` | **Master switch.** OTel v2 does nothing until this is `true`. |
| `OTEL_EXPORTER` (alias `OTEL_EXPORTER_OTLP_PROTOCOL`) | `console` | Exporter kind: `console`, `otlp_http`, `otlp_grpc`. |
| `OTEL_ENDPOINT` (alias `OTEL_EXPORTER_OTLP_ENDPOINT`) | none | OTLP collector URL. Setting an endpoint implies `otlp_http` unless you override `OTEL_EXPORTER`. |
| `OTEL_HEADERS` (alias `OTEL_EXPORTER_OTLP_HEADERS`) | none | Comma-separated `key=value` auth headers for your backend. |
| `OTEL_SERVICE_NAME` | `litellm` | `service.name` resource attribute shown in your backend. |
| `OTEL_ENVIRONMENT_NAME` | none | `deployment.environment` resource attribute (e.g. `production`). |
| `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` | `no_content` | Prompt/response capture: `no_content`, `span_only`, `event_only`, `span_and_event`. |
| `OTEL_PYTHON_FASTAPI_EXCLUDED_URLS` | health/metrics/UI routes | Comma-separated paths to exclude from tracing (substring match). Set to `""` to trace everything. |
| `LITELLM_OTEL_INTEGRATION_ENABLE_METRICS` | `false` | Also emit the GenAI client metrics (duration, token usage, cost, streaming timings). See [Metrics](#metrics). |

### What's on an LLM-call span

Every `chat <model>` (LLM-call) span carries standard GenAI attributes, including:

| Attribute | Meaning |
|---|---|
| `gen_ai.operation.name` | The operation, e.g. `chat`, `embeddings`. |
| `gen_ai.provider.name` / `gen_ai.system` | The provider, e.g. `openai`, `anthropic`. |
| `gen_ai.request.model` | The model requested. |
| `gen_ai.response.model` | The model that answered. |
| `gen_ai.usage.input_tokens` / `output_tokens` | Token counts. |
| `gen_ai.request.temperature`, `max_tokens`, `top_p`, … | Request parameters, when set. |
| `gen_ai.response.finish_reasons` | Why generation stopped. |
| `gen_ai.input.messages` / `gen_ai.output.messages` | Prompt/response — **only** when content capture is enabled. |

## Troubleshooting

**No traces showing up?**

1. Confirm `LITELLM_OTEL_V2=true` is set in the proxy's environment.
2. Try `OTEL_EXPORTER="console"` first — if spans print to stdout, the problem is your exporter endpoint/headers, not LiteLLM.
3. Make sure you hit an LLM route (e.g. `/v1/chat/completions`). Health checks and UI routes are excluded by default.
4. Check that `opentelemetry-instrumentation-fastapi` is installed (see [Requirements](#requirements)).

**Only see the LLM call but no `auth`/`postgres`/server span?** Those server and DB spans require the FastAPI instrumentation package — install `opentelemetry-instrumentation-fastapi`.

**I see metadata but no prompts/responses.** That's the default. Set `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=span_only` to capture content.

## Support

For questions, open an issue at [BerriAI/litellm](https://github.com/BerriAI/litellm/issues).
