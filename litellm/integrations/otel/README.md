# OpenTelemetry instrumentation

This package produces OpenTelemetry traces for LiteLLM. It is enabled by the
`LITELLM_OTEL_V2` environment variable (`is_otel_v2_enabled()` in
[`config.py`](./model/config.py)); when unset, nothing in this package runs.

## What gets traced

A traced proxy request produces one trace with two kinds of spans:

```
SERVER span  "POST /v1/chat/completions"        ← FastAPI instrumentation
├── INTERNAL span  "auth /v1/chat/completions"   ← auth phase     ┐
│   ├── CLIENT span  "postgres get_key_object"    ← datastore call │
│   └── CLIENT span  "postgres get_team_membership"                │
├── INTERNAL span  "execute_guardrail …"         ← guardrail       │ this package
├── CLIENT span    "chat gpt-4o"                  ← LLM call        │
└── CLIENT span    "batch_write_to_db …"          ← spend write    ┘
```

The gen-ai spans are siblings under the server span. In particular the guardrail
span is a sibling of the LLM call, not a child of it: pre/during/post-call
guardrail hooks are part of the request lifecycle (a pre-call guardrail runs
before the LLM call even starts), so they belong directly under the server span,
alongside the LLM call.

Request-level spans (LLM call, guardrail) parent to the server span via an
**explicit anchor** — `context.set_request_root_span` captures the server span
once at request entry, and `resolve_request_span_context` reads it — rather than
to whatever span is momentarily active. Ambient-only parenting was wrong at two
boundaries: inside the live `auth` phase span the active span is `auth` (so the
span would nest under auth), and a pass-through request closes its span from a
detached `asyncio.create_task` where the server span is no longer active (so the
span orphaned into its own trace). The anchor — a contextvar inherited by those
child tasks — gives a stable parent in both cases. DB/service spans keep ambient
parenting so an auth DB lookup still nests under `auth`.

**Which service calls become spans (`spans.span_role_for_service`).** LiteLLM's
service-logging layer instruments many internal functions, but only some are
traceable units of work:

- **`DB_CALL` (CLIENT)** — outbound datastore calls (redis, postgres,
  `batch_write_to_db`), carrying `db.system.name` / `db.operation.name` semconv.
- **`SERVICE` (INTERNAL)** — genuine internal work worth a span (background
  budget/reset jobs, pod-lock manager).
- **metrics-only (no span)** — `self` (the `track_llm_api_timing` wrapper, which
  duplicates the LLM-call span), `router` (duplicates the request), and
  `proxy_pre_call` (a guardrail's real span is `execute_guardrail …`). These
  still feed Prometheus/Datadog through their own hooks; they just never enter
  the trace. `auth` is also excluded here because it gets a **live phase span**
  instead (see below).

Spans are named `"{service} {call_type}"` (e.g. `"redis set"`) so repeated calls
to one service stay distinguishable. Like every other span they parent to the
**ambient** context, falling back to the threaded `litellm_parent_otel_span` only
when ambient has no live span; a background job with neither starts its own root
trace. Caller-supplied `event_metadata` is **sanitized** before it reaches a span
(primitives only, no live objects, no secrets/headers, bounded) — see
`payloads.sanitize_event_metadata`.

**Live phase spans.** `auth` is wrapped in a real, active span
(`logger.phase_span`) for the duration of authentication, so the DB lookups it
triggers nest **under** it instead of flattening onto the server span. Identity
Baggage (team/key/user) is seeded once the key resolves, so every post-auth span
inherits it; auth-internal DB lookups that run before the key is known stay
unlabeled, which is correct.

**Status.** On success a span's status is left `UNSET` (the semconv default,
matching the FastAPI server span); only a genuine error sets `ERROR`.

- **Server spans** (one per HTTP route) are created by the
  `opentelemetry-instrumentation-fastapi` package. It stamps `http.*` attributes
  and extracts inbound `traceparent` headers. This package does **not** create
  or modify server spans — request routes never touch spans.
- **Gen-AI spans** (LLM calls, guardrails, internal service calls) are created
  by this package from LiteLLM's logging callbacks. Request-level spans parent to
  the server span via the captured anchor; DB/service spans parent to the active
  span (ambient) so they nest under the request phase that triggered them.

Both kinds share a single `TracerProvider`, so they belong to the same trace
and export through the same configured exporters. FastAPI middleware can only be
added before the app starts serving, so the app is instrumented at
import time **without** a provider — it binds to the OTel global
`ProxyTracerProvider`. Once config (and the callbacks) is loaded, the proxy
publishes the chosen logger's `TracerProvider` as the global via
`trace.set_tracer_provider(...)`, and the server spans delegate to it. When a
preset callback (`arize`, `langfuse_otel`, …) is configured, its provider
becomes the global, so server spans export to that backend too.

## How a request flows

1. **App creation** (`proxy_server` import): when the gate is on,
   `mount.instrument_fastapi_app(app)` calls `FastAPIInstrumentor.instrument_app`
   with no provider (the middleware stack is frozen once the app serves, so this
   can't wait for startup). It binds to the OTel global `ProxyTracerProvider`. Noisy
   non-LLM routes are excluded by default (`mount._DEFAULT_EXCLUDED_ROUTES`): health
   checks (`/health*`), the Prometheus scrape (`/metrics`), and static UI/docs assets
   (`/litellm-asset-prefix`, `/_next`, `/ui`, `/swagger`, `/docs`, `/redoc`,
   `/openapi.json`, favicons, `/.well-known`) — so load-balancer polling, metric
   scrapes, and asset fetches don't flood traces. Entries are substring-matched, so
   `/metrics` also drops the `/model/metrics` admin-analytics spans. Set
   `OTEL_PYTHON_FASTAPI_EXCLUDED_URLS` to override the whole set (e.g. `""` to trace
   everything, or your own comma-separated path list).
2. **Startup** (`proxy_server.proxy_startup_event`): after the config (and
   callbacks) is loaded, the already-registered preset `OpenTelemetryV2` logger
   is reused — or a generic one reading `OTEL_*` envs is built when no preset is
   configured — and its `TracerProvider` is published as the OTel global with
   `trace.set_tracer_provider(...)`. The proxy tracer then delegates to it, so
   server spans and gen-ai spans share one provider and the same trace.
3. **Request**: the FastAPI instrumentation starts the server span and makes it
   the active context for the request task. The proxy's first call into the V2
   logger (`create_litellm_proxy_request_started_span`, at the auth boundary)
   **captures it as the request anchor** (`set_request_root_span`), so every later
   request-level span has a stable explicit parent regardless of what is active
   when it emits.
4. **LLM call span (born at the boundary)**: `OpenTelemetryV2.log_pre_api_call`
   runs synchronously in the request task, just before the upstream call, and
   **opens** the LLM-call span there, parented to the anchored server span
   (`resolve_request_span_context`). The open span is held in a bounded cache keyed
   by `litellm_call_id` (a primitive the callback kwargs carry at both `pre_call`
   and close), so no live `Span` ever travels through a `litellm_params` metadata
   dict. For the boundary hook to fire at all, the logger is registered into
   `litellm.input_callback` — the list `Logging.pre_call` iterates. The async
   success/failure callback later
   **closes** it: it builds an `LLMCallSpanData` from the typed
   `standard_logging_object` (token usage and cost are computed only by then),
   stamps the attributes, sets status, and ends the span. The sync callback is a
   no-op (closing is async-only). When `pre_call` runs off the request task — a
   sync-only provider driven through a thread pool, where contextvars (and so the
   anchor) don't follow — no parent is visible there, so creation is **deferred**
   to the async callback, whose worker context was copied from the request task at
   enqueue and so still carries the anchor. **Pass-through** endpoints call
   `logging_obj.pre_call` in the request task too, then close from a detached
   `asyncio.create_task`; the anchor (not the by-then-inactive server span) keeps
   their LLM-call span in the request's trace. `pre_call` is litellm's generic
   "log the attempt" hook, so it also fires for synthetic proxy-gate error logs
   (auth/rate-limit rejections); those carry `LITELLM_LOGGING_NO_UPSTREAM_LLM_CALL`
   and are skipped, so a request rejected before reaching a provider never produces
   a phantom CLIENT span.
5. **Guardrails / services**: the post-call and service hooks emit guardrail and
   service spans the same way — typed data → engine → span. Service spans
   (Redis/Postgres) are dispatched by `litellm/_service_logger.py`, which
   recognizes the V2 `OpenTelemetryV2` logger (a plain `CustomLogger`, not a
   subclass of the legacy `OpenTelemetry`). It hands every service call to the
   logger — including calls with no parent span — and the V2 adapter decides the
   role (`DB_CALL` vs `SERVICE`), the parent (ambient → threaded → root), and
   whether the call is a traceable operation or a metrics-only ping. Guardrail
   span data is built from the typed, provider-agnostic
   `StandardLoggingGuardrailInformation` — no single provider's field shape is
   assumed.
6. **Export**: each span ends and is handed to the provider's span processors,
   which export to the configured backends (OTLP, console, in-memory, …).

## Components

### Sources of truth (`model/`, no OpenTelemetry import)

These define the shape of a span without depending on the OTel SDK, so they can
be imported anywhere. They live in [`model/`](./model) and form a closed set —
nothing here imports outside it:

- [`semconv.py`](./model/semconv.py) — attribute-key constants (`gen_ai.*`, `http.*`,
  `litellm.*`), the GenAI operation/provider enums, and the functions that map
  LiteLLM provider/call-type strings onto convention values.
- [`spans.py`](./model/spans.py) — the span registry: every span role, its OTel span
  kind, its place in the hierarchy, and its name builder.
- [`payloads.py`](./model/payloads.py) — frozen dataclasses (`LLMCallSpanData`,
  `GuardrailSpanData`, `ServiceSpanData`, …) built from heterogeneous logging
  payloads via `from_*` classmethods.
- [`config.py`](./model/config.py) — `OpenTelemetryV2Config`, a pydantic-settings
  model that reads `OTEL_*` / `LITELLM_OTEL_*` env vars, plus the feature gate.
  `capture_span_content` gates whether prompt/response bodies may be written as
  span attributes; it defaults **off** (`no_content`). The Baggage allowlists are
  configurable, not hard-coded: set `LITELLM_OTEL_BAGGAGE_PROMOTED_KEYS` /
  `LITELLM_OTEL_BAGGAGE_METADATA_KEYS` /
  `LITELLM_OTEL_BAGGAGE_TEAM_METADATA_KEYS` (comma-separated) as env vars, or
  `baggage_promoted_keys` / `baggage_metadata_keys` /
  `baggage_team_metadata_keys` (YAML lists) under `callback_settings.otel` in
  `config.yaml` — the latter reach the config through the logger's constructor
  kwargs. `baggage_team_metadata_keys` is empty by default, so none of a team's
  free-form metadata is promoted until each sub-key is explicitly allowlisted.
- [`baggage.py`](./model/baggage.py) — the single definition of which request-identity
  values are promoted into Baggage (so child spans inherit them) and under which
  attribute keys.
- [`utils.py`](./model/utils.py) — value coercion, JSON serialization, and
  extractor-table application, shared across the package.

### Engine

- [`emitter.py`](./emitter.py) — `SpanEmitter.emit(role, data)`: dedupe → start
  the span → run the mapper chain to stamp attributes → set status → end. It
  owns no attribute keys. The dedupe set (which coalesces the sync+async firing
  of one request) is a bounded LRU so it can't grow without limit.
- [`mappers/`](./mappers) — each mapper turns typed span data into a flat
  `{attribute key: value}` dict. They compose: listing several mapper names in
  the config layers multiple attribute vocabularies onto the same span.
  - `genai` — the canonical OpenTelemetry GenAI vocabulary, always present.
  - `legacy` — an additional vocabulary using the older semconv-ai / Traceloop
    attribute key names, for backends that read those.
  - `openinference`, `langfuse`, `weave`, `langtrace` — vendor vocabularies.
  - `resolve_mappers(names)` turns config names into mapper instances.

### Plumbing (`plumbing/`)

The OTel-SDK wiring. Everything here imports only `model/` and each other; it
lives in [`plumbing/`](./plumbing):

- [`providers.py`](./plumbing/providers.py) — builds the `TracerProvider`, its exporters
  (from `ExporterSpec`s), and the span processor that copies allowlisted Baggage
  entries onto every span. `register_exporter_factory(kind, factory)` lets a
  preset contribute a custom exporter `kind` (e.g. one that fetches an auth
  token lazily) without coupling this module to any vendor.
- [`context.py`](./plumbing/context.py) — trace-context and Baggage read/write helpers.
- [`routing.py`](./plumbing/routing.py) — `TenantTracerCache`: when a request carries
  team/key-scoped vendor credentials, route its spans through a credential-keyed
  `TracerProvider` so one logger serves many tenants. The cache is a bounded LRU
  that flushes + shuts down evicted providers, since the key derives from
  request-supplied credentials and must not grow (or leak threads) without limit.
- [`metrics.py`](./plumbing/metrics.py) — GenAI client metric instruments. The
  six `gen_ai.client.*` histograms are recorded through the meter resolved by
  `providers.resolve_meter_provider`: an injected provider wins (tests/DI),
  otherwise the operator's globally configured `MeterProvider` is reused so its
  readers/exporters receive them alongside the server metrics, and one is built
  and registered as the global only when none is set (mirroring how V2 owns trace
  export).

### Adapter

- [`logger.py`](./logger.py) — `OpenTelemetryV2`, a `CustomLogger` that
  translates LiteLLM's logging callbacks into typed span data and hands them to
  the engine. The LLM-call span is opened at the `log_pre_api_call` boundary
  (parented to the live server span via ambient context) and closed at the async
  success/failure callback; the open span is held in a bounded cache keyed by
  `litellm_call_id`, never threaded through a metadata dict. The logger registers
  itself into `litellm.input_callback` so `Logging.pre_call` fires the boundary
  hook.
- [`mount.py`](./mount.py) — `instrument_fastapi_app(app)`, the single call site
  that attaches `opentelemetry-instrumentation-fastapi` for SERVER spans. It owns
  the health-check exclusion default (`OTEL_PYTHON_FASTAPI_EXCLUDED_URLS`) and the
  passthrough span-naming hook (`PASSTHROUGH_PREFIXES`) so `proxy_server` carries
  no OTel detail. A safe no-op when the gate is off or the instrumentation package
  is absent; must be called at app-creation time (the middleware stack freezes
  once the app serves).

### Presets

- [`presets/`](./presets) — each preset reads one integration's env vars and
  returns an `OpenTelemetryV2Config` (exporter destination + mapper vocabularies
  + resource attributes). `PRESET_BY_CALLBACK` maps a callback name (`"arize"`,
  `"langfuse_otel"`, …) to its preset. Integrations that support team/key-scoped
  credentials also provide a per-request OTLP header builder
  (`DYNAMIC_HEADERS_BY_CALLBACK`). Presets do **no** network I/O at build time:
  AgentOps, for example, mints its JWT lazily inside a custom exporter on the
  first export (in the `BatchSpanProcessor` worker thread), never on the event
  loop.

## Extending

- **A new attribute vocabulary for a backend**: add a mapper in `mappers/`
  (a class with a `map(data) -> AttributeMap` method, typically built from
  `key -> extractor` tables) and register it in `mappers/__init__._MAPPER_BY_NAME`.
- **A new integration**: add a preset in `presets/` that returns an
  `OpenTelemetryV2Config`, and register it in `presets/__init__.PRESET_BY_CALLBACK`.
  If it supports dynamic credentials, add a header builder to
  `DYNAMIC_HEADERS_BY_CALLBACK`.
- **A new span kind**: add a role to `spans.py` (registry entry + name builder),
  a payload dataclass in `payloads.py`, and a branch in the relevant mapper(s).
