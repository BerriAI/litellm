# RFC: Rewrite of LiteLLM OpenTelemetry Instrumentation

Status: Proposed
Scope: `litellm/integrations/opentelemetry.py` and everything that depends on it
Target package: `litellm/integrations/otel/`

---

## 0. TL;DR

The current OTel integration is a single 3,227-line god-class
(`litellm/integrations/opentelemetry.py`) that passes untyped `kwargs` dicts and
raw response objects around, reads config from a scatter of env vars, and has no
single place that answers "what spans exist", "how are they nested", or "what
attributes do they carry". Attribute keys live in **three** different places plus
~30 inline string literals, and two conventions (legacy `llm.*` / stale
`gen_ai.usage.completion_tokens` and modern `gen_ai.*`) coexist incoherently.

We replace it with a new package, `litellm/integrations/otel/`, built on three
**declarative sources of truth**:

1. **`semconv.py`** — every attribute key, as typed constants/enums. The only
   place an attribute string is ever written.
2. **`spans.py`** — every span that exists, declared as data (name pattern, kind,
   **and its parent**). This is simultaneously the span inventory **and** the
   hierarchy.
3. **Typed span-data models** (`payloads.py`) — every span is built from a typed
   input model (derived from the existing typed `StandardLoggingPayload`,
   `ServiceLoggerPayload`, `UserAPIKeyAuth`). **No `**kwargs`, no `dict`, no `Any`
   at any boundary.**

Per the agreed decisions:
- **Strict OTel GenAI semconv** (`gen_ai.*`, span name `"{operation} {model}"`)
  **with dual-emit**: during a deprecation window the engine also emits the
  legacy keys behind a flag, then we drop them.
- **Full scope**: base LLM-call tracing, the `ServiceLogging` reuse, the six
  derived integrations (Arize, Arize-Phoenix, Levo, Langfuse-OTEL, Weave-OTEL,
  AgentOps) + the `callback_name`-dispatched Langtrace, and HTTP context
  propagation.

The old module is removed only at the end, after the new package is at behavioral
parity and the public import path (`litellm.integrations.opentelemetry.OpenTelemetry`)
has been turned into a thin shim that re-exports the new code.

---

## 1. What is wrong today (evidence)

All references are to the current worktree.

### 1.1 No typing — `kwargs`/god-objects everywhere
- Callback entry points have **no annotations at all**:
  `log_success_event(self, kwargs, response_obj, start_time, end_time)` and the
  async variants — `opentelemetry.py:527-537`.
- The OTel runtime types are aliased to `Any`: `Span = Any`, `Tracer = Any`,
  `Context = Any` — `opentelemetry.py:50-56`. Every `span: Span` annotation is
  therefore `Any` at runtime.
- The single largest god-object sink: `set_attributes(self, span, kwargs,
  response_obj)` — `opentelemetry.py:1968` — receives the entire untyped
  `model_call_details` dict + raw response and stamps dozens of attributes.
- Synthetic `kwargs` dicts are *fabricated* to reuse the untyped helpers, e.g.
  `_emit_guardrail_spans_from_request_data` (`opentelemetry.py:776-782`),
  failure/management hooks (`699-703`, `3123-3126`).

### 1.2 No source of truth for **what spans exist**
There are 10 span-creation sites with names coming from a mix of module constants,
inline literals, and dynamic builders (`opentelemetry.py`): `litellm_request`/`"{op}
{model}"`/`generation_name` (`_get_span_name` @ 2569), `"Received Proxy Server
Request"` (const @ 61), `raw_gen_ai_request` (const @ 68), `"guardrail"` (literal @
1646), `"Failed Proxy Server Request"` (literal @ 717), `payload.service` (dynamic),
`logging_payload.route` (dynamic). Nothing enumerates them.

### 1.3 No source of truth for **hierarchy**
Parenting is done by hand at every call site via
`trace.set_span_in_context(parent)` threaded through a 4-level priority resolver
`_get_span_context` (`opentelemetry.py:2600`) and a separate guardrail resolver
`_resolve_guardrail_context` (`1568`). The tree shape is implicit, reconstructable
only by reading all 10 sites.

### 1.4 No source of truth for **attributes**
Keys come from: (a) `litellm/proxy/_types.py:3574 SpanAttributes` enum (mixes
`gen_ai.system`, stale `gen_ai.usage.completion_tokens`, and Traceloop-style
`llm.*`); (b) `litellm/integrations/_types/open_inference.py`; (c) the opt-in
`opentelemetry_utils/gen_ai_semconv.py` maps; **plus ~30 inline literals** in the
main file (`"gen_ai.response.id"` @ 2128, `"gen_ai.provider.name"` @ 2064,
`f"gen_ai.cost.{key}"` @ 2035, `f"metadata.{key}"` @ 2013, the `guardrail_*` keys
@ 1657-1710, etc.).

### 1.5 Config read ad hoc from env
`OpenTelemetryConfig.from_env()` (`opentelemetry.py:132`) + `__post_init__` (`101`)
read 12+ env vars (`OTEL_EXPORTER_OTLP_PROTOCOL`, `OTEL_EXPORTER_OTLP_ENDPOINT`,
`OTEL_EXPORTER_OTLP_HEADERS`, `OTEL_SERVICE_NAME`, `OTEL_ENVIRONMENT_NAME`,
`OTEL_MODEL_ID`, `OTEL_IGNORE_CONTEXT_PROPAGATION`, `OTEL_SEMCONV_STABILITY_OPT_IN`,
`LITELLM_OTEL_INTEGRATION_ENABLE_METRICS/EVENTS`, `DEBUG_OTEL`,
`OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`, `OTEL_LOGS_EXPORTER`) plus
`get_secret_bool("USE_OTEL_LITELLM_REQUEST_SPAN")` checked inline in the hot path
(`1030`, `1748`).

### 1.6 Invisible coupling that makes it un-deletable today
- `set_attributes` branches on `self.callback_name` to call Langtrace / Langfuse /
  Weave attribute setters (`opentelemetry.py:1968-1994`). **Langtrace has no
  subclass** and dies if this dispatch is removed.
- Six subclasses override internal methods (`_init_tracing`,
  `_init_otel_logger_on_litellm_proxy`, `set_attributes`, `_start_primary_span`,
  `_maybe_log_raw_request`, `_handle_success`, `construct_dynamic_otel_headers`):
  `ArizeLogger` (`arize/arize.py:30`), `ArizePhoenixLogger`
  (`arize/arize_phoenix.py:40`), `LevoLogger`, `LangfuseOtelLogger`
  (`langfuse/langfuse_otel.py:30`), `WeaveOtelLogger` (`weave/weave_otel.py:189`),
  `AgentOps` (`agentops/agentops.py:31`). Any signature change ripples to all six.
- `proxy_server.open_telemetry_logger` global (`proxy_server.py:1846`) is claimed
  by a **"first-registered-wins"** guard inside `__init__`
  (`_init_otel_logger_on_litellm_proxy`, `opentelemetry.py:237-264`) and read by
  `_service_logger.py`, `litellm_pre_call_utils.py:2782`,
  `user_api_key_auth.py:682`, `management_helpers/utils.py:456`,
  `common_utils/debug_utils.py:727`.
- Dedup state for sync+async dual-fire lives in `kwargs` metadata
  (`_otel_internal.spans_logged`, `_emit_once` @ 919).

---

## 2. Goals & non-goals

### Goals
- One typed input per span; zero `**kwargs`/`dict`/`Any` at boundaries (mypy-clean
  under the repo's existing mypy config).
- `semconv.py` is the **only** file containing attribute-key strings.
- `spans.py` is the **only** place a span name/kind/parent is declared.
- Strict OTel GenAI semconv emission; legacy keys emitted only via an explicit
  dual-emit shim behind a flag, then removed.
- Derived integrations migrate to **composition** (an injected `AttributeMapper`
  strategy) instead of subclassing internals + string dispatch.
- Behavioral parity verified against the existing test suite before any deletion.

### Non-goals
- Changing how users *enable* OTel (`litellm.callbacks=["otel"]`,
  `config.yaml`, env vars all keep working).
- Adding agent/tool spans now (the registry is designed to host
  `create_agent`/`invoke_agent`/`execute_tool` later, but they are out of scope).
- Re-pricing/re-costing or changing `StandardLoggingPayload`.

---

## 3. Decisions (locked)

| Decision | Choice | Consequence |
|---|---|---|
| Semconv adoption | **Strict semconv + dual-emit** | New `gen_ai.*` keys + `"{operation} {model}"` span name are authoritative; legacy keys/span-names emitted behind `legacy_compat` flag during the deprecation window, then dropped. |
| Scope | Base LLM tracing **+** ServiceLogger **+** derived integrations **+** context propagation | All four migrate in this RFC. |
| Delivery | RFC doc (this file) + draft PR + chat summary | This file is the durable artifact. |

---

## 4. The three sources of truth

### 4.1 Attributes — `litellm/integrations/otel/semconv.py`

The **only** module that contains attribute-key strings. Modeled on the OTel GenAI
semantic-convention registry (experimental, current as of 2026). Two namespaces:

- `GenAI` — canonical OTel keys.
- `LiteLLM` — vendor-extension keys for things with no semconv equivalent
  (cost, guardrails, internal service spans, team/key metadata). Always prefixed
  `litellm.` per OTel guidance on custom attributes.

```python
from enum import Enum
from typing import Final

class GenAIOperation(str, Enum):           # gen_ai.operation.name values
    CHAT = "chat"
    TEXT_COMPLETION = "text_completion"
    EMBEDDINGS = "embeddings"
    GENERATE_CONTENT = "generate_content"
    CREATE_AGENT = "create_agent"          # reserved for future
    INVOKE_AGENT = "invoke_agent"          # reserved for future
    EXECUTE_TOOL = "execute_tool"          # reserved for future

class GenAIProvider(str, Enum):            # gen_ai.provider.name values (replaces gen_ai.system)
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AWS_BEDROCK = "aws.bedrock"
    AZURE_AI_OPENAI = "azure.ai.openai"
    AZURE_AI_INFERENCE = "azure.ai.inference"
    GCP_GEMINI = "gcp.gemini"
    GCP_VERTEX_AI = "gcp.vertex_ai"
    COHERE = "cohere"
    MISTRAL_AI = "mistral_ai"
    DEEPSEEK = "deepseek"
    GROQ = "groq"
    PERPLEXITY = "perplexity"
    X_AI = "x_ai"
    IBM_WATSONX_AI = "ibm.watsonx.ai"
    # ... full provider map maintained here, keyed off litellm's custom_llm_provider

class GenAI:
    # --- request (CLIENT) ---
    OPERATION_NAME: Final = "gen_ai.operation.name"     # Required
    PROVIDER_NAME:  Final = "gen_ai.provider.name"      # Required (replaces gen_ai.system)
    REQUEST_MODEL:  Final = "gen_ai.request.model"      # Required if available
    REQUEST_TEMPERATURE:       Final = "gen_ai.request.temperature"
    REQUEST_TOP_P:             Final = "gen_ai.request.top_p"
    REQUEST_TOP_K:             Final = "gen_ai.request.top_k"
    REQUEST_MAX_TOKENS:        Final = "gen_ai.request.max_tokens"
    REQUEST_FREQUENCY_PENALTY: Final = "gen_ai.request.frequency_penalty"
    REQUEST_PRESENCE_PENALTY:  Final = "gen_ai.request.presence_penalty"
    REQUEST_STOP_SEQUENCES:    Final = "gen_ai.request.stop_sequences"
    REQUEST_SEED:              Final = "gen_ai.request.seed"
    REQUEST_CHOICE_COUNT:      Final = "gen_ai.request.choice.count"
    REQUEST_ENCODING_FORMATS:  Final = "gen_ai.request.encoding_formats"
    # --- response ---
    RESPONSE_ID:             Final = "gen_ai.response.id"
    RESPONSE_MODEL:          Final = "gen_ai.response.model"
    RESPONSE_FINISH_REASONS: Final = "gen_ai.response.finish_reasons"
    # --- usage ---
    USAGE_INPUT_TOKENS:  Final = "gen_ai.usage.input_tokens"
    USAGE_OUTPUT_TOKENS: Final = "gen_ai.usage.output_tokens"
    # --- content (Opt-In, gated by capture mode) ---
    INPUT_MESSAGES:      Final = "gen_ai.input.messages"
    OUTPUT_MESSAGES:     Final = "gen_ai.output.messages"
    SYSTEM_INSTRUCTIONS: Final = "gen_ai.system_instructions"
    OUTPUT_TYPE:         Final = "gen_ai.output.type"
    CONVERSATION_ID:     Final = "gen_ai.conversation.id"
    # --- agent / tool (reserved) ---
    AGENT_ID: Final = "gen_ai.agent.id"; AGENT_NAME: Final = "gen_ai.agent.name"
    TOOL_NAME: Final = "gen_ai.tool.name"; TOOL_CALL_ID: Final = "gen_ai.tool.call.id"

class Error:
    TYPE: Final = "error.type"                 # Conditionally Required on error

class Server:
    ADDRESS: Final = "server.address"
    PORT:    Final = "server.port"

class HTTP:                                     # SERVER span (proxy) follows HTTP semconv
    REQUEST_METHOD:       Final = "http.request.method"
    ROUTE:                Final = "http.route"
    RESPONSE_STATUS_CODE: Final = "http.response.status_code"
    URL_PATH:             Final = "url.path"

class LiteLLM:                                  # vendor extensions (no semconv equivalent)
    CALL_ID:              Final = "litellm.call_id"
    COST_PREFIX:          Final = "litellm.cost."          # + breakdown key
    METADATA_PREFIX:      Final = "litellm.metadata."      # + metadata key
    TEAM_ID:              Final = "litellm.team.id"
    TEAM_ALIAS:           Final = "litellm.team.alias"
    KEY_HASH:             Final = "litellm.api_key.hash"
    GUARDRAIL_NAME:       Final = "litellm.guardrail.name"
    GUARDRAIL_MODE:       Final = "litellm.guardrail.mode"
    GUARDRAIL_STATUS:     Final = "litellm.guardrail.status"
    GUARDRAIL_ACTION:     Final = "litellm.guardrail.action"
    SERVICE_NAME:         Final = "litellm.service.name"
    PREPROCESSING_MS:     Final = "litellm.preprocessing.duration_ms"
```

Metrics keys also live here: `gen_ai.client.token.usage` (histogram),
`gen_ai.client.operation.duration` (histogram).

### 4.2 Spans + hierarchy — `litellm/integrations/otel/spans.py`

A single declarative registry. Each entry encodes the span's identity **and its
parent**, so the inventory and the hierarchy are the same artifact.

```python
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional
from opentelemetry.trace import SpanKind

class SpanRole(str, Enum):
    PROXY_REQUEST = "proxy_request"   # HTTP SERVER root (proxy)
    LLM_CALL      = "llm_call"        # gen_ai CLIENT span
    GUARDRAIL     = "guardrail"       # INTERNAL
    SERVICE       = "service"         # INTERNAL (redis/db/etc.)
    MANAGEMENT    = "management"      # admin endpoints (SERVER root)

@dataclass(frozen=True)
class SpanSpec:
    role: SpanRole
    kind: SpanKind
    parent: Optional[SpanRole]                 # <-- the hierarchy, as data
    name: Callable[["SpanData"], str]          # name builder over the typed input

SPAN_REGISTRY: dict[SpanRole, SpanSpec] = {
    SpanRole.PROXY_REQUEST: SpanSpec(
        SpanRole.PROXY_REQUEST, SpanKind.SERVER, parent=None,
        name=lambda d: f"{d.http_method} {d.route}",        # HTTP semconv
    ),
    SpanRole.LLM_CALL: SpanSpec(
        SpanRole.LLM_CALL, SpanKind.CLIENT, parent=SpanRole.PROXY_REQUEST,
        name=lambda d: f"{d.operation.value} {d.request_model}",  # "chat gpt-4o"
    ),
    SpanRole.GUARDRAIL: SpanSpec(
        SpanRole.GUARDRAIL, SpanKind.INTERNAL, parent=SpanRole.LLM_CALL,
        name=lambda d: f"execute_guardrail {d.guardrail_name}",
    ),
    SpanRole.SERVICE: SpanSpec(
        SpanRole.SERVICE, SpanKind.INTERNAL, parent=SpanRole.PROXY_REQUEST,
        name=lambda d: d.service_name,
    ),
    SpanRole.MANAGEMENT: SpanSpec(
        SpanRole.MANAGEMENT, SpanKind.SERVER, parent=None,
        name=lambda d: d.route,
    ),
}
```

**Canonical hierarchy** (the only tree the engine can produce):

```
PROXY_REQUEST  (SERVER, root — or rooted in the remote traceparent if present)
├── LLM_CALL   (CLIENT)            content captured as attrs/events on THIS span
│                                  (no separate raw_gen_ai_request child span)
├── GUARDRAIL  (INTERNAL)          falls back to PROXY_REQUEST if no LLM_CALL
└── SERVICE    (INTERNAL)          redis/db/etc.

MANAGEMENT     (SERVER, root)      admin endpoints, independent root
```

When LiteLLM is used as a **library** (no proxy), `LLM_CALL` is the root, or a
child of the caller's active span via extracted context.

**Deliberate consolidations vs. today** (semconv-aligned):
- `raw_gen_ai_request` child span (`opentelemetry.py:1145`) is **removed**; raw
  request/response become opt-in content attributes/events on `LLM_CALL`.
- `"Failed Proxy Server Request"` child span (`717`) is **removed**; failures set
  `error.type` + `Status(ERROR)` + `span.record_exception` on `PROXY_REQUEST`.
- Both old behaviors remain reproducible **only** under `legacy_compat=True`
  during the deprecation window (so the existing tests that assert
  `RAW_REQUEST_SPAN_NAME` / `"Failed Proxy Server Request"` keep passing until
  those tests are migrated).

### 4.3 Typed span-data inputs — `litellm/integrations/otel/payloads.py`

Every span is built from a frozen, typed model. These are *constructed once* from
the existing typed payloads — never raw `kwargs`.

```python
@dataclass(frozen=True)
class LLMCallSpanData:
    operation: GenAIOperation
    provider: GenAIProvider
    request_model: str
    response_model: Optional[str]
    response_id: Optional[str]
    request_params: "LLMRequestParams"          # temperature/top_p/max_tokens/...
    usage: "LLMUsage"                            # input/output/total tokens
    finish_reasons: list[str]
    messages_in: Optional[list[ChatMessage]]     # only if content capture on
    messages_out: Optional[list[ChatMessage]]
    error: Optional["SpanError"]
    cost: Optional["CostBreakdown"]
    guardrails: list["GuardrailSpanData"]
    server: Optional["ServerInfo"]
    identity: "RequestIdentity"                  # call_id, team, key, metadata
    timing: "SpanTiming"

    @classmethod
    def from_standard_logging_payload(
        cls, payload: StandardLoggingPayload
    ) -> "LLMCallSpanData": ...                  # the ONLY adapter from litellm internals
```

Equivalent typed models: `ProxyRequestSpanData.from_auth(UserAPIKeyAuth, request)`,
`ServiceSpanData.from_payload(ServiceLoggerPayload)`,
`ManagementSpanData.from_payload(ManagementEndpointLoggingPayload)`. **The
`from_*` adapters are the single chokepoint where untyped litellm internals are
read; everything downstream is typed.**

---

## 5. Component architecture (`litellm/integrations/otel/`)

```
litellm/integrations/otel/
  __init__.py        # public re-exports: OpenTelemetry, OpenTelemetryConfig, span-name consts
  config.py          # OpenTelemetryConfig (typed); ALL env reads isolated here
  semconv.py         # SOURCE OF TRUTH #1: attribute keys + enums + metric names
  spans.py           # SOURCE OF TRUTH #2: span registry + hierarchy
  payloads.py        # SOURCE OF TRUTH #3: typed span-data models + from_* adapters
  providers.py       # tracer/meter/logger provider + exporter factory + baggage span processor
  context.py         # traceparent extract/inject; parent resolution; baggage set/get
  emitter.py         # the engine: SpanData + SpanSpec -> emitted span (start/attrs/end)
  metrics.py         # gen_ai.client.* histograms
  mappers/
    __init__.py
    base.py          # AttributeMapper Protocol
    genai.py         # canonical gen_ai.* mapper (default)
    legacy.py        # dual-emit: also writes legacy keys (gated by config.legacy_compat)
    arize.py langfuse.py weave.py langtrace.py   # one mapper per derived integration
  logger.py          # OpenTelemetry(CustomLogger): thin lifecycle adapter
```

### 5.1 The mapper Protocol (replaces subclassing + `callback_name` dispatch)

```python
class AttributeMapper(Protocol):
    def map(self, data: SpanData) -> dict[str, AttrValue]: ...
```

- The engine composes a list of mappers: `[GenAIMapper(), *integration_mappers,
  LegacyMapper() if cfg.legacy_compat else None]`.
- Each integration ships **one** mapper instead of overriding base internals.
  `GenAIMapper` is always present → strict semconv is guaranteed regardless of
  integration.
- This deletes the `callback_name`-string `if/elif` in `set_attributes`
  (`opentelemetry.py:1968-1994`) and the per-subclass `set_attributes` overrides.

### 5.2 The engine (`emitter.py`)

- Single `emit(role: SpanRole, data: SpanData, parent: SpanContextRef)` entry.
- Looks up `SPAN_REGISTRY[role]`, builds the name, resolves the parent from the
  registry's `parent` field + `context.py`, starts the span with the correct
  `SpanKind`, runs the mapper chain (each writes via `safe_set_attribute`), sets
  status, ends with explicit timing.
- **Idempotency**: an in-instance `set[tuple[str, SpanRole]]` keyed by
  `StandardLoggingPayload["id"]` replaces the `kwargs`-metadata `_emit_once`
  hack (`opentelemetry.py:919`). Survives the sync+async streaming dual-fire
  because both callbacks carry the same payload `id`.
- Keeps the **"only close LiteLLM-owned proxy spans"** rule
  (`opentelemetry.py:985-991`): externally created spans (user/Langfuse/HTTP
  header) are never ended by the engine.

### 5.3 Config (`config.py`)

`OpenTelemetryConfig` becomes a Pydantic v2 model (repo standard) with a single
`from_env()` classmethod that is the **only** place env vars are read. Fields are
explicit and typed; no inline `get_secret_bool` in the hot path. New field:
`legacy_compat: bool` (default `True` during the deprecation window) controlling
dual-emit of legacy keys and legacy span names/child-spans.

### 5.4 Context propagation (`context.py`)

Folds `get_traceparent_from_header` (`opentelemetry.py:2583`), `_get_span_context`
(`2600`), `_resolve_guardrail_context` (`1568`) into typed functions:
`extract_parent(headers) -> Context`, `resolve_parent(role, request_ctx) ->
SpanContextRef`. Outgoing propagation (`litellm_pre_call_utils.py:2781
_add_otel_traceparent_to_data`, gated by
`litellm.forward_traceparent_to_llm_provider`) is preserved and moved behind a
typed `inject_traceparent(headers) -> headers`.

### 5.5 Cross-cutting request-scoped attributes via Baggage

Some consumers want request-scoped *identity* — `litellm.team.id`,
`litellm.team.alias`, `gen_ai.request.model`, and selected `metadata` — present on
**every** span (LLM_CALL, GUARDRAIL, SERVICE), not just the root. This is already a
behavioral contract today: `test_otel_team_attributes_matrix.py` asserts
`team_id`/`team_alias` land on every span. The rewrite formalizes *how*.

**Why this is legitimate (not just denormalization).** Most backends cannot cheaply
join a child span back to its root, so slicing a slow GUARDRAIL or SERVICE sub-span
by team or model requires the attribute *on that span*. Tail-based sampling,
retention, per-team cost attribution, and access control all operate per-span, and
the root may be sampled out while a child is kept.

**Mechanism — one span processor, not per-call-site copy-paste.** At request entry
(proxy auth / first SDK call) `context.py` writes the allowlisted identity values
into **W3C Baggage**. A `LiteLLMBaggageSpanProcessor` (registered in `providers.py`)
implements `on_start(span, parent_context)` and stamps those baggage entries onto
every span as it is created. This keeps the behavior in **one place**, makes the
key set a single source of truth (`semconv.py`), and propagates correctly across
async tasks and the sync+async streaming dual-fire (baggage rides the context).

```python
# semconv.py — the SINGLE allowlist of keys promoted onto every span
BAGGAGE_PROMOTED_KEYS: Final[tuple[str, ...]] = (
    LiteLLM.TEAM_ID, LiteLLM.TEAM_ALIAS, LiteLLM.KEY_HASH, GenAI.REQUEST_MODEL,
)
# metadata is promoted only for an explicit, config-driven allowlist of sub-keys
DEFAULT_BAGGAGE_METADATA_KEYS: Final[tuple[str, ...]] = (...)  # reuse the existing
                                          # ~15-key metrics allowlist, not the blob

# providers.py
class LiteLLMBaggageSpanProcessor(SpanProcessor):
    def on_start(self, span: Span, parent_context: Optional[Context]) -> None:
        for k, v in baggage.get_all(parent_context).items():
            if k in self._promoted_keys and v is not None:
                span.set_attribute(k, v)
```

**Two explicit pushbacks (the antipattern boundary):**
1. **HTTP attributes stay on the SERVER span only.** `http.request.method`,
   `http.route`, `http.response.status_code` semantically describe the HTTP entry
   point; stamping them on a `gen_ai` CLIENT span or a redis INTERNAL span is
   semconv-nonconformant and misleading (a redis call is not an HTTP operation). If
   a child needs a correlation key, that need is already served by the promoted
   identity keys above — **`http.*` is deliberately excluded from
   `BAGGAGE_PROMOTED_KEYS`.** A customer asking for "HTTP attributes on all spans"
   should be steered to the identity keys, not to copying `http.route` everywhere.
2. **Never promote the full `metadata` dict.** `metadata` is arbitrary,
   high-cardinality, and a PII vector; replicated across every span it multiplies
   storage and can blow up attribute indexes. Only an explicit, config-bounded
   allowlist of metadata sub-keys is promoted (`DEFAULT_BAGGAGE_METADATA_KEYS`);
   the full blob remains a `litellm.metadata.*` set on the root LLM_CALL span only.

**Config knobs (in `config.py`):** `baggage_promoted_keys: list[str]` (defaults to
`BAGGAGE_PROMOTED_KEYS`) and `baggage_metadata_keys: list[str]` let operators widen
or narrow the promoted set without code changes — but the *default* is the bounded,
semconv-safe set above.

---

## 6. Dual-emit / backward-compatibility mapping

`LegacyMapper` (in `mappers/legacy.py`) emits these **in addition** to the
canonical keys, only when `config.legacy_compat` is true. This is the exhaustive
old→new table (legacy keys sourced from `proxy/_types.py:3574` + inline literals):

| Legacy key (today) | Canonical key (new) | Notes |
|---|---|---|
| `gen_ai.system` | `gen_ai.provider.name` | value mapped via `GenAIProvider` |
| `gen_ai.usage.prompt_tokens` | `gen_ai.usage.input_tokens` | |
| `gen_ai.usage.completion_tokens` | `gen_ai.usage.output_tokens` | |
| `gen_ai.usage.total_tokens` | (dropped; derivable) | kept under legacy only |
| `gen_ai.prompt` | `gen_ai.input.messages` | content-capture gated |
| `gen_ai.completion` | `gen_ai.output.messages` | content-capture gated |
| `llm.is_streaming` | `gen_ai.request.*` n/a → keep as `litellm.request.streaming` | not in semconv |
| `llm.user` | (dropped) | PII; keep legacy only |
| `llm.top_k` | `gen_ai.request.top_k` | |
| `llm.frequency_penalty` | `gen_ai.request.frequency_penalty` | |
| `llm.presence_penalty` | `gen_ai.request.presence_penalty` | |
| `llm.chat.stop_sequences` | `gen_ai.request.stop_sequences` | |
| `llm.request.functions` | (tool attrs / events) | |
| `gen_ai.request.id` | `gen_ai.response.id` | today's key is misnamed |
| span `litellm_request` / `raw_gen_ai_request` | span `"{operation} {model}"` | name change |
| span `"Received Proxy Server Request"` | span `"{method} {route}"` | HTTP semconv |
| span `"Failed Proxy Server Request"` | status ERROR on PROXY_REQUEST | child span removed |

Content capture continues to honor the existing env var
`OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` and litellm's
`turn_off_message_logging` / presidio masking path
(`test_opentelemetry_unit_tests.py`).

---

## 7. Derived-integration migration

Each derived integration becomes a **mapper + optional provider override**, not a
subclass of the engine. Migration is mechanical and one-per-PR.

| Integration | Today | New form | Risk |
|---|---|---|---|
| Arize (`arize/arize.py:30`) | subclass; private TracerProvider; `set_attributes` override | `ArizeMapper` + `provider_factory=private` | private provider preserved (coexistence test `test_arize_otel_coexistence.py`) |
| Arize-Phoenix (`arize/arize_phoenix.py:40`) | subclass; auto-init | `PhoenixMapper`; auto-init via factory (`litellm_logging.py:4324`) keeps working | |
| Levo (`levo/`) | subclass | `LevoMapper` | low |
| Langfuse-OTEL (`langfuse/langfuse_otel.py:30`) | subclass; static `set_langfuse_otel_attributes`; `skip_set_global` | `LangfuseMapper` + `skip_set_global=True` in config | must keep NOT claiming `open_telemetry_logger` |
| Weave-OTEL (`weave/weave_otel.py:189`) | subclass; overrides `_start_primary_span`, `_handle_success` (user owns span lifecycle), `_maybe_log_raw_request` | `WeaveMapper` + `owns_root_span=False` engine flag | heaviest; needs `test_weave_otel.py` parity |
| AgentOps (`agentops/agentops.py:31`) | subclass; token fetch in `__init__` | `AgentOpsMapper` + config builder | low |
| Langtrace (`langtrace.py`) | **no subclass**; reached only via `callback_name` dispatch | `LangtraceMapper` registered for `callback_name="langtrace"` | this is why the base can't be deleted before this step |

The two distinct `"logfire"` paths (legacy `LogfireLogger` standalone vs. the
factory's `OpenTelemetry`-backed Logfire at `litellm_logging.py:4046`) are
documented and the OTel-backed one moves to the new engine; the legacy
`LogfireLogger` is left untouched (separate SDK, out of scope).

---

## 8. Public-surface preservation (must not break)

These are load-bearing and kept stable:
- Import path `from litellm.integrations.opentelemetry import OpenTelemetry,
  OpenTelemetryConfig` — preserved by turning `opentelemetry.py` into a shim that
  re-exports from `otel/__init__.py`.
- Module constants imported by tests: `LITELLM_REQUEST_SPAN_NAME`,
  `RAW_REQUEST_SPAN_NAME`, `LITELLM_PROXY_REQUEST_SPAN_NAME`
  (`test_opentelemetry.py:1715,1726,3438`) — re-exported (and still emitted when
  `legacy_compat=True`).
- Constructor signature `OpenTelemetry(config, callback_name, tracer_provider,
  logger_provider, meter_provider)` — kept (the three provider args are the test
  injection points, `opentelemetry.py:178-186`).
- `proxy_server.open_telemetry_logger` global + first-registered-wins guard
  (`opentelemetry.py:237-264`) — preserved verbatim.
- Proxy API used by auth/server: `create_litellm_proxy_request_started_span`,
  `set_proxy_request_route_attributes`, `set_response_status_code_attribute`,
  `set_preprocessing_duration_attribute` (`opentelemetry.py:3130-3208`,
  callers in `user_api_key_auth.py:693`, `proxy_server.py`,
  `management_helpers/utils.py:484`) — kept as thin methods delegating to the
  engine.
- `litellm.forward_traceparent_to_llm_provider` flag (`__init__.py:370`).
- Service hooks `async_service_success_hook` / `async_service_failure_hook`
  consumed by `_service_logger.py:147,258`.

---

## 9. Phased migration & removal plan

Each phase is independently shippable and reversible. **No old code is deleted
until Phase 6.** Exit criteria are explicit so there is no ambiguity about when a
phase is "done".

### Phase 0 — Scaffolding (no behavior change)
- Create `litellm/integrations/otel/` with `semconv.py`, `spans.py`, `payloads.py`
  fully populated, plus empty `emitter.py`/`mappers/`/`config.py`/`providers.py`/
  `context.py` stubs. No wiring into litellm yet.
- **Exit:** `semconv.py`, `spans.py`, `payloads.py` merged; unit tests assert the
  registry is internally consistent (every `parent` references a real role; every
  span name builds from its `SpanData`). mypy-clean.

### Phase 1 — Engine + providers + config behind a hard-off flag
- Implement `config.py` (Pydantic, single `from_env`), `providers.py` (exporter
  factory ported from `_get_span_processor`/`_get_log_exporter`/`_get_metric_reader`,
  `opentelemetry.py:2656-2911`) **plus the `LiteLLMBaggageSpanProcessor`**),
  `context.py` (incl. baggage set/get), `emitter.py`, `metrics.py`,
  `mappers/genai.py`, `mappers/legacy.py`.
- New flag `LITELLM_OTEL_V2` (default **off**). When off, nothing changes.
- **Exit:** with `LITELLM_OTEL_V2=on` in a test harness, the engine emits the
  `LLM_CALL` span with correct semconv attributes for a unit `StandardLoggingPayload`
  fixture. mypy-clean.

### Phase 2 — `logger.py` adapter at parity for the LLM-call + proxy spans
- Implement `OpenTelemetry(CustomLogger)` in `logger.py`: `async_log_success_event`/
  `async_log_failure_event`/`log_*` build `LLMCallSpanData.from_standard_logging_payload`
  and call the engine; proxy SERVER-span API methods delegate to the engine.
- Keep the public import path working via the shim (Phase 5 finalizes it).
- Run the **existing** suites with `LITELLM_OTEL_V2=on` +
  `legacy_compat=on`: `test_opentelemetry.py`,
  `test_opentelemetry_unit_tests.py`, `test_otel_logging.py`,
  `test_dynamic_otel_keys.py`, `otel_tests/test_otel.py`,
  `test_otel_thread_leak.py`, `test_otel_team_attributes_matrix.py`,
  `test_otel_guardrail_violation_spans.py`, the `open_telemetry/` admin/exception/
  passthrough/unified suites, `test_otel_load_test.py`.
- **Exit:** all the above pass with V2 on (legacy_compat on). New tests added:
  one per `SpanRole` asserting name/kind/parent against `spans.py`; a
  semconv-conformance test asserting only registry keys appear.

### Phase 3 — ServiceLogger on the new engine
- Port `async_service_success_hook`/`async_service_failure_hook`
  (`opentelemetry.py:539-660`) to `ServiceSpanData` + engine. `_service_logger.py`
  is unchanged (still calls the same hook names on the same instance).
- **Exit:** `test_service_logger_otel.py` passes with V2 on.

### Phase 4 — Derived integrations to mappers (one PR each)
Order (low-risk → high-risk): AgentOps → Levo → Langtrace → Arize →
Arize-Phoenix → Langfuse-OTEL → Weave-OTEL.
- Each PR adds `mappers/<x>.py`, switches the factory branch in
  `litellm_logging.py` (`3873-4200`) to construct the V2 `OpenTelemetry` with that
  mapper + config flags (`skip_set_global`, `owns_root_span`,
  `provider_factory`), and deletes that subclass.
- **Exit per PR:** the integration's test passes with V2 on
  (`test_weave_otel.py`, `test_langfuse_otel.py`,
  `test_arize_otel_coexistence.py`, etc.); the old subclass file deleted.

### Phase 5 — Flip default + shim the public path
- Default `LITELLM_OTEL_V2=on`. `opentelemetry.py` becomes a re-export shim:
  `from litellm.integrations.otel import OpenTelemetry, OpenTelemetryConfig,
  LITELLM_REQUEST_SPAN_NAME, RAW_REQUEST_SPAN_NAME, LITELLM_PROXY_REQUEST_SPAN_NAME`.
- **Exit:** full `make test-unit` green with V2 default; one full
  integration/load run green; a canary deploy shows spans in a real collector
  with both new + legacy keys present.

### Phase 6 — Remove legacy
- Delete the old implementation body from history of the shim, delete
  `opentelemetry_utils/gen_ai_semconv.py` (folded into `semconv.py` +
  `mappers/genai.py`) and `base_otel_llm_obs_attributes.py` (folded into
  `emitter.safe_set_attribute`). Remove the `SpanAttributes` enum usages from
  `proxy/_types.py:3574` (or leave the enum as deprecated re-export if any
  external code imports it — verify via grep first).
- Announce legacy-key deprecation; after the window, set `legacy_compat` default
  to `False`, then delete `mappers/legacy.py`.
- **Exit:** `git grep` finds no inline attribute literals outside `semconv.py`; no
  `**kwargs`/`Any` in the `otel/` boundary signatures; the 3,227-line file is gone.

### Removal checklist (files)
- `litellm/integrations/opentelemetry.py` → shrinks to a ~15-line shim, then the
  shim only.
- `litellm/integrations/opentelemetry_utils/gen_ai_semconv.py` → **deleted**.
- `litellm/integrations/opentelemetry_utils/base_otel_llm_obs_attributes.py` →
  **deleted** (after Arize/Phoenix mappers absorb its helpers).
- `litellm/integrations/arize/arize.py`, `arize_phoenix.py`, `levo/*`,
  `langfuse/langfuse_otel.py`, `weave/weave_otel.py`, `agentops/agentops.py` →
  subclass bodies replaced by `*Mapper` + factory config.
- `litellm/proxy/_types.py:3574 SpanAttributes` → deleted or deprecated re-export.

---

## 10. Testing strategy

- **Registry consistency tests** (new): parent integrity, name builders,
  no-orphan guarantee — pure, fast.
- **Semconv conformance test** (new): emit a span for a fixture payload, assert
  the set of attribute keys ⊆ `semconv.py` constants, and that required keys are
  present.
- **Golden parity tests** (new): for a fixed `StandardLoggingPayload`, snapshot
  the emitted span tree (names/kind/parent/attrs) in both `legacy_compat` on/off
  modes.
- **Baggage promotion test** (new + reuse `test_otel_team_attributes_matrix.py`):
  assert the promoted identity keys appear on **every** span role, that `http.*`
  appears **only** on the SERVER span, and that the full `metadata` blob is **not**
  promoted (only the allowlisted sub-keys).
- **Reuse all existing suites** (listed in Phase 2) as the behavioral contract;
  they must pass with V2 on before any deletion.
- Run `make lint` (Ruff + **MyPy** + Black) — the rewrite must be mypy-clean,
  which is the structural proof the "no `Any`/`kwargs`" goal was met.

---

## 11. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Dashboards break on key rename | `legacy_compat` dual-emit + deprecation window (decision §3). |
| Subclass signature ripple | Composition (mappers) removes the override surface entirely. |
| Langtrace silently dies (no subclass) | Explicit `LangtraceMapper` step in Phase 4 with its own test gate. |
| `open_telemetry_logger` first-wins regressions | Preserved verbatim; covered by coexistence + service-logger tests. |
| Streaming sync+async double spans | Engine idempotency set keyed by payload `id` (replaces `_emit_once`). |
| Removing `raw_gen_ai_request`/`Failed Proxy` child spans | Reproducible under `legacy_compat` until those tests migrate. |
| Hidden env-var consumers | All env reads centralized in `config.from_env`; grep audit in Phase 6. |
| Attribute bloat / high cardinality from "put X on all spans" | Baggage promotion is a bounded, explicit allowlist (§5.5); `http.*` excluded; full `metadata` blob never promoted. |

---

## 12. Appendix — current → new file map

| Current | New home |
|---|---|
| `opentelemetry.py` config dataclass + `from_env` | `otel/config.py` |
| `_get_span_processor`/`_get_log_exporter`/`_get_metric_reader`/`_get_or_create_provider` | `otel/providers.py` |
| `_get_span_context`/`_resolve_guardrail_context`/`get_traceparent_from_header` | `otel/context.py` |
| per-call-site team/metadata stamping (`test_otel_team_attributes_matrix.py` contract) | `otel/context.py` baggage + `otel/providers.py` `LiteLLMBaggageSpanProcessor` |
| `_get_span_name`/span constants/10 creation sites | `otel/spans.py` + `otel/emitter.py` |
| `set_attributes`/`set_tools_attributes`/`set_raw_request_attributes`/inline literals | `otel/semconv.py` + `otel/mappers/genai.py` |
| `opentelemetry_utils/gen_ai_semconv.py` | `otel/semconv.py` + `otel/mappers/genai.py` |
| `proxy/_types.py:3574 SpanAttributes` + legacy keys | `otel/mappers/legacy.py` |
| `callback_name` dispatch + 6 subclasses | `otel/mappers/{arize,langfuse,weave,langtrace,...}.py` |
| `_record_metrics` | `otel/metrics.py` |
| `_handle_success`/`_handle_failure`/`_emit_once`/lifecycle hooks | `otel/logger.py` + `otel/emitter.py` |
