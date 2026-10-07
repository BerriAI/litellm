# Native diagnostic tracing

`litellm-tracing` connects standard `tracing` spans and events to a host-provided `Sink`. It has no Python dependency. Hosts choose scoped dispatch or explicitly install a global subscriber

Use upstream `tracing` macros and `#[tracing::instrument(skip_all, fields(...))]` in native code. A host creates a `Logger` with its sink, uses `scope` for synchronous operations, and wraps futures with `instrument`. Propagate both spans and dispatch into spawned work and returned streams. `Logger::current()` captures the current dispatch. Existing event macro re-exports remain available

Bindings implement `litellm_tracing::Sink` to connect events to their host runtime:

```rust
pub trait Sink: Send + Sync + 'static {
    fn enabled(&self, metadata: &Metadata<'_>) -> bool;
    fn emit(&self, record: &Record);
}
```

Pass the implementation to `litellm_tracing::Logger::new(sink)`, then call `logger.scope(|| litellm_tracing::info!(attempt = 1, "request started"))`. The sink owns host access, level mapping, correlation capture, and delivery failures. `enabled` runs before event fields are evaluated or formatted. `emit` borrows a record; an adapter that queues delivery must copy the data it needs into an owned value

Records retain event metadata, the message, and typed event fields. Sink filtering runs for each event so runtime level changes take effect. Logging from inside a sink is suppressed to prevent recursion

`sink_layer(sink)` exposes the same adapter as a composable `tracing_subscriber::Layer`, with an independent dynamic filter. It records span field updates and inherits fields from outer to inner spans, with event fields taking precedence. Closing a span emits `span closed` at the span's level with `span_name` and monotonic `duration_ms`. Delivery rechecks the sink filter. Core records route outcomes and retains route spans until a stream ends, fails, or is dropped

The Python bridge scopes native execution to a sink that uses LiteLLM's existing Python logger. It preserves request correlation, redacts before delivering to handlers, maps Rust trace events to Python debug, and reports handler failures through `sys.unraisablehook`. It accepts LiteLLM targets only, keeping dependency wire diagnostics out of the application logger

Python consumers continue using `litellm._logging` and its existing loggers, filters, formatters, and context setters. Catalog dispatch selects the processing backend for both Python and native diagnostics. The pure `Processor` takes explicit settings and never emits events

A future Node bridge can implement the same sink with runtime-specific delivery and expose the same processor through N-API. Node callback scheduling, queue limits, and shutdown belong in that bridge; this crate has no interpreter handles

This is diagnostic logging. Request lifecycle hooks and `CustomLogger` dispatch remain separate

## Configured diagnostic export

`DiagnosticsConfig` describes export intent, service identity, a default diagnostic policy, and named destination wiring. `Diagnostics` constructs the official SDK adapters and owns reconfiguration, flush, and shutdown. These types have no Python dependency and are usable directly by Rust SDK hosts

The Python SDK exposes the same contract through `litellm.diagnostics`. Configuration does not enable Rust inference routes or change Python logging levels and handlers

```python
from litellm import diagnostics

configuration = diagnostics.DiagnosticsConfig.from_sources()
diagnostics.configure(configuration)
```

Both gateways accept this block in `config.yaml`:

```yaml
general_settings:
  diagnostics:
    enabled: true
    service_name: litellm-gateway
    policy:
      minimum_level: INFO
      sample_rate: 0.1
    destinations:
      - name: events
        transport: posthog
        api_key: os.environ/POSTHOG_PROJECT_KEY
      - name: logs
        transport: otlp
        endpoint: os.environ/OTLP_LOGS_ENDPOINT
        headers:
          Authorization: os.environ/OTLP_AUTHORIZATION
```

`LITELLM_DIAGNOSTICS` accepts a JSON diagnostics object and replaces the complete YAML block. Endpoint, project key, and header values support `os.environ/NAME` references. The Rust gateway resolves its YAML environment variables before process environment values, matching the Python gateway's loaded environment

Policy has `minimum_level` (`TRACE`, `DEBUG`, `INFO`, `WARN`, `ERROR`), `target_prefixes`, and finite `sample_rate` between zero and one. A destination may provide a complete policy override. Errors bypass sampling while retaining severity and target admission. Correlated routine logs use a stable trace-ID decision, and uncorrelated records use random per-record sampling. Export sampling leaves existing Python output unchanged

Calling `configure` replaces the complete destination set. Disabled configuration and `shutdown` remove all destinations, making both Python forwarding and native export inactive. The Python SDK returns `False` if the native binding is missing; invalid configuration or an unavailable transport raises an error. Gateways initialize in workers after forking and drain their exporters when their lifespan ends. SDK hosts call `diagnostics.shutdown()` explicitly

`LoggerRule(Rollout.RUST_OPT_IN)` remains an internal migration gate for the existing diagnostic processing backend. Export intent comes from diagnostics configuration, independently of that rule. Our owned PyO3 adapter lives in `python-bridge/src/logger/python.rs`; it owns integer severity mapping, context capture, Python callbacks, and process ownership, with JSON argument decoding delegated to `litellm-host-python`. The additive Python handler owns snapshot conversion and native-origin checks. There is no `pyo3-pylogger` dependency or `log` crate hop. Generic `source.target` and `source.timestamp` fields let exporters consume normalized records without Python-specific interpretation

The `posthog` Cargo feature compiles SDK support by default but starts no worker or network traffic by itself. PostHog captures personless `litellm diagnostic` events; OTLP sends HTTP/protobuf logs to a full logs endpoint. Destination transport names and credentials are integration wiring, while SDK users configure one diagnostics policy API

The subscriber remains the upstream `tracing_subscriber::Registry` with composable layers and filters. `Sink` is the existing normalized-record adapter used by those layers, not a separate subscriber or transport framework. Rust hosts may compose `sink_layer(diagnostics.clone())` with standard `fmt` or other layers. SDKs own batching, queues, retries, and HTTP transport

Both adapters redact before enqueueing. Delivery is best effort; flush completion does not prove remote receipt. Queue limits bound records rather than bytes, and per-destination drop counters and distributed span export are not implemented

Read [the shared pipeline notes](../../.agents/skills/rust-tracing/references/unified-python-logging.md) for normalization, compatibility, and process ownership

Payload shape extraction

`PayloadShape::extract` walks a borrowed JSON value and returns sorted, unique JSONPath expressions containing object keys and array wildcards, with no scalar values or array positions. `ShapeLimits` bounds visited nodes, depth, path count, and path bytes. Exceeding a limit discards partial paths and marks the shape truncated

Children of `metadata`, `properties`, `$defs`, `definitions`, and `headers` use wildcards for dynamic key names. Strings containing JSON remain opaque. The shared synthetic fixture covers Python/Rust parity; capture policy and provider boundary wiring belong to the payload-shape feature above this PR

## Opt-in payload shape capture

Add `"payload_shapes": true` to the same diagnostics configuration used by the SDK, gateway YAML, or `LITELLM_DIAGNOSTICS`. Capture defaults to false and is independent of the inference rollout. A destination must admit INFO events with target `litellm_payload_shapes`. For a destination dedicated to shapes, use `"policy": {"target_prefixes": ["litellm_payload_shapes"]}`

The Python SDK records the supplied arguments before setup, the body seen by provider pre-call logging, the final body in the shared chat HTTP transport or the body passed to the OpenAI chat SDK, the provider response before normalization, and the returned normalized object. A cached result has no provider send or receive event. Other provider SDK transports that bypass these handlers do not produce a sent event. OpenAI SDK internal retries and serialization are below the observed SDK boundary. Missing stages mean that boundary was not observed, rather than an empty payload being sent

The Rust extractor is shared through a source visitor. The Python adapter borrows dict/list/tuple and Pydantic field values while attached to the interpreter, without serializing scalar payload values. Extraction is bounded while holding the GIL; event processing detaches before export. Only owned paths leave that synchronous borrow. The common event contains `event.name = llm.payload.shape`, `schema_version = 1`, `trace_id`, `payload.stage`, `payload.field_paths`, `payload.shape_truncated`, and optional `payload.outcome`. The subscriber drops inherited span and context fields for shape events before either destination processes them. Existing Python handlers do not receive these events. PostHog and OTLP share the same field projection, redaction and destination sampling policy

Streaming captures union paths during normal iteration and emit one summary per observed response stage when exhausted, failed or closed. Final synthetic finish and usage chunks participate in the normalized summary. An incomplete JSON frame or a traversal limit discards partial paths and marks the summary truncated. This does not change chunks returned to the caller. Shape capture does not inspect strings containing tool arguments as another JSON document

Limits are 4,096 visited nodes, depth 16, 256 paths, 16 KiB of path text and 128 bytes per key. Known dynamic maps, including metadata, headers and schema property names, use wildcards. Other key names are still supplied by callers and can contain private data, so this remains an opt-in diagnostic feature. Do not use field paths as metric labels
