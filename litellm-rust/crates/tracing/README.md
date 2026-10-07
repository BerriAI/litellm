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

`DiagnosticsConfig.from_sources` accepts a settings mapping and an optional environment mapping. `LITELLM_DIAGNOSTICS` accepts a JSON diagnostics object and replaces the complete settings object. Endpoint, project key, and header values support `os.environ/NAME` references, resolved at the native boundary before exporter startup

Policy has `minimum_level` (`TRACE`, `DEBUG`, `INFO`, `WARN`, `ERROR`), `target_prefixes`, and finite `sample_rate` between zero and one. A destination may provide a complete policy override. Errors bypass sampling while retaining severity and target admission. Correlated routine logs use a stable trace-ID decision, and uncorrelated records use random per-record sampling. Export sampling leaves existing Python output unchanged

Calling `configure` replaces the complete destination set. Disabled configuration and `shutdown` remove all destinations, making both Python forwarding and native export inactive. The Python SDK returns `False` if the native binding is missing; invalid configuration or an unavailable transport raises an error. Configure exporters in workers after forking. SDK hosts call `diagnostics.shutdown()` explicitly

`LoggerRule(Rollout.RUST_OPT_IN)` remains an internal migration gate for the existing diagnostic processing backend. Export intent comes from diagnostics configuration, independently of that rule. Our owned PyO3 adapter lives in `python-bridge/src/logger/python.rs`; it owns integer severity mapping, context capture, Python callbacks, and process ownership, with JSON argument decoding delegated to `litellm-host-python`. The additive Python handler owns snapshot conversion and native-origin checks. There is no `pyo3-pylogger` dependency or `log` crate hop. Generic `source.target` and `source.timestamp` fields let exporters consume normalized records without Python-specific interpretation

The `posthog` Cargo feature compiles SDK support by default but starts no worker or network traffic by itself. PostHog captures personless `litellm diagnostic` events; OTLP sends HTTP/protobuf logs to a full logs endpoint. Destination transport names and credentials are integration wiring, while SDK users configure one diagnostics policy API

The subscriber remains the upstream `tracing_subscriber::Registry` with composable layers and filters. `Sink` is the existing normalized-record adapter used by those layers, not a separate subscriber or transport framework. Rust hosts may compose `sink_layer(diagnostics.clone())` with standard `fmt` or other layers. SDKs own batching, queues, retries, and HTTP transport

Both adapters redact before enqueueing. Delivery is best effort; flush completion does not prove remote receipt. Queue limits bound records rather than bytes, and per-destination drop counters and distributed span export are not implemented

Read [the shared pipeline notes](../../.agents/skills/rust-tracing/references/unified-python-logging.md) for normalization, compatibility, and process ownership

Payload shape extraction

`PayloadShape::extract` walks a borrowed JSON value and returns sorted, unique JSONPath expressions containing object keys and array wildcards, with no scalar values or array positions. `ShapeLimits` bounds visited nodes, depth, path count, and path bytes. Exceeding a limit discards partial paths and marks the shape truncated

Children of `metadata`, `properties`, `$defs`, `definitions`, and `headers` use wildcards for dynamic key names. Strings containing JSON remain opaque. The shared synthetic fixture covers Python/Rust parity; capture policy and provider boundary wiring belong to the payload-shape feature above this PR
