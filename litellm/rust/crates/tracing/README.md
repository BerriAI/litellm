# Native diagnostic tracing

`litellm-tracing` connects standard `tracing` spans and events to a host-provided `Sink`. It has no Python dependency. Hosts choose scoped dispatch or explicitly install a global subscriber

Use upstream `tracing` macros and `#[tracing::instrument(skip_all, fields(...))]` in native code. A host creates a `Logger` with its sink, uses `scope` for synchronous operations, and wraps futures with `instrument`. Propagate both spans and dispatch into spawned work and returned streams. `Logger::current()` captures the current dispatch. `logger.within(|| span)` gives a logger whose scoped work runs inside that span, which is how a host attaches fields such as request ids to everything a call logs. This crate exports no event macros, so log with `tracing::warn!` and its siblings

Bindings implement `litellm_tracing::Sink` to connect events to their host runtime:

```rust
pub trait Sink: Send + Sync + 'static {
    fn enabled(&self, metadata: &Metadata<'_>) -> bool;
    fn emit(&self, record: &Record);
}
```

Pass the implementation to `litellm_tracing::Logger::new(sink)`, then call `logger.scope(|| tracing::info!(attempt = 1, "request started"))`. The sink owns host access, level mapping, correlation capture, and delivery failures. `enabled` runs before event fields are evaluated or formatted. `emit` borrows a record; an adapter that queues delivery must copy the data it needs into an owned value

Records retain event metadata, the message, and typed event fields. Sink filtering runs for each event so runtime level changes take effect. Logging from inside a sink is suppressed to prevent recursion

The subscriber records span field updates and inherits fields from outer to inner spans, with event fields taking precedence. Closing a span emits `span closed` at the span's level with `span_name` and monotonic `duration_ms`. Delivery rechecks the sink filter. Core records route outcomes and retains route spans until a stream ends, fails, or is dropped

The Python bridge installs one sink for the whole process when the extension loads, and it delivers to LiteLLM's existing Python logger. Each call into the bridge runs inside a span that carries its request's session and trace ids, so an event inside the call is delivered with them and an event outside every call has none. The sink redacts before delivering to handlers, maps Rust trace events to Python debug, and reports handler failures through `sys.unraisablehook`. It accepts LiteLLM targets only, keeping dependency wire diagnostics out of the application logger. Records from runtime tasks go through a queue to one task that delivers them from the blocking pool. The queue is bounded by a record count and by the bytes of text it holds, and a record past either bound is dropped and counted. Records from any other thread are delivered on that thread

Python consumers continue using `litellm._logging` and its existing loggers, filters, formatters, and context setters. Catalog dispatch selects the processing backend for both Python and native diagnostics. The pure `Processor` takes explicit settings and never emits events. It scrubs secrets with `SecretRedactor` from `litellm-redaction`, which this crate does not re-export

`Scrubbed::new(sink, minimum_custom_key_length)` wraps a sink so that every record is scrubbed of secrets first, the message as text and each text field under its name. A field that holds a JSON object or array, such as a logged request body, is scrubbed value by value under each value's own key and written back as JSON. The JSON on each `data:` line of a logged server-sent event is scrubbed the same way, a number under a secret key name is hidden like a string, and a secret used as a key is scrubbed too. A host that prints records itself, as the gateway does to stderr, wraps its sink in it. `scrubbed.key_length()` gives a handle that sets the key length after the sink is installed, for a host that reads the length from a config it loads later

A future Node bridge can implement the same sink with runtime-specific delivery and expose the same processor through N-API. Node callback scheduling, queue limits, and shutdown belong in that bridge; this crate has no interpreter handles or output queue

This is diagnostic logging. Request lifecycle hooks and `CustomLogger` dispatch remain separate
