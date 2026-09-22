# Native diagnostic logging

`litellm-logger` connects standard `tracing` events to a host-provided `Sink`. It has no Python dependency and does not install a global subscriber

Use the exported `debug!`, `info!`, `warn!`, and `error!` macros in native code. A host creates a `Logger` with its sink, uses `scope` for synchronous operations, and wraps futures with `instrument`. Instrument spawned futures explicitly because thread-local subscribers do not automatically follow spawned work

Records retain event metadata, the message, and typed event fields. Sink filtering runs for each event so runtime level changes take effect. Logging from inside a sink is suppressed to prevent recursion

The Python bridge scopes native execution to a sink that uses LiteLLM's existing Python logger. It preserves request correlation, redacts before delivering to handlers, maps Rust trace events to Python debug, and reports handler failures through `sys.unraisablehook`. It accepts LiteLLM targets only, keeping dependency wire diagnostics out of the application logger

This is diagnostic logging. Request lifecycle hooks and `CustomLogger` dispatch remain separate
