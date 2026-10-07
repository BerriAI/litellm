---
name: rust-tracing
description: Add or change Rust diagnostic tracing in litellm-rust, including route spans, subscriber layers, and Python logger delivery
---

# Rust tracing

Use upstream `tracing` throughout Rust, including `#[tracing::instrument]`, events, and span propagation. Centralize collection and delivery infrastructure in `crates/tracing`. Direct upstream imports still reach our configured subscriber; re-exporting macros does not control delivery. Do not introduce Rust `log` or `pyo3-log` for this path

`litellm-tracing` owns shared subscriber layers, span field collection, and diagnostic processing. Keep adapters composable as `tracing_subscriber::Layer`s, with `Logger` providing host setup. Runtime-specific delivery belongs in the host bridge. The Python bridge delivers directly to the existing Python SDK logger, preserving its handlers, filtering, redaction, and request correlation. Keep Python dependencies out of `crates/tracing`

Hosts configure subscribers. Keep Python execution scoped to its captured dispatch rather than installing a process-wide subscriber. Propagate both span context and dispatch across spawned work and returned streams

Built-in analytics defaults on without a configured license and off when a license is declared, regardless of verification. `DO_NOT_TRACK=1` wins over explicit opt-in, and invalid controls fail closed. Freeze the decision after gateway configuration and secret resolution; suppress SDK initialization during gateway bootstrap. Missing build-time analytics project configuration must create no client or network work. Preserve customer diagnostics and Python handlers independently. Queue only the versioned event allowlist, never inherited diagnostic fields, payload shapes, model strings, or messages

In core, instrument execution shared by native calls and hosted machines. Use consistent route, model, provider, streaming, and outcome fields. Put status recording at shared provider boundaries instead of scattering basic logging through handlers. Keep upstream HTTP status separate from route success

Use `skip_all` and explicitly selected fields. Basic tracing excludes bodies, credentials, headers, and raw error strings. Avoid automatic `ret` or `err` capture of sensitive values. Keep payload diagnostics separate and subject to existing redaction

A returned stream retains its route span until exhaustion, error, or drop, with exactly one terminal outcome. Builder construction does not start a trace. Never hold a span entry guard across an await. Diagnostic tracing remains separate from lifecycle callbacks and `CustomLogger` dispatch

Use `litellm_tracing::sink_layer` to compose a sink with other subscriber layers. It inherits span fields into events and emits span-close summaries with elapsed time. Test observable records, concurrent isolation, dynamic filtering, sensitive-field exclusion, and stream cancellation when changing this behavior

Consult the [tracing API](https://docs.rs/tracing/latest/tracing/) and [subscriber layers](https://docs.rs/tracing-subscriber/latest/tracing_subscriber/layer/index.html) for implementation details
