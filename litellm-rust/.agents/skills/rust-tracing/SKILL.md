---
name: rust-tracing
description: Add or change Rust diagnostic tracing in litellm-rust, including route spans, subscriber layers, and Python logger delivery
---

# Rust tracing

Use upstream `tracing` throughout Rust, including `#[tracing::instrument]`, events, and span propagation. Centralize collection and delivery infrastructure in `crates/tracing`. Direct upstream imports still reach our configured subscriber; re-exporting macros does not control delivery. Do not introduce Rust `log` or `pyo3-log` for this path

`litellm-tracing` owns shared subscriber layers, span field collection, and diagnostic processing. Keep adapters composable as `tracing_subscriber::Layer`s, with `Logger` providing host setup. Runtime-specific delivery belongs in the host bridge. The Python bridge delivers directly to the existing Python SDK logger, preserving its handlers, filtering, redaction, and request correlation. Keep Python dependencies out of `crates/tracing`

Hosts configure subscribers. Keep Python execution scoped to its captured dispatch rather than installing a process-wide subscriber. Propagate both span context and dispatch across spawned work and returned streams

In core, instrument execution shared by native calls and hosted machines. Use consistent route, model, provider, streaming, and outcome fields. Put status recording at shared provider boundaries instead of scattering basic logging through handlers. Keep upstream HTTP status separate from route success

Use `skip_all` and explicitly selected fields. Basic tracing excludes bodies, credentials, headers, and raw error strings. Avoid automatic `ret` or `err` capture of sensitive values. Keep payload diagnostics separate and subject to existing redaction

A returned stream retains its route span until exhaustion, error, or drop, with exactly one terminal outcome. Builder construction does not start a trace. Never hold a span entry guard across an await. Diagnostic tracing remains separate from lifecycle callbacks and `CustomLogger` dispatch

The existing sink forwards events only. Span collection and composable adapter exposure are intended changes, not capabilities to assume already exist. When implementing them, test observable records, concurrent isolation, filtering, sensitive-field exclusion, and stream cancellation

Consult the [tracing API](https://docs.rs/tracing/latest/tracing/) and [subscriber layers](https://docs.rs/tracing-subscriber/latest/tracing_subscriber/layer/index.html) for implementation details
