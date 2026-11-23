---
name: rust-tracing
description: Add or change Rust diagnostic tracing in litellm-rust, including route spans, subscriber layers, and Python logger delivery
metadata:
  internal: true
---

# Rust tracing

Use upstream `tracing` throughout Rust, including `#[tracing::instrument]`, events, and span propagation. Centralize collection and delivery infrastructure in `crates/tracing`. Log with the upstream macros, `tracing::warn!` and its siblings, which reach our configured subscriber. `litellm-tracing` exports none of its own. Do not introduce Rust `log` or `pyo3-log` for this path

`litellm-tracing` owns shared subscriber layers, span field collection, and diagnostic processing. Keep adapters composable as `tracing_subscriber::Layer`s, with `Logger` providing host setup. Runtime-specific delivery belongs in the host bridge. The Python bridge delivers directly to the existing Python SDK logger, preserving its handlers, filtering, redaction, and request correlation. Keep Python dependencies out of `crates/tracing`

The bridge alone decides which Python logger a Rust target logs on, in `python-bridge/src/logger/route.rs`. Python's host module is told the logger's name and never matches on a Rust target, so the rule is written once

Secret scrubbing of free text lives in `crates/redaction` (`litellm-redaction`). `Processor` uses it for log records, and request logging callbacks such as spend logs depend on it directly, never through `litellm-tracing`

A host whose output does no redaction of its own wraps its sink in `litellm_tracing::Scrubbed`, which scrubs the message and every text field before the sink sees the record. The gateway does. The Python bridge does not, because Python's logging filter scrubs each record with the user's settings

Hosts configure subscribers. The Python bridge installs one process-wide subscriber when the extension loads, and every Rust event in the process goes through it. A call into the bridge opens a span that carries its request's session and trace ids, so an event anywhere inside the call is delivered with them, and an event outside every call, such as one from a detached task, has none. An event on a runtime task goes through a queue bounded by record count and by text bytes, and is delivered from the blocking pool, so a Python handler never runs on an async worker. An event on any other thread, such as the Python thread of a sync call, is delivered there and then. The level each Python logger takes is read when a call starts and kept for the whole process, so a level that is lowered takes effect at the next call. Keep propagating span context across spawned work and returned streams, since the ids travel in the span. Tests still scope a `Logger` to their own sink, so keep propagating the dispatch too

In core, instrument execution shared by native calls and hosted machines. Use consistent route, model, provider, streaming, and outcome fields. Put status recording at shared provider boundaries instead of scattering basic logging through handlers. Keep upstream HTTP status separate from route success

Put what a reader needs to act on, such as the reason a call failed, in the message or in a field. Both hosts print every field. The gateway prints them as JSON, and Python prints them after the message as `[name=value ...]` in plain logs and as `rust_fields` in JSON logs

Use `skip_all` and explicitly selected fields. Basic tracing excludes bodies, credentials, headers, and raw error strings. Avoid automatic `ret` or `err` capture of sensitive values. Keep payload diagnostics separate and subject to existing redaction

A returned stream retains its route span until exhaustion, error, or drop, with exactly one terminal outcome. Builder construction does not start a trace. Never hold a span entry guard across an await. Diagnostic tracing remains separate from lifecycle callbacks and `CustomLogger` dispatch

The subscriber inherits span fields into events and emits span-close summaries with elapsed time. Test observable records, concurrent isolation, dynamic filtering, sensitive-field exclusion, and stream cancellation when changing this behavior

Consult the [tracing API](https://docs.rs/tracing/latest/tracing/) and [subscriber layers](https://docs.rs/tracing-subscriber/latest/tracing_subscriber/layer/index.html) for implementation details
