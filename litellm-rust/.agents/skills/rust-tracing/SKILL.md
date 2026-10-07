---
name: rust-tracing
description: Add or change diagnostic tracing in litellm-rust, including route spans, Python logging compatibility, shared subscriber policy, and exporters
---

# Rust tracing

Use upstream `tracing` throughout Rust, including `#[tracing::instrument]`, events, and span propagation. Centralize collection, filtering, sampling, and delivery infrastructure in `crates/tracing`. Direct upstream imports still reach our configured subscriber; re-exporting macros does not control delivery. Keep Rust instrumentation on `tracing` rather than introducing a parallel `log` pipeline

`litellm-tracing` owns shared subscriber layers, span field collection, and diagnostic processing. Keep adapters composable as `tracing_subscriber::Layer`s, with the host owning initialization and dispatch. Keep Python dependencies and Python object conversion in the explicit `python-bridge/src/logger/python.rs` adapter. Core configuration, lifecycle, and exporter source fields must remain language-neutral

Existing Python logging calls and Rust tracing events share one scoped Rust dispatch per host process when the host configures export. Read [the Python logging pipeline](references/unified-python-logging.md) before changing forwarding, subscriber ownership, or exporters. `DiagnosticsConfig` controls export independently of inference rollout. `LoggerRule(Rollout.RUST_OPT_IN)` remains an internal migration gate only for the legacy diagnostic processing backend. The compatibility sink still delivers native records to existing Python output

Preserve Python logging call sites, logger names, levels, filters, handlers, propagation, formatters, exception information, and redaction. Forward an owned copy through an explicit `logging.Handler` integration. Do not replace `logging.basicConfig`, globally change levels, or take over an embedding application's root logger. Export disabled or native initialization unavailable must retain existing Python behavior. Export configuration must not enable inference routes or alter the logger rollout decision. Guard records delivered back to Python against forwarding loops and duplicate delivery

Use our owned PyO3 adapter in `python-bridge/src/logger/python.rs`, paired with the additive Python forwarding handler. Emit directly into `tracing`; do not introduce `pyo3-pylogger`, `pyo3-log`, or a `log` crate hop. The reference records compatibility gaps in the audited `pyo3-pylogger` 0.5.2 release. Detach from Python during Rust-only exporter configuration and draining, and attach only inside the Python compatibility callbacks

Hosts configure subscribers once for their runtime. A library must not unconditionally install a process-wide subscriber. A shared dispatch may remain scoped in an embedding host; do not create a new registry for each request or log record as part of this pipeline. Capture request correlation before crossing threads or queues, and propagate both span context and dispatch across spawned work and returned streams. Python context variables and Rust's current span are not automatically shared

Apply metadata filtering early and destination filtering with `Layer::with_filter`. Use event-aware filtering or normalized records for Python logger names and dynamic fields that are not tracing metadata. Keep rollout gating separate from verbosity filtering and sampling. Make trace sampling decisions consistently for the whole trace, with collector tail sampling when retention depends on final errors or latency. Remote sampling must not silently suppress existing Python handler output

Prefer existing OpenTelemetry bridges and OTLP exporters for logs and traces. `tracing-opentelemetry` handles traces; `opentelemetry-appender-tracing` bridges log events. Keep PostHog product analytics explicit and separate from diagnostic logs and `CustomLogger` lifecycle callbacks. The official `posthog-rs` diagnostic sink is compiled by the default `posthog` Cargo feature and starts only through explicit host configuration. It captures personless `litellm diagnostic` events, not PostHog Logs or product conversions. Network delivery belongs behind bounded queues with batching and host-owned shutdown. SDK queue warnings and handler ingress failures are distinct from confirmed delivery; do not imply per-destination drop counters exist

In core, instrument execution shared by native calls and hosted machines. Use consistent route, model, provider, streaming, and outcome fields. Put status recording at shared provider boundaries instead of scattering basic logging through handlers. Keep upstream HTTP status separate from route success

Use `skip_all` and explicitly selected fields. Basic tracing excludes bodies, credentials, headers, and raw error strings. Avoid automatic `ret` or `err` capture of sensitive values. Keep payload diagnostics separate and subject to existing redaction

A returned stream retains its route span until exhaustion, error, or drop, with exactly one terminal outcome. Builder construction does not start a trace. Never hold a span entry guard across an await. Diagnostic tracing remains separate from lifecycle callbacks and `CustomLogger` dispatch

Use `litellm_tracing::sink_layer` to compose a sink with other subscriber layers. It inherits span fields into events and emits span-close summaries with elapsed time. Test observable records, concurrent isolation, dynamic filtering, sensitive-field exclusion, and stream cancellation when changing this behavior. For Python forwarding changes, also verify existing handler output, traceback and extra fields, export independence from rollout, repeated initialization, shutdown, and loop prevention

Consult the [tracing API](https://docs.rs/tracing/latest/tracing/) and [subscriber layers](https://docs.rs/tracing-subscriber/latest/tracing_subscriber/layer/index.html) for implementation details
