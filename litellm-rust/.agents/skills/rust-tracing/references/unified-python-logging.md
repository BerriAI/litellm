# Shared Python and Rust diagnostic pipeline

## Implemented behavior

The Python bridge lazily owns one scoped `Logger` and shared destination registry per process. Existing native route capture clones that dispatch and captures request correlation through `Logger.with_fields`. Context is restored on each future poll and saved in span data so closure retains the creation context

`litellm.diagnostics.configure` projects the shared `DiagnosticsConfig` into the language-neutral Rust `Diagnostics` runtime, then attaches an additive standard `logging.Handler`. PostHog and OTLP are internal transport adapters selected through destination wiring. The defaults cover `LiteLLM`, `LiteLLM Proxy`, and `LiteLLM Router`. A host can explicitly select other logger objects. Repeated installation avoids duplicate handlers and overlapping selected parent/child coverage while respecting `propagate=False`

`LoggerRule(Rollout.RUST_OPT_IN)` selects the legacy diagnostic processing backend only. Configured export works while that decision remains Python and does not enable Rust inference routes. Imports do not start workers or install a global subscriber. A missing binding makes configuration return `False`; malformed configuration or an unavailable transport raises. Existing Python output continues

```text
Existing Python logging calls
  -> existing logger admission and filters
  -> explicit forwarding handler and export-active guard
  -> copied record, captured context, owned JSON
  -> shared scoped Rust dispatch

Native Rust tracing events and spans
  -> shared scoped Rust dispatch

Shared subscriber
  -> independent destination level and target filters
  -> log sampling and mandatory secret redaction
  -> official SDK batch queues and transports

Native compatibility sink
  -> existing Python output with an origin guard
```

## Surface configuration and Python boundary

The Python SDK accepts `DiagnosticsConfig` or a validated mapping through `litellm.diagnostics.configure`. The SDK configuration loader accepts a settings mapping and a complete overriding JSON object in `LITELLM_DIAGNOSTICS`. `os.environ/NAME` references are resolved for endpoints, headers, and project keys. There is no implicit exporter startup on package import

Rust SDK hosts use `DiagnosticsConfig` and `Diagnostics` directly, or compose `sink_layer(diagnostics.clone())` into an upstream subscriber. The generic tracing crate contains neither PyO3 nor Python callbacks. The explicit `python-bridge/src/logger/python.rs` adapter owns Python severity mapping, context capture, callbacks, process admission, and exception mapping

The common runtime uses the existing normalized-record `Sink` interface behind standard `tracing_subscriber` layers and filters. Official SDKs own transport and queues. Preserve this boundary rather than creating a second subscriber framework or a custom vendor HTTP client

## Python compatibility

Preserve logging call sites, logger identities, levels, filters, handlers, propagation, and formatters. No replacement of `logging.basicConfig` or automatic takeover of an embedding application's root logger. The forwarder reads an owned snapshot without changing original interpolation arguments, traceback information, or nested extras

Normalization supplies language-neutral `source.target`, `source.timestamp`, and `source.language` fields. The Python adapter additionally retains `logger.name`, `python.levelno`, `python.levelname`, `python.created`, code location, rendered exception and stack text, structured `extra`, and the caller's session and trace IDs. Conversion uses existing bounded-depth safe serialization. Queued delivery retains no live Python objects or traceback frames

Native-origin Python records carry an identity marker excluded from redaction and JSON output. The handler skips them and guards reentrant serialization with thread-local state. Logging-time native failures increment `ForwardingHandler.failures` without interrupting subsequent Python handlers. Explicit configuration, flush, and shutdown can report errors

Python logger admission remains authoritative: Rust cannot recover messages already rejected by Python levels or filters. Existing handler-specific filters affect those handlers; the forwarding handler has its own destination policy

## Filtering and sampling

Each named destination has its own minimum severity, target prefixes, and sample rate. Configured minimum levels use the shared uppercase severity enum. Python integer severities map to Rust buckets; custom Python level numbers and names remain attributes. Metadata level admission runs early; spans remain enabled to collect fields for admitted events. Normalized source targets are dynamic fields, so destination target selection runs over normalized records. Native records use their tracing target

The finite sample rate is between zero and one. A nonempty LiteLLM trace ID determines a stable SHA256 fraction, so routine logs sharing that ID receive the same decision. Uncorrelated records use per-record random sampling. Error and critical records bypass sampling, but still respect level and target filters. Sampling never changes existing Python output

This is log-event sampling. It does not export OpenTelemetry spans, automatically create distributed trace context, or perform collector tail sampling. LiteLLM correlation IDs remain attributes because they may not be valid OpenTelemetry trace/span IDs

## Exporters and lifecycle

`OtlpSink` uses the official OpenTelemetry 0.33 log SDK and HTTP/protobuf exporter with a batch processor. Pass a full logs endpoint and authentication headers. A collector or compatible direct endpoint can ingest these logs for Loki and other backends. Loki is an open-source log backend with self-managed and hosted deployments, not only a Grafana Cloud service

`PostHogSink` uses the official `posthog-rs` 0.14 SDK documented by PostHog, with its nonblocking capture API and bounded background queue. The `posthog` Cargo feature is enabled by default in `litellm-tracing` and the Python extension. A feature-disabled extension rejects unavailable transport configuration

PostHog receives `litellm diagnostic` events with redacted message, severity, source target, service, and structured fields. The service name is the distinct ID and `$process_person_profile=false` prevents person creation. Python source timestamps are preserved when valid. This sink is explicitly configured diagnostic event capture, not an inference lifecycle callback, product conversion, or the separate PostHog Logs ingestion protocol

Use distinct destination names to fan out. Configuration replaces the complete destination set, and all policies and references are validated before starting workers or publishing a replacement. Rust releases destination locks before SDK work or Python callbacks. Configuration and flush/shutdown release the GIL. Exporter initialization enters the existing host process gate, so fork masters cannot start workers; configure in workers after forking. An inherited exporter is refused before acquiring its locks

The host must flush and shut down before exiting. Shutdown removes all destinations and is idempotent. The forwarding handler rechecks runtime activity, so Python ingress and native remote export both stop while existing handler output remains available. SDK delivery is best effort: queue overflow, failed transport, and exiting before flush can lose diagnostics. PostHog flush has no delivery acknowledgement and SDK timeout settings do not guarantee a total five-second drain across multiple batches. Queue capacity limits records rather than total payload bytes

`force_flush` and `shutdown` returning `True` means the native operation was available and completed, not that a remote service acknowledged every record. SDK warnings cover transport and queue errors; there are no per-destination delivery/drop counters yet

## Bridge choice

The published `pyo3-pylogger` 0.5.2 is not a drop-in adapter for this contract. Its setup replaces `logging.basicConfig`, does not attach existing LiteLLM loggers, and omits exception/stack text and the original creation timestamp. Extras are a JSON field rather than native dynamic metadata. Its README mentions `register_tracing`, but the released source exposes `register` and `setup_logging`

The implementation has no `pyo3-pylogger` dependency and no `log` crate hop. The additive Python handler snapshots the record, and our owned adapter in `python-bridge/src/logger/python.rs` emits directly into `tracing`. That module owns Python severity conversion, context capture, interpreter attachment, and compatibility callbacks. Rust-only exporter configuration and draining run detached from Python, following the [PyO3 performance guide](https://pyo3.rs/main/performance.html#detach-from-the-interpreter-for-long-running-rust-only-work). Keep dynamic fields inside an owned normalized record rather than creating unbounded tracing callsites

The audit used the [0.5.2 package](https://docs.rs/crate/pyo3-pylogger/0.5.2), whose VCS metadata identifies [commit 68a5ae9](https://github.com/dylanbstorey/pyo3-pylogger/blob/68a5ae90b2b839bc4109551ab6ca10659d09c92b/src/lib.rs). The [PyO3 logging guide](https://pyo3.rs/main/ecosystem/logging.html) distinguishes Python ingress from `pyo3-log`, which sends Rust logs into Python

## Validation and remaining scope

Rust tests exercise typed OTLP HTTP payloads and authentication, official PostHog batch requests, original timestamps, nested redaction, minimum levels, target filtering, error retention, stable correlation sampling, independent composable filters, and concurrent context restoration

Python tests cover original handler output and object identity, custom levels, exception/extra projection, disabled export and unavailable native handling, native errors, repeated setup, propagation boundaries, reentrant serialization, and the redaction-safe native origin guard

A shared configuration golden fixture checks Python and Rust wire schemas. The installed-extension recording-server tests send both a Python diagnostic and an actual native OCR route summary through one PostHog sink, then verify named destinations sample independently. They use local recording services and require no production keys

Future work can add explicit delivery/drop metrics, payload-size limits, additional backend authentication/configuration surfaces, and separately scoped OpenTelemetry trace export. Do not claim those are implemented

## References

Use the [Python logging contract](https://docs.python.org/3/library/logging.html), [subscriber per-layer filtering](https://docs.rs/tracing-subscriber/latest/tracing_subscriber/layer/index.html#per-layer-filtering), [OTLP log exporter](https://docs.rs/opentelemetry-otlp/0.33.0/opentelemetry_otlp/), and [PostHog Rust SDK](https://posthog.com/docs/libraries/rust.md) for the implemented boundaries

Use [Loki OTLP ingestion](https://grafana.com/docs/loki/latest/send-data/otel/), [OpenTelemetry sampling](https://opentelemetry.io/docs/concepts/sampling/), and [Alloy tail sampling](https://grafana.com/docs/alloy/latest/reference/components/otelcol/otelcol.processor.tail_sampling/) when designing further destination or trace support
