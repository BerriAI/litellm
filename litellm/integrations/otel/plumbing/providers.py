"""Provider / exporter factory + the Baggage span processor."""

from typing import TYPE_CHECKING, Any, Callable, Iterable

from opentelemetry import _logs, baggage, metrics
from opentelemetry._events import EventLogger
from opentelemetry._logs import LoggerProvider, NoOpLoggerProvider
from opentelemetry.context import Context
from opentelemetry.metrics import MeterProvider, NoOpMeterProvider
from opentelemetry.sdk._events import EventLoggerProvider
from opentelemetry.sdk._logs import LoggerProvider as SDKLoggerProvider
from opentelemetry.sdk._logs.export import (
    BatchLogRecordProcessor,
    ConsoleLogExporter,
    InMemoryLogExporter,
    LogExporter,
    SimpleLogRecordProcessor,
)
from opentelemetry.sdk.metrics import MeterProvider as SDKMeterProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
    SpanExporter,
)
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace import Span, SpanKind, Tracer

from litellm._version import version as litellm_version
from litellm.integrations.otel.model.config import ExporterSpec, OpenTelemetryV2Config
from litellm.integrations.otel.model.semconv import LiteLLM
from litellm.integrations.otel.model.spans import LiteLLMSpanKind

# Re-exported so ``providers.parse_headers`` remains a stable entry point.
from litellm.integrations.otel.model.utils import parse_headers as parse_headers

if TYPE_CHECKING:
    from opentelemetry.metrics import Meter
    from opentelemetry.sdk.metrics.export import MetricReader

    from litellm.integrations.otel.model.destination import OtelDestination

_SPAN_KIND_BY_ROLE_KIND: dict[LiteLLMSpanKind, SpanKind] = {
    LiteLLMSpanKind.SERVER: SpanKind.SERVER,
    LiteLLMSpanKind.CLIENT: SpanKind.CLIENT,
    LiteLLMSpanKind.INTERNAL: SpanKind.INTERNAL,
    LiteLLMSpanKind.PRODUCER: SpanKind.PRODUCER,
    LiteLLMSpanKind.CONSUMER: SpanKind.CONSUMER,
}


def to_otel_span_kind(kind: LiteLLMSpanKind) -> SpanKind:
    return _SPAN_KIND_BY_ROLE_KIND[kind]


# Custom exporter factories keyed by ``ExporterSpec.kind``. A preset registers
# one here when its destination needs construction logic the built-in kinds
# can't express — e.g. an exporter that fetches an auth token lazily on its
# first export (off the event loop) instead of blocking at config-build time.
# Keeping the registry here lets this module stay vendor-agnostic: the factory
# lives with the integration that needs it.
_EXPORTER_FACTORIES: dict[str, Callable[[ExporterSpec], SpanExporter]] = {}


def register_exporter_factory(kind: str, factory: Callable[[ExporterSpec], SpanExporter]) -> None:
    """Register a custom exporter ``factory`` for the exporter ``kind``."""
    _EXPORTER_FACTORIES[kind.lower()] = factory


class LiteLLMBaggageSpanProcessor(SpanProcessor):
    """Stamps an allowlisted set of Baggage entries onto every span at start."""

    def __init__(
        self,
        allowed_keys: Iterable[str],
        allowed_prefixes: tuple[str, ...] = (LiteLLM.METADATA_PREFIX,),
    ) -> None:
        self._allowed_keys = frozenset(allowed_keys)
        self._allowed_prefixes = tuple(allowed_prefixes)

    def _is_allowed(self, key: str) -> bool:
        return key in self._allowed_keys or any(key.startswith(prefix) for prefix in self._allowed_prefixes)

    def on_start(self, span: Span, parent_context: Context | None = None) -> None:
        for key, value in baggage.get_all(parent_context).items():
            if self._is_allowed(key) and isinstance(value, (str, bool, int, float)):
                span.set_attribute(key, value)

    def on_end(self, span: ReadableSpan) -> None:  # noqa: D401 - no-op
        return None

    def shutdown(self) -> None:
        return None

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


def _otlp_traces_endpoint(endpoint: str | None) -> str | None:
    """Point an OTLP/HTTP base endpoint at the ``/v1/traces`` signal path.

    ``OTEL_EXPORTER_OTLP_ENDPOINT`` is a base URL (e.g. ``http://host:4318``).
    The OTLP/HTTP exporter only appends the ``/v1/traces`` path when it reads
    that env var itself; when an endpoint is passed explicitly it is used
    verbatim, so a base URL would POST to the root and the collector returns
    404. Append the signal path here (leaving an already-correct path intact).
    """
    if not endpoint:
        return endpoint
    endpoint = endpoint.rstrip("/")
    # Some vendors expose a complete traces ingest path that is NOT the OTLP-standard
    # ``/v1/traces`` base: Splunk Observability uses ``/v2/trace/otlp`` and Langtrace
    # ingests at ``/api/trace``. Appending ``/v1/traces`` to those 404s, so never
    # rewrite them.
    if endpoint.endswith("/v1/traces") or "/v2/trace/otlp" in endpoint or endpoint.endswith("/api/trace"):
        return endpoint
    for other_signal in ("/v1/logs", "/v1/metrics"):
        if endpoint.endswith(other_signal):
            return endpoint[: -len(other_signal)] + "/v1/traces"
    return endpoint + "/v1/traces"


# Backends whose OTLP transport is gRPC. Arize's OTLP endpoint
# (``otlp.arize.com``) speaks gRPC; every other current preset speaks OTLP/HTTP.
# Single source of truth shared by the per-tenant fan-out processor and the
# ``TenantTracerCache`` so the two never disagree on a destination's transport.
_GRPC_BACKENDS = frozenset({"arize"})


def default_otlp_kind_for_backend(callback_name: "str | None") -> str:
    """The intrinsic OTLP transport for a backend's own OTLP endpoint."""
    return "otlp_grpc" if callback_name in _GRPC_BACKENDS else "otlp_http"


def destination_resource_attrs(destination: "OtelDestination") -> dict[str, str]:
    """The backend-required Resource attributes a destination carries on every span.

    Backend-agnostic: each backend's destination builder (``presets.destinations``)
    declares whatever Resource attributes its ingestion needs, and this just reads
    them. Backends that route by auth header (langfuse, weave, generic OTLP) declare
    none; Arize declares ``model_id`` / ``arize.project.name`` because it selects the
    project from the Resource, not a header. New backends needing Resource-level
    routing only have to populate ``resource_attributes`` in their builder.

    Shared by the two export paths that reach a per-tenant destination -- the
    ``TenantFanOutSpanProcessor`` (proxy-internal spans) and the ``TenantTracerCache``
    clone provider (the gen-AI span) -- so the gen-AI span and its parents always
    carry the SAME Resource and a backend like Arize renders one connected trace
    instead of an orphaned subtree.
    """
    return dict(destination.resource_attributes)


def _exporter_from_spec(spec: ExporterSpec) -> SpanExporter:
    kind = (spec.kind or "console").lower()
    factory = _EXPORTER_FACTORIES.get(kind)
    if factory is not None:
        return factory(spec)
    if kind in ("in_memory", "inmemory", "memory"):
        return InMemorySpanExporter()
    if kind in ("otlp_http", "http", "http/protobuf", "http/json"):
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter as HTTPExporter,
        )

        return HTTPExporter(
            endpoint=_otlp_traces_endpoint(spec.endpoint),
            headers=parse_headers(spec.headers),
        )
    if kind in ("otlp_grpc", "grpc"):
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter as GRPCExporter,
        )

        return GRPCExporter(endpoint=spec.endpoint, headers=parse_headers(spec.headers))
    return ConsoleSpanExporter()


def _processor_for(exporter: SpanExporter, use_simple: bool | None) -> SpanProcessor:
    """Pick a Simple or Batch span processor for ``exporter``.

    When ``use_simple`` is unset, default to Simple for console and in-memory
    exporters (spans export synchronously, which tests rely on) and Batch for
    everything else (the right export semantics for production).
    """
    if use_simple is None:
        use_simple = isinstance(exporter, (ConsoleSpanExporter, InMemorySpanExporter))
    return SimpleSpanProcessor(exporter) if use_simple else BatchSpanProcessor(exporter)


def build_span_exporter(config: OpenTelemetryV2Config) -> SpanExporter:
    """Build a single exporter from the top-level config fields.

    Convenience for the common single-exporter case (and for tests): reads the
    ``exporter`` / ``endpoint`` / ``headers`` fields. To configure multiple
    exporters, populate ``config.exporters`` directly.
    """
    return _exporter_from_spec(ExporterSpec(kind=config.exporter, endpoint=config.endpoint, headers=config.headers))


def _otlp_metrics_endpoint(endpoint: str | None) -> str | None:
    """Point an OTLP/HTTP base endpoint at the ``/v1/metrics`` signal path.

    The OTLP/HTTP exporter only appends ``/v1/metrics`` when it reads
    ``OTEL_EXPORTER_OTLP_ENDPOINT`` itself; an explicitly passed endpoint is used
    verbatim, so a base URL would POST to the root. Mirror ``_otlp_traces_endpoint``
    for the metrics signal (rewriting a sibling signal path when present).
    """
    if not endpoint:
        return endpoint
    endpoint = endpoint.rstrip("/")
    if endpoint.endswith("/v1/metrics"):
        return endpoint
    for other_signal in ("/v1/traces", "/v1/logs"):
        if endpoint.endswith(other_signal):
            return endpoint[: -len(other_signal)] + "/v1/metrics"
    return endpoint + "/v1/metrics"


def build_metric_reader(config: OpenTelemetryV2Config) -> "MetricReader":
    """Build a metric reader mirroring v1's exporter selection.

    ``console`` (and any unrecognized kind) exports to the console; ``otlp_http``
    and ``otlp_grpc`` export over OTLP with the configured endpoint/headers. The
    reader exports on a 5s period, matching v1.
    """
    from opentelemetry.sdk.metrics.export import (
        ConsoleMetricExporter,
        PeriodicExportingMetricReader,
    )

    kind = (config.exporter or "console").lower()
    if kind in ("otlp_http", "http", "http/protobuf", "http/json"):
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
            OTLPMetricExporter as HTTPMetricExporter,
        )
        from opentelemetry.sdk.metrics import Histogram
        from opentelemetry.sdk.metrics.export import AggregationTemporality

        exporter: Any = HTTPMetricExporter(
            endpoint=_otlp_metrics_endpoint(config.endpoint),
            headers=parse_headers(config.headers),
            preferred_temporality={Histogram: AggregationTemporality.DELTA},
        )
    elif kind in ("otlp_grpc", "grpc"):
        from opentelemetry.sdk.metrics import Histogram
        from opentelemetry.sdk.metrics.export import AggregationTemporality

        try:
            from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
                OTLPMetricExporter as GRPCMetricExporter,
            )
        except ImportError as exc:
            raise ImportError(
                "OpenTelemetry OTLP gRPC metric exporter is not available. Install "
                "`opentelemetry-exporter-otlp` and `grpcio` (or `litellm[grpc]`)."
            ) from exc

        exporter = GRPCMetricExporter(
            endpoint=config.endpoint,
            headers=parse_headers(config.headers),
            preferred_temporality={Histogram: AggregationTemporality.DELTA},
        )
    else:
        exporter = ConsoleMetricExporter()

    return PeriodicExportingMetricReader(exporter, export_interval_millis=5000)


def _otlp_logs_endpoint(endpoint: str | None) -> str | None:
    """Point an OTLP/HTTP base endpoint at the ``/v1/logs`` signal path.

    The OTLP/HTTP exporter only appends ``/v1/logs`` when it reads
    ``OTEL_EXPORTER_OTLP_ENDPOINT`` itself; an explicitly passed endpoint is used
    verbatim, so a base URL would POST to the root. Mirror ``_otlp_traces_endpoint``
    for the logs signal (rewriting a sibling signal path when present).
    """
    if not endpoint:
        return endpoint
    endpoint = endpoint.rstrip("/")
    if endpoint.endswith("/v1/logs"):
        return endpoint
    for other_signal in ("/v1/traces", "/v1/metrics"):
        if endpoint.endswith(other_signal):
            return endpoint[: -len(other_signal)] + "/v1/logs"
    return endpoint + "/v1/logs"


def build_log_exporter(config: OpenTelemetryV2Config) -> LogExporter:
    """Build a log exporter mirroring the exporter selection of the other signals.

    ``console`` (and any unrecognized kind) exports to the console; ``otlp_http``
    and ``otlp_grpc`` export over OTLP with the configured endpoint/headers;
    ``in_memory`` buffers for tests. Like GenAI metrics, events ride the
    single-destination shorthand fields, not the multi-exporter ``exporters`` list.
    """
    kind = (config.exporter or "console").lower()
    if kind in ("in_memory", "inmemory", "memory"):
        return InMemoryLogExporter()
    if kind in ("otlp_http", "http", "http/protobuf", "http/json"):
        from opentelemetry.exporter.otlp.proto.http._log_exporter import (
            OTLPLogExporter as HTTPLogExporter,
        )

        return HTTPLogExporter(
            endpoint=_otlp_logs_endpoint(config.endpoint),
            headers=parse_headers(config.headers),
        )
    if kind in ("otlp_grpc", "grpc"):
        try:
            from opentelemetry.exporter.otlp.proto.grpc._log_exporter import (
                OTLPLogExporter as GRPCLogExporter,
            )
        except ImportError as exc:
            raise ImportError(
                "OpenTelemetry OTLP gRPC log exporter is not available. Install "
                "`opentelemetry-exporter-otlp` and `grpcio` (or `litellm[grpc]`)."
            ) from exc

        return GRPCLogExporter(endpoint=config.endpoint, headers=parse_headers(config.headers))
    return ConsoleLogExporter()


def build_logger_provider(
    config: OpenTelemetryV2Config,
    log_exporter: LogExporter | None = None,
) -> SDKLoggerProvider:
    """Build the :class:`LoggerProvider` GenAI events export through.

    ``log_exporter`` is an explicit override (tests inject an
    ``InMemoryLogExporter``); otherwise the exporter is selected from the config's
    exporter kind via :func:`build_log_exporter`. Console and in-memory exporters
    get a Simple processor (synchronous export, which tests rely on), everything
    else a Batch processor — the same split as span processing.
    """
    exporter = log_exporter if log_exporter is not None else build_log_exporter(config)
    provider = SDKLoggerProvider(resource=build_resource(config))
    use_simple = isinstance(exporter, (ConsoleLogExporter, InMemoryLogExporter))
    provider.add_log_record_processor(
        SimpleLogRecordProcessor(exporter) if use_simple else BatchLogRecordProcessor(exporter)
    )
    return provider


def resolve_logger_provider(
    config: OpenTelemetryV2Config,
    logger_provider: SDKLoggerProvider | None = None,
) -> SDKLoggerProvider | None:
    """Resolve the :class:`LoggerProvider` GenAI events record through, or ``None``
    when the operator has opted out of the logs signal.

    Same resolution order as :func:`resolve_meter_provider`: an injected provider
    wins (DI/tests); an operator-configured SDK global is reused so events ride
    their pipeline; an explicit ``NoOpLoggerProvider`` global is an opt-out and
    yields ``None``, so no event is ever built. Only the default placeholder
    global makes V2 build a provider from the config and publish it as the global.
    """
    if logger_provider is not None:
        return logger_provider

    existing: LoggerProvider = _logs.get_logger_provider()
    if isinstance(existing, SDKLoggerProvider):
        return existing
    if isinstance(existing, NoOpLoggerProvider):
        return None

    provider = build_logger_provider(config)
    _logs.set_logger_provider(provider)
    return provider


def get_event_logger(provider: SDKLoggerProvider, name: str = "litellm") -> EventLogger:
    return EventLoggerProvider(logger_provider=provider).get_event_logger(name, litellm_version)


def build_meter_provider(
    config: OpenTelemetryV2Config,
    metric_reader: "MetricReader | None" = None,
) -> SDKMeterProvider:
    """Build the :class:`MeterProvider` for GenAI metrics.

    ``metric_reader`` is an explicit override (tests inject an
    ``InMemoryMetricReader``); otherwise the reader is selected from the config's
    exporter kind via :func:`build_metric_reader`.
    """
    reader = metric_reader if metric_reader is not None else build_metric_reader(config)
    return SDKMeterProvider(metric_readers=[reader], resource=build_resource(config))


def resolve_meter_provider(
    config: OpenTelemetryV2Config,
    meter_provider: MeterProvider | None = None,
) -> MeterProvider:
    """Resolve the :class:`MeterProvider` GenAI metrics record through.

    An injected provider wins (DI/tests). Otherwise reuse whatever the operator has
    configured as the global, whether a real SDK provider or an explicit
    ``NoOpMeterProvider``, so the GenAI histograms ride the operator's
    readers/exporters and an explicit opt-out is honored. Only when the global is
    still the default proxy placeholder does V2 build one from the config and
    publish it as the global, mirroring how V2 owns trace export. The built
    provider is the one returned, so its reader thread is always live, never
    orphaned.
    """
    if meter_provider is not None:
        return meter_provider

    existing = metrics.get_meter_provider()
    if isinstance(existing, (SDKMeterProvider, NoOpMeterProvider)):
        return existing

    provider = build_meter_provider(config)
    metrics.set_meter_provider(provider)
    return provider


def get_meter(provider: MeterProvider, name: str = "litellm") -> "Meter":
    return provider.get_meter(name, litellm_version)


def build_resource(config: OpenTelemetryV2Config) -> Resource:
    attributes: dict[str, str] = {"service.name": config.service_name}
    if config.deployment_environment:
        attributes["deployment.environment"] = config.deployment_environment
    attributes.update(config.resource_attributes)
    return Resource.create(attributes)


def build_tracer_provider(
    config: OpenTelemetryV2Config,
    exporter: SpanExporter | None = None,
    baggage_processor: SpanProcessor | None = None,
    use_simple_processor: bool | None = None,
    tenant_fan_out_owner: str | None = None,
    attach_tenant_fan_out: bool = False,
) -> TracerProvider:
    """Build the shared :class:`TracerProvider`.

    Attach the Baggage processor first (so identity attributes land on each
    span before any export decision), then add one ``SpanProcessor`` per
    ``config.exporters`` entry — this is what fans spans out to multiple
    backends. ``exporter`` and ``use_simple_processor`` are explicit overrides:
    pass a single exporter to attach exactly that one (used by tests).

    ``attach_tenant_fan_out`` — attach a ``TenantFanOutSpanProcessor`` that forwards
    each finished proxy-internal span (FastAPI server, ``auth`` phase, DB lookups, the
    cost ledger) to the request's admin-resolved destinations. The MAIN v2 logger
    provider always opts in, EVEN when no backend is named (the generic global logger
    published for a destination-only deployment) -- otherwise the server span never
    reaches the destination and its gen-AI child is orphaned. ``tenant_fan_out_owner``
    is the owning backend name when one exists; it is informational (the fan-out skips
    the gen-AI span by attribute and forwards internal spans to every destination).
    The per-tenant clone providers (built by ``TenantTracerCache``) pass neither, so
    the LLM-call span exported through them is not also fanned out here.
    """
    provider = TracerProvider(resource=build_resource(config))
    if baggage_processor is None:
        baggage_processor = LiteLLMBaggageSpanProcessor(allowed_keys=config.baggage_promoted_keys)
    provider.add_span_processor(baggage_processor)

    if attach_tenant_fan_out or tenant_fan_out_owner is not None:
        from litellm.integrations.otel.plumbing.routing import (
            TenantFanOutSpanProcessor,
        )

        provider.add_span_processor(TenantFanOutSpanProcessor(owner_callback_name=tenant_fan_out_owner))

    if exporter is not None:
        provider.add_span_processor(_processor_for(exporter, use_simple_processor))
        return provider

    # ``config._normalize`` guarantees at least one spec (it folds the top-level
    # ``exporter``/``endpoint``/``headers`` fields in when ``exporters`` is empty).
    for spec in config.exporters:
        exp = _exporter_from_spec(spec)
        provider.add_span_processor(
            _processor_for(
                exp,
                (spec.use_simple_processor if spec.use_simple_processor is not None else use_simple_processor),
            )
        )
    return provider


def get_tracer(provider: TracerProvider, name: str = "litellm") -> Tracer:
    # Stamp the instrumentation scope with the LiteLLM package version so every
    # emitted span carries a deterministic ``scope.version`` (the standard OTel
    # location for the emitting library's version) for downstream consumers.
    return provider.get_tracer(name, litellm_version)


def in_memory_provider(
    config: OpenTelemetryV2Config | None = None,
) -> tuple[TracerProvider, InMemorySpanExporter]:
    """Convenience for tests: a provider exporting to an in-memory buffer."""
    cfg = config or OpenTelemetryV2Config(exporter="in_memory")
    exporter = InMemorySpanExporter()
    provider = build_tracer_provider(cfg, exporter=exporter)
    return provider, exporter
