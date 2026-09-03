"""Provider / exporter factory + the Baggage span processor."""

import threading
from collections import OrderedDict
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any, Final, Literal

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
from opentelemetry.sdk.trace import Span as SDKSpan
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
from opentelemetry.util.re import parse_env_headers

from litellm._logging import verbose_logger
from litellm._version import version as litellm_version
from litellm.integrations.otel.model.config import ExporterSpec, OpenTelemetryV2Config
from litellm.integrations.otel.model.semconv import LiteLLM
from litellm.integrations.otel.model.spans import LiteLLMSpanKind
from litellm.integrations.otel.plumbing.context import (
    overridden_backends,
    request_destinations,
)

if TYPE_CHECKING:
    from opentelemetry.metrics import Meter
    from opentelemetry.sdk.metrics.export import MetricReader

    from litellm.integrations.otel.model.destination import OtelDestination

_SPAN_KIND_BY_ROLE_KIND: Final[dict[LiteLLMSpanKind, SpanKind]] = {
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
_EXPORTER_FACTORIES: Final[dict[str, Callable[[ExporterSpec], SpanExporter]]] = {}


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
    # Splunk Observability uses ``/v2/trace/otlp``; never rewrite it.
    if endpoint.endswith("/v1/traces") or "/v2/trace/otlp" in endpoint:
        return endpoint
    for other_signal in ("/v1/logs", "/v1/metrics"):
        if endpoint.endswith(other_signal):
            return endpoint[: -len(other_signal)] + "/v1/traces"
    return endpoint + "/v1/traces"


def parse_headers(raw: str | None) -> dict[str, str]:
    """Parse an OTLP ``"k=v,k=v"`` header string into a dict.

    ``OTEL_EXPORTER_OTLP_HEADERS`` is W3C Baggage encoded per the OTLP spec, so
    values are percent-decoded: a vendor that documents
    ``Authorization=Basic%20<token>`` (Grafana Cloud does, because a bare space
    is not representable there) has to reach the exporter as ``Basic <token>``,
    not with a literal ``%20`` that the backend rejects as malformed. The SDK's
    own parser is used so litellm decodes exactly what the OTLP exporters do
    when they read the env var themselves; ``liberal`` keeps values that are not
    percent-encoded (``Authorization=Bearer <token>``) working unchanged.
    """
    if not raw:
        return {}
    return dict(parse_env_headers(raw, liberal=True))


_IN_MEMORY_KINDS: Final = ("in_memory", "inmemory", "memory")
_OTLP_HTTP_KINDS: Final = ("otlp_http", "http", "http/protobuf", "http/json")
_OTLP_GRPC_KINDS: Final = ("otlp_grpc", "grpc")


def exporter_transport(kind: str) -> Literal["http", "grpc", "headerless"]:
    """How an exporter of this ``kind`` carries credentials, per ``_exporter_from_spec``.

    ``http``/``grpc`` exporters (and any registered factory, which builds an
    OTLP exporter) stamp ``spec.headers``; ``console``, ``in_memory``, and any
    unrecognized kind (which falls back to a header-ignoring console exporter)
    are ``headerless``. Routability decisions must read this rather than a
    denylist, so a typo'd or unavailable kind is not mistaken for OTLP.
    """
    resolved: Final = kind.lower()
    if resolved in _OTLP_HTTP_KINDS or resolved in _EXPORTER_FACTORIES:
        return "http"
    if resolved in _OTLP_GRPC_KINDS:
        return "grpc"
    return "headerless"


def _exporter_from_spec(spec: ExporterSpec) -> SpanExporter:
    kind: Final = (spec.kind or "console").lower()
    factory: Final = _EXPORTER_FACTORIES.get(kind)
    if factory is not None:
        return factory(spec)
    if kind in _IN_MEMORY_KINDS:
        return InMemorySpanExporter()
    if kind in _OTLP_HTTP_KINDS:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter as HTTPExporter,
        )

        return HTTPExporter(
            endpoint=_otlp_traces_endpoint(spec.endpoint),
            headers=parse_headers(spec.headers),
        )
    if kind in _OTLP_GRPC_KINDS:
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


#: Distinct tenant destinations whose exporters stay alive. Each holds a connection
#: pool and a batch thread, so the cache is bounded and evicts least-recently-used.
_MAX_CACHED_DESTINATION_PROCESSORS: Final = 32
_MAX_RETIRED_DESTINATION_PROCESSORS: Final = 8


class _ResourceWrappedReadableSpan(ReadableSpan):
    """A ``ReadableSpan`` view with an overridden Resource, leaving the original alone."""

    def __init__(self, inner: ReadableSpan, resource: Resource) -> None:
        super().__init__(
            name=inner.name,
            context=inner.context,
            parent=inner.parent,
            resource=resource,
            attributes=inner.attributes,
            events=inner.events,
            links=inner.links,
            kind=inner.kind,
            status=inner.status,
            start_time=inner.start_time,
            end_time=inner.end_time,
            instrumentation_scope=inner.instrumentation_scope,
        )


def _with_destination_resource(span: ReadableSpan, destination: "OtelDestination") -> ReadableSpan:
    extra: Final = destination.resource_attributes
    if not extra:
        return span
    merged: Final = Resource.create(
        {**dict(span.resource.attributes), **dict(extra)}  # mutable-ok: the OTel SDK takes a concrete attribute mapping
    )
    return _ResourceWrappedReadableSpan(span, merged)


class TenantFanOutSpanProcessor(SpanProcessor):
    """Export every finished span to each destination this request resolved.

    Destinations ride a request-scoped ``ContextVar`` set during auth, so concurrent
    requests stay isolated. The forwarded view keeps the original trace and parent
    ids, so the tenant gets the same tree the operator would have received.
    """

    def __init__(
        self,
        processor_factory: 'Callable[["OtelDestination"], SpanProcessor | None] | None' = None,
    ) -> None:
        self._lock: Final = threading.Lock()
        self._build: Final = processor_factory if processor_factory is not None else _destination_processor
        self._processors: OrderedDict[object, SpanProcessor] = OrderedDict()  # mutable-ok: bounded LRU
        self._retired: OrderedDict[object, SpanProcessor] = OrderedDict()  # mutable-ok: bounded drain list

    def on_start(self, span: SDKSpan, parent_context: Context | None = None) -> None:
        return None

    def on_end(self, span: ReadableSpan) -> None:
        for destination in request_destinations():
            processor = self._processor_for(destination)  # rebind-ok: loop variable; pyright forbids Final in a loop
            if processor is None:
                continue
            try:
                processor.on_end(_with_destination_resource(span, destination))
            except Exception as exc:  # noqa: BLE001  # one destination's failure must not cost the others their span
                verbose_logger.debug("OTel V2 fan-out: forwarding to %s failed: %s", destination.endpoint, exc)

    def shutdown(self) -> None:
        # Snapshot first: ``on_end`` mutates the cache on whichever thread ends a span
        # and can run concurrently with this SDK-driven shutdown, so iterating the live
        # mapping risks a "mutated during iteration" the per-item except cannot catch.
        for processor in self._snapshot():
            try:
                processor.shutdown()
            except Exception as exc:  # noqa: BLE001  # one processor's shutdown must not abort the rest
                verbose_logger.debug("OTel V2 fan-out: processor shutdown failed: %s", exc)
        with self._lock:
            self._processors.clear()
            self._retired.clear()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        results: Final = tuple(self._flush_one(processor, timeout_millis) for processor in self._snapshot())
        return all(results)

    def _snapshot(self) -> tuple[SpanProcessor, ...]:
        with self._lock:
            return (*self._processors.values(), *self._retired.values())

    @staticmethod
    def _flush_one(processor: SpanProcessor, timeout_millis: int) -> bool:
        try:
            return processor.force_flush(timeout_millis)
        except Exception:  # noqa: BLE001  # one exporter's flush failure must not fail the whole flush
            return False

    def _processor_for(self, destination: "OtelDestination") -> SpanProcessor | None:
        key: Final = destination.cache_key()
        with self._lock:
            cached: Final = self._processors.get(key)
            if cached is not None:
                self._processors.move_to_end(key)
                return cached
        built: Final = self._build(destination)
        if built is None:
            return None
        with self._lock:
            existing: Final = self._processors.get(key)
            if existing is not None:
                # Another thread won the race; drop ours rather than leak its thread.
                _shutdown_quietly(built)
                return existing
            self._processors[key] = built
            overflowed: Final = self._retired_on_overflow_locked()
        if overflowed is not None:
            _shutdown_quietly(overflowed)
        return built

    def _retired_on_overflow_locked(self) -> SpanProcessor | None:
        """Drop the LRU processor past the cap; return one only once it is safe to close.

        ``on_end`` hands a processor back and then exports outside the lock, so shutting
        an evicted one down there loses that span. Evictions retire to drain instead, and
        the retirees are capped so they cannot accumulate a thread each.
        """
        if len(self._processors) <= _MAX_CACHED_DESTINATION_PROCESSORS:
            return None
        _, evicted = self._processors.popitem(last=False)
        self._retired[id(evicted)] = evicted
        if len(self._retired) <= _MAX_RETIRED_DESTINATION_PROCESSORS:
            return None
        return self._retired.popitem(last=False)[1]


def _destination_processor(destination: "OtelDestination") -> SpanProcessor | None:
    """A batching OTLP processor aimed at ``destination``, or ``None`` if unbuildable."""
    try:
        spec: Final = ExporterSpec(
            kind=destination.protocol or "otlp_http",
            endpoint=destination.endpoint,
            headers=destination.header_string(),
            owner=None,
        )
        return _processor_for(_exporter_from_spec(spec), use_simple=False)
    except Exception as exc:  # noqa: BLE001  # a malformed destination must not break the request or the other destinations
        verbose_logger.debug("OTel V2 fan-out: no processor for %s: %s", destination.endpoint, exc)
        return None


def _shutdown_quietly(processor: SpanProcessor) -> None:
    try:
        processor.shutdown()
    except Exception as exc:  # noqa: BLE001  # defensive: shedding a spare processor must not raise
        verbose_logger.debug("OTel V2 fan-out: discarding processor failed: %s", exc)


class _OverriddenBackendFilter(SpanProcessor):
    """Hold a span back from ``owner``'s operator-level exporter when the request
    pointed ``owner`` at a tenant's own account.

    Wrapping is the only place this works: ``SynchronousMultiSpanProcessor.on_end``
    ignores return values, so a sibling processor can never veto the export.
    """

    def __init__(self, inner: SpanProcessor, owner: str) -> None:
        self._inner: Final = inner
        self._owner: Final = owner

    def on_start(self, span: SDKSpan, parent_context: Context | None = None) -> None:
        self._inner.on_start(span, parent_context)

    def on_end(self, span: ReadableSpan) -> None:
        if self._owner in overridden_backends():
            return
        self._inner.on_end(span)

    def shutdown(self) -> None:
        self._inner.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self._inner.force_flush(timeout_millis)


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

    Histograms keep the SDK's default cumulative temporality. Prometheus-backed
    OTLP receivers (Grafana Cloud / Mimir, and the Prometheus OTLP endpoint)
    reject delta histograms outright with ``invalid temporality and type
    combination``, which drops the whole metric batch, while backends that
    prefer delta still accept cumulative. The enterprise billing exporter
    already relies on the same default.
    """
    from opentelemetry.sdk.metrics.export import (
        ConsoleMetricExporter,
        PeriodicExportingMetricReader,
    )

    kind: Final = (config.exporter or "console").lower()
    if kind in ("otlp_http", "http", "http/protobuf", "http/json"):
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
            OTLPMetricExporter as HTTPMetricExporter,
        )

        exporter: Any = HTTPMetricExporter(
            endpoint=_otlp_metrics_endpoint(config.endpoint),
            headers=parse_headers(config.headers),
        )
    elif kind in ("otlp_grpc", "grpc"):
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
    kind: Final = (config.exporter or "console").lower()
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
    exporter: Final = log_exporter if log_exporter is not None else build_log_exporter(config)
    provider: Final = SDKLoggerProvider(resource=build_resource(config))
    use_simple: Final = isinstance(exporter, (ConsoleLogExporter, InMemoryLogExporter))
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

    existing: Final[LoggerProvider] = _logs.get_logger_provider()
    if isinstance(existing, SDKLoggerProvider):
        return existing
    if isinstance(existing, NoOpLoggerProvider):
        return None

    provider: Final = build_logger_provider(config)
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
    reader: Final = metric_reader if metric_reader is not None else build_metric_reader(config)
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

    existing: Final = metrics.get_meter_provider()
    if isinstance(existing, (SDKMeterProvider, NoOpMeterProvider)):
        return existing

    provider: Final = build_meter_provider(config)
    metrics.set_meter_provider(provider)
    return provider


def get_meter(provider: MeterProvider, name: str = "litellm") -> "Meter":
    return provider.get_meter(name, litellm_version)


def build_resource(config: OpenTelemetryV2Config) -> Resource:
    attributes: Final[dict[str, str]] = {"service.name": config.service_name}
    if config.deployment_environment:
        attributes["deployment.environment"] = config.deployment_environment
    attributes.update(config.resource_attributes)
    return Resource.create(attributes)


def build_tracer_provider(
    config: OpenTelemetryV2Config,
    exporter: SpanExporter | None = None,
    baggage_processor: SpanProcessor | None = None,
    use_simple_processor: bool | None = None,
    tenant_overrides: bool = False,
) -> TracerProvider:
    """Build the shared :class:`TracerProvider`.

    Attach the Baggage processor first (so identity attributes land on each
    span before any export decision), then add one ``SpanProcessor`` per
    ``config.exporters`` entry — this is what fans spans out to multiple
    backends. ``exporter`` and ``use_simple_processor`` are explicit overrides:
    pass a single exporter to attach exactly that one (used by tests).

    ``tenant_overrides`` belongs to the operator-level provider alone: it wraps each
    owned exporter so a request that pointed that backend at a key's or team's own
    account skips it, and adds the fan-out processor that delivers to that account
    instead. The per-tenant providers this same function builds must leave it off,
    or they would filter out the very spans they exist to carry.
    """
    provider: Final = TracerProvider(resource=build_resource(config))
    if baggage_processor is None:
        baggage_processor = LiteLLMBaggageSpanProcessor(allowed_keys=config.baggage_promoted_keys)
    provider.add_span_processor(baggage_processor)

    if exporter is not None:
        provider.add_span_processor(_processor_for(exporter, use_simple_processor))
        return provider

    # ``config._normalize`` guarantees at least one spec (it folds the top-level
    # ``exporter``/``endpoint``/``headers`` fields in when ``exporters`` is empty).
    for spec in config.exporters:
        if spec.requires_headers and not spec.headers:
            continue
        exp = _exporter_from_spec(spec)
        processor = _processor_for(
            exp,
            (spec.use_simple_processor if spec.use_simple_processor is not None else use_simple_processor),
        )
        owner = spec.owner.value if spec.owner is not None else None
        provider.add_span_processor(
            _OverriddenBackendFilter(processor, owner) if tenant_overrides and owner is not None else processor
        )
    if tenant_overrides:
        provider.add_span_processor(TenantFanOutSpanProcessor())
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
    cfg: Final = config or OpenTelemetryV2Config(exporter="in_memory")
    exporter: Final = InMemorySpanExporter()
    provider: Final = build_tracer_provider(cfg, exporter=exporter)
    return provider, exporter
