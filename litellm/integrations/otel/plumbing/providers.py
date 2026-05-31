"""Provider / exporter factory + the Baggage span processor."""

from typing import Callable, Iterable

from opentelemetry import baggage
from opentelemetry.context import Context
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

from litellm.integrations.otel.model.config import ExporterSpec, OpenTelemetryV2Config
from litellm.integrations.otel.model.semconv import LiteLLM
from litellm.integrations.otel.model.spans import LiteLLMSpanKind

# Re-exported so ``providers.parse_headers`` remains a stable entry point.
from litellm.integrations.otel.model.utils import parse_headers as parse_headers

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


def register_exporter_factory(
    kind: str, factory: Callable[[ExporterSpec], SpanExporter]
) -> None:
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
        return key in self._allowed_keys or any(
            key.startswith(prefix) for prefix in self._allowed_prefixes
        )

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
    return _exporter_from_spec(
        ExporterSpec(
            kind=config.exporter, endpoint=config.endpoint, headers=config.headers
        )
    )


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
) -> TracerProvider:
    """Build the shared :class:`TracerProvider`.

    Attach the Baggage processor first (so identity attributes land on each
    span before any export decision), then add one ``SpanProcessor`` per
    ``config.exporters`` entry — this is what fans spans out to multiple
    backends. ``exporter`` and ``use_simple_processor`` are explicit overrides:
    pass a single exporter to attach exactly that one (used by tests).
    """
    provider = TracerProvider(resource=build_resource(config))
    if baggage_processor is None:
        baggage_processor = LiteLLMBaggageSpanProcessor(
            allowed_keys=config.baggage_promoted_keys
        )
    provider.add_span_processor(baggage_processor)

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
                (
                    spec.use_simple_processor
                    if spec.use_simple_processor is not None
                    else use_simple_processor
                ),
            )
        )
    return provider


def get_tracer(provider: TracerProvider, name: str = "litellm") -> Tracer:
    return provider.get_tracer(name)


def in_memory_provider(
    config: OpenTelemetryV2Config | None = None,
) -> tuple[TracerProvider, InMemorySpanExporter]:
    """Convenience for tests: a provider exporting to an in-memory buffer."""
    cfg = config or OpenTelemetryV2Config(exporter="in_memory")
    exporter = InMemorySpanExporter()
    provider = build_tracer_provider(cfg, exporter=exporter)
    return provider, exporter
