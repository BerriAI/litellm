"""Provider/exporter factory and the Baggage span processor.

OTLP exporter imports are performed lazily inside :func:`build_span_exporter` so
that importing this module never pulls in ``grpc`` (a contract enforced by
``tests/.../test_opentelemetry_dynamic_imports.py``).
"""

from typing import Dict, Iterable, List, Optional, Tuple

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

from litellm.integrations.otel.config import OpenTelemetryV2Config
from litellm.integrations.otel.semconv import LiteLLM
from litellm.integrations.otel.spans import LiteLLMSpanKind

_SPAN_KIND_BY_ROLE_KIND: Dict[LiteLLMSpanKind, SpanKind] = {
    LiteLLMSpanKind.SERVER: SpanKind.SERVER,
    LiteLLMSpanKind.CLIENT: SpanKind.CLIENT,
    LiteLLMSpanKind.INTERNAL: SpanKind.INTERNAL,
    LiteLLMSpanKind.PRODUCER: SpanKind.PRODUCER,
    LiteLLMSpanKind.CONSUMER: SpanKind.CONSUMER,
}


def to_otel_span_kind(kind: LiteLLMSpanKind) -> SpanKind:
    return _SPAN_KIND_BY_ROLE_KIND[kind]


class LiteLLMBaggageSpanProcessor(SpanProcessor):
    """Stamps an allowlisted set of Baggage entries onto every span at start.

    Only exact keys in ``allowed_keys`` or keys matching ``allowed_prefixes`` are
    promoted, so arbitrary upstream Baggage and the full ``metadata`` blob are
    never copied onto spans.
    """

    def __init__(
        self,
        allowed_keys: Iterable[str],
        allowed_prefixes: Tuple[str, ...] = (LiteLLM.METADATA_PREFIX,),
    ) -> None:
        self._allowed_keys = frozenset(allowed_keys)
        self._allowed_prefixes = tuple(allowed_prefixes)

    def _is_allowed(self, key: str) -> bool:
        return key in self._allowed_keys or any(
            key.startswith(prefix) for prefix in self._allowed_prefixes
        )

    def on_start(self, span: Span, parent_context: Optional[Context] = None) -> None:
        for key, value in baggage.get_all(parent_context).items():
            if self._is_allowed(key) and isinstance(value, (str, bool, int, float)):
                span.set_attribute(key, value)

    def on_end(self, span: ReadableSpan) -> None:  # noqa: D401 - no-op
        return None

    def shutdown(self) -> None:
        return None

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


def parse_headers(raw: Optional[str]) -> Dict[str, str]:
    """Parse an ``"k=v,k2=v2"`` header string into a dict."""
    headers: Dict[str, str] = {}
    if not raw:
        return headers
    for pair in raw.split(","):
        if "=" in pair:
            key, _, value = pair.partition("=")
            headers[key.strip()] = value.strip()
    return headers


def build_span_exporter(config: OpenTelemetryV2Config) -> SpanExporter:
    exporter = (config.exporter or "console").lower()
    if exporter in ("in_memory", "inmemory", "memory"):
        return InMemorySpanExporter()
    if exporter in ("otlp_http", "http", "http/protobuf", "http/json"):
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter as HTTPExporter,
        )

        return HTTPExporter(
            endpoint=config.endpoint, headers=parse_headers(config.headers)
        )
    if exporter in ("otlp_grpc", "grpc"):
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter as GRPCExporter,
        )

        return GRPCExporter(
            endpoint=config.endpoint, headers=parse_headers(config.headers)
        )
    return ConsoleSpanExporter()


def build_resource(config: OpenTelemetryV2Config) -> Resource:
    attributes: Dict[str, str] = {"service.name": config.service_name}
    if config.deployment_environment:
        attributes["deployment.environment"] = config.deployment_environment
    return Resource.create(attributes)


def build_tracer_provider(
    config: OpenTelemetryV2Config,
    exporter: Optional[SpanExporter] = None,
    baggage_processor: Optional[SpanProcessor] = None,
    use_simple_processor: Optional[bool] = None,
) -> TracerProvider:
    """Build a :class:`TracerProvider` wired with the baggage + export processors.

    The baggage processor is registered first so identity attributes are stamped
    on ``on_start`` before any export decision.
    """
    provider = TracerProvider(resource=build_resource(config))
    if baggage_processor is None:
        baggage_processor = LiteLLMBaggageSpanProcessor(
            allowed_keys=config.baggage_promoted_keys
        )
    provider.add_span_processor(baggage_processor)

    span_exporter = exporter if exporter is not None else build_span_exporter(config)
    if use_simple_processor is None:
        use_simple_processor = isinstance(
            span_exporter, (ConsoleSpanExporter, InMemorySpanExporter)
        )
    export_processor: SpanProcessor = (
        SimpleSpanProcessor(span_exporter)
        if use_simple_processor
        else BatchSpanProcessor(span_exporter)
    )
    provider.add_span_processor(export_processor)
    return provider


def get_tracer(provider: TracerProvider, name: str = "litellm") -> Tracer:
    return provider.get_tracer(name)


def in_memory_provider(
    config: Optional[OpenTelemetryV2Config] = None,
) -> Tuple[TracerProvider, InMemorySpanExporter]:
    """Convenience for tests: a provider exporting to an in-memory buffer."""
    cfg = config or OpenTelemetryV2Config(exporter="in_memory")
    exporter = InMemorySpanExporter()
    provider = build_tracer_provider(cfg, exporter=exporter)
    return provider, exporter


def promoted_baggage_processor_keys(
    config: OpenTelemetryV2Config,
) -> List[str]:
    """Exact baggage keys the processor promotes (excludes metadata-prefix keys)."""
    return list(config.baggage_promoted_keys) + [LiteLLM.END_USER]
