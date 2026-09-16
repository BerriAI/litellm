"""The span engine: dedup, start, run the mapper chain, set status, end."""

from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from types import MappingProxyType
from typing import Final

from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan, SpanLimits
from opentelemetry.sdk.trace import Tracer as SdkTracer
from opentelemetry.trace import Link, Span, Tracer
from opentelemetry.trace.status import Status, StatusCode

from litellm.integrations.otel.mappers import resolve_mappers
from litellm.integrations.otel.mappers.base import AttributeMapper, AttrValue, SpanData
from litellm.integrations.otel.mappers.openinference import fit_indexed_messages
from litellm.integrations.otel.model.config import OpenTelemetryV2Config
from litellm.integrations.otel.model.payloads import (
    GuardrailSpanData,
    LLMCallSpanData,
    MCPListToolsSpanData,
    MCPToolCallSpanData,
    ServiceSpanData,
    SpanError,
)
from litellm.integrations.otel.model.semconv import Error, ExceptionEvent, LiteLLMError
from litellm.integrations.otel.model.spans import (
    SPAN_REGISTRY,
    SpanRole,
    guardrail_span_name,
    llm_call_span_name,
    mcp_list_tools_span_name,
    mcp_tool_call_span_name,
    service_span_name,
)
from litellm.integrations.otel.plumbing.events import GenAIEventRecorder
from litellm.integrations.otel.plumbing.providers import to_otel_span_kind

# Roles emit() knows how to name and emit. PROXY_REQUEST and the management
# routes are SERVER spans owned by the mounted FastAPI instrumentor, so they
# have no builder here.
_NAME_BUILDERS: Final[dict[SpanRole, Callable[..., str]]] = {
    SpanRole.LLM_CALL: llm_call_span_name,
    SpanRole.MCP_TOOL_CALL: mcp_tool_call_span_name,
    SpanRole.MCP_LIST_TOOLS: mcp_list_tools_span_name,
    SpanRole.GUARDRAIL: guardrail_span_name,
    # DB_CALL and SERVICE are both built from ServiceSpanData; they differ only in
    # span kind (CLIENT vs INTERNAL) and attribute vocabulary, not in naming.
    SpanRole.DB_CALL: service_span_name,
    SpanRole.SERVICE: service_span_name,
}

# Cap on the dedup cache. It only needs to coalesce the sync+async firing window
# of a single in-flight request, so a bounded LRU keeps memory flat on a
# long-running proxy while still covering every concurrently-open call.
_DEDUP_CACHE_MAX: Final = 10_000


def _resolve_error(error: SpanError) -> tuple[str, str] | None:
    """The ``(error_type, message)`` fallback chain shared by the status, the event and the attributes, or
    ``None`` when ``error`` carries neither a type nor a message."""
    if not (error.error_type or error.message):
        return None
    return error.error_type or "error", error.message or error.error_type or "error"


_NO_ATTRIBUTES: Final[Mapping[str, AttrValue]] = MappingProxyType({})


def error_attributes(error: SpanError) -> Mapping[str, AttrValue]:
    """The v2 error attribute set: the OTel-semconv ``error.*`` pair plus the litellm detail keys that are
    populated, so guardrail-shape errors carrying only a message aren't polluted with empty detail keys."""
    resolved: Final = _resolve_error(error)
    if resolved is None:
        return _NO_ATTRIBUTES
    error_type, message = resolved
    pairs: Final = (
        (Error.TYPE, error_type),
        (Error.MESSAGE, message),
        (LiteLLMError.CODE, error.code),
        (LiteLLMError.STACK_TRACE, error.stack_trace),
        (LiteLLMError.LLM_PROVIDER, error.llm_provider),
    )
    return MappingProxyType({key: value for key, value in pairs if value})


def span_attribute_limit(tracer: Tracer) -> int | None:
    """The attribute count limit spans started by ``tracer`` are built with, ``None`` when unbounded."""
    if not isinstance(tracer, SdkTracer):
        return SpanLimits().max_span_attributes
    return tracer._span_limits.max_span_attributes  # pyright: ignore[reportPrivateUsage]  # SDK has no public getter


def stamp_error(
    span: Span,
    error: SpanError,
    *,
    record_event: bool = True,
    set_status: bool = True,
) -> tuple[str, str] | None:
    """Stamp the full v2 error attribute set on ``span`` and return the resolved
    ``(error_type, message)`` pair, or ``None`` when the error carries neither a
    type nor a message.

    Shared by the LLM-call span (``finish_span``) and the proxy-level failure
    spans (the FastAPI SERVER span and the ``auth`` phase span) so every v2 error
    span carries identical keys. The semconv ``exception`` event rides alongside
    the attributes so backends that map unknown string attrs to a truncated
    ``keyword`` (e.g. Elasticsearch's 1024-char ``ignore_above``) still see the
    full untruncated message on the recognized event field. ``record_event`` and
    ``set_status`` are opt-outs for callers whose span lifecycle (``use_span``) or
    owner (the FastAPI instrumentor) already records the event or the status.
    """
    resolved: Final = _resolve_error(error)
    if resolved is None:
        return None
    error_type, message = resolved
    for key, value in error_attributes(error).items():
        span.set_attribute(key, value)
    if set_status:
        span.set_status(Status(StatusCode.ERROR, message))
    if record_event:
        span.add_event(
            ExceptionEvent.NAME,
            {ExceptionEvent.TYPE: error_type, ExceptionEvent.MESSAGE: message},
        )
    return error_type, message


class SpanEmitter:
    def __init__(
        self,
        tracer: Tracer,
        config: OpenTelemetryV2Config,
        mappers: Sequence[AttributeMapper] | None = None,
        event_recorder: GenAIEventRecorder | None = None,
    ) -> None:
        self._tracer = tracer
        self._config = config
        self._event_recorder = event_recorder
        self._span_attribute_limit: int | None = span_attribute_limit(tracer)
        # The mapper chain is the sole source of span attributes. When not
        # passed in, resolve it from the config so there's one source of truth.
        self._mappers: list[AttributeMapper] = (
            list(mappers) if mappers is not None else resolve_mappers(config.mapper_names)
        )
        # Bounded LRU (ordered by insertion / most-recent touch). Storing keys
        # only — the value is unused — so it behaves like a capped set.
        self._emitted: OrderedDict[tuple[str, SpanRole], None] = OrderedDict()

    # -- low-level helpers --------------------------------------------------- #

    def start_span(
        self,
        role: SpanRole,
        name: str,
        parent_context: Context | None = None,
        start_time_ns: int | None = None,
        *,
        tracer: Tracer | None = None,
        links: Sequence[Link] | None = None,
    ) -> Span:
        """Start a span for ``role`` without dedup or attribute mapping.

        For callers that own and manage their own span lifecycle. ``tracer``
        overrides the bound tracer for this span only, used for per-request
        multi-tenant credential routing. ``links`` records related-but-not-parent
        spans (e.g. the trace context an MCP client propagated in ``params._meta``).
        """
        return (tracer or self._tracer).start_span(
            name,
            context=parent_context,
            kind=to_otel_span_kind(SPAN_REGISTRY[role].kind),
            start_time=start_time_ns,
            links=list(links) if links else None,
        )

    def mark_emitted(self, dedup_key: str | None, role: SpanRole) -> None:
        """Register a span emitted outside :meth:`emit` (the boundary-opened
        LLM-call span closed via :meth:`finish_span`) so a later :meth:`emit`
        for the same ``(dedup_key, role)`` deduplicates against it."""
        self._seen(dedup_key, role)

    def _seen(self, dedup_key: str | None, role: SpanRole) -> bool:
        """Return True once a ``(dedup_key, role)`` pair has been emitted.

        Guards against emitting the same span twice when a streaming call
        fires both a sync and an async logging callback.
        """
        if not dedup_key:
            return False
        marker: Final = (dedup_key, role)
        if marker in self._emitted:
            self._emitted.move_to_end(marker)
            return True
        self._emitted[marker] = None
        if len(self._emitted) > _DEDUP_CACHE_MAX:
            self._emitted.popitem(last=False)  # evict least-recently-used
        return False

    # -- the engine ---------------------------------------------------------- #

    def emit(
        self,
        role: SpanRole,
        data: SpanData,
        parent_context: Context | None = None,
        *,
        start_time_ns: int | None = None,
        end_time_ns: int | None = None,
        tracer: Tracer | None = None,
        links: Sequence[Link] | None = None,
    ) -> Span | None:
        """Emit one complete span: dedup, start, map attributes, status, end.

        Return the span, or ``None`` if it was deduplicated away. ``tracer``
        overrides the bound tracer for this span, used for per-request routing.
        ``links`` records related-but-not-parent spans (e.g. the trace context an
        MCP client propagated in ``params._meta``).
        """
        # LLM-call and MCP tool-call spans carry a dedup key (their request's
        # call id), so a sync+async double-firing coalesces. ``isinstance`` narrows
        # the type for mypy and keeps the engine free of duck-typed attribute reads.
        dedup_key: Final = (
            data.identity.call_id
            if isinstance(data, (LLMCallSpanData, MCPToolCallSpanData, MCPListToolsSpanData))
            else None
        )
        if self._seen(dedup_key, role):
            return None
        span: Final = self.start_span(
            role,
            _NAME_BUILDERS[role](data),
            parent_context=parent_context,
            start_time_ns=start_time_ns,
            tracer=tracer,
            links=links,
        )
        self.finish_span(role, span, data, end_time_ns=end_time_ns)
        return span

    def finish_span(
        self,
        role: SpanRole,
        span: Span,
        data: SpanData,
        *,
        end_time_ns: int | None = None,
    ) -> None:
        """Stamp attributes + status on an already-started ``span`` and end it.

        The counterpart to :meth:`start_span` for callers that own a span's
        lifecycle — the LLM-call span is opened at the request's ``pre_call``
        boundary (so it parents to the live server span via real ambient context,
        never a span threaded through a metadata dict) and closed here once the
        typed payload is available. The span name is (re)built from the now-known
        data, since the boundary opener only has a provisional name.
        """
        span.update_name(_NAME_BUILDERS[role](data))
        error: Final = (
            data.error
            if isinstance(
                data,
                (
                    LLMCallSpanData,
                    MCPToolCallSpanData,
                    MCPListToolsSpanData,
                    ServiceSpanData,
                    GuardrailSpanData,
                ),
            )
            else None
        )
        mapped: Final = MappingProxyType(
            {key: value for mapper in self._mappers for key, value in mapper.map(data).items()}
        )
        stamped_later: Final = error_attributes(error) if error else _NO_ATTRIBUTES
        reserved: Final = len(stamped_later.keys() - mapped.keys())
        for key, value in fit_indexed_messages(mapped, self._attribute_budget(span, reserved)).items():
            span.set_attribute(key, value)
        if error:
            stamped: Final = stamp_error(span, error)
            if stamped is not None and self._event_recorder is not None and role is SpanRole.LLM_CALL:
                error_type, message = stamped
                self._event_recorder.record_operation_exception(
                    span_context=span.get_span_context(),
                    error_type=error_type,
                    message=message,
                    stack_trace=error.stack_trace,
                    timestamp_ns=end_time_ns,
                )
        # On success leave the status UNSET (the semconv default) rather than
        # forcing OK — that matches the FastAPI server span and avoids implying a
        # span-level health signal litellm doesn't actually evaluate. Only a
        # genuine error sets a status.
        span.end(end_time=end_time_ns)

    def _attribute_budget(self, span: Span, reserved: int) -> int | None:
        """How many mapped attributes fit on ``span`` next to what it already carries and ``reserved`` more."""
        if self._span_attribute_limit is None:
            return None
        on_span: Final = len(span.attributes or ()) if isinstance(span, ReadableSpan) else 0
        return self._span_attribute_limit - on_span - reserved
