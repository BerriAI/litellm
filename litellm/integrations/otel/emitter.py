"""The span engine: dedup, start, run the mapper chain, set status, end."""

from collections import OrderedDict
from typing import Callable, Sequence

from opentelemetry.context import Context
from opentelemetry.trace import Link, Span, Tracer
from opentelemetry.trace.status import Status, StatusCode

from litellm.integrations.otel.model.config import OpenTelemetryV2Config
from litellm.integrations.otel.mappers import resolve_mappers
from litellm.integrations.otel.mappers.base import AttributeMapper, SpanData
from litellm.integrations.otel.model.payloads import (
    GuardrailSpanData,
    LLMCallSpanData,
    MCPListToolsSpanData,
    MCPToolCallSpanData,
    ServiceSpanData,
    SpanError,
)
from litellm.integrations.otel.plumbing.events import GenAIEventRecorder
from litellm.integrations.otel.plumbing.providers import to_otel_span_kind
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

# Roles emit() knows how to name and emit. PROXY_REQUEST and the management
# routes are SERVER spans owned by the mounted FastAPI instrumentor, so they
# have no builder here.
_NAME_BUILDERS: dict[SpanRole, Callable[..., str]] = {
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
_DEDUP_CACHE_MAX = 10_000


def _stamp_otel_error_attributes(span: Span, error_type: str, resolved_message: str) -> None:
    """Stamp the OTel-semconv error attributes (``error.type`` + ``error.message``).
    ``error_type`` and ``resolved_message`` are ``finish_span``'s already-computed
    fallback chains, so the pair on the status, event, and attributes stays in
    lockstep."""
    span.set_attribute(Error.TYPE, error_type)
    span.set_attribute(Error.MESSAGE, resolved_message)


def _stamp_litellm_error_attributes(span: Span, error: SpanError) -> None:
    """Stamp litellm-specific error detail attributes. Emitted only when the
    corresponding field is populated so guardrail-shape errors carrying only a
    message aren't polluted with empty detail keys."""
    if error.code:
        span.set_attribute(LiteLLMError.CODE, error.code)
    if error.stack_trace:
        span.set_attribute(LiteLLMError.STACK_TRACE, error.stack_trace)
    if error.llm_provider:
        span.set_attribute(LiteLLMError.LLM_PROVIDER, error.llm_provider)


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
        # The mapper chain is the sole source of span attributes. When not
        # passed in, resolve it from the config so there's one source of truth.
        self._mappers: list[AttributeMapper] = (
            list(mappers) if mappers is not None else resolve_mappers(config.mapper_names)
        )
        # Bounded LRU (ordered by insertion / most-recent touch). Storing keys
        # only — the value is unused — so it behaves like a capped set.
        self._emitted: "OrderedDict[tuple[str, SpanRole], None]" = OrderedDict()

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
        spans (e.g. the transport span of an MCP message, per MCP semconv).
        """
        return (tracer or self._tracer).start_span(
            name,
            context=parent_context,
            kind=to_otel_span_kind(SPAN_REGISTRY[role].kind),
            start_time=start_time_ns,
            links=list(links) if links else None,
        )

    def _seen(self, dedup_key: str | None, role: SpanRole) -> bool:
        """Return True once a ``(dedup_key, role)`` pair has been emitted.

        Guards against emitting the same span twice when a streaming call
        fires both a sync and an async logging callback.
        """
        if not dedup_key:
            return False
        marker = (dedup_key, role)
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
        ``links`` records related-but-not-parent spans (the transport span of an
        MCP message).
        """
        # LLM-call and MCP tool-call spans carry a dedup key (their request's
        # call id), so a sync+async double-firing coalesces. ``isinstance`` narrows
        # the type for mypy and keeps the engine free of duck-typed attribute reads.
        dedup_key = (
            data.identity.call_id
            if isinstance(data, (LLMCallSpanData, MCPToolCallSpanData, MCPListToolsSpanData))
            else None
        )
        if self._seen(dedup_key, role):
            return None
        span = self.start_span(
            role,
            _NAME_BUILDERS[role](data),
            parent_context=parent_context,
            start_time_ns=start_time_ns,
            tracer=tracer,
            links=links,
        )
        self.finish_span(role, span, data, end_time_ns=end_time_ns)
        return span

    def emit_fanout(
        self,
        role: SpanRole,
        data: SpanData,
        parent_context: Context | None = None,
        *,
        start_time_ns: int | None = None,
        end_time_ns: int | None = None,
        tracers: Sequence[Tracer],
    ) -> Span | None:
        """Emit one logical span once per tracer, deduping the call ONCE.

        A backend that selects its target from the Resource (Arize's project) needs a
        separately-tagged span per Resource group, so ``tracers`` is one tracer per group
        (from ``TenantTracerCache.tracers_for``). Dedup runs once on the call id so a
        sync+async double-firing still coalesces, then a span is started and finished on
        each tracer (each provider stamps its own Resource). Returns the first span (or
        ``None`` if deduped/empty).
        """
        dedup_key = data.identity.call_id if isinstance(data, (LLMCallSpanData, MCPToolCallSpanData)) else None
        if self._seen(dedup_key, role):
            return None
        name = _NAME_BUILDERS[role](data)
        first: Span | None = None
        for tracer in tracers:
            span = self.start_span(
                role,
                name,
                parent_context=parent_context,
                start_time_ns=start_time_ns,
                tracer=tracer,
            )
            self.finish_span(role, span, data, end_time_ns=end_time_ns)
            if first is None:
                first = span
        return first

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
        for mapper in self._mappers:
            for key, value in mapper.map(data).items():
                span.set_attribute(key, value)
        error = (
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
        if error and (error.error_type or error.message):
            error_type = error.error_type or "error"
            message = error.message or error.error_type or "error"
            _stamp_otel_error_attributes(span, error_type, message)
            _stamp_litellm_error_attributes(span, error)
            span.set_status(Status(StatusCode.ERROR, message))
            # Also emit the semconv ``exception`` event so backends that
            # dynamic-map unknown string span attrs to ``keyword`` (e.g.
            # Elasticsearch with a 1024-char ``ignore_above``) still see the
            # full untruncated message on the recognized event field.
            span.add_event(
                ExceptionEvent.NAME,
                {ExceptionEvent.TYPE: error_type, ExceptionEvent.MESSAGE: message},
            )
            if self._event_recorder is not None and role is SpanRole.LLM_CALL:
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
