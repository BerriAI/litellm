"""Trace-context + Baggage helpers."""

from contextvars import ContextVar, Token
from typing import TYPE_CHECKING, Mapping

from opentelemetry import baggage
from opentelemetry.context import Context, get_current
from opentelemetry.trace import (
    Link,
    NonRecordingSpan,
    Span,
    SpanContext,
    get_current_span,
    set_span_in_context,
)
from opentelemetry.trace.propagation.tracecontext import (
    TraceContextTextMapPropagator,
)

if TYPE_CHECKING:
    from litellm.integrations.otel.model.destination import OtelDestination

_PROPAGATOR = TraceContextTextMapPropagator()

# The request's root span (the FastAPI-owned SERVER span), captured once so request-level
# spans (LLM call, guardrails) parent to it explicitly rather than to whatever span is active —
# which under the ``auth`` phase span or a detached success task would misnest or orphan them.
# A ``ContextVar`` so it rides the request task and its ``create_task`` children (the async
# logging callbacks that close the span).
_request_root_span: "ContextVar[Span | None]" = ContextVar("litellm_otel_request_root_span", default=None)

_request_destinations: 'ContextVar[tuple["OtelDestination", ...]]' = ContextVar(
    "litellm_otel_request_destinations", default=()
)


def set_request_destinations(
    destinations: 'tuple["OtelDestination", ...]',
) -> None:
    """Anchor the admin-resolved destinations for this request."""
    _request_destinations.set(tuple(destinations))


def request_destinations() -> 'tuple["OtelDestination", ...]':
    """Destinations the request fans out to, or empty when none were resolved."""
    return _request_destinations.get()


def set_request_root_span(span: Span) -> None:
    """Anchor the request's root (server) span for explicit child parenting.

    No-ops for a non-recordable span (a bad capture can't replace a good one); idempotent.
    """
    if is_recordable_span(span):
        _request_root_span.set(span)


def request_root_span() -> "Span | None":
    """The anchored request root span, or ``None`` outside a proxy request."""
    span = _request_root_span.get()
    return span if is_recordable_span(span) else None


# The W3C trace-context carrier (``traceparent``/``tracestate``/``baggage``) the MCP client
# propagated in the current request's ``params._meta``; set per message. A ``ContextVar`` so it
# rides the request task and is readable by the inline success-logging callback.
_mcp_message_trace_carrier: "ContextVar[Mapping[str, str] | None]" = ContextVar(
    "litellm_otel_mcp_message_trace_carrier", default=None
)


def set_mcp_message_trace_carrier(
    carrier: "Mapping[str, str] | None",
) -> "Token[Mapping[str, str] | None]":
    """Stash the current MCP message's propagated trace-context carrier.

    Returns the reset token; the caller must reset it once the message is handled
    so the carrier never leaks to the next message on the same session task.
    """
    return _mcp_message_trace_carrier.set(carrier)


def reset_mcp_message_trace_carrier(token: "Token[Mapping[str, str] | None]") -> None:
    _mcp_message_trace_carrier.reset(token)


# The transport span of the HTTP request carrying the CURRENT MCP message.
# ``_request_root_span`` can't serve MCP: a stateful streamable-HTTP session runs every message
# on the task of its ``initialize`` POST, so that anchor is frozen at ``initialize`` and never
# sees later ``tools/call`` POSTs. The gateway instead resolves each message's transport span on
# the request task and the handler publishes it here.
_mcp_message_transport_span: "ContextVar[Span | None]" = ContextVar(
    "litellm_otel_mcp_message_transport_span", default=None
)


def set_mcp_message_transport_span(span: object) -> "Token[Span | None]":
    """Publish the transport span of the request carrying the current MCP message.

    Also re-anchors the request root so everything the message emits lands on this request, not
    the one that opened the session; only a transport still open for writes is anchored. Takes
    ``object`` (untyped ASGI-scope value); a non-span is stored as ``None``. Returns the reset
    token, which the caller must reset once the message is handled to avoid leaking it.
    """
    transport = span if isinstance(span, Span) and is_recordable_span(span) else None
    if transport is not None and transport.is_recording():
        set_request_root_span(transport)
    return _mcp_message_transport_span.set(transport)


def reset_mcp_message_transport_span(token: "Token[Span | None]") -> None:
    _mcp_message_transport_span.reset(token)


def mcp_message_transport_span() -> "Span | None":
    """The published transport span, only while it is still recording (open for writes).

    Recording — not merely valid — is the bar because this span is the target of ``error.*``
    stamping from another task, and a finished span keeps a valid context forever but would
    refuse the write. Returns ``None`` for a transport that has already answered.
    """
    span = _mcp_message_transport_span.get()
    if span is None or not span.is_recording():
        return None
    return span


def _mcp_transport_span_context() -> "SpanContext | None":
    """The transport span an MCP message span should attach to.

    Prefers the transport the gateway published for this message; falls back to the ambient
    request anchor for paths that emit an MCP span on the request task (REST MCP endpoints, SDK).
    Only the immutable context is needed, so unlike ``mcp_message_transport_span`` this does not
    require the span to still be recording.
    """
    published = _mcp_message_transport_span.get()
    if published is not None:
        return published.get_span_context()
    span = request_root_span()
    return span.get_span_context() if span is not None else None


def set_request_baggage(values: Mapping[str, str], context: Context | None = None) -> Context:
    """Return a context with ``values`` written into Baggage."""
    ctx = context
    for key, value in values.items():
        ctx = baggage.set_baggage(key, value, context=ctx)
    return ctx if ctx is not None else (context or get_current())


def get_baggage_attributes(context: Context | None = None) -> dict[str, str]:
    """All Baggage entries on ``context`` as strings."""
    return {key: str(value) for key, value in baggage.get_all(context).items()}


def context_from_span(span: Span, context: Context | None = None) -> Context:
    """A context with ``span`` as the active span (for explicit parenting)."""
    return set_span_in_context(span, context=context)


def resolve_parent_context(threaded: Span | None = None) -> Context:
    """The context a child span should parent under.

    Ambient-first: parent to the active OTel context, falling back to an explicitly passed
    ``threaded`` span only when the ambient context has no recordable span (a background service
    call with no request on the stack); when neither is recordable the ambient context is returned
    unchanged, so the span starts a new root trace. Only service/DB spans pass ``threaded``.
    """
    ctx = get_current()
    if is_recordable_span(threaded) and not is_recordable_span(get_current_span(ctx)):
        ctx = context_from_span(threaded, context=ctx)  # type: ignore[arg-type]
    return ctx


def resolve_request_span_context() -> Context:
    """The parent context for a request-level span (the LLM call, a guardrail).

    These are direct children of the request's root server span, never nested under whatever span
    is momentarily active, so prefer the anchored root span; fall back to ambient only on the
    SDK/no-proxy path with no anchor. Unlike :func:`resolve_parent_context`, this never returns
    the active span when an anchor exists.
    """
    root = request_root_span()
    if root is not None:
        return context_from_span(root)
    return get_current()


def resolve_mcp_span_context(
    carrier: "Mapping[str, str] | None" = None,
) -> "tuple[Context, tuple[Link, ...]]":
    """Parent context + links for an MCP message span.

    Per the OTel GenAI MCP semconv: when the client propagates W3C trace context in
    ``params._meta`` (SEP-414), parent to that remote context and link the transport span.
    Almost no client implements SEP-414, so with no remote parent, parent to this message's
    transport span (from :func:`_mcp_transport_span_context`, the current message's POST, so a
    long-lived session doesn't glue every message under its first request) and add no link. With
    neither, the span starts its own root trace.

    Only ``traceparent``/``tracestate`` is extracted, never the client's Baggage: ``params._meta``
    is caller-controlled and honoring remote baggage would let a client spoof identity attribution.
    The extraction base context is empty so a malformed ``traceparent`` can't fall through to the
    ambient (stale session) span.
    """
    source = carrier if carrier is not None else _mcp_message_trace_carrier.get()
    parent = _PROPAGATOR.extract(dict(source or {}), context=Context())
    transport = _mcp_transport_span_context()
    if is_recordable_span(get_current_span(parent)):
        return parent, (Link(transport),) if transport is not None else ()
    if transport is not None:
        return context_from_span(NonRecordingSpan(transport)), ()
    return parent, ()


def is_recordable_span(obj: object) -> bool:
    """True if ``obj`` is a live span with a valid context (safe to parent under)."""
    if not isinstance(obj, Span):
        return False
    try:
        ctx = obj.get_span_context()
    except Exception:
        return False
    return ctx is not None and ctx.is_valid


def extract_traceparent(headers: Mapping[str, str]) -> Context | None:
    """Extract a remote parent context from incoming HTTP headers, if present."""
    if not any(key.lower() == "traceparent" for key in headers):
        return None
    carrier = {str(key).lower(): value for key, value in headers.items()}
    return _PROPAGATOR.extract(carrier)
