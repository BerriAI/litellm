"""Trace-context + Baggage helpers."""

from contextvars import ContextVar, Token
from typing import Mapping

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

_PROPAGATOR = TraceContextTextMapPropagator()

# The request's root span — the FastAPI-owned SERVER span — captured ONCE when the
# proxy first resolves it, so request-level spans (the LLM call, guardrails) can
# parent to it EXPLICITLY instead of to whatever span happens to be active at the
# instant they are emitted. Ambient-only parenting (``get_current_span()``) is
# wrong at two boundaries:
#   * inside the ``auth`` phase span the active span is the auth span, so an LLM /
#     guardrail span emitted there would nest under auth instead of being its
#     sibling; and
#   * in a detached success task (pass-through logs success from a fire-and-forget
#     ``asyncio.create_task``) the server span may not be active at all, orphaning
#     the span into a brand-new trace.
# A ``ContextVar`` (not a request attribute) so it rides the request task's context
# and is inherited by ``asyncio.create_task`` children — i.e. the async logging
# callbacks that close the span. It is never reset: the contextvar dies with the
# request task, so there is nothing to leak.
_request_root_span: "ContextVar[Span | None]" = ContextVar("litellm_otel_request_root_span", default=None)


def set_request_root_span(span: Span) -> None:
    """Anchor the request's root (server) span for explicit child parenting.

    No-ops for a non-recordable span so a bad capture can never replace a good one
    with a phantom parent. Idempotent — the proxy captures the same server span at
    more than one entry point.
    """
    if is_recordable_span(span):
        _request_root_span.set(span)


def request_root_span() -> "Span | None":
    """The anchored request root span, or ``None`` outside a proxy request."""
    span = _request_root_span.get()
    return span if is_recordable_span(span) else None


# The W3C trace-context carrier (``traceparent``/``tracestate``/``baggage``) the
# MCP client propagated in the current request's ``params._meta``. The MCP gateway
# sets it per message so the MCP span can parent to the client's span rather than
# to the transport. A ``ContextVar`` because, like the root-span anchor, it must
# ride the request task and be readable by the inline success-logging callback.
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
#
# ``_request_root_span`` above cannot be used for MCP: a *stateful* streamable-HTTP
# session runs every message on the single task spawned by that session's
# ``initialize`` POST, so the ContextVar the ASGI request task writes at auth time
# is frozen at ``initialize`` there and never sees the later ``tools/call`` POSTs.
# Reading it from the message handler parents every tool call in the session to the
# first request's server span and aims that call's ``error.*`` at it — a span that
# ended long ago, so the SDK drops the write and the failure reaches no request at
# all. The gateway instead resolves the current message's transport span on the
# request task and hands it over the same way it hands over per-request auth, and
# the handler publishes it here for the span emitter and the failure hook.
_mcp_message_transport_span: "ContextVar[Span | None]" = ContextVar(
    "litellm_otel_mcp_message_transport_span", default=None
)


def set_mcp_message_transport_span(span: object) -> "Token[Span | None]":
    """Publish the transport span of the request carrying the current MCP message.

    Also re-anchors the request root, so everything else the message emits or stamps
    — the identity attributes seeded onto the server span, a guardrail span, a
    proxy-level failure — lands on this request instead of on the one that opened
    the session. The MCP SDK dispatches each message on its own task, so the anchor
    is scoped to this message; the handler re-publishes it for the next one either
    way. Only a transport still open for writes is anchored: replacing the anchor
    with a request that already answered would just move the dropped writes from one
    finished span to another.

    Takes ``object`` because the gateway reads it back out of the ASGI scope, whose
    values are untyped; anything that is not a usable span is stored as ``None``
    rather than trusted.

    Returns the reset token; the caller must reset it once the message is handled
    so the transport never leaks to the next message on the same session task.
    """
    transport = span if isinstance(span, Span) and is_recordable_span(span) else None
    if transport is not None and transport.is_recording():
        set_request_root_span(transport)
    return _mcp_message_transport_span.set(transport)


def reset_mcp_message_transport_span(token: "Token[Span | None]") -> None:
    _mcp_message_transport_span.reset(token)


def mcp_message_transport_span() -> "Span | None":
    """The published transport span, only while it is still open for writes.

    Recording — not merely valid — is the bar here because this span is the target
    of ``error.*`` stamping from another task, and the publisher's validity check
    cannot speak for a span that has since ended. A finished span keeps a valid
    context forever, so it would otherwise be handed back for a write the SDK then
    refuses. The POST carrying a ``tools/call`` stays open until the result is
    written, so it is recording for the life of the call; a notification POST can
    answer first, and this returns ``None`` for it rather than writing into the void.
    """
    span = _mcp_message_transport_span.get()
    if span is None or not span.is_recording():
        return None
    return span


def _mcp_transport_span_context() -> "SpanContext | None":
    """The transport span an MCP message span should attach to.

    Prefers the transport the gateway published for this specific message; falls
    back to the ambient request anchor for paths that emit an MCP span on the
    request task itself (the REST MCP endpoints, the SDK). Parenting and linking
    only need the immutable context, and unlike ``mcp_message_transport_span`` they
    stay correct against a transport that has already finished, so this does not
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

    Ambient-first: parent to the active OTel context (the server span, restored
    by the logging worker or active in the request task), falling back to a span
    passed explicitly (``threaded``) only when the ambient context has no
    recordable span — e.g. a background service call with no request on the
    stack. When neither is recordable the ambient context is returned unchanged,
    so the span starts a new root trace.

    Only service/DB spans pass ``threaded`` (the ``parent_otel_span`` handed to
    the service hook). Request-level spans — the LLM call and guardrails — are
    created where the server span is genuinely ambient, so they never need it.
    """
    ctx = get_current()
    if is_recordable_span(threaded) and not is_recordable_span(get_current_span(ctx)):
        ctx = context_from_span(threaded, context=ctx)  # type: ignore[arg-type]
    return ctx


def resolve_request_span_context() -> Context:
    """The parent context for a request-level span (the LLM call, a guardrail).

    These are direct children of the request's root server span — siblings of the
    ``auth`` phase span and of each other, never nested under whatever span is
    momentarily active. So prefer the explicitly anchored root span; fall back to
    ambient context only when there is no anchor (the SDK / no-proxy path), where
    the span legitimately starts its own root trace.

    Unlike :func:`resolve_parent_context` (used by DB/service spans, which DO want
    to nest under the active phase span, e.g. an auth DB lookup under ``auth``),
    this never returns the active span when an anchor exists.
    """
    root = request_root_span()
    if root is not None:
        return context_from_span(root)
    return get_current()


def resolve_mcp_span_context(
    carrier: "Mapping[str, str] | None" = None,
) -> "tuple[Context, tuple[Link, ...]]":
    """Parent context + links for an MCP message span.

    When the client propagates W3C trace context in the request's ``params._meta``
    (SEP-414), MCP and the underlying transport are independent lifecycles — one
    streamable-HTTP session multiplexes many messages, and the client's own span is
    the truthful parent. So, per the OTel GenAI MCP semconv:

    * parent to the trace context the client propagated (a *remote* parent), and
    * record the transport span as a *link*, never the parent.

    Almost no client implements SEP-414 yet, so in practice nothing is propagated.
    Rooting the span there splits a single tool call into two disconnected traces
    joined only by a link, which is how it surfaces in APM: the ``POST`` transaction
    and the ``tools/call`` span share no trace. With no remote parent to honor,
    parent to the transport span of the request carrying this message instead, so
    the call stays in one trace; no link is added since the transport is now the
    real parent. The transport comes from :func:`_mcp_transport_span_context`, which
    is the *current message's* POST rather than whatever request happened to open
    the session, so a long-lived session does not glue every message under its
    first request. With neither a remote parent nor a transport the returned context
    carries no span and the span legitimately starts its own root trace.

    Only trace context (``traceparent``/``tracestate``) is extracted, never the
    client's W3C Baggage: ``params._meta`` is caller-controlled, and the otel
    baggage processor stamps allowlisted baggage keys (``litellm.team.id``,
    ``litellm.metadata.*``, ...) onto the span as attributes, so honoring remote
    baggage would let a client spoof a span's identity attribution. The base context
    for extraction is explicitly empty so an absent or malformed ``traceparent`` can
    never fall through to the ambient (stale session) span.
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
