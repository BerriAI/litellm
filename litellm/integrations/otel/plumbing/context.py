"""Trace-context + Baggage helpers."""

from contextvars import ContextVar, Token
from typing import Mapping

from opentelemetry import baggage
from opentelemetry.context import Context, get_current
from opentelemetry.trace import Link, Span, get_current_span, set_span_in_context
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
    """Parent context + links for an MCP message span, per the OTel GenAI MCP semconv.

    MCP and the underlying transport (HTTP) are independent lifecycles — one
    streamable-HTTP session multiplexes many messages, so nesting the message span
    under the HTTP/session span is wrong (it renders the message at the session's
    start, skewed by however long the session has been open). Instead:

    * parent to the trace context the client propagated in the request's
      ``params._meta`` (a *remote* parent), and
    * record the transport/session span as a *link*, never the parent.

    Only trace context (``traceparent``/``tracestate``) is extracted, never the
    client's W3C Baggage: ``params._meta`` is caller-controlled, and the otel
    baggage processor stamps allowlisted baggage keys (``litellm.team.id``,
    ``litellm.metadata.*``, ...) onto the span as attributes, so honoring remote
    baggage would let a client spoof a span's identity attribution.

    With no propagated context the returned context carries no span, so the span
    starts its own root trace (still linked to the transport). The base context is
    explicitly empty so an absent ``traceparent`` can never fall through to the
    ambient (stale session) span.
    """
    source = carrier if carrier is not None else _mcp_message_trace_carrier.get()
    parent = _PROPAGATOR.extract(dict(source or {}), context=Context())
    transport = request_root_span()
    links = (Link(transport.get_span_context()),) if transport is not None else ()
    return parent, links


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
