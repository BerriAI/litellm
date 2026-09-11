"""Trace-context + Baggage helpers."""

import os
from collections.abc import Mapping
from contextvars import ContextVar, Token
from typing import TYPE_CHECKING, Final

from opentelemetry import baggage
from opentelemetry.context import Context, get_current
from opentelemetry.sdk.trace import ReadableSpan
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

from litellm.integrations.otel.model.semconv import HTTP

if TYPE_CHECKING:
    from litellm.integrations.otel.model.destination import OtelDestination

_PROPAGATOR: Final = TraceContextTextMapPropagator()

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
_request_root_span: Final["ContextVar[Span | None]"] = ContextVar("litellm_otel_request_root_span", default=None)


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
    span: Final = _request_root_span.get()
    return span if is_recordable_span(span) else None


def request_root_http_route() -> str | None:
    """``http.route`` exactly as the request's root SERVER span reports it.

    Read off the span rather than re-derived, so the LLM call span cannot disagree
    with its own parent about which endpoint served the request: the template the
    instrumentation matched, or the literal path where
    ``mount._passthrough_span_name_hook`` rewrote it, are already in the attribute.
    An MCP call anchors that same server span, so it reports the ``/mcp`` mount
    point the instrumentation matched. Attributes stay readable after a span ends,
    so this answers just as well from the async logging callback.

    None when no server span is anchored, which is the SDK path and any deployment
    where the FastAPI instrumentation did not mount.
    """
    span: Final = request_root_span()
    route: Final = span.attributes.get(HTTP.ROUTE) if isinstance(span, ReadableSpan) and span.attributes else None
    return route if isinstance(route, str) and route else None


# The W3C trace-context carrier (``traceparent``/``tracestate``/``baggage``) the
# MCP client propagated in the current request's ``params._meta``. The MCP gateway
# sets it per message so the MCP span can record the client's span as a span
# link. A ``ContextVar`` because, like the root-span anchor, it must
# ride the request task and be readable by the inline success-logging callback.
_mcp_message_trace_carrier: Final["ContextVar[Mapping[str, str] | None]"] = ContextVar(
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
_mcp_message_transport_span: Final["ContextVar[Span | None]"] = ContextVar(
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
    transport: Final = span if isinstance(span, Span) and is_recordable_span(span) else None
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
    span: Final = _mcp_message_transport_span.get()
    if span is None or not span.is_recording():
        return None
    return span


def _mcp_transport_span_context() -> "SpanContext | None":
    """The transport span an MCP message span should attach to.

    Prefers the transport the gateway published for this specific message; falls
    back to the ambient request anchor for paths that emit an MCP span on the
    request task itself (the REST MCP endpoints). Parenting needs only the
    immutable context, and unlike ``mcp_message_transport_span`` it stays correct
    against a transport that has already finished, so this does not require the
    span to still be recording.
    """
    published: Final = _mcp_message_transport_span.get()
    if published is not None:
        return published.get_span_context()
    span: Final = request_root_span()
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
        ctx = context_from_span(threaded, context=ctx)
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
    root: Final = request_root_span()
    if root is not None:
        return context_from_span(root)
    return get_current()


def resolve_mcp_span_context(
    carrier: "Mapping[str, str] | None" = None,
) -> "tuple[Context, tuple[Link, ...]]":
    """Parent context + links for an MCP message span.

    The span always nests under the transport span of the request carrying this
    message, so a tool call and the ``POST`` that carried it stay in one trace.
    The transport comes from :func:`_mcp_transport_span_context`, which is the
    *current message's* POST rather than whatever request happened to open the
    session, so a long-lived session does not glue every message under its first
    request.

    When the client propagates W3C trace context in the request's ``params._meta``
    (SEP-414), that remote context is recorded as a span *link*, never the parent.
    The OTel GenAI MCP semconv prefers the inverse (remote parent, transport link),
    but the gateway's tracing backend only ever receives the gateway's half of such
    a trace: parenting into the client's trace id roots the span in a trace whose
    root span never reaches the backend, so the span is unreachable from the trace
    view and the transport transaction shows a dangling link (observed with
    clients that propagate synthetic trace ids). Anchoring to the gateway's own
    request and linking the client's context keeps every trace renderable while
    preserving the client-side correlation.

    With no transport at all the span starts its own root trace, still carrying
    the link — the client context is only ever a link, so this event keeps one
    shape everywhere. Both returned contexts are built on an explicitly empty
    base, so ambient (stale session) state can never leak in, and the span
    inherits the transport's sampling decision exactly like every other
    request-level span — a client's sampled flag neither forces nor suppresses
    recording.

    Only trace context (``traceparent``/``tracestate``) is extracted, never the
    client's W3C Baggage: ``params._meta`` is caller-controlled, and the otel
    baggage processor stamps allowlisted baggage keys (``litellm.team.id``,
    ``litellm.metadata.*``, ...) onto the span as attributes, so honoring remote
    baggage would let a client spoof a span's identity attribution. The base context
    for extraction is explicitly empty so an absent or malformed ``traceparent`` can
    never fall through to the ambient (stale session) span.
    """
    source: Final = carrier if carrier is not None else _mcp_message_trace_carrier.get()
    propagated: Final = get_current_span(_PROPAGATOR.extract(dict(source or {}), context=Context()))
    links: Final = (Link(propagated.get_span_context()),) if is_recordable_span(propagated) else ()
    transport: Final = _mcp_transport_span_context()
    if transport is None:
        return Context(), links
    return context_from_span(NonRecordingSpan(transport), context=Context()), links


def is_recordable_span(obj: object) -> bool:
    """True if ``obj`` is a live span with a valid context (safe to parent under)."""
    if not isinstance(obj, Span):
        return False
    try:
        ctx: Final = obj.get_span_context()
    except Exception:
        return False
    return ctx is not None and ctx.is_valid


def extract_traceparent(headers: Mapping[str, str]) -> Context | None:
    """Extract a remote parent context from incoming HTTP headers, if present."""
    if not any(key.lower() == "traceparent" for key in headers):
        return None
    carrier: Final = {str(key).lower(): value for key, value in headers.items()}
    return _PROPAGATOR.extract(carrier)


# The OTLP destinations this request's key or team pointed its traces at, resolved
# once during auth. A ``ContextVar`` for the same reason the root span above is one:
# it rides the request task's context into the ``asyncio.create_task`` children that
# close the LLM span, and it is visible to every ``SpanProcessor.on_end`` that fires
# on the request task. Stateful MCP handlers set and reset it per message; the
# request-task value otherwise dies with that task.
_request_destinations: Final['ContextVar[tuple["OtelDestination", ...]]'] = ContextVar(
    "litellm_otel_request_destinations", default=()
)


def set_request_destinations(destinations: 'tuple["OtelDestination", ...]') -> "Token[tuple[OtelDestination, ...]]":
    """Anchor the destinations this request exports to and return a reset token."""
    return _request_destinations.set(destinations)


def reset_request_destinations(token: "Token[tuple[OtelDestination, ...]]") -> None:
    _request_destinations.reset(token)


def request_destinations() -> 'tuple["OtelDestination", ...]':
    """The destinations resolved for this request, empty outside a proxy request."""
    return _request_destinations.get()


#: ``litellm_settings: otel_tenant_destination_mode`` and its env equivalent.
ADDITIVE_DESTINATION_MODE: Final = "additive"
OTEL_TENANT_DESTINATION_MODE_ENV: Final = "LITELLM_OTEL_TENANT_DESTINATION_MODE"


def tenant_destinations_are_additive() -> bool:
    """Whether a tenant destination exports alongside the operator's own exporter.

    Override is the default: the tenant's traffic reaches the tenant's account and
    nowhere else. Operators running one org-wide backend across every team set this
    to ``additive`` so the same trace lands in both places.
    """
    import litellm

    configured: Final = litellm.otel_tenant_destination_mode or os.environ.get(OTEL_TENANT_DESTINATION_MODE_ENV)
    return isinstance(configured, str) and configured.strip().lower() == ADDITIVE_DESTINATION_MODE


def destination_backends() -> frozenset[str]:
    """Backends this request resolved a tenant destination for.

    The fan-out already carries the whole trace to those destinations, so the
    per-request tracer route must never send a second copy, in either mode.
    """
    return frozenset(d.callback_name for d in _request_destinations.get() if d.callback_name)


def suppressed_backends() -> frozenset[str]:
    """Backends whose operator-level exporters this request must NOT reach.

    Empty under ``additive``, where the operator keeps its copy of every span.
    """
    if tenant_destinations_are_additive():
        return frozenset()
    return destination_backends()
