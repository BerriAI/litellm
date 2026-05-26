"""Trace-context + Baggage helpers.

Loaded only when OpenTelemetry is in use, so it imports the SDK at module level.
Baggage is the mechanism by which request-scoped identity (team, model, an
allowlisted metadata subset) is propagated so a single span processor can stamp
it onto every span — instead of per-call-site duplication.
"""

from typing import Dict, Mapping, Optional

from opentelemetry import baggage
from opentelemetry.context import Context, get_current
from opentelemetry.trace import Span, set_span_in_context
from opentelemetry.trace.propagation.tracecontext import (
    TraceContextTextMapPropagator,
)

_PROPAGATOR = TraceContextTextMapPropagator()


def set_request_baggage(
    values: Mapping[str, str], context: Optional[Context] = None
) -> Context:
    """Return a context with ``values`` written into Baggage."""
    ctx = context
    for key, value in values.items():
        ctx = baggage.set_baggage(key, value, context=ctx)
    return ctx if ctx is not None else (context or get_current())


def get_baggage_attributes(context: Optional[Context] = None) -> Dict[str, str]:
    """All Baggage entries on ``context`` as strings."""
    return {key: str(value) for key, value in baggage.get_all(context).items()}


def context_from_span(span: Span, context: Optional[Context] = None) -> Context:
    """A context with ``span`` as the active span (for explicit parenting)."""
    return set_span_in_context(span, context=context)


def extract_traceparent(headers: Mapping[str, str]) -> Optional[Context]:
    """Extract a remote parent context from incoming HTTP headers, if present."""
    if not any(key.lower() == "traceparent" for key in headers):
        return None
    carrier = {str(key).lower(): value for key, value in headers.items()}
    return _PROPAGATOR.extract(carrier)
