from typing import Any  # noqa:F401
from typing import Dict  # noqa:F401
from typing import Optional  # noqa:F401

from opentracing import SpanContext as OpenTracingSpanContext

from ddtrace._trace.context import Context as DatadogContext
from ddtrace.internal.compat import NumericType  # noqa:F401


class SpanContext(OpenTracingSpanContext):
    """Implementation of the OpenTracing span context."""

    def __init__(
        self,
        trace_id=None,  # type: Optional[int]
        span_id=None,  # type: Optional[int]
        sampling_priority=None,  # type: Optional[NumericType]
        baggage=None,  # type: Optional[Dict[str, Any]]
        ddcontext=None,  # type: Optional[DatadogContext]
    ):
        # type: (...) -> None
        # create a new dict for the baggage if it is not provided
        # NOTE: it would be preferable to use opentracing.SpanContext.EMPTY_BAGGAGE
        #       but it is mutable.
        # see: opentracing-python/blob/8775c7bfc57fd66e1c8bcf9a54d3e434d37544f9/opentracing/span.py#L30
        baggage = baggage or {}

        if ddcontext is not None:
            self._dd_context = ddcontext
        else:
            self._dd_context = DatadogContext(
                trace_id=trace_id,
                span_id=span_id,
                sampling_priority=sampling_priority,
            )

        self._baggage = dict(baggage)

    @property
    def baggage(self):
        # type: () -> Dict[str, Any]
        return self._baggage

    def set_baggage_item(self, key, value):
        # type: (str, Any) -> None
        """Sets a baggage item in this span context.

        Note that this operation mutates the baggage of this span context
        """
        self.baggage[key] = value

    def with_baggage_item(self, key, value):
        # type: (str, Any) -> SpanContext
        """Returns a copy of this span with a new baggage item.

        Useful for instantiating new child span contexts.
        """
        baggage = dict(self._baggage)
        baggage[key] = value
        return SpanContext(ddcontext=self._dd_context, baggage=baggage)

    def get_baggage_item(self, key):
        # type: (str) -> Optional[Any]
        """Gets a baggage item in this span context."""
        return self.baggage.get(key, None)
