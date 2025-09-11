from opentelemetry import trace
from opentelemetry.context import (
    Context,
    get_current,
    set_value,
)
from opentelemetry.propagators.textmap import (
    CarrierT,
    Getter,
    Setter,
    TextMapPropagator,
    default_getter,
    default_setter,
)
from opentelemetry.trace import (
    NonRecordingSpan,
    SpanContext,
    TraceFlags,
)

from sentry_sdk.integrations.opentelemetry.consts import (
    SENTRY_BAGGAGE_KEY,
    SENTRY_TRACE_KEY,
)
from sentry_sdk.integrations.opentelemetry.span_processor import (
    SentrySpanProcessor,
)
from sentry_sdk.tracing import (
    BAGGAGE_HEADER_NAME,
    SENTRY_TRACE_HEADER_NAME,
)
from sentry_sdk.tracing_utils import Baggage, extract_sentrytrace_data

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Optional, Set


class SentryPropagator(TextMapPropagator):
    """
    Propagates tracing headers for Sentry's tracing system in a way OTel understands.
    """

    def extract(self, carrier, context=None, getter=default_getter):
        # type: (CarrierT, Optional[Context], Getter[CarrierT]) -> Context
        if context is None:
            context = get_current()

        sentry_trace = getter.get(carrier, SENTRY_TRACE_HEADER_NAME)
        if not sentry_trace:
            return context

        sentrytrace = extract_sentrytrace_data(sentry_trace[0])
        if not sentrytrace:
            return context

        context = set_value(SENTRY_TRACE_KEY, sentrytrace, context)

        trace_id, span_id = sentrytrace["trace_id"], sentrytrace["parent_span_id"]

        span_context = SpanContext(
            trace_id=int(trace_id, 16),  # type: ignore
            span_id=int(span_id, 16),  # type: ignore
            # we simulate a sampled trace on the otel side and leave the sampling to sentry
            trace_flags=TraceFlags(TraceFlags.SAMPLED),
            is_remote=True,
        )

        baggage_header = getter.get(carrier, BAGGAGE_HEADER_NAME)

        if baggage_header:
            baggage = Baggage.from_incoming_header(baggage_header[0])
        else:
            # If there's an incoming sentry-trace but no incoming baggage header,
            # for instance in traces coming from older SDKs,
            # baggage will be empty and frozen and won't be populated as head SDK.
            baggage = Baggage(sentry_items={})

        baggage.freeze()
        context = set_value(SENTRY_BAGGAGE_KEY, baggage, context)

        span = NonRecordingSpan(span_context)
        modified_context = trace.set_span_in_context(span, context)
        return modified_context

    def inject(self, carrier, context=None, setter=default_setter):
        # type: (CarrierT, Optional[Context], Setter[CarrierT]) -> None
        if context is None:
            context = get_current()

        current_span = trace.get_current_span(context)
        current_span_context = current_span.get_span_context()

        if not current_span_context.is_valid:
            return

        span_id = trace.format_span_id(current_span_context.span_id)

        span_map = SentrySpanProcessor().otel_span_map
        sentry_span = span_map.get(span_id, None)
        if not sentry_span:
            return

        setter.set(carrier, SENTRY_TRACE_HEADER_NAME, sentry_span.to_traceparent())

        if sentry_span.containing_transaction:
            baggage = sentry_span.containing_transaction.get_baggage()
            if baggage:
                baggage_data = baggage.serialize()
                if baggage_data:
                    setter.set(carrier, BAGGAGE_HEADER_NAME, baggage_data)

    @property
    def fields(self):
        # type: () -> Set[str]
        return {SENTRY_TRACE_HEADER_NAME, BAGGAGE_HEADER_NAME}
