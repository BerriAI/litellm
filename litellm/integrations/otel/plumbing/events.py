"""GenAI client events: the ``gen_ai.client.operation.exception`` log event.

The GenAI semantic conventions define exception recording for client
operations as a log-based event (severity WARN) carrying the ``exception.*``
attribute trio, correlated to the failed span through the trace/span ids —
not as a span attribute or span event. This module owns building and
emitting that event; the exporter pipeline it rides is built in
:mod:`litellm.integrations.otel.plumbing.providers`.
"""

from dataclasses import dataclass

from opentelemetry._events import Event, EventLogger
from opentelemetry._logs.severity import SeverityNumber
from opentelemetry.trace import SpanContext

from litellm.integrations.otel.model.semconv import ExceptionEvent, GenAIEvent


@dataclass(frozen=True, slots=True)
class GenAIEventRecorder:
    event_logger: EventLogger

    def record_operation_exception(
        self,
        span_context: SpanContext,
        error_type: str,
        message: str,
        stack_trace: str | None,
        timestamp_ns: int | None,
    ) -> None:
        # ``exception.type`` and ``exception.message`` are the semconv-required
        # pair and always ride the event; only the recommended stacktrace is
        # conditional on the payload carrying one.
        stacktrace = ((ExceptionEvent.STACKTRACE, stack_trace),) if stack_trace else ()
        self.event_logger.emit(
            Event(
                name=GenAIEvent.OPERATION_EXCEPTION,
                timestamp=timestamp_ns,
                trace_id=span_context.trace_id,
                span_id=span_context.span_id,
                trace_flags=span_context.trace_flags,
                severity_number=SeverityNumber.WARN,
                attributes=dict(
                    (
                        (ExceptionEvent.TYPE, error_type),
                        (ExceptionEvent.MESSAGE, message),
                        *stacktrace,
                    )
                ),
            )
        )
