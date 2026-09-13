"""GenAI client events: the ``gen_ai.client.operation.exception`` log event.

The GenAI semantic conventions define exception recording for client
operations as a log-based event (severity WARN) carrying the ``exception.*``
attribute trio, correlated to the failed span through the trace/span ids —
not as a span attribute or span event. This module owns building and
emitting that event; the exporter pipeline it rides is built in
:mod:`litellm.integrations.otel.plumbing.providers`.
"""

from dataclasses import dataclass
from typing import Final

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
        stacktrace: Final = ((ExceptionEvent.STACKTRACE, stack_trace),) if stack_trace else ()
        self.event_logger.emit(
            Event(
                name=GenAIEvent.OPERATION_EXCEPTION,
                # The body MUST be set. ``Event.body`` defaults to ``None``, and
                # OTLP's ``AnyValue`` has no representation for ``None``: the
                # exporter's ``_encode_value`` raises ``Invalid type <class
                # 'NoneType'>``, and ``BatchLogRecordProcessor._export_batch``
                # swallows that and discards the WHOLE batch — so one such event
                # silently destroys every log record batched with it. The
                # encoder grew a ``None`` branch in opentelemetry-exporter-otlp-
                # proto-common 1.43.0, but litellm pins 1.28.0, so this event is
                # unexportable as shipped. The message is the natural body for a
                # WARN record and adds no data the attributes don't already
                # carry.
                body=message,
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
