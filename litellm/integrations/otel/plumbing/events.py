"""GenAI client events: the ``gen_ai.client.operation.exception`` log event.

The GenAI semantic conventions define exception recording for client
operations as a log-based event (severity WARN) carrying the ``exception.*``
attribute trio, correlated to the failed span through the trace/span ids —
not as a span attribute or span event. This module owns building and
emitting that event; the exporter pipeline it rides is built in
:mod:`litellm.integrations.otel.plumbing.providers`.
"""

from dataclasses import dataclass
from time import time_ns
from typing import Final

from opentelemetry._logs import Logger, LogRecord
from opentelemetry._logs.severity import SeverityNumber
from opentelemetry.sdk.resources import Resource
from opentelemetry.trace import SpanContext

from litellm.integrations.otel.model.semconv import ExceptionEvent, GenAIEvent

try:
    from opentelemetry.sdk._logs import LogRecord as _SDKLogRecord
except ImportError:
    _SDKLogRecord = None

SDK_LOG_RECORD: Final[type[LogRecord] | None] = _SDKLogRecord


@dataclass(frozen=True, slots=True)
class GenAIEventRecorder:
    event_logger: Logger
    resource: Resource | None = None

    def record_operation_exception(
        self,
        span_context: SpanContext,
        error_type: str,
        message: str,
        stack_trace: str | None,
        timestamp_ns: int | None,
    ) -> None:
        stacktrace: Final = ((ExceptionEvent.STACKTRACE, stack_trace),) if stack_trace else ()
        attributes: Final = dict(
            (
                (GenAIEvent.NAME_KEY, GenAIEvent.OPERATION_EXCEPTION),
                (ExceptionEvent.TYPE, error_type),
                (ExceptionEvent.MESSAGE, message),
                *stacktrace,
            )
        )
        record: Final[LogRecord] = (
            SDK_LOG_RECORD(
                timestamp=timestamp_ns or time_ns(),
                trace_id=span_context.trace_id,
                span_id=span_context.span_id,
                trace_flags=span_context.trace_flags,
                severity_number=SeverityNumber.WARN,
                body=message,
                attributes=attributes,
                resource=self.resource,  # pyright: ignore[reportCallIssue]  # SDK-only kwarg absent from the API LogRecord signature on the pin
            )
            if SDK_LOG_RECORD is not None
            else LogRecord(
                timestamp=timestamp_ns or time_ns(),
                trace_id=span_context.trace_id,
                span_id=span_context.span_id,
                trace_flags=span_context.trace_flags,
                severity_number=SeverityNumber.WARN,
                body=message,
                attributes=attributes,
                event_name=GenAIEvent.OPERATION_EXCEPTION,  # pyright: ignore[reportCallIssue]  # kwarg exists only on OTel 1.38+, absent from the pinned API signature
            )
        )
        self.event_logger.emit(record)
