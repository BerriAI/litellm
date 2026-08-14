"""Regression pin for #36863: OTel v2 GenAI exception events must not leave
`Event.body` as ``None`` — the OTLP protobuf encoder rejects ``None`` and
discards the entire batch.

The v2 events path is separate from the v1 metrics/events path fixed in
cycle 11 (#36759) and lives in `litellm/integrations/otel/plumbing/events.py`
under the v2 directory tree (`litellm/integrations/otel/`).
"""

import pytest

pytest.importorskip("opentelemetry")

from unittest.mock import MagicMock  # noqa: E402

from litellm.integrations.otel.plumbing.events import GenAIEventRecorder  # noqa: E402


class TestGenAIEventRecorderBodyNotNone:
    """
    Regression pin for #36863.

    `GenAIEventRecorder.record_operation_exception` previously built the
    `Event` without a `body`, which defaults to `None`. The OTLP protobuf
    encoder has no representation for `None` and raises
    `Invalid type <class 'NoneType'> of value None` during serialization.
    `BatchLogRecordProcessor._export_batch` catches the exception and
    silently discards the entire batch — every event batched alongside
    the failing one is lost too. Because `record_operation_exception` is
    the *only* emit site on the v2 logs signal, every batch on that
    pipeline contains only unencodable records and the signal never
    delivers anything at all.

    Pin: `record_operation_exception` must pass a non-`None` `body` to
    the `Event` constructor. The exception message is the natural body
    for a WARN record and adds nothing the attributes don't already
    carry.
    """

    def _make_recorder(self) -> tuple[GenAIEventRecorder, MagicMock]:
        """Construct a recorder with a MagicMock event_logger so the
        `emit` call can be captured and asserted on without needing the
        real OTel SDK's logger pipeline."""
        event_logger = MagicMock()
        recorder = GenAIEventRecorder(event_logger=event_logger)
        return recorder, event_logger

    def test_record_operation_exception_sets_non_none_body(self):
        """
        The regression pin: the Event emitted to the event_logger must
        have a non-None body. Before the fix the body defaulted to None,
        which the OTLP encoder rejects.

        Note: the recorder is sync and calls ``self.event_logger.emit(...)``
        without ``await`` (matching the issue's standalone repro and the
        pre-existing v2 emit site), so the test mocks ``emit`` as a plain
        ``MagicMock`` rather than an ``AsyncMock``.
        """
        recorder, event_logger = self._make_recorder()

        recorder.record_operation_exception(
            span_context=MagicMock(trace_id=1, span_id=2, trace_flags=0),
            error_type="BadRequestError",
            message="Provider returned 400: invalid top_p",
            stack_trace=None,
            timestamp_ns=1234567890,
        )

        # The event_logger must have been called exactly once.
        assert event_logger.emit.call_count == 1, (
            "record_operation_exception must call event_logger.emit "
            f"exactly once; got call_count={event_logger.emit.call_count}"
        )
        # Capture the Event that was passed in.
        event = event_logger.emit.call_args.args[0]
        # The fix: body is no longer None. The OTLP encoder rejects
        # None; the natural body for a WARN record is the exception
        # message.
        assert event.body is not None, (
            "Event.body must not be None — the OTLP encoder rejects None "
            "and BatchLogRecordProcessor discards the entire batch "
            "(see #36863). Got: %r" % (event.body,)
        )
        assert event.body == "Provider returned 400: invalid top_p"
        # And the rest of the Event shape stays intact.
        assert event.name == "gen_ai.client.operation.exception"
        # Exception attributes are still required by the semconv and
        # should not have moved to body.
        assert event.attributes["exception.type"] == "BadRequestError"
        assert event.attributes["exception.message"] == "Provider returned 400: invalid top_p"

    def test_record_operation_exception_with_stacktrace_still_sets_body(self):
        """
        When the payload carries a stacktrace, the Event must still
        have a non-None body. Same regression pin, with the
        ``exception.stacktrace`` attribute present too.
        """
        recorder, event_logger = self._make_recorder()

        recorder.record_operation_exception(
            span_context=MagicMock(trace_id=1, span_id=2, trace_flags=0),
            error_type="RateLimitError",
            message="rate limited",
            stack_trace="Traceback (most recent call last):\n  ...",
            timestamp_ns=1234567890,
        )

        event = event_logger.emit.call_args.args[0]
        assert event.body is not None
        assert event.body == "rate limited"
        assert event.attributes["exception.stacktrace"].startswith("Traceback")
