"""Harness coverage for the gen-AI span selection in `test_otel_trace_e2e`.

Carries no `e2e` marker: this exercises the selection helper itself against
Jaeger-shaped payloads, so it runs whether or not a proxy is up. The live
assertions it protects are expensive to reproduce (they need an upstream that
fails the first attempt), which is exactly why the helper is worth pinning
here.
"""

from __future__ import annotations

import pytest
from otel_client import JaegerTrace
from test_otel_trace_e2e import TTFT_TAG, one_served_genai_span, served_genai_spans

GENAI_SPAN = "chat claude-haiku-4-5"


def _span(name: str, *, failed: bool = False, ttft: float | None = None) -> dict[str, object]:
    tags: list[dict[str, object]] = []
    if failed:
        tags.append({"key": "otel.status_code", "value": "ERROR"})
        tags.append({"key": "error.type", "value": "AuthenticationError"})
    if ttft is not None:
        tags.append({"key": TTFT_TAG, "value": ttft})
    return {"spanID": f"{name}-{len(tags)}-{failed}-{ttft}", "operationName": name, "tags": tags}


def _trace(*spans: dict[str, object]) -> JaegerTrace:
    return JaegerTrace.model_validate({"traceID": "t1", "spans": list(spans)})


def test_served_span_is_the_only_one_when_nothing_was_retried() -> None:
    trace = _trace(_span("POST /chat/completions"), _span(GENAI_SPAN, ttft=0.3))

    assert [span.operation_name for span in served_genai_spans(trace, GENAI_SPAN)] == [GENAI_SPAN]


def test_retried_attempt_span_is_excluded() -> None:
    """The real shape from a stage trace: the first attempt 401s and records no
    TTFT, the retry serves the stream. The served attempt is the one the TTFT
    assertions must run against."""
    trace = _trace(
        _span(GENAI_SPAN, failed=True),
        _span(GENAI_SPAN, ttft=0.52),
    )

    served = one_served_genai_span(trace, GENAI_SPAN)

    assert [tag.value for tag in served.tags if tag.key == TTFT_TAG] == [0.52]


def test_several_failed_attempts_still_leave_one_served_span() -> None:
    trace = _trace(
        _span(GENAI_SPAN, failed=True),
        _span(GENAI_SPAN, failed=True),
        _span(GENAI_SPAN, failed=True),
        _span(GENAI_SPAN, ttft=0.1),
    )

    assert len(served_genai_spans(trace, GENAI_SPAN)) == 1


def test_two_served_spans_still_fail() -> None:
    """The regression the count assertion exists for: one streamed call must
    not be logged as two served gen-AI spans."""
    trace = _trace(_span(GENAI_SPAN, ttft=0.2), _span(GENAI_SPAN, ttft=0.4))

    with pytest.raises(AssertionError, match="exactly ONE served gen-AI span, got 2"):
        one_served_genai_span(trace, GENAI_SPAN)


def test_all_attempts_failed_is_a_failure_not_a_pass() -> None:
    trace = _trace(_span(GENAI_SPAN, failed=True), _span(GENAI_SPAN, failed=True))

    with pytest.raises(AssertionError, match="exactly ONE served gen-AI span, got 0"):
        one_served_genai_span(trace, GENAI_SPAN)


def test_other_operations_are_not_counted() -> None:
    trace = _trace(_span("chat gpt-5.5", ttft=0.3), _span(GENAI_SPAN, ttft=0.3))

    assert len(served_genai_spans(trace, GENAI_SPAN)) == 1
