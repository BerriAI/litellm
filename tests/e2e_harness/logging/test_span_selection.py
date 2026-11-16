"""Harness coverage for the gen-AI span selection `logging/test_otel_trace_e2e` relies on.

This exercises the selection helper itself against Jaeger-shaped payloads, so it
runs without a proxy. The live assertions it protects are expensive to reproduce
(they need an upstream that fails the first attempt), which is exactly why the
helper is worth pinning here.
"""

from __future__ import annotations

from typing import Final

import pytest
from otel_client import (
    TTFT_TAG,
    CallTraces,
    JaegerReference,
    JaegerSpan,
    JaegerTag,
    JaegerTrace,
    one_served_genai_span,
    served_genai_spans,
    trace_settled,
)
from test_otel_trace_e2e import assert_complete_trace, assert_real_ttft

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


def _request_trace(
    *,
    root_name: str = "POST",
    root_kind: str = "server",
    route: str = "/v1/messages",
    genai_parent: str = "root",
    ttft: float = 0.2,
) -> JaegerTrace:
    parent: Final = JaegerReference(refType="CHILD_OF", traceID="request", spanID="root")
    return JaegerTrace(
        traceID="request",
        spans=[
            JaegerSpan(
                spanID="root",
                operationName=root_name,
                tags=[JaegerTag(key="span.kind", value=root_kind), JaegerTag(key="http.target", value=route)],
            ),
            JaegerSpan(spanID="auth", operationName="auth /v1/messages", references=[parent]),
            JaegerSpan(spanID="db", operationName="postgres.select tokens", references=[parent]),
            JaegerSpan(
                spanID="genai",
                operationName=GENAI_SPAN,
                duration=1_000_000,
                references=[JaegerReference(refType="CHILD_OF", traceID="request", spanID=genai_parent)],
                tags=[JaegerTag(key="span.kind", value="client"), JaegerTag(key=TTFT_TAG, value=ttft)],
            ),
        ],
    )


@pytest.mark.parametrize("root_name", ("POST", "POST /v1/messages"))
def test_connected_request_trace_is_independent_of_server_display_name(root_name: str) -> None:
    trace: Final = _request_trace(root_name=root_name)
    assert_complete_trace(CallTraces((trace,), ()), route="/v1/messages", genai_span=GENAI_SPAN)
    assert_real_ttft((trace,), genai_span=GENAI_SPAN)
    assert trace_settled(trace, {"auth /v1/messages", GENAI_SPAN}, {"postgres."})


def test_incomplete_trace_cannot_settle_before_the_server_span_arrives() -> None:
    trace: Final = _request_trace()
    incomplete: Final = trace.model_copy(update={"spans": trace.spans[1:]})
    assert not trace_settled(incomplete, {"auth /v1/messages", GENAI_SPAN}, {"postgres."})


@pytest.mark.parametrize("route", ("/v1/responses", ""))
def test_request_trace_must_still_identify_the_requested_route(route: str) -> None:
    trace: Final = _request_trace(root_name="POST /v1/messages", route=route)
    with pytest.raises(AssertionError, match="route"):
        assert_complete_trace(CallTraces((trace,), ()), route="/v1/messages", genai_span=GENAI_SPAN)


def test_request_trace_must_still_be_rooted_at_a_server_span() -> None:
    trace: Final = _request_trace(root_name="POST /v1/messages", root_kind="client")
    with pytest.raises(AssertionError, match="kind=server"):
        assert_complete_trace(CallTraces((trace,), ()), route="/v1/messages", genai_span=GENAI_SPAN)


def test_request_trace_must_still_connect_the_model_call_to_the_request() -> None:
    trace: Final = _request_trace(genai_parent="absent")
    with pytest.raises(AssertionError, match="orphaned trace"):
        assert_complete_trace(CallTraces((trace,), ()), route="/v1/messages", genai_span=GENAI_SPAN)


@pytest.mark.parametrize("ttft", (0.0, 1.0))
def test_server_name_tolerance_does_not_accept_invalid_first_token_timing(ttft: float) -> None:
    trace: Final = _request_trace(ttft=ttft)
    with pytest.raises(AssertionError, match="TTFT"):
        assert_real_ttft((trace,), genai_span=GENAI_SPAN)
