"""Live e2e: DataDog log delivery for successful non-streaming calls.

Covers logging.datadog.success.exports_metric: one successful call on each
route must reach the DataDog logs intake as EXACTLY ONE log event whose
message (the StandardLoggingPayload) carries the model, the token counts, and
the response cost. Delivery is judged on what DataDog itself ingested: the
proxy ships with DD_API_KEY exactly as in production, and the tests search the
events back through the DataDog Logs Search API (DD_APP_KEY, keys from the
secret manager on the cluster), so a dropped event, a duplicated event, or a
payload missing the cost all fail here.

Both halves of the contract are asserted: the recorded state (the proxy
reports the DataDogLogger callback active via /health/readiness/details) and
the enforced behavior (the event at the intake, with the cost cross-checked
against the x-litellm-response-cost header of the very response the caller
received).
"""

from __future__ import annotations

import math

import pytest
from pydantic import BaseModel, ConfigDict

from datadog_reader import DdLogEvent, DdLogsReader
from e2e_config import CHEAP_ANTHROPIC_MODEL, CHEAP_OPENAI_MODEL, unique_marker
from e2e_http import NoBody
from lifecycle import ResourceManager
from logging_client import LoggingClient, first_ok

pytestmark = pytest.mark.e2e

#: The active DataDog callback's name in /health/readiness/details success_callbacks.
DD_LOGGER_NAME = "DataDogLogger"


class _DdMessagePayload(BaseModel):
    """The fields of the StandardLoggingPayload the scenario pins."""

    model_config = ConfigDict(extra="ignore")

    model_group: str
    total_tokens: int
    response_cost: float
    status: str
    call_type: str
    stream: bool | None = None


def _assert_datadog_configured(client: LoggingClient) -> None:
    """Recorded state: the proxy reports the DataDog callback among its active
    callbacks, so a missing destination config fails here, before any
    delivery-based assertion can time out confusingly."""
    result = client.gateway.probe("/health/readiness/details", params=NoBody())
    assert result.status_code == 200, (
        f"/health/readiness/details must answer 200, got {result.status_code}: {result.body[:300]}"
    )
    assert DD_LOGGER_NAME in result.body, (
        f"the proxy must report the {DD_LOGGER_NAME} callback active "
        f"(callbacks + DD_* env in the compose config); got: {result.body[:400]}"
    )


def _assert_exactly_one_event(
    events: list[DdLogEvent],
    *,
    model_group: str,
    call_type: str,
    cost_anchor: float,
    expect_stream: bool = False,
) -> _DdMessagePayload:
    """The enforced behavior: the intake holds exactly one event for the call,
    sourced from litellm, whose payload names the model group and call type,
    counts real tokens, and carries the same cost as ``cost_anchor`` - the
    x-litellm-response-cost header for non-streaming calls, or the /spend/logs
    row for streamed calls (headers ship before a stream's cost exists)."""
    assert events, "no DataDog log event for this call reached the intake within the deadline"
    assert len(events) == 1, (
        f"expected exactly ONE DataDog log event for the call, got {len(events)} - "
        "more than one event for one call is the duplicate-delivery bug (see LIT-4447 "
        "for the currently known non-streaming /v1/messages instance)"
    )
    event = events[0]
    assert "source:litellm" in event.tags, (
        f"the ingested event must carry the litellm source (shipped as ddsource), got tags {event.tags!r}"
    )
    # The proxy ships the envelope at status "info", but DataDog re-derives the
    # indexed event status from the parsed payload's status attribute
    # ("success") and normalizes it to its OK severity - so "ok" is what a
    # successfully ingested success event looks like on the search API.
    assert event.status == "ok", (
        f"success events must index at DataDog's ok severity, got {event.status!r}"
    )

    payload = _DdMessagePayload.model_validate(event.attributes)
    assert payload.status == "success", f"payload status must be success, got {payload.status!r}"
    assert payload.model_group == model_group, (
        f"payload model_group must be {model_group!r}, got {payload.model_group!r}"
    )
    assert payload.call_type == call_type, (
        f"payload call_type must be {call_type!r}, got {payload.call_type!r}"
    )
    assert payload.total_tokens > 0, f"payload must count real tokens, got {payload.total_tokens}"
    # Relative tolerance, not bit-equality: the cost round-trips through
    # DataDog's attribute indexing, whose float serialization may drift in the
    # last bits; 9 significant digits still catches any real cost discrepancy.
    assert math.isclose(payload.response_cost, cost_anchor, rel_tol=1e-9), (
        f"payload response_cost {payload.response_cost} must equal the anchor cost {cost_anchor}"
    )
    if expect_stream:
        assert payload.stream is True, (
            f"a streamed call's payload must record stream=true, got {payload.stream!r}"
        )
    return payload


class TestDataDogLogDelivery:
    @pytest.mark.covers("logging.datadog.success.exports_metric", exercised_on=["chat_completions"])
    def test_chat_completions_emits_one_log_event(
        self, client: LoggingClient, dd_logs: DdLogsReader, resources: ResourceManager
    ) -> None:
        """One successful non-streaming /chat/completions call must reach the
        DataDog logs intake as exactly one log event whose payload carries the
        model, the token counts, and the response cost."""
        _assert_datadog_configured(client)

        key = client.key_with_alias(f"dd-chat-{unique_marker()}", models=[CHEAP_ANTHROPIC_MODEL])
        resources.defer(lambda: client.delete_key(key))

        marker = unique_marker()
        outcome = first_ok(
            client,
            lambda: client.chat_raw(key, CHEAP_ANTHROPIC_MODEL, f"reply with one word {marker}", max_tokens=16),
        )
        assert outcome.response_cost is not None and outcome.response_cost > 0, (
            f"the response must report x-litellm-response-cost, got {outcome.response_cost!r}"
        )
        events = dd_logs.poll_events_for_marker(marker)
        _assert_exactly_one_event(
            events, model_group=CHEAP_ANTHROPIC_MODEL, call_type="acompletion", cost_anchor=outcome.response_cost
        )

    @pytest.mark.covers("logging.datadog.success.exports_metric", exercised_on=["messages"])
    def test_messages_emits_one_log_event(
        self, client: LoggingClient, dd_logs: DdLogsReader, resources: ResourceManager
    ) -> None:
        """One successful non-streaming /v1/messages call must reach the
        DataDog logs intake as exactly one log event whose payload carries the
        model, the token counts, and the response cost.

        This currently fails on the known /v1/messages double-log (LIT-4447); it goes green when the fix lands."""
        _assert_datadog_configured(client)

        key = client.key_with_alias(f"dd-messages-{unique_marker()}", models=[CHEAP_ANTHROPIC_MODEL])
        resources.defer(lambda: client.delete_key(key))

        marker = unique_marker()
        outcome = first_ok(
            client,
            lambda: client.messages_raw(key, CHEAP_ANTHROPIC_MODEL, f"reply with one word {marker}", max_tokens=16),
        )
        assert outcome.response_cost is not None and outcome.response_cost > 0, (
            f"the response must report x-litellm-response-cost, got {outcome.response_cost!r}"
        )
        events = dd_logs.poll_events_for_marker(marker)
        _assert_exactly_one_event(
            events, model_group=CHEAP_ANTHROPIC_MODEL, call_type="anthropic_messages", cost_anchor=outcome.response_cost
        )

    @pytest.mark.covers("logging.datadog.success.exports_metric", exercised_on=["responses"])
    def test_responses_emits_one_log_event(
        self, client: LoggingClient, dd_logs: DdLogsReader, resources: ResourceManager
    ) -> None:
        """One successful non-streaming /v1/responses call must reach the
        DataDog logs intake as exactly one log event whose payload carries the
        model, the token counts, and the response cost."""
        _assert_datadog_configured(client)

        key = client.key_with_alias(f"dd-responses-{unique_marker()}", models=[CHEAP_OPENAI_MODEL])
        resources.defer(lambda: client.delete_key(key))

        marker = unique_marker()
        outcome = first_ok(
            client,
            lambda: client.responses_raw(key, CHEAP_OPENAI_MODEL, f"reply with one word {marker}"),
        )
        assert outcome.response_cost is not None and outcome.response_cost > 0, (
            f"the response must report x-litellm-response-cost, got {outcome.response_cost!r}"
        )
        events = dd_logs.poll_events_for_marker(marker)
        _assert_exactly_one_event(
            events, model_group=CHEAP_OPENAI_MODEL, call_type="aresponses", cost_anchor=outcome.response_cost
        )

    @pytest.mark.covers("logging.datadog.stream.exports_metric", exercised_on=["chat_completions"])
    def test_chat_completions_stream_emits_one_log_event(
        self, client: LoggingClient, dd_logs: DdLogsReader, resources: ResourceManager
    ) -> None:
        """One successful STREAMED /chat/completions call must reach real
        DataDog as exactly one log event whose payload carries the model, the
        token counts aggregated across the stream, stream=true, and a response
        cost equal to the /spend/logs row for the same call (a stream's
        headers ship before its cost exists, so the spend row is the
        cross-check anchor)."""
        _assert_datadog_configured(client)

        key = client.key_with_alias(f"dd-stream-chat-{unique_marker()}", models=[CHEAP_ANTHROPIC_MODEL])
        resources.defer(lambda: client.delete_key(key))

        marker = unique_marker()
        outcome = first_ok(
            client,
            lambda: client.chat_raw(key, CHEAP_ANTHROPIC_MODEL, f"reply with one word {marker}", stream=True, max_tokens=16),
        )
        assert outcome.is_streaming, f"response must be an event stream, got content-type {outcome.content_type!r}"
        assert outcome.chunks > 0, "the stream must deliver at least one event"
        assert outcome.stream_error is None, (
            f"the stream carried an upstream error event despite the 200: {outcome.stream_error}"
        )

        spend_row = client.poll_proxy_spend_for_key(key)
        assert spend_row is not None and spend_row.spend is not None and spend_row.spend > 0, (
            f"the streamed call must record a positive-spend row, got {spend_row!r}"
        )
        events = dd_logs.poll_events_for_marker(marker)
        payload = _assert_exactly_one_event(
            events,
            model_group=CHEAP_ANTHROPIC_MODEL,
            call_type="acompletion",
            cost_anchor=spend_row.spend,
            expect_stream=True,
        )
        assert spend_row.total_tokens is not None, (
            "the spend row must record total_tokens for the token cross-check"
        )
        assert spend_row.total_tokens == payload.total_tokens, (
            f"the spend row and the DataDog event must agree on tokens: "
            f"{spend_row.total_tokens} vs {payload.total_tokens}"
        )

    @pytest.mark.covers("logging.datadog.stream.exports_metric", exercised_on=["messages"])
    def test_messages_stream_emits_one_log_event(
        self, client: LoggingClient, dd_logs: DdLogsReader, resources: ResourceManager
    ) -> None:
        """One successful STREAMED /v1/messages call must reach real DataDog
        as exactly one log event whose payload carries the model, the token
        counts aggregated across the stream, stream=true, and a response cost
        equal to the /spend/logs row for the same call.

        The non-streaming messages route double-logs today (LIT-4447); the
        streamed path verified clean during evidence probing, so this test is
        strict and expected green."""
        _assert_datadog_configured(client)

        key = client.key_with_alias(f"dd-stream-messages-{unique_marker()}", models=[CHEAP_ANTHROPIC_MODEL])
        resources.defer(lambda: client.delete_key(key))

        marker = unique_marker()
        outcome = first_ok(
            client,
            lambda: client.messages_raw(key, CHEAP_ANTHROPIC_MODEL, f"reply with one word {marker}", max_tokens=16, stream=True),
        )
        assert outcome.is_streaming, f"response must be an event stream, got content-type {outcome.content_type!r}"
        assert outcome.chunks > 0, "the stream must deliver at least one event"
        assert outcome.stream_error is None, (
            f"the stream carried an upstream error event despite the 200: {outcome.stream_error}"
        )

        spend_row = client.poll_proxy_spend_for_key(key)
        assert spend_row is not None and spend_row.spend is not None and spend_row.spend > 0, (
            f"the streamed call must record a positive-spend row, got {spend_row!r}"
        )
        events = dd_logs.poll_events_for_marker(marker)
        payload = _assert_exactly_one_event(
            events,
            model_group=CHEAP_ANTHROPIC_MODEL,
            call_type="anthropic_messages",
            cost_anchor=spend_row.spend,
            expect_stream=True,
        )
        assert spend_row.total_tokens is not None, (
            "the spend row must record total_tokens for the token cross-check"
        )
        assert spend_row.total_tokens == payload.total_tokens, (
            f"the spend row and the DataDog event must agree on tokens: "
            f"{spend_row.total_tokens} vs {payload.total_tokens}"
        )

    @pytest.mark.covers("logging.datadog.stream.exports_metric", exercised_on=["responses"])
    def test_responses_stream_emits_one_log_event(
        self, client: LoggingClient, dd_logs: DdLogsReader, resources: ResourceManager
    ) -> None:
        """One successful STREAMED /v1/responses call must reach real DataDog
        as exactly one log event whose payload carries the model, the token
        counts aggregated across the stream, stream=true, and a response cost
        equal to the /spend/logs row for the same call."""
        _assert_datadog_configured(client)

        key = client.key_with_alias(f"dd-stream-responses-{unique_marker()}", models=[CHEAP_OPENAI_MODEL])
        resources.defer(lambda: client.delete_key(key))

        marker = unique_marker()
        outcome = first_ok(
            client,
            lambda: client.responses_raw(key, CHEAP_OPENAI_MODEL, f"reply with one word {marker}", stream=True),
        )
        assert outcome.is_streaming, f"response must be an event stream, got content-type {outcome.content_type!r}"
        assert outcome.chunks > 0, "the stream must deliver at least one event"
        assert outcome.stream_error is None, (
            f"the stream carried an upstream error event despite the 200: {outcome.stream_error}"
        )

        spend_row = client.poll_proxy_spend_for_key(key)
        assert spend_row is not None and spend_row.spend is not None and spend_row.spend > 0, (
            f"the streamed call must record a positive-spend row, got {spend_row!r}"
        )
        events = dd_logs.poll_events_for_marker(marker)
        payload = _assert_exactly_one_event(
            events,
            model_group=CHEAP_OPENAI_MODEL,
            call_type="aresponses",
            cost_anchor=spend_row.spend,
            expect_stream=True,
        )
        assert spend_row.total_tokens is not None, (
            "the spend row must record total_tokens for the token cross-check"
        )
        assert spend_row.total_tokens == payload.total_tokens, (
            f"the spend row and the DataDog event must agree on tokens: "
            f"{spend_row.total_tokens} vs {payload.total_tokens}"
        )
