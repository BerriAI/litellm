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
from e2e_http import NoBody, StreamingResponse
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
    events: list[DdLogEvent], *, model_group: str, call_type: str, outcome: StreamingResponse
) -> None:
    """The enforced behavior: the intake holds exactly one event for the call,
    sourced from litellm, whose payload names the model group and call type,
    counts real tokens, and carries the same cost the response header reported."""
    assert events, "no DataDog log event for this call reached the intake within the deadline"
    assert len(events) == 1, (
        f"expected exactly ONE DataDog log event for the call, got {len(events)} - "
        "more than one event for one call is the duplicate-delivery bug (see LIT-4447 "
        "for the currently known /v1/messages instance)"
    )
    event = events[0]
    assert "source:litellm" in event.tags, (
        f"the ingested event must carry the litellm source (shipped as ddsource), got tags {event.tags!r}"
    )
    assert event.status == "info", f"success events ship at status info, got {event.status!r}"

    payload = _DdMessagePayload.model_validate(event.attributes)
    assert payload.status == "success", f"payload status must be success, got {payload.status!r}"
    assert payload.model_group == model_group, (
        f"payload model_group must be {model_group!r}, got {payload.model_group!r}"
    )
    assert payload.call_type == call_type, (
        f"payload call_type must be {call_type!r}, got {payload.call_type!r}"
    )
    assert payload.total_tokens > 0, f"payload must count real tokens, got {payload.total_tokens}"
    assert outcome.response_cost is not None and outcome.response_cost > 0, (
        f"the response must report x-litellm-response-cost, got {outcome.response_cost!r}"
    )
    # Relative tolerance, not bit-equality: the cost round-trips through
    # DataDog's attribute indexing, whose float serialization may drift in the
    # last bits; 9 significant digits still catches any real cost discrepancy.
    assert math.isclose(payload.response_cost, outcome.response_cost, rel_tol=1e-9), (
        f"payload response_cost {payload.response_cost} must equal the response header "
        f"cost {outcome.response_cost}"
    )


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
        events = dd_logs.poll_events_for_marker(marker)
        _assert_exactly_one_event(
            events, model_group=CHEAP_ANTHROPIC_MODEL, call_type="acompletion", outcome=outcome
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
        events = dd_logs.poll_events_for_marker(marker)
        _assert_exactly_one_event(
            events, model_group=CHEAP_ANTHROPIC_MODEL, call_type="anthropic_messages", outcome=outcome
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
        events = dd_logs.poll_events_for_marker(marker)
        _assert_exactly_one_event(
            events, model_group=CHEAP_OPENAI_MODEL, call_type="aresponses", outcome=outcome
        )
