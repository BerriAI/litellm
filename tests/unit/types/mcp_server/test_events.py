from typing import Final

import pytest
from pydantic import ValidationError

from litellm.types.mcp_server.events import (
    EventOccurrence,
    GapEnvelope,
    PollEventsParams,
    PollEventsResult,
    SubscribeEventsParams,
    SubscribeEventsResult,
    TerminatedEnvelope,
)


@pytest.mark.parametrize(
    ("result", "wire"),
    (
        (
            SubscribeEventsResult(id="sub_1", refresh_before=None),
            {"id": "sub_1", "refreshBefore": None, "truncated": False},
        ),
        (
            SubscribeEventsResult(id="sub_1", refresh_before="2026-11-21T00:00:00Z", cursor="c"),
            {"id": "sub_1", "refreshBefore": "2026-11-21T00:00:00Z", "cursor": "c", "truncated": False},
        ),
    ),
)
def test_a_subscribe_result_always_carries_refresh_before(
    result: SubscribeEventsResult, wire: dict[str, object]
) -> None:
    assert result.model_dump(by_alias=True, exclude_none=True) == wire


def test_wire_fields_use_the_spec_camel_case_names() -> None:
    params: Final = PollEventsParams.model_validate({"name": "build.failed", "maxAgeMs": 10, "maxEvents": 5})
    assert (params.max_age_ms, params.max_events) == (10, 5)
    occurrence: Final = EventOccurrence(
        event_id="evt_1", name="build.failed", timestamp="2026-11-20T00:00:00Z", data={}
    )
    assert occurrence.model_dump(by_alias=True, exclude_none=True)["eventId"] == "evt_1"


@pytest.mark.parametrize(
    "delivery",
    ({"mode": "webhook", "url": "https://h/x"}, {"mode": "carrier-pigeon", "url": "https://h/x", "secret": "s"}),
)
def test_a_webhook_delivery_needs_a_known_mode_and_a_secret(delivery: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        SubscribeEventsParams.model_validate({"name": "build.failed", "delivery": delivery})


def test_control_envelopes_carry_their_type_discriminator() -> None:
    assert GapEnvelope(cursor="c").model_dump(by_alias=True) == {"type": "gap", "cursor": "c"}
    terminated: Final = TerminatedEnvelope.model_validate({"error": {"code": -32024, "message": "Forbidden"}})
    assert terminated.model_dump(by_alias=True, exclude_none=True)["type"] == "terminated"


def test_refresh_before_preserves_the_requested_key_style() -> None:
    result: Final = SubscribeEventsResult(id="sub_1", refresh_before=None)
    assert result.model_dump(exclude_none=True) == {"id": "sub_1", "refresh_before": None, "truncated": False}


@pytest.mark.parametrize("field", ["events", "truncated", "hasMore", "nextPollMs"])
def test_a_poll_without_replay_still_carries_progress_and_timing(field: str) -> None:
    wire: Final = {"events": [], "cursor": None, "truncated": False, "hasMore": False, "nextPollMs": 1000}
    assert PollEventsResult.model_validate(wire).model_dump(by_alias=True, mode="json") == wire
    incomplete: Final = {key: value for key, value in wire.items() if key != field}
    with pytest.raises(ValidationError):
        PollEventsResult.model_validate(incomplete)


def test_a_poll_result_without_a_cursor_means_replay_is_unavailable() -> None:
    wire: Final = {"events": [], "truncated": False, "hasMore": False, "nextPollMs": 1000}
    assert PollEventsResult.model_validate(wire) == PollEventsResult.model_validate({**wire, "cursor": None})
