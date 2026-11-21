from typing import Final

import pytest
from pydantic import ValidationError

from litellm.proxy.mcp_server.events.errors import (
    EventsFailure,
    callback_error,
    event_not_found,
    exhausted,
    forbidden,
    header_mismatch,
    subscription_not_found,
    unsupported,
)
from litellm.types.mcp_server.events import (
    EventOccurrence,
    EventsErrorCode,
    GapEnvelope,
    PollEventsParams,
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


@pytest.mark.parametrize(
    ("failure", "code", "data"),
    (
        (event_not_found(), EventsErrorCode.NOT_FOUND, {"kind": "event"}),
        (subscription_not_found(), EventsErrorCode.NOT_FOUND, {"kind": "subscription"}),
        (forbidden("Key revoked"), EventsErrorCode.FORBIDDEN, {"reason": "Key revoked"}),
        (exhausted("subscriptions", 50), EventsErrorCode.RESOURCE_EXHAUSTED, {"limit": "subscriptions", "max": 50}),
        (
            unsupported("deliveryMode", "push"),
            EventsErrorCode.UNSUPPORTED,
            {"feature": "deliveryMode", "value": "push"},
        ),
        (unsupported("webhook"), EventsErrorCode.UNSUPPORTED, {"feature": "webhook"}),
        (callback_error("timeout"), EventsErrorCode.CALLBACK_ENDPOINT_ERROR, {"reason": "timeout"}),
        (header_mismatch("Mcp-Name"), EventsErrorCode.HEADER_MISMATCH, {"header": "Mcp-Name"}),
    ),
)
def test_each_failure_maps_to_its_spec_error(
    failure: EventsFailure, code: EventsErrorCode, data: dict[str, object]
) -> None:
    error: Final = failure.to_mcp()
    assert (error.error.code, error.error.data) == (int(code), data)
