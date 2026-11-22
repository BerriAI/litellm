import pytest

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
from litellm.types.mcp_server.events import EventsErrorCode


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
