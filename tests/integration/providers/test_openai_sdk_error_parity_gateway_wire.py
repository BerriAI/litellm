import json
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.wire import Reply, wire_server
from pydantic import JsonValue, TypeAdapter

_ERROR_ENVELOPE: Final = TypeAdapter(dict[str, dict[str, JsonValue]])


def _error(status: int) -> Reply:
    return Reply(
        status=status,
        body=json.dumps({"error": {"message": f"upstream {status}", "type": "x", "code": None}}).encode(),
    )


@pytest.mark.parametrize(
    ("status", "error_type", "message_prefix"),
    (
        (409, "invalid_request_error", "litellm.APIError: "),
        (422, "x", "litellm.BadRequestError: "),
        (502, "internal_server_error", "litellm.BadGatewayError: "),
        (503, "internal_server_error", "litellm.ServiceUnavailableError: "),
        (504, "internal_server_error", "litellm.Timeout: "),
    ),
)
def test_guard_gateway_error_shape_is_unchanged(
    gateway: Gateway, status: int, error_type: str, message_prefix: str
) -> None:
    with wire_server(lambda _request: _error(status)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=wire.url, max_retries=0)
        response: Final = gateway.client.post(
            "/chat/completions",
            json={"model": model, "messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": f"Bearer {gateway.key}"},
            timeout=30,
        )
    assert response.status_code == status, response.text
    error: Final = _ERROR_ENVELOPE.validate_json(response.content)["error"]
    assert error["type"] == error_type, error
    assert str(error["message"]).startswith(message_prefix), error
