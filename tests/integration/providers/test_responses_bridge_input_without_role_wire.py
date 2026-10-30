import json
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue, TypeAdapter

_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_BRIDGED_MODEL: Final = "deepseek/gpt-4o-mini"
_USER_TURN: Final[dict[str, JsonValue]] = {"role": "user", "content": "hello"}


def _completion(_: Request) -> Reply:
    return Reply(
        body=json.dumps(
            {
                "id": "chatcmpl-input-without-role",
                "object": "chat.completion",
                "created": 1789788254,
                "model": "gpt-4o-mini",
                "choices": [
                    {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "hi"}},
                ],
                "usage": {"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5},
            }
        ).encode()
    )


@pytest.mark.parametrize(
    "item",
    [
        pytest.param({"content": "no role"}, id="content-only"),
        pytest.param({"type": "message", "content": "no role"}, id="typed-message"),
        pytest.param({"role": None, "content": "no role"}, id="null-role"),
        pytest.param({"role": "", "content": "no role"}, id="empty-role"),
    ],
)
def test_bridged_input_item_without_a_role_answers_400_and_reaches_no_provider(
    gateway: Gateway, item: dict[str, JsonValue]
) -> None:
    with wire_server(_completion) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=_BRIDGED_MODEL, api_base=wire.url + "/v1")
        response: Final = gateway.request(
            "POST", "/v1/responses", {"model": model, "input": [_USER_TURN, item], "store": False}
        )

        assert response.status_code == 400, response.text
        error: Final = _JSON_OBJECT.validate_python(_JSON_OBJECT.validate_json(response.content)["error"])
        assert error["type"] == "invalid_request_error", response.text
        assert error["param"] == "input[1].role", response.text
        assert "input[1].role" in str(error["message"]), response.text
        assert wire.drain() == ()


def test_bridged_role_less_item_reference_still_reaches_the_provider(gateway: Gateway) -> None:
    with wire_server(_completion) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=_BRIDGED_MODEL, api_base=wire.url + "/v1")
        response: Final = gateway.request(
            "POST", "/v1/responses", {"model": model, "input": [{"id": "msg_1"}, _USER_TURN], "store": False}
        )

        assert response.status_code == 200, response.text
        assert len(wire.drain()) == 1
