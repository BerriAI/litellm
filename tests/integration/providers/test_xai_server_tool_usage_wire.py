import json
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue, TypeAdapter

_BACKEND: Final = "grok-test"
_API_KEY: Final = "synthetic-xai-key"
_REPLY_TEXT: Final = "server tool usage control"
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_PLAIN_USAGE: Final[dict[str, JsonValue]] = {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17}
_ABSENT: Final = "absent"


def _reply(usage: JsonValue) -> bytes:
    return json.dumps(
        {
            "id": "chatcmpl-xai-usage",
            "object": "chat.completion",
            "created": 1,
            "model": _BACKEND,
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": _REPLY_TEXT}, "finish_reason": "stop"}
            ],
            "usage": usage,
        }
    ).encode()


def _peer(usage: JsonValue):
    def respond(request: Request) -> Reply:
        assert request.method == "POST" and request.target == "/v1/chat/completions", request.target
        assert request.headers["authorization"] == f"Bearer {_API_KEY}"
        assert _JSON_OBJECT.validate_json(request.body)["messages"] == [
            {"role": "user", "content": "synthetic usage request"}
        ], request.body
        return Reply(body=_reply(usage))

    return respond


def _usage_with(details: JsonValue) -> dict[str, JsonValue]:
    return {**_PLAIN_USAGE, "server_side_tool_usage_details": details}


@pytest.mark.parametrize(
    ("usage", "expected_details", "expected_web_search_requests"),
    (
        pytest.param(
            _usage_with({"web_search_calls": 2, "x_search_calls": 1}),
            {"web_search_calls": 2, "x_search_calls": 1},
            2,
            id="web-search-calls-counted",
        ),
        pytest.param(_usage_with({"x_search_calls": 3}), {"x_search_calls": 3}, None, id="no-web-search-calls"),
        pytest.param(
            _usage_with({"web_search_calls": "many"}), {"web_search_calls": "many"}, None, id="call-count-is-a-word"
        ),
        pytest.param(_usage_with([1, 2]), [1, 2], None, id="details-are-a-list"),
        pytest.param(_usage_with(None), _ABSENT, None, id="details-are-null"),
        pytest.param(_PLAIN_USAGE, _ABSENT, None, id="details-absent"),
    ),
)
def test_xai_chat_reports_server_side_tool_usage_from_the_provider_reply(
    gateway: Gateway,
    usage: dict[str, JsonValue],
    expected_details: JsonValue,
    expected_web_search_requests: int | None,
) -> None:
    with wire_server(_peer(usage)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"xai/{_BACKEND}", api_base=f"{wire.url}/v1", api_key=_API_KEY)
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": "synthetic usage request"}]},
        )
        assert response.status_code == 200, response.text
        payload: Final = _JSON_OBJECT.validate_json(response.content)
        assert payload["choices"][0]["message"]["content"] == _REPLY_TEXT, response.text
        reported: Final = payload["usage"]
        assert isinstance(reported, dict), response.text
        assert {name: reported[name] for name in _PLAIN_USAGE} == _PLAIN_USAGE, response.text
        assert reported.get("server_side_tool_usage_details", _ABSENT) == expected_details, response.text
        prompt_details: Final = reported.get("prompt_tokens_details")
        web_search_requests: Final = (
            prompt_details.get("web_search_requests") if isinstance(prompt_details, dict) else None
        )
        assert web_search_requests == expected_web_search_requests, response.text
        assert len(wire.drain()) == 1


def test_xai_chat_reports_zero_usage_when_the_provider_usage_is_null(gateway: Gateway) -> None:
    with wire_server(_peer(None)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"xai/{_BACKEND}", api_base=f"{wire.url}/v1", api_key=_API_KEY)
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": "synthetic usage request"}]},
        )
        assert response.status_code == 200, response.text
        reported: Final = _JSON_OBJECT.validate_json(response.content)["usage"]
        assert isinstance(reported, dict), response.text
        assert {name: reported[name] for name in _PLAIN_USAGE} == dict.fromkeys(_PLAIN_USAGE, 0), response.text
        assert "server_side_tool_usage_details" not in reported, response.text
        assert len(wire.drain()) == 1


@pytest.mark.parametrize("usage", (pytest.param([1, 2], id="usage-is-a-list"), pytest.param("lots", id="usage-is-text")))
def test_xai_chat_answers_500_when_the_provider_usage_is_not_an_object(gateway: Gateway, usage: JsonValue) -> None:
    with wire_server(_peer(usage)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"xai/{_BACKEND}", api_base=f"{wire.url}/v1", api_key=_API_KEY)
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": "synthetic usage request"}]},
        )
        assert response.status_code == 500, response.text
        error: Final = _JSON_OBJECT.validate_json(response.content)["error"]
        assert isinstance(error, dict), response.text
        assert error["type"] == "internal_server_error", response.text
        assert "XaiException - Invalid response object" in str(error["message"]), response.text
        assert len(wire.drain()) >= 1
