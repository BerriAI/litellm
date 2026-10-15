import json
from collections.abc import Iterator
from typing import Final

import pytest
from integration._support.client import Scenario, gateway_from_environment
from integration._support.process import OwnedProxy, graceful_stop_seconds, owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

pytestmark: Final = pytest.mark.timeout(2 * graceful_stop_seconds() + 120)

_COUNT: Final = 4242
_TOOL_SCHEMA: Final[dict[str, JsonValue]] = {"type": "object", "properties": {}}
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])


@pytest.fixture(scope="module")
def counting_proxy(tmp_path_factory: pytest.TempPathFactory) -> Iterator[OwnedProxy]:
    with (
        gateway_from_environment() as gateway,
        owned_proxy_process(gateway, tmp_path_factory.mktemp("openai-count-tools"), {}, workers=1) as owned,
    ):
        yield owned


def _deployment(scenario: Scenario, api_base: str) -> str:
    return scenario.model(
        model="openai/gpt-4o-mini",
        api_base=f"{api_base}/v1",
        api_key="sk-integration-openai",
    )


def _count_response(_request: Request) -> Reply:
    return Reply(body=json.dumps({"input_tokens": _COUNT}).encode())


def _count_body(wire: Wire) -> dict[str, JsonValue]:
    requests: Final = wire.drain()
    assert len(requests) == 1, requests
    request: Final = requests[0]
    assert request.method == "POST"
    assert request.target == "/v1/responses/input_tokens", request.target
    return _JSON_OBJECT.validate_json(request.body)


def test_openai_count_tokens_uses_deployment_api_base(counting_proxy: OwnedProxy) -> None:
    gateway: Final = counting_proxy.gateway
    with wire_server(_count_response) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire.url)
        response: Final = gateway.request(
            "POST",
            "/v1/messages/count_tokens",
            {"model": model, "messages": [{"role": "user", "content": "hello"}]},
        )
        assert response.status_code == 200, response.text
        assert _JSON_OBJECT.validate_json(response.content)["input_tokens"] == _COUNT
        assert "tools" not in _count_body(wire)


def test_google_count_tokens_maps_anthropic_tools_to_openai(counting_proxy: OwnedProxy) -> None:
    gateway: Final = counting_proxy.gateway
    with wire_server(_count_response) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire.url)
        response: Final = gateway.request(
            "POST",
            f"/v1beta/models/{model}:countTokens",
            {
                "contents": [{"role": "user", "parts": [{"text": "hello"}]}],
                "tools": [
                    {
                        "functionDeclarations": [
                            {
                                "name": "lookup",
                                "description": "Find a record",
                                "parameters": _TOOL_SCHEMA,
                            }
                        ]
                    }
                ],
            },
        )
        assert response.status_code == 200, response.text
        assert _JSON_OBJECT.validate_json(response.content)["totalTokens"] == _COUNT
        assert _count_body(wire)["tools"] == [
            {
                "type": "function",
                "name": "lookup",
                "description": "Find a record",
                "parameters": _TOOL_SCHEMA,
                "strict": False,
            }
        ]


def test_messages_count_tokens_maps_anthropic_tools_to_openai(counting_proxy: OwnedProxy) -> None:
    gateway: Final = counting_proxy.gateway
    with wire_server(_count_response) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire.url)
        response: Final = gateway.request(
            "POST",
            "/v1/messages/count_tokens",
            {
                "model": model,
                "messages": [{"role": "user", "content": "hello"}],
                "tools": [
                    {
                        "name": "lookup",
                        "description": "Find a record",
                        "input_schema": _TOOL_SCHEMA,
                    }
                ],
            },
        )
        assert response.status_code == 200, response.text
        assert _JSON_OBJECT.validate_json(response.content)["input_tokens"] == _COUNT
        assert _count_body(wire)["tools"] == [
            {
                "type": "function",
                "name": "lookup",
                "description": "Find a record",
                "parameters": _TOOL_SCHEMA,
                "strict": False,
            }
        ]


def test_utils_token_counter_keeps_chat_tool_conversion(counting_proxy: OwnedProxy) -> None:
    gateway: Final = counting_proxy.gateway
    with wire_server(_count_response) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire.url)
        response: Final = gateway.request(
            "POST",
            "/utils/token_counter",
            {
                "model": model,
                "messages": [{"role": "user", "content": "hello"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "description": "Find a record",
                            "parameters": _TOOL_SCHEMA,
                        },
                    }
                ],
            },
            params={"call_endpoint": "true"},
        )
        assert response.status_code == 200, response.text
        assert _JSON_OBJECT.validate_json(response.content)["total_tokens"] == _COUNT
        assert _count_body(wire)["tools"] == [
            {
                "type": "function",
                "name": "lookup",
                "description": "Find a record",
                "parameters": _TOOL_SCHEMA,
            }
        ]
