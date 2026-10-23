from typing import Final

from integration._support.client import Gateway
from integration._support.openai_wire import chat_reply
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue, TypeAdapter

_DEPLOYMENT: Final = "gpt-6.1-sol"
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_MESSAGES: Final[list[JsonValue]] = [{"role": "user", "content": "no reasoning please"}]


def _reply(_request: Request) -> Reply:
    return chat_reply("chatcmpl-effort-none", _DEPLOYMENT, "answered without reasoning", stream=False)


def test_reasoning_effort_none_is_rejected_by_litellm_before_reaching_azure(gateway: Gateway) -> None:
    with wire_server(_reply) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"azure/{_DEPLOYMENT}",
            api_base=wire.url,
            api_key="synthetic-azure-key",
            api_version="2025-04-01-preview",
        )
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": _MESSAGES, "reasoning_effort": "none"},
            key=scenario.key(),
        )

        assert response.status_code == 400, response.text
        assert "reasoning_effort='none'" in response.text, response.text
        assert wire.drain() == ()


def test_reasoning_effort_none_is_dropped_when_the_deployment_drops_params(gateway: Gateway) -> None:
    with wire_server(_reply) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"azure/{_DEPLOYMENT}",
            api_base=wire.url,
            api_key="synthetic-azure-key",
            api_version="2025-04-01-preview",
            drop_params=True,
        )
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": _MESSAGES, "reasoning_effort": "none"},
            key=scenario.key(),
        )

        assert response.status_code == 200, response.text
        assert "answered without reasoning" in response.text, response.text
        received: Final = wire.drain()
        assert [request.method for request in received] == ["POST"], [request.target for request in received]
        body: Final = _JSON_OBJECT.validate_json(received[0].body)
        assert "reasoning_effort" not in body and "reasoning" not in body, body
