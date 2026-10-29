from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

import pytest
from integration._support.anthropic_sse import message_json
from integration._support.client import Gateway
from integration._support.openai_wire import answering_model_discovery, chat_reply, responses_reply
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue, TypeAdapter

_PROVIDER_KEY: Final = "synthetic-allowed-tools-key"
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_TOOLS: Final[list[JsonValue]] = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
        },
    },
    {"type": "function", "function": {"name": "get_time", "parameters": {"type": "object", "properties": {}}}},
]
_NESTED: Final[dict[str, JsonValue]] = {
    "type": "allowed_tools",
    "allowed_tools": {"mode": "required", "tools": [{"type": "function", "function": {"name": "get_time"}}]},
}
_FLAT: Final[dict[str, JsonValue]] = {
    "type": "allowed_tools",
    "mode": "required",
    "tools": [{"type": "function", "name": "get_time"}],
}


@dataclass(frozen=True, slots=True)
class _Route:
    name: str
    model: str
    api_base_path: str
    extra_params: Mapping[str, JsonValue]
    request_fields: Mapping[str, JsonValue]
    upstream_path: str
    upstream_choice: Mapping[str, JsonValue]


_ROUTES: Final = (
    _Route(
        name="openai_chat",
        model="openai/gpt-5.6-luna",
        api_base_path="/v1",
        extra_params={},
        request_fields={"reasoning_effort": "none"},
        upstream_path="/v1/chat/completions",
        upstream_choice=_NESTED,
    ),
    _Route(
        name="openai_bridged_to_responses",
        model="openai/gpt-5.6-luna",
        api_base_path="/v1",
        extra_params={},
        request_fields={},
        upstream_path="/v1/responses",
        upstream_choice=_FLAT,
    ),
    _Route(
        name="azure_ai_custom_gpt5_name_with_base_model",
        model="azure_ai/gpt-5-my-deployment",
        api_base_path="/models",
        extra_params={"base_model": "gpt-5.4"},
        request_fields={},
        upstream_path="/models/chat/completions",
        upstream_choice=_NESTED,
    ),
)


def _upstream(request: Request) -> Reply:
    if request.target.startswith("/v1/messages"):
        return Reply(body=message_json("msg_allowed_tools", "claude-haiku-5-5", "allowed tools control"))
    if request.target.startswith("/v1/responses"):
        return responses_reply("resp_allowed_tools", "gpt-5.6-luna", "allowed tools control", stream=False)
    return chat_reply("chatcmpl-allowed-tools", "gpt-5.6-luna", "allowed tools control", stream=False)


def _posted(requests: tuple[Request, ...]) -> tuple[Request, ...]:
    return tuple(request for request in requests if request.method == "POST")


@pytest.mark.parametrize("choice", (_NESTED, _FLAT), ids=("nested", "flat"))
@pytest.mark.parametrize("route", _ROUTES, ids=tuple(route.name for route in _ROUTES))
def test_allowed_tools_choice_reaches_openai_shaped_upstream_in_the_shape_its_endpoint_takes(
    gateway: Gateway, route: _Route, choice: dict[str, JsonValue]
) -> None:
    with wire_server(answering_model_discovery(_upstream)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=route.model, api_key=_PROVIDER_KEY, api_base=f"{wire.url}{route.api_base_path}", **route.extra_params
        )
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": "Weather in Paris?"}],
                "tools": _TOOLS,
                "tool_choice": choice,
                "cache": {"no-cache": True},
                **route.request_fields,
            },
        )
        assert response.status_code == 200, response.text
        posted: Final = _posted(wire.drain())
        assert len(posted) == 1, posted
        assert posted[0].target.split("?")[0] == route.upstream_path, posted[0].target
        assert _JSON_OBJECT.validate_json(posted[0].body)["tool_choice"] == route.upstream_choice


_ANTHROPIC_CALLS: Final = (
    (
        "/v1/chat/completions",
        {"messages": [{"role": "user", "content": "Weather in Paris?"}], "tools": _TOOLS, "tool_choice": _NESTED},
    ),
    (
        "/v1/responses",
        {
            "input": "Weather in Paris?",
            "tools": [
                {"type": "function", "name": "get_weather", "parameters": {"type": "object", "properties": {}}},
                {"type": "function", "name": "get_time", "parameters": {"type": "object", "properties": {}}},
            ],
            "tool_choice": _FLAT,
        },
    ),
)
_ANTHROPIC_IDS: Final = ("chat_completions", "responses")


@pytest.mark.parametrize(("path", "body"), _ANTHROPIC_CALLS, ids=_ANTHROPIC_IDS)
def test_anthropic_answers_400_to_allowed_tools_choice_without_calling_the_provider(
    gateway: Gateway, path: str, body: dict[str, JsonValue]
) -> None:
    with wire_server(answering_model_discovery(_upstream)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model="anthropic/claude-haiku-5-5", api_key=_PROVIDER_KEY, api_base=wire.url)
        response: Final = gateway.request("POST", path, {"model": model, "cache": {"no-cache": True}, **body})
        assert response.status_code == 400, response.text
        assert "does not support an allowed_tools tool_choice" in response.text, response.text
        assert _posted(wire.drain()) == ()


@pytest.mark.parametrize(("path", "body"), _ANTHROPIC_CALLS, ids=_ANTHROPIC_IDS)
def test_anthropic_drops_allowed_tools_choice_under_drop_params(
    gateway: Gateway, path: str, body: dict[str, JsonValue]
) -> None:
    with wire_server(answering_model_discovery(_upstream)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="anthropic/claude-haiku-5-5", api_key=_PROVIDER_KEY, api_base=wire.url, drop_params=True
        )
        response: Final = gateway.request("POST", path, {"model": model, "cache": {"no-cache": True}, **body})
        assert response.status_code == 200, response.text
        posted: Final = _posted(wire.drain())
        assert len(posted) == 1, posted
        assert posted[0].target.split("?")[0] == "/v1/messages", posted[0].target
        assert "tool_choice" not in _JSON_OBJECT.validate_json(posted[0].body), posted[0].body
