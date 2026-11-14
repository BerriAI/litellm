"""Claude Fable 5.2 over Anthropic direct. Its cost-map row says `supports_sampling_params: false`,
`supports_forced_tool_use: false` and `thinking_always_on: true`, and Anthropic answers 400 to a
request that ignores any of them, so the proxy has to rewrite or refuse those requests before they
leave. The scripted upstream here is what proves the outbound body, a leaked field is invisible in
the response."""

import json
import uuid
from typing import Final

from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue, TypeAdapter

_MODEL: Final = "claude-fable-5-2"
_API_KEY: Final = "synthetic-anthropic-key"
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_WEATHER_TOOL: Final[JsonValue] = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
}
_NAMED_TOOL_CHOICE: Final[JsonValue] = {"type": "function", "function": {"name": "get_weather"}}


def _anthropic_reply(identity: str, usage: dict[str, JsonValue] | None = None) -> bytes:
    return json.dumps(
        {
            "id": identity,
            "type": "message",
            "role": "assistant",
            "model": _MODEL,
            "content": [
                {"type": "thinking", "thinking": "scripted thought", "signature": "scripted-signature"},
                {"type": "text", "text": "done"},
            ],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": usage or {"input_tokens": 12, "output_tokens": 3},
        }
    ).encode()


def _chat_body(model: str, **extra: JsonValue) -> dict[str, JsonValue]:
    return {
        "model": model,
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "hi"}],
        "cache": {"no-cache": True},
        **extra,
    }


def _outbound(request: Request) -> dict[str, JsonValue]:
    assert request.method == "POST" and request.target == "/v1/messages", (request.method, request.target)
    assert request.headers["x-api-key"] == _API_KEY
    body: Final = _JSON_OBJECT.validate_json(request.body)
    assert body["model"] == _MODEL, body
    return body


def test_drop_params_removes_temperature_and_top_p_before_anthropic(gateway: Gateway) -> None:
    identity: Final = f"fable-sampling-{uuid.uuid4().hex}"

    def respond(request: Request) -> Reply:
        body: Final = _outbound(request)
        assert "temperature" not in body and "top_p" not in body, body
        return Reply(body=_anthropic_reply(identity))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"anthropic/{_MODEL}", api_base=wire.url, api_key=_API_KEY, drop_params=True
        )
        response: Final = gateway.request("POST", "/v1/chat/completions", _chat_body(model, temperature=0.2, top_p=0.5))
        assert response.status_code == 200, response.text
        assert len(wire.drain()) == 1


def test_sampling_params_are_refused_before_reaching_anthropic_without_drop_params(gateway: Gateway) -> None:
    def respond(request: Request) -> Reply:
        raise AssertionError(f"a request with sampling params reached Anthropic: {request.body!r}")

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"anthropic/{_MODEL}", api_base=wire.url, api_key=_API_KEY)
        response: Final = gateway.request("POST", "/v1/chat/completions", _chat_body(model, temperature=0.2))
        assert response.status_code == 400, response.text
        assert "temperature" in response.text and "drop_params" in response.text, response.text
        assert wire.drain() == ()


def test_drop_params_downgrades_forced_and_named_tool_choice_to_auto(gateway: Gateway) -> None:
    identity: Final = f"fable-tool-choice-{uuid.uuid4().hex}"

    def respond(request: Request) -> Reply:
        body: Final = _outbound(request)
        assert body["tool_choice"] == {"type": "auto"}, body
        tools: Final = body["tools"]
        assert isinstance(tools, list) and len(tools) == 1, body
        first: Final = tools[0]
        assert isinstance(first, dict) and first["name"] == "get_weather", body
        return Reply(body=_anthropic_reply(identity))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"anthropic/{_MODEL}", api_base=wire.url, api_key=_API_KEY, drop_params=True
        )
        responses: Final = tuple(
            gateway.request(
                "POST", "/v1/chat/completions", _chat_body(model, tools=[_WEATHER_TOOL], tool_choice=tool_choice)
            )
            for tool_choice in ("required", _NAMED_TOOL_CHOICE)
        )
        assert [response.status_code for response in responses] == [200, 200], [r.text for r in responses]
        assert len(wire.drain()) == 2


def test_forced_tool_choice_is_refused_before_reaching_anthropic_without_drop_params(gateway: Gateway) -> None:
    def respond(request: Request) -> Reply:
        raise AssertionError(f"a forced tool_choice reached Anthropic: {request.body!r}")

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"anthropic/{_MODEL}", api_base=wire.url, api_key=_API_KEY)
        response: Final = gateway.request(
            "POST", "/v1/chat/completions", _chat_body(model, tools=[_WEATHER_TOOL], tool_choice="required")
        )
        assert response.status_code == 400, response.text
        assert "tool" in response.text and "drop_params" in response.text, response.text
        assert wire.drain() == ()


def test_thinking_disabled_is_dropped_for_an_always_on_model(gateway: Gateway) -> None:
    identity: Final = f"fable-thinking-off-{uuid.uuid4().hex}"

    def respond(request: Request) -> Reply:
        body: Final = _outbound(request)
        assert "thinking" not in body, body
        return Reply(body=_anthropic_reply(identity))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"anthropic/{_MODEL}", api_base=wire.url, api_key=_API_KEY)
        response: Final = gateway.request(
            "POST", "/v1/chat/completions", _chat_body(model, thinking={"type": "disabled"})
        )
        assert response.status_code == 200, response.text
        payload: Final = _JSON_OBJECT.validate_json(response.content)
        choices: Final = payload["choices"]
        assert isinstance(choices, list) and isinstance(choices[0], dict), response.text
        message: Final = choices[0]["message"]
        assert isinstance(message, dict) and message["reasoning_content"] == "scripted thought", response.text
        assert len(wire.drain()) == 1


def test_reasoning_effort_becomes_adaptive_thinking_with_output_config_effort(gateway: Gateway) -> None:
    identity: Final = f"fable-effort-{uuid.uuid4().hex}"

    def respond(request: Request) -> Reply:
        body: Final = _outbound(request)
        assert body["thinking"] == {"type": "adaptive", "display": "summarized"}, body
        assert body["output_config"] == {"effort": "max"}, body
        assert "budget_tokens" not in json.dumps(body), body
        return Reply(body=_anthropic_reply(identity))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"anthropic/{_MODEL}", api_base=wire.url, api_key=_API_KEY)
        response: Final = gateway.request("POST", "/v1/chat/completions", _chat_body(model, reasoning_effort="max"))
        assert response.status_code == 200, response.text
        assert len(wire.drain()) == 1


def test_messages_one_hour_cache_control_reaches_anthropic_and_its_usage_comes_back(gateway: Gateway) -> None:
    identity: Final = f"fable-cache-1h-{uuid.uuid4().hex}"
    policy: Final = f"policy {identity}"
    upstream_usage: Final[dict[str, JsonValue]] = {
        "input_tokens": 12,
        "output_tokens": 3,
        "cache_creation_input_tokens": 9216,
        "cache_read_input_tokens": 0,
        "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 9216},
    }

    def respond(request: Request) -> Reply:
        body: Final = _outbound(request)
        assert body["system"] == [
            {"type": "text", "text": policy, "cache_control": {"type": "ephemeral", "ttl": "1h"}}
        ], body
        return Reply(body=_anthropic_reply(identity, upstream_usage))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"anthropic/{_MODEL}", api_base=wire.url, api_key=_API_KEY)
        response: Final = gateway.request(
            "POST",
            "/v1/messages",
            {
                "model": model,
                "max_tokens": 16,
                "system": [{"type": "text", "text": policy, "cache_control": {"type": "ephemeral", "ttl": "1h"}}],
                "messages": [{"role": "user", "content": "hi"}],
                "cache": {"no-cache": True},
            },
        )
        assert response.status_code == 200, response.text
        usage: Final = _JSON_OBJECT.validate_json(response.content)["usage"]
        assert isinstance(usage, dict), response.text
        assert usage["cache_creation_input_tokens"] == 9216, usage
        assert usage["cache_creation"] == {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 9216}, usage
        assert len(wire.drain()) == 1
