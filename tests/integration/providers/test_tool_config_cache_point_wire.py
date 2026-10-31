import json
import uuid
from typing import Final
from urllib.parse import unquote, urlsplit

from integration._support.client import Gateway
from integration._support.openai_wire import chat_reply, responses_reply
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue, TypeAdapter

_ANTHROPIC_MODEL: Final = "claude-sonnet-4-6"
_BEDROCK_MODEL: Final = "us.anthropic.claude-sonnet-4-6"
_OPENAI_MODEL: Final = "gpt-4.1"
_GEMINI_MODEL: Final = "gemini-3.8-flash"
_NOVA_MODEL: Final = "us.amazon.nova-pro-v1:0"
_API_KEY: Final = "synthetic-provider-key"
_CONTROL: Final = {"type": "ephemeral", "ttl": "1h"}
_TOOL_CONFIG_POINT: Final = [{"location": "tool_config", "control": _CONTROL}]
_AWS: Final = {
    "aws_access_key_id": "AKIAINTEGRATIONTEST",
    "aws_secret_access_key": "integration-secret",
    "aws_region_name": "us-east-1",
}
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_TOOL_LIST: Final = TypeAdapter(list[dict[str, JsonValue]])


def _openai_tools() -> list[JsonValue]:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": f"{name} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for name in ("get_weather", "get_forecast")
    ]


def _anthropic_tools() -> list[JsonValue]:
    return [
        {"name": name, "description": f"{name} tool", "input_schema": {"type": "object", "properties": {}}}
        for name in ("get_weather", "get_forecast")
    ]


def _anthropic_reply(identity: str) -> bytes:
    return json.dumps(
        {
            "id": identity,
            "type": "message",
            "role": "assistant",
            "model": _ANTHROPIC_MODEL,
            "content": [{"type": "text", "text": "done"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 12, "output_tokens": 3, "cache_creation_input_tokens": 10},
        }
    ).encode()


def _converse_reply() -> bytes:
    return json.dumps(
        {
            "output": {"message": {"role": "assistant", "content": [{"text": "done"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 12, "outputTokens": 3, "totalTokens": 15},
            "metrics": {"latencyMs": 1},
        }
    ).encode()


def _assert_last_anthropic_tool_marked(body: dict[str, JsonValue]) -> None:
    assert "cache_control_injection_points" not in body, body
    tools: Final = _TOOL_LIST.validate_python(body["tools"])
    assert [tool["name"] for tool in tools] == ["get_weather", "get_forecast"], tools
    assert "cache_control" not in tools[0], tools
    assert tools[1]["cache_control"] == _CONTROL, tools


def test_chat_completions_tool_config_point_marks_the_last_anthropic_tool(gateway: Gateway) -> None:
    identity: Final = f"anthropic-tool-config-{uuid.uuid4().hex}"

    def respond(request: Request) -> Reply:
        assert request.method == "POST" and request.target == "/v1/messages", request.target
        _assert_last_anthropic_tool_marked(_JSON_OBJECT.validate_json(request.body))
        return Reply(body=_anthropic_reply(identity))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"anthropic/{_ANTHROPIC_MODEL}",
            api_base=wire.url,
            api_key=_API_KEY,
            cache_control_injection_points=_TOOL_CONFIG_POINT,
        )
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "hi"}],
                "tools": _openai_tools(),
            },
        )
        assert response.status_code == 200, response.text
        assert response.json()["usage"]["cache_creation_input_tokens"] == 10, response.text
        assert len(wire.drain()) == 1


def test_messages_tool_config_point_marks_the_last_anthropic_tool(gateway: Gateway) -> None:
    identity: Final = f"anthropic-native-tool-config-{uuid.uuid4().hex}"

    def respond(request: Request) -> Reply:
        assert request.method == "POST" and request.target == "/v1/messages", request.target
        _assert_last_anthropic_tool_marked(_JSON_OBJECT.validate_json(request.body))
        return Reply(body=_anthropic_reply(identity))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"anthropic/{_ANTHROPIC_MODEL}",
            api_base=wire.url,
            api_key=_API_KEY,
            cache_control_injection_points=_TOOL_CONFIG_POINT,
        )
        response: Final = gateway.request(
            "POST",
            "/v1/messages",
            {
                "model": model,
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "hi"}],
                "tools": _anthropic_tools(),
            },
        )
        assert response.status_code == 200, response.text
        assert response.json()["usage"]["cache_creation_input_tokens"] == 10, response.text
        assert len(wire.drain()) == 1


def test_chat_completions_tool_config_point_appends_a_bedrock_converse_cache_point(gateway: Gateway) -> None:
    def respond(request: Request) -> Reply:
        path: Final = unquote(urlsplit(request.target).path)
        assert request.method == "POST" and path == f"/model/{_BEDROCK_MODEL}/converse", request.target
        body: Final = _JSON_OBJECT.validate_json(request.body)
        assert "cache_control_injection_points" not in body, body
        tools: Final = _TOOL_LIST.validate_python(_JSON_OBJECT.validate_python(body["toolConfig"])["tools"])
        assert [tuple(tool) for tool in tools] == [("toolSpec",), ("toolSpec",), ("cachePoint",)], tools
        assert tools[2] == {"cachePoint": {"type": "default", "ttl": "1h"}}, tools
        return Reply(body=_converse_reply())

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"bedrock/{_BEDROCK_MODEL}",
            api_base=wire.url,
            api_key=None,
            cache_control_injection_points=_TOOL_CONFIG_POINT,
            **_AWS,
        )
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "hi"}],
                "tools": _openai_tools(),
            },
        )
        assert response.status_code == 200, response.text
        assert len(wire.drain()) == 1


def test_chat_completions_tool_config_point_never_reaches_openai(gateway: Gateway) -> None:
    identity: Final = f"openai-tool-config-{uuid.uuid4().hex}"

    def respond(request: Request) -> Reply:
        assert request.method == "POST" and request.target == "/chat/completions", request.target
        body: Final = _JSON_OBJECT.validate_json(request.body)
        assert "cache_control_injection_points" not in body, body
        tools: Final = _TOOL_LIST.validate_python(body["tools"])
        assert [tuple(sorted(tool)) for tool in tools] == [("function", "type"), ("function", "type")], tools
        return chat_reply(identity, _OPENAI_MODEL, "done", stream=False)

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"openai/{_OPENAI_MODEL}",
            api_base=wire.url,
            api_key=_API_KEY,
            cache_control_injection_points=_TOOL_CONFIG_POINT,
        )
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "hi"}],
                "tools": _openai_tools(),
            },
        )
        assert response.status_code == 200, response.text
        assert response.json()["id"] == identity, response.text
        assert len(wire.drain()) == 1


def _gemini_reply() -> bytes:
    return json.dumps(
        {
            "candidates": [
                {"content": {"role": "model", "parts": [{"text": "done"}]}, "finishReason": "STOP", "index": 0}
            ],
            "usageMetadata": {"promptTokenCount": 12, "candidatesTokenCount": 3, "totalTokenCount": 15},
            "modelVersion": _GEMINI_MODEL,
        }
    ).encode()


def test_messages_tool_config_point_never_reaches_openai_through_the_responses_bridge(gateway: Gateway) -> None:
    identity: Final = f"openai-bridge-tool-config-{uuid.uuid4().hex}"

    def respond(request: Request) -> Reply:
        assert request.method == "POST" and request.target == "/responses", request.target
        body: Final = _JSON_OBJECT.validate_json(request.body)
        assert "cache_control_injection_points" not in body, body
        return responses_reply(identity, _OPENAI_MODEL, "done", stream=False)

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"openai/{_OPENAI_MODEL}",
            api_base=wire.url,
            api_key=_API_KEY,
            cache_control_injection_points=_TOOL_CONFIG_POINT,
        )
        response: Final = gateway.request(
            "POST",
            "/v1/messages",
            {
                "model": model,
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "hi"}],
                "tools": _anthropic_tools(),
            },
        )
        assert response.status_code == 200, response.text
        assert len(wire.drain()) == 1


def test_messages_tool_config_point_never_reaches_gemini_through_the_chat_bridge(gateway: Gateway) -> None:
    def respond(request: Request) -> Reply:
        path: Final = urlsplit(request.target).path
        assert request.method == "POST" and path == f"/models/{_GEMINI_MODEL}:generateContent", request.target
        body: Final = _JSON_OBJECT.validate_json(request.body)
        assert "cache_control_injection_points" not in body, body
        return Reply(body=_gemini_reply())

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"gemini/{_GEMINI_MODEL}",
            api_base=wire.url,
            api_key=_API_KEY,
            cache_control_injection_points=_TOOL_CONFIG_POINT,
        )
        response: Final = gateway.request(
            "POST",
            "/v1/messages",
            {
                "model": model,
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "hi"}],
                "tools": _anthropic_tools(),
            },
        )
        assert response.status_code == 200, response.text
        assert len(wire.drain()) == 1


def test_messages_tool_config_point_appends_a_bedrock_converse_cache_point_through_the_chat_bridge(
    gateway: Gateway,
) -> None:
    def respond(request: Request) -> Reply:
        path: Final = unquote(urlsplit(request.target).path)
        assert request.method == "POST" and path == f"/model/{_NOVA_MODEL}/converse", request.target
        body: Final = _JSON_OBJECT.validate_json(request.body)
        assert "cache_control_injection_points" not in body, body
        tools: Final = _TOOL_LIST.validate_python(_JSON_OBJECT.validate_python(body["toolConfig"])["tools"])
        assert [tuple(tool) for tool in tools] == [("toolSpec",), ("toolSpec",), ("cachePoint",)], tools
        assert tools[2] == {"cachePoint": {"type": "default"}}, tools
        return Reply(body=_converse_reply())

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"bedrock/{_NOVA_MODEL}",
            api_base=wire.url,
            api_key=None,
            cache_control_injection_points=_TOOL_CONFIG_POINT,
            **_AWS,
        )
        response: Final = gateway.request(
            "POST",
            "/v1/messages",
            {
                "model": model,
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "hi"}],
                "tools": _anthropic_tools(),
            },
        )
        assert response.status_code == 200, response.text
        assert len(wire.drain()) == 1
