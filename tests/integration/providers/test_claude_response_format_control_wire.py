import json
from typing import Final
from uuid import uuid4

import httpx
from integration._support.client import Gateway
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_SONNET_5_5: Final = "claude-sonnet-5-5"
_ANTHROPIC_API_KEY: Final = "synthetic-anthropic-key"
_AZURE_API_KEY: Final = "synthetic-azure-key"
_BEDROCK_BEARER: Final = "synthetic-bedrock-bearer"
_BEDROCK_MODEL: Final = "global.anthropic.claude-sonnet-5-5"
_CITY_SCHEMA: Final = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "country": {"type": "string"}},
    "required": ["name", "country"],
    "additionalProperties": False,
}
_RESPONSE_FORMAT: Final = {
    "type": "json_schema",
    "json_schema": {"name": "city", "strict": True, "schema": _CITY_SCHEMA},
}
_JSON_TOOL_INPUT: Final = {"name": "Paris", "country": "France"}


def _anthropic_tool_use_message(model: str) -> bytes:
    return json.dumps(
        {
            "id": "msg_scripted",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [{"type": "tool_use", "id": "toolu_1", "name": "json_tool_call", "input": _JSON_TOOL_INPUT}],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 11, "output_tokens": 7},
        }
    ).encode()


def _chat(gateway: Gateway, model: str) -> httpx.Response:
    return gateway.client.post(
        "/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": f"Give me Paris as JSON. Trace {uuid4().hex[:12]}"}],
            "max_tokens": 1024,
            "response_format": _RESPONSE_FORMAT,
        },
        headers={"Authorization": f"Bearer {gateway.key}"},
        timeout=60,
    )


def _bodies(wire: Wire) -> tuple[dict[str, JsonValue], ...]:
    return [_JSON_OBJECT.validate_json(request.body) for request in wire.drain()]


def test_anthropic_sonnet_5_5_response_format_sends_native_output_format(gateway: Gateway) -> None:
    def respond(request: Request) -> Reply:
        assert request.method == "POST" and request.target == "/v1/messages", request.target
        return Reply(body=_anthropic_tool_use_message(_SONNET_5_5))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"anthropic/{_SONNET_5_5}", api_base=wire.url, api_key=_ANTHROPIC_API_KEY)
        response: Final = _chat(gateway, model)
        assert response.status_code == 200, response.text
        upstream: Final = _bodies(wire)
        assert len(upstream) == 1
        assert upstream[0]["output_format"] == {"type": "json_schema", "schema": _CITY_SCHEMA}
        assert "tools" not in upstream[0]
        assert "tool_choice" not in upstream[0]


def test_azure_ai_sonnet_5_5_response_format_sends_native_output_format(gateway: Gateway) -> None:
    def respond(request: Request) -> Reply:
        assert request.method == "POST" and request.target == "/anthropic/v1/messages", request.target
        return Reply(body=_anthropic_tool_use_message(_SONNET_5_5))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"azure_ai/{_SONNET_5_5}", api_base=wire.url, api_key=_AZURE_API_KEY)
        response: Final = _chat(gateway, model)
        assert response.status_code == 200, response.text
        upstream: Final = _bodies(wire)
        assert len(upstream) == 1
        assert upstream[0]["output_format"] == {"type": "json_schema", "schema": _CITY_SCHEMA}
        assert "tools" not in upstream[0]
        assert "tool_choice" not in upstream[0]


def test_bedrock_invoke_sonnet_5_5_response_format_sends_unforced_json_tool(gateway: Gateway) -> None:
    def respond(request: Request) -> Reply:
        assert request.method == "POST" and request.target == f"/model/{_BEDROCK_MODEL}/invoke", request.target
        return Reply(body=_anthropic_tool_use_message(_BEDROCK_MODEL))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"bedrock/invoke/{_BEDROCK_MODEL}",
            api_base=wire.url,
            api_key=_BEDROCK_BEARER,
            aws_region_name="us-east-1",
        )
        response: Final = _chat(gateway, model)
        assert response.status_code == 200, response.text
        upstream: Final = _bodies(wire)
        assert len(upstream) == 1
        assert [tool["name"] for tool in upstream[0]["tools"]] == ["json_tool_call"]
        assert "tool_choice" not in upstream[0]
        assert "output_format" not in upstream[0]
