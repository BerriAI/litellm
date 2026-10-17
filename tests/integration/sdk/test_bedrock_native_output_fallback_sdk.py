import json
import re
import uuid
from collections.abc import Mapping
from typing import Final

import pytest
from integration._support.client import list_value, object_value, string_value
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

import litellm
from litellm import CustomStreamWrapper, ModelResponse
from litellm.types.utils import Choices, StreamingChoices

_OPUS_4_8: Final = "us.anthropic.claude-opus-4-8"
_JSON_TOOL: Final = "json_tool_call"
_MARKER: Final = re.compile(r"probe ([0-9a-f]{32})")
_REJECTION: Final = "The model returned the following errors: output_config.format: Extra inputs are not permitted"
_SCHEMA: Final = {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"],
    "additionalProperties": False,
}
_JSON: Final = TypeAdapter(dict[str, JsonValue])
_ROUTES: Final = pytest.mark.parametrize(
    ("route", "stream"),
    [("converse", False), ("converse", True), ("invoke", False)],
    ids=["converse", "converse_stream", "invoke"],
)


@pytest.fixture
def marker_hex() -> str:
    return uuid.uuid4().hex


def _is_invoke(request: Request) -> bool:
    return "/invoke" in request.target


def _native(body: Mapping[str, JsonValue], invoke: bool) -> bool:
    output_config: Final = body.get("output_config" if invoke else "outputConfig")
    return isinstance(output_config, dict) and ("format" if invoke else "textFormat") in output_config


def _tool_names(body: Mapping[str, JsonValue], invoke: bool) -> frozenset[str]:
    if invoke:
        return frozenset(string_value(object_value(tool)["name"]) for tool in list_value(body.get("tools", [])))
    tools: Final = list_value(object_value(body.get("toolConfig") or {"tools": []})["tools"])
    return frozenset(string_value(object_value(object_value(tool)["toolSpec"])["name"]) for tool in tools)


def _tool_reply(invoke: bool, answer: Mapping[str, JsonValue]) -> Reply:
    if invoke:
        message: Final = {
            "id": "msg_scripted",
            "type": "message",
            "role": "assistant",
            "model": "claude-scripted",
            "content": [{"type": "tool_use", "id": "toolu_scripted", "name": _JSON_TOOL, "input": answer}],
            "stop_reason": "tool_use",
            "stop_sequence": None,
            "usage": {"input_tokens": 11, "output_tokens": 7},
        }
        return Reply(body=json.dumps(message).encode())
    converse: Final = {
        "output": {
            "message": {
                "role": "assistant",
                "content": [{"toolUse": {"toolUseId": "tooluse_scripted", "name": _JSON_TOOL, "input": answer}}],
            }
        },
        "stopReason": "tool_use",
        "usage": {"inputTokens": 11, "outputTokens": 7, "totalTokens": 18},
        "metrics": {"latencyMs": 1},
    }
    return Reply(body=json.dumps(converse).encode())


def _bedrock_without_native_output(request: Request) -> Reply:
    body: Final = _JSON.validate_json(request.body)
    invoke: Final = _is_invoke(request)
    if _native(body, invoke):
        return Reply(status=400, body=json.dumps({"message": _REJECTION}).encode())
    found: Final = _MARKER.search(json.dumps(body))
    assert found is not None and _JSON_TOOL in _tool_names(body, invoke), body
    return _tool_reply(invoke, {"city": found.group(1)})


def _request(route: str, stream: bool, wire: Wire, marker_hex: str) -> Mapping[str, object]:
    return {
        "model": f"bedrock/{route}/{_OPUS_4_8}",
        "messages": [{"role": "user", "content": f"Name a city. probe {marker_hex}"}],
        "api_base": wire.url,
        "api_key": "synthetic-bedrock-bearer",
        "aws_region_name": "us-east-1",
        "max_tokens": 2048,
        "thinking": {"type": "adaptive"},
        "response_format": {"type": "json_schema", "json_schema": {"name": "city", "schema": _SCHEMA}},
        "num_retries": 0,
        "stream": stream,
    }


def _message_content(response: ModelResponse) -> str:
    (choice,) = response.choices
    assert isinstance(choice, Choices), response
    return string_value(choice.message.content)


def _delta_content(chunk: ModelResponse) -> str:
    (choice,) = chunk.choices
    assert isinstance(choice, StreamingChoices), chunk
    return choice.delta.content or ""


def _assert_one_retry_without_native_output(wire: Wire, route: str) -> None:
    rejected, retried = (_JSON.validate_json(request.body) for request in wire.drain())
    invoke: Final = route == "invoke"
    assert _native(rejected, invoke), rejected
    assert not _native(retried, invoke), retried
    assert _JSON_TOOL in _tool_names(retried, invoke), retried


@_ROUTES
def test_sync_call_retries_a_native_output_rejection_once_with_the_json_tool(
    route: str, stream: bool, marker_hex: str
) -> None:
    with wire_server(_bedrock_without_native_output) as wire:
        response: Final = litellm.completion(**_request(route, stream, wire, marker_hex))
        content: Final = (
            "".join(_delta_content(chunk) for chunk in response)
            if isinstance(response, CustomStreamWrapper)
            else _message_content(response)
        )
        assert json.loads(content) == {"city": marker_hex}
        _assert_one_retry_without_native_output(wire, route)


@_ROUTES
async def test_async_call_retries_a_native_output_rejection_once_with_the_json_tool(
    route: str, stream: bool, marker_hex: str
) -> None:
    with wire_server(_bedrock_without_native_output) as wire:
        response: Final = await litellm.acompletion(**_request(route, stream, wire, marker_hex))
        content: Final = (
            "".join([_delta_content(chunk) async for chunk in response])
            if isinstance(response, CustomStreamWrapper)
            else _message_content(response)
        )
        assert json.loads(content) == {"city": marker_hex}
        _assert_one_retry_without_native_output(wire, route)
