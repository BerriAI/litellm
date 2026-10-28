import json
import re
import uuid
from collections.abc import Mapping
from typing import Final

import boto3
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


_CONSTRAINED_SCHEMA: Final = {
    "type": "object",
    "properties": {
        "city": {"type": "string", "maxLength": 20},
        "population": {"type": "integer", "minimum": 0},
        "districts": {"type": "array", "items": {"type": "string"}, "minItems": 2, "uniqueItems": True},
        "code": {"oneOf": [{"type": "string"}, {"type": "integer"}]},
    },
    "required": ["city", "population", "districts", "code"],
    "additionalProperties": False,
}


def _response_format(schema: Mapping[str, object]) -> Mapping[str, object]:
    return {"type": "json_schema", "json_schema": {"name": "city", "schema": schema}}


def _consume(response: ModelResponse | CustomStreamWrapper) -> None:
    if isinstance(response, CustomStreamWrapper):
        tuple(response)


def _sent_bodies(
    route: str,
    stream: bool,
    marker_hex: str,
    request_params: Mapping[str, object],
    model_info: Mapping[str, object] | None,
) -> tuple[Mapping[str, JsonValue], ...]:
    with wire_server(_bedrock_without_native_output) as wire:
        request: Final = {
            **_request(route, stream, wire, marker_hex),
            **{"thinking": None, "response_format": None},
            **request_params,
            **({} if model_info is None else {"model_info": model_info}),
        }
        _consume(litellm.completion(**{key: value for key, value in request.items() if value is not None}))
        return tuple(_JSON.validate_json(sent.body) for sent in wire.drain())


@_ROUTES
@pytest.mark.parametrize(
    "request_params",
    [
        {"reasoning_effort": "low", "response_format": _response_format(_SCHEMA)},
        {"thinking": {"type": "adaptive"}, "response_format": _response_format(_CONSTRAINED_SCHEMA)},
        {
            "thinking": {"type": "adaptive"},
            "response_format": _response_format(_SCHEMA),
            "output_config": {"format": {"type": "json_schema", "schema": _SCHEMA}},
        },
        {
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "low"},
            "response_format": _response_format(_SCHEMA),
        },
    ],
    ids=[
        "reasoning_effort",
        "constrained_schema",
        "response_format_and_output_config_format",
        "adaptive_thinking_with_effort",
    ],
)
def test_the_retry_sends_what_a_deployment_without_native_output_sends(
    route: str, stream: bool, marker_hex: str, request_params: Mapping[str, object]
) -> None:
    rejected, retried = _sent_bodies(route, stream, marker_hex, request_params, None)
    (opted_out,) = _sent_bodies(
        route, stream, marker_hex, request_params, {"supports_bedrock_native_structured_output": False}
    )

    assert _native(rejected, route == "invoke"), rejected
    assert retried == opted_out


@_ROUTES
def test_the_retry_keeps_the_callers_continuation_message(route: str, stream: bool, marker_hex: str) -> None:
    request_params: Final = {
        "thinking": {"type": "adaptive"},
        "response_format": _response_format(_SCHEMA),
        "messages": [
            {"role": "user", "content": f"Name a city. probe {marker_hex}"},
            {"role": "assistant", "content": "Which country should it be in?"},
        ],
        "user_continue_message": {"role": "user", "content": "Any country in South America"},
    }

    rejected, retried = _sent_bodies(route, stream, marker_hex, request_params, None)
    (opted_out,) = _sent_bodies(
        route, stream, marker_hex, request_params, {"supports_bedrock_native_structured_output": False}
    )

    assert _native(rejected, route == "invoke"), rejected
    assert retried == opted_out


_MAX_LENGTH_ONE_OF_SCHEMA: Final = {
    "type": "object",
    "properties": {
        "city": {"type": "string", "maxLength": 20},
        "code": {"oneOf": [{"type": "string"}, {"type": "integer"}]},
    },
    "required": ["city", "code"],
    "additionalProperties": False,
}


def _messages_bedrock_without_native_output(request: Request) -> Reply:
    if not _is_invoke(request) or _native(_JSON.validate_json(request.body), invoke=True):
        return _bedrock_without_native_output(request)
    message: Final = {
        "id": "msg_scripted",
        "type": "message",
        "role": "assistant",
        "model": "claude-scripted",
        "content": [{"type": "text", "text": '{"city": "Paris"}'}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 11, "output_tokens": 7},
    }
    return Reply(body=json.dumps(message).encode())


async def _messages_bodies(
    route: str, marker_hex: str, model_info: Mapping[str, object] | None
) -> tuple[Mapping[str, JsonValue], ...]:
    with wire_server(_messages_bedrock_without_native_output) as wire:
        await litellm.anthropic_messages(
            model=f"bedrock/{route}/{_OPUS_4_8}",
            messages=[{"role": "user", "content": f"Name a city. probe {marker_hex}"}],
            api_base=wire.url,
            api_key="synthetic-bedrock-bearer",
            aws_region_name="us-east-1",
            max_tokens=2048,
            thinking={"type": "adaptive"},
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": _MAX_LENGTH_ONE_OF_SCHEMA}},
            **({} if model_info is None else {"model_info": model_info}),
        )
        return tuple(_JSON.validate_json(sent.body) for sent in wire.drain())


@pytest.mark.parametrize("route", ["converse", "invoke"])
async def test_a_messages_retry_sends_what_a_deployment_without_native_output_sends(
    route: str, marker_hex: str
) -> None:
    rejected, retried = await _messages_bodies(route, marker_hex, None)
    (opted_out,) = await _messages_bodies(route, marker_hex, {"supports_bedrock_native_structured_output": False})

    assert _native(rejected, invoke=route == "invoke"), rejected
    assert retried == opted_out


@_ROUTES
def test_the_retry_signs_with_the_credentials_of_a_deprecated_aws_bedrock_client(
    route: str, stream: bool, marker_hex: str
) -> None:
    aws_bedrock_client: Final = boto3.client(
        "bedrock-runtime",
        region_name="us-west-2",
        aws_access_key_id="AKIASYNTHETICCLIENT",
        aws_secret_access_key="synthetic-client-secret",
        aws_session_token="synthetic-client-session",
    )
    with wire_server(_bedrock_without_native_output) as wire:
        request: Final = {
            key: value
            for key, value in _request(route, stream, wire, marker_hex).items()
            if key not in {"api_key", "aws_region_name"}
        }
        _consume(litellm.completion(**request, aws_bedrock_client=aws_bedrock_client))
        sent: Final = wire.drain()

    assert len(sent) == 2
    for attempt in sent:
        headers = {name.lower(): value for name, value in attempt.headers.items()}
        assert "Credential=AKIASYNTHETICCLIENT/" in headers["authorization"]
        assert "/us-west-2/bedrock/aws4_request" in headers["authorization"]
        assert headers["x-amz-security-token"] == "synthetic-client-session"
