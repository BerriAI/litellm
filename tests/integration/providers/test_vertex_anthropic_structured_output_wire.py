import base64
import functools
import json
import re
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

import httpx
import pytest
from integration._support.client import Gateway, Scenario, list_value, object_value, string_value
from integration._support.upstream import aws_event_stream_frame
from integration._support.vertex import service_account_json
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

_JSON_TOOL: Final = "json_tool_call"


_USER_TOOL: Final = "lookup"


_TOOL_MODEL: Final = "claude-sonnet-5"


_NATIVE_MODEL: Final = "claude-opus-5-5"


_FORCED_TOOL_NATIVE_MODEL: Final = "claude-haiku-5-5"


_BEDROCK_JSON_TOOL_MODEL: Final = "us.anthropic.claude-opus-4-8"


_THINKING_SIGNATURE: Final = "c2NyaXB0ZWQtc2lnbmF0dXJl"


_PROJECT: Final = "scripted-project"


_LOCATION: Final = "us-east5"


_VERTEX_MODELS: Final = f"/v1/projects/{_PROJECT}/locations/{_LOCATION}/publishers/anthropic/models/"


_VERTEX_STREAM_SUFFIX: Final = ":streamRawPredict?alt=sse"


_ANTHROPIC_KEY: Final = "synthetic-anthropic-key"


_AZURE_KEY: Final = "synthetic-azure-ai-key"


_BEDROCK_TOKEN: Final = "synthetic-bedrock-bearer"


_BAD_OUTPUT_CONFIG: Final = "output_config: Input should be a valid dictionary"


_BAD_FORMAT: Final = "output_config.format: Input should be a json_schema object"


_SCHEMA: Final[dict[str, JsonValue]] = {
    "type": "object",
    "properties": {"tier": {"type": "integer"}, "summary": {"type": "string"}},
    "required": ["tier", "summary"],
    "additionalProperties": False,
}


_RESPONSE_FORMAT: Final[dict[str, JsonValue]] = {
    "type": "json_schema",
    "json_schema": {"name": "risk", "schema": _SCHEMA},
}


_VALUES_SCHEMA: Final[dict[str, JsonValue]] = {
    "type": "object",
    "properties": {"values": {"type": "object"}},
    "required": ["values"],
}


_VALUES_RESPONSE_FORMAT: Final[dict[str, JsonValue]] = {
    "type": "json_schema",
    "json_schema": {"name": "valued", "schema": _VALUES_SCHEMA},
}


_FORCED_JSON_TOOL: Final[dict[str, JsonValue]] = {"type": "tool", "name": _JSON_TOOL}


_USER_TOOL_SPEC: Final[dict[str, JsonValue]] = {
    "type": "function",
    "function": {
        "name": _USER_TOOL,
        "parameters": {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]},
    },
}


_MARKER: Final = re.compile(r"\b([a-z_]+) ([0-9a-f]{32})\b")


_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])


_AS_WRITTEN_SHAPES: Final = ("label_non_ascii", "label_two_objects_non_ascii")


_TEXT_ANSWER_SHAPES: Final = frozenset({"undeclared_wrapper"})


_LABEL: Final = "RiskAssessment"


_NON_ASCII_LABEL: Final = "Évaluation des risques 风险"


_ENCODED_DEPTH: Final = 200_000


_FRAGMENT_SIZE: Final = 20


_TIMING_BYTES: Final = 512 * 1024


_TIMING_FRAGMENT: Final = 4096


_TIMING_SMALL_BYTES: Final = 128 * 1024


_SSE: Final = "text/event-stream"


_EVENT_STREAM: Final = "application/vnd.amazon.eventstream"


@dataclass(frozen=True, slots=True)
class _Delivery:
    target: str
    beta: str | None
    body: dict[str, JsonValue]


def _payload(marker_hex: str) -> dict[str, JsonValue]:
    return {"tier": 2, "summary": marker_hex}


def _nest(payload: JsonValue, depth: int) -> JsonValue:
    return functools.reduce(lambda inner, _: {"parameters": inner}, range(depth), payload)


def _tool_input(shape: str, marker_hex: str) -> JsonValue:
    payload: Final = _payload(marker_hex)
    match shape:
        case "plain":
            return payload
        case "wrapped":
            return {"tool_name": _JSON_TOOL, "parameters": payload}
        case "named":
            return {"name": _JSON_TOOL, "parameters": payload}
        case "values":
            return {"values": payload}
        case "large":
            return {"tool_name": _JSON_TOOL, "parameters": {**payload, "summary": marker_hex + "x" * 5000}}
        case "two_candidates":
            return {"parameters": payload, "alternate": {**payload, "tier": 3}}
        case "inner_incomplete":
            return {"parameters": {"tier": 2}}
        case "self_shaped":
            return {**payload, "nested": payload}
        case "scalar":
            return marker_hex
        case "empty":
            return {}
        case "list_input":
            return [payload]
        case "deep":
            return _nest(payload, 60)
        case "other_tool":
            return {"tool_name": "other_tool", "parameters": payload}
        case "named_input":
            return {"name": _JSON_TOOL, "input": payload}
        case "named_two_objects":
            return {"tool_name": _JSON_TOOL, "parameters": payload, "extra": {"note": marker_hex}}
        case "named_string":
            return {"tool_name": _JSON_TOOL, "parameters": marker_hex}
        case "named_partial":
            return {"tool_name": _JSON_TOOL, "parameters": {"tier": 2}}
        case "name_property":
            return {"name": _JSON_TOOL, "parameters": {"name": marker_hex, "tier": 1}}
        case "name_value_plain":
            return {"name": marker_hex, "tier": 1}
        case "parameter_label":
            return {"parameter_name": _LABEL, "parameter_value": payload}
        case "function_label":
            return {"$FUNCTION_NAME": _JSON_TOOL, "PARAMETER_VALUE": payload}
        case "label_unshared":
            return {"parameter_name": _LABEL, "parameter_value": {"verdict": marker_hex}}
        case "label_two_objects":
            return {"parameter_name": _LABEL, "parameter_value": payload, "alternate": {"tier": 3}}
        case "label_encoded":
            return {"parameter_name": _LABEL, "parameter_value": json.dumps(payload)}
        case "label_then_declared":
            return {"parameter_name": _LABEL, **payload}
        case "named_encoded":
            return {"name": _JSON_TOOL, "parameters": json.dumps(payload)}
        case "undeclared_encoded":
            return {"parameters": json.dumps(payload)}
        case "encoded_deep":
            return {"tool_name": _JSON_TOOL, "parameters": "[" * _ENCODED_DEPTH + "]" * _ENCODED_DEPTH}
        case "number_first":
            return {"rank": 3, "parameter_value": payload}
        case "bool_first":
            return {"rank": True, "parameter_value": payload}
        case "null_first":
            return {"rank": None, "parameter_value": payload}
        case "undeclared_wrapper":
            return {"parameters": payload}
        case "label_non_ascii":
            return {"parameter_name": _NON_ASCII_LABEL, "parameter_value": {"verdict": f"résumé ✓ {marker_hex}"}}
        case "label_two_objects_non_ascii":
            return {"parameter_name": _NON_ASCII_LABEL, "parameter_value": payload, "alternate": {"note": "风险 ✓"}}
    raise AssertionError(shape)


def _tool_text(shape: str, marker_hex: str) -> str:
    tool_input: Final = _tool_input(shape, marker_hex)
    if shape in _AS_WRITTEN_SHAPES:
        return json.dumps(tool_input, ensure_ascii=False, separators=(",", ":"))
    return json.dumps(tool_input)


def _structured_text(shape: str, marker_hex: str) -> str:
    return _tool_text(shape, marker_hex) if shape in _TEXT_ANSWER_SHAPES else json.dumps(_payload(marker_hex))


def _timing_digits(size: int) -> str:
    return "".join(f"{index:08d}" for index in range(size // 8))


def _timing_text(shape: str) -> str | None:
    match shape:
        case "unclosed_key":
            return '{"' + _timing_digits(_TIMING_BYTES)
        case "unclosed_key_small":
            return '{"' + _timing_digits(_TIMING_SMALL_BYTES)
        case "unclosed_name":
            return '{"name": "' + _timing_digits(_TIMING_BYTES)
        case "unclosed_name_small":
            return '{"name": "' + _timing_digits(_TIMING_SMALL_BYTES)
        case "unclosed_label":
            return '{"parameter_name": "' + _timing_digits(_TIMING_BYTES)
        case "unclosed_label_small":
            return '{"parameter_name": "' + _timing_digits(_TIMING_SMALL_BYTES)
        case "unclosed_label_key":
            return f'{{"parameter_name": "{_LABEL}", "' + _timing_digits(_TIMING_BYTES)
        case "unclosed_label_key_small":
            return f'{{"parameter_name": "{_LABEL}", "' + _timing_digits(_TIMING_SMALL_BYTES)
    return None


def _fragments(text: str, size: int) -> tuple[str, ...]:
    return tuple(text[start : start + size] for start in range(0, len(text), size))


def _fragment_size(shape: str) -> int:
    return _TIMING_FRAGMENT if shape == "encoded_deep" or _timing_text(shape) is not None else _FRAGMENT_SIZE


def _message(message_id: str, content: list[JsonValue], stop_reason: str) -> dict[str, JsonValue]:
    return {
        "id": message_id,
        "type": "message",
        "role": "assistant",
        "model": _TOOL_MODEL,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 5, "output_tokens": 3},
    }


def _events(
    message: dict[str, JsonValue], block: dict[str, JsonValue], deltas: tuple[dict[str, JsonValue], ...]
) -> tuple[tuple[str, dict[str, JsonValue]], ...]:
    return (
        (
            "message_start",
            {
                "type": "message_start",
                "message": {
                    **message,
                    "content": [],
                    "stop_reason": None,
                    "usage": {"input_tokens": 5, "output_tokens": 1},
                },
            },
        ),
        ("content_block_start", {"type": "content_block_start", "index": 0, "content_block": block}),
        *(("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": delta}) for delta in deltas),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": message["stop_reason"], "stop_sequence": None},
                "usage": {"output_tokens": 3},
            },
        ),
        ("message_stop", {"type": "message_stop"}),
    )


def _aws_chunk(data: Mapping[str, JsonValue]) -> bytes:
    payload: Final = json.dumps({"bytes": base64.b64encode(json.dumps(data).encode()).decode()}).encode()
    return aws_event_stream_frame(
        {":event-type": "chunk", ":content-type": "application/json", ":message-type": "event"}, payload
    )


def _stream_reply(
    message: dict[str, JsonValue], block: dict[str, JsonValue], deltas: tuple[dict[str, JsonValue], ...], framing: str
) -> Reply:
    events: Final = _events(message, block, deltas)
    if framing == "aws":
        return Reply(content_type=_EVENT_STREAM, chunks=tuple(_aws_chunk(data) for _, data in events))
    return Reply(
        content_type=_SSE,
        chunks=tuple(f"event: {name}\ndata: {json.dumps(data)}\n\n".encode() for name, data in events),
    )


def _error(status: int, message: str) -> Reply:
    body: Final[dict[str, JsonValue]] = {
        "type": "error",
        "error": {"type": "invalid_request_error", "message": message},
    }
    return Reply(status=status, body=json.dumps(body).encode())


def _format_fault(body: Mapping[str, JsonValue], vertex: bool) -> str | None:
    output_config: Final = (
        {"format": body["output_format"]} if vertex and "output_format" in body else body.get("output_config")
    )
    if output_config is None:
        return None
    if not isinstance(output_config, dict):
        return _BAD_OUTPUT_CONFIG
    output_format: Final = output_config.get("format")
    if output_format is None:
        return None
    if not isinstance(output_format, dict) or output_format.get("type") != "json_schema":
        return _BAD_FORMAT
    return None if isinstance(output_format.get("schema"), dict) else _BAD_FORMAT


def _marker_of(body: Mapping[str, JsonValue]) -> tuple[str, str]:
    found: Final = _MARKER.search(json.dumps(body.get("messages")))
    assert found is not None, body
    return found.group(1), found.group(2)


def _tool_names(body: Mapping[str, JsonValue]) -> frozenset[str]:
    return frozenset(string_value(object_value(tool)["name"]) for tool in list_value(body.get("tools", [])))


def _tool_use(block_id: str, name: str, tool_input: JsonValue) -> dict[str, JsonValue]:
    return {"type": "tool_use", "id": block_id, "name": name, "input": tool_input}


def _tool_reply(message_id: str, shape: str, marker_hex: str, user_tool: bool, framing: str | None) -> Reply:
    timing: Final = _timing_text(shape)
    tool_input: Final = {} if timing is not None else _tool_input(shape, marker_hex)
    json_block: Final = _tool_use(f"toolu_{marker_hex}", _JSON_TOOL, tool_input)
    user_blocks: Final[list[JsonValue]] = (
        [_tool_use(f"toolu_user_{marker_hex}", _USER_TOOL, {"q": marker_hex})] if user_tool else []
    )
    message: Final = _message(message_id, [*user_blocks, json_block], "tool_use")
    if framing is None:
        return Reply(body=json.dumps(message).encode())
    partial: Final = _tool_text(shape, marker_hex) if timing is None else timing
    deltas: Final = tuple(
        {"type": "input_json_delta", "partial_json": piece} for piece in _fragments(partial, _fragment_size(shape))
    )
    return _stream_reply(message, {**json_block, "input": {}}, deltas, framing)


def _text_reply(message_id: str, text: str, framing: str | None) -> Reply:
    message: Final = _message(message_id, [{"type": "text", "text": text}], "end_turn")
    if framing is None:
        return Reply(body=json.dumps(message).encode())
    return _stream_reply(message, {"type": "text", "text": ""}, ({"type": "text_delta", "text": text},), framing)


def _scripted_target(request: Request) -> bool:
    return (
        request.target.startswith(_VERTEX_MODELS)
        or request.target.endswith("/v1/messages")
        or request.target.endswith("/invoke")
        or request.target.endswith("/invoke-with-response-stream")
    )


def _framing(request: Request, body: Mapping[str, JsonValue]) -> str | None:
    if request.target.endswith("/invoke-with-response-stream"):
        return "aws"
    if request.target.endswith(_VERTEX_STREAM_SUFFIX) or body.get("stream") is True:
        return "sse"
    return None


def _structured(body: Mapping[str, JsonValue]) -> bool:
    output_config: Final = body.get("output_config")
    return "output_format" in body or (isinstance(output_config, dict) and "format" in output_config)


def _peer(request: Request) -> Reply:
    vertex: Final = request.target.startswith(_VERTEX_MODELS)
    if vertex:
        assert request.headers["authorization"] == "Bearer scripted-token", request.headers
    elif not _scripted_target(request):
        return _error(404, f"unscripted target {request.target}")
    body: Final = _JSON_OBJECT.validate_json(request.body)
    fault: Final = _format_fault(body, vertex)
    if fault is not None:
        return _error(400, fault)
    shape, marker_hex = _marker_of(body)
    framing: Final = _framing(request, body)
    tools: Final = _tool_names(body)
    message_id: Final = f"msg_{marker_hex}"
    if _JSON_TOOL in tools:
        return _tool_reply(message_id, shape, marker_hex, _USER_TOOL in tools, framing)
    if _structured(body):
        return _text_reply(message_id, _structured_text(shape, marker_hex), framing)
    return _text_reply(message_id, f"plain text {marker_hex}", framing)


def _delivered(wire: Wire) -> tuple[_Delivery, ...]:
    return tuple(
        _Delivery(request.target, request.headers.get("anthropic-beta"), _JSON_OBJECT.validate_json(request.body))
        for request in wire.drain()
    )


def _one_delivery(wire: Wire, marker_hex: str) -> _Delivery:
    (sent,) = _delivered(wire)
    assert marker_hex in json.dumps(sent.body), sent.body
    return sent


def _json_tool_schema(delivery: _Delivery) -> dict[str, JsonValue]:
    tools: Final = tuple(object_value(tool) for tool in list_value(delivery.body["tools"]))
    (json_tool,) = tuple(tool for tool in tools if tool["name"] == _JSON_TOOL)
    assert json_tool.keys() == {"name", "input_schema"}, json_tool
    return object_value(json_tool["input_schema"])


def _native_format(delivery: _Delivery) -> dict[str, JsonValue]:
    return object_value(delivery.body["output_format"])


def _marker(shape: str) -> tuple[str, str]:
    marker_hex: Final = uuid.uuid4().hex
    return f"{shape} {marker_hex}", marker_hex


def _chat_body(
    model: str,
    marker: str,
    *,
    stream: bool = False,
    response_format: JsonValue = _RESPONSE_FORMAT,
    tools: JsonValue = None,
) -> dict[str, JsonValue]:
    body: Final[dict[str, JsonValue]] = {
        "model": model,
        "messages": [{"role": "user", "content": marker}],
        "stream": stream,
        "response_format": response_format,
    }
    return body if tools is None else {**body, "tools": tools}


def _chat_message(response: httpx.Response) -> dict[str, JsonValue]:
    assert response.status_code == 200, response.text
    choice: Final = object_value(list_value(_JSON_OBJECT.validate_json(response.content)["choices"])[0])
    assert choice["finish_reason"] == "stop", response.text
    return object_value(choice["message"])


def _chat_content(response: httpx.Response) -> JsonValue:
    return json.loads(string_value(_chat_message(response)["content"]))


def _vertex_deployment(gateway: Gateway, scenario: Scenario, api_base: str, backend: str) -> str:
    return scenario.model(
        model=f"vertex_ai/{backend}",
        api_base=api_base,
        api_key=None,
        vertex_project=_PROJECT,
        vertex_location=_LOCATION,
        vertex_credentials=service_account_json(_PROJECT, gateway.upstream_url.rstrip("/")),
    )


def _bedrock_invoke_deployment(
    scenario: Scenario, api_base: str, backend: str, model_info: Mapping[str, JsonValue] | None = None
) -> str:
    return scenario.model(
        model=f"bedrock/invoke/{backend}",
        api_base=api_base,
        api_key=_BEDROCK_TOKEN,
        aws_region_name="us-east-1",
        model_info=model_info,
    )


def _native_deployment(gateway: Gateway, scenario: Scenario, api_base: str, provider: str, backend: str) -> str:
    match provider:
        case "vertex_ai":
            return _vertex_deployment(gateway, scenario, api_base, backend)
        case "anthropic":
            return scenario.model(model=f"anthropic/{backend}", api_base=api_base, api_key=_ANTHROPIC_KEY)
        case "azure_ai":
            return scenario.model(model=f"azure_ai/{backend}", api_base=api_base, api_key=_AZURE_KEY)
        case _:
            raise AssertionError(provider)


@pytest.mark.covers(
    "other.provider_wire.native_model_json_schema_with_an_open_object_takes_the_json_tool_and_keeps_the_object_open"
)
@pytest.mark.parametrize("provider", ["anthropic", "azure_ai"])
def test_native_model_json_schema_with_an_open_object_takes_the_json_tool_and_keeps_the_object_open(
    gateway: Gateway, provider: str
) -> None:
    marker, marker_hex = _marker("values")
    with wire_server(_peer) as wire, gateway.scenario() as scenario:
        model: Final = _native_deployment(gateway, scenario, wire.url, provider, _FORCED_TOOL_NATIVE_MODEL)
        response: Final = gateway.request(
            "POST", "/v1/chat/completions", _chat_body(model, marker, response_format=_VALUES_RESPONSE_FORMAT)
        )
        assert _chat_content(response) == _tool_input("values", marker_hex), response.text
        sent: Final = _one_delivery(wire, marker_hex)
        assert not _structured(sent.body), sent.body
        assert _json_tool_schema(sent)["properties"] == _VALUES_SCHEMA["properties"], sent.body


@pytest.mark.covers("other.provider_wire.open_object_schema_keeps_native_output_when_the_json_tool_cannot_be_forced")
@pytest.mark.parametrize(
    ("provider", "backend", "extra"),
    [
        ("anthropic", _NATIVE_MODEL, {}),
        ("vertex_ai", _NATIVE_MODEL, {}),
        ("anthropic", _FORCED_TOOL_NATIVE_MODEL, {"thinking": {"type": "adaptive"}}),
        ("anthropic", _FORCED_TOOL_NATIVE_MODEL, {"tools": [_USER_TOOL_SPEC]}),
        ("anthropic", _FORCED_TOOL_NATIVE_MODEL, {"reasoning_effort": "none"}),
        ("azure_ai", _FORCED_TOOL_NATIVE_MODEL, {"reasoning_effort": "none"}),
    ],
    ids=[
        "forced_tool_refused-anthropic",
        "forced_tool_refused-vertex_ai",
        "thinking",
        "caller_tools",
        "effort_none-anthropic",
        "effort_none-azure_ai",
    ],
)
def test_open_object_schema_keeps_native_output_when_the_json_tool_cannot_be_forced(
    gateway: Gateway, provider: str, backend: str, extra: dict[str, JsonValue]
) -> None:
    marker, marker_hex = _marker("values")
    expected_tools: Final[frozenset[str]] = frozenset({_USER_TOOL}) if "tools" in extra else frozenset()
    with wire_server(_peer) as wire, gateway.scenario() as scenario:
        model: Final = _native_deployment(gateway, scenario, wire.url, provider, backend)
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {**_chat_body(model, marker, response_format=_VALUES_RESPONSE_FORMAT), **extra},
        )
        assert _chat_content(response) == _payload(marker_hex), response.text
        sent: Final = _one_delivery(wire, marker_hex)
        assert _structured(sent.body), sent.body
        assert _tool_names(sent.body) == expected_tools, sent.body
        assert "tool_choice" not in sent.body, sent.body
        assert ("thinking" in sent.body) is ("thinking" in extra), sent.body


def _thinking_then_json_tool(request: Request) -> Reply:
    body: Final = _JSON_OBJECT.validate_json(request.body)
    _, marker_hex = _marker_of(body)
    assert _JSON_TOOL in _tool_names(body), body
    thinking: Final[dict[str, JsonValue]] = {
        "type": "thinking",
        "thinking": f"thinking {marker_hex}",
        "signature": _THINKING_SIGNATURE,
    }
    answer: Final = _tool_use(f"toolu_{marker_hex}", _JSON_TOOL, _payload(marker_hex))
    return Reply(body=json.dumps(_message(f"msg_{marker_hex}", [thinking, answer], "tool_use")).encode())


def _json_tool_deployment(gateway: Gateway, scenario: Scenario, api_base: str, provider: str) -> str:
    match provider:
        case "vertex_ai":
            return _vertex_deployment(gateway, scenario, api_base, _TOOL_MODEL)
        case "bedrock_invoke":
            return _bedrock_invoke_deployment(
                scenario, api_base, _BEDROCK_JSON_TOOL_MODEL, {"supports_bedrock_native_structured_output": False}
            )
        case _:
            raise AssertionError(provider)


@pytest.mark.covers("other.provider_wire.json_mode_answer_keeps_the_thinking_that_preceded_the_json_tool_call")
@pytest.mark.parametrize("provider", ["vertex_ai", "bedrock_invoke"])
def test_json_mode_answer_keeps_the_thinking_that_preceded_the_json_tool_call(gateway: Gateway, provider: str) -> None:
    marker, marker_hex = _marker("plain")
    with wire_server(_thinking_then_json_tool) as wire, gateway.scenario() as scenario:
        model: Final = _json_tool_deployment(gateway, scenario, wire.url, provider)
        response: Final = gateway.request(
            "POST", "/v1/chat/completions", {**_chat_body(model, marker), "reasoning_effort": "high"}
        )
        message: Final = _chat_message(response)
        assert json.loads(string_value(message["content"])) == _payload(marker_hex), response.text
        assert message.get("reasoning_content") == f"thinking {marker_hex}", response.text
        (block,) = tuple(object_value(item) for item in list_value(message.get("thinking_blocks") or []))
        assert (block["thinking"], block["signature"]) == (f"thinking {marker_hex}", _THINKING_SIGNATURE), block
        sent: Final = _one_delivery(wire, marker_hex)
        assert "thinking" in sent.body, sent.body
        assert sent.body.get("tool_choice") != _FORCED_JSON_TOOL, sent.body


@pytest.mark.covers("other.provider_wire.vertex_chat_native_schema_keeps_min_items_one_and_closes_a_nullable_object")
def test_vertex_chat_native_schema_keeps_min_items_one_and_closes_a_nullable_object(gateway: Gateway) -> None:
    marker, marker_hex = _marker("native")
    schema: Final[dict[str, JsonValue]] = {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "tags": {"type": "array", "items": {"type": "string"}, "minItems": 1},
            "backup": {"type": ["object", "null"], "properties": {"name": {"type": "string"}}, "required": ["name"]},
        },
        "required": ["summary", "tags", "backup"],
        "additionalProperties": False,
    }
    expected: Final[dict[str, JsonValue]] = {
        **schema,
        "properties": {
            "summary": {"type": "string"},
            "tags": {"type": "array", "items": {"type": "string"}, "minItems": 1},
            "backup": {
                "type": ["object", "null"],
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
                "additionalProperties": False,
            },
        },
    }
    response_format: Final[dict[str, JsonValue]] = {
        "type": "json_schema",
        "json_schema": {"name": "tagged", "schema": schema},
    }
    with wire_server(_peer) as wire, gateway.scenario() as scenario:
        model: Final = _vertex_deployment(gateway, scenario, wire.url, _NATIVE_MODEL)
        response: Final = gateway.request(
            "POST", "/v1/chat/completions", _chat_body(model, marker, response_format=response_format)
        )
        assert _chat_content(response) == _payload(marker_hex), response.text
        sent: Final = _one_delivery(wire, marker_hex)
        assert _native_format(sent) == {"type": "json_schema", "schema": expected}, sent.body
        assert "tools" not in sent.body, sent.body
