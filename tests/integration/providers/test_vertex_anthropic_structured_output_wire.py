import asyncio
import base64
import functools
import json
import re
import signal
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from queue import SimpleQueue
from typing import Final

import anthropic
import httpx
import openai
import psutil
import pytest
import yaml
from integration._support.client import Gateway, Scenario, eventually, list_value, object_value, string_value
from integration._support.database import read_rows
from integration._support.process import OwnedProxy, graceful_stop_seconds, owned_proxy_process
from integration._support.upstream import aws_event_stream_frame
from integration._support.vertex import service_account_json
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

_JSON_TOOL: Final = "json_tool_call"
_USER_TOOL: Final = "lookup"
_STRUCTURED_OUTPUTS_BETA: Final = "structured-outputs-2025-11-13"
_TOOL_MODEL: Final = "claude-sonnet-5"
_EFFORT_GATED_MODEL: Final = "claude-haiku-4-5"
_NATIVE_MODEL: Final = "claude-opus-5-5"
_UNMAPPED_MODEL: Final = "claude-scripted-9"
_BEDROCK_INVOKE_MODEL: Final = "anthropic.claude-sonnet-5"
_PROJECT: Final = "scripted-project"
_LOCATION: Final = "us-east5"
_VERTEX_MODELS: Final = f"/v1/projects/{_PROJECT}/locations/{_LOCATION}/publishers/anthropic/models/"
_VERTEX_STREAM_SUFFIX: Final = ":streamRawPredict?alt=sse"
_ANTHROPIC_KEY: Final = "synthetic-anthropic-key"
_AZURE_KEY: Final = "synthetic-azure-ai-key"
_BEDROCK_TOKEN: Final = "synthetic-bedrock-bearer"
_PLATFORM_KEY: Final = "synthetic-platform-key"
_MANTLE_KEY: Final = "synthetic-bedrock-mantle-key"
_SCRIPTED_KEY: Final = "scripted"
_NIM_GROUP: Final = "scripted-nim"
_WATSONX_VERSION: Final = "2024-03-13"
_OWNED_MODEL: Final = "partner-claude"
_DEPRECATED_OUTPUT_FORMAT: Final = "output_format: this field is deprecated, use output_config.format"
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
_TEXT_FORMAT: Final[dict[str, JsonValue]] = {
    "format": {"type": "json_schema", "name": "risk", "schema": _SCHEMA},
}
_OUTPUT_FORMAT: Final[dict[str, JsonValue]] = {"type": "json_schema", "schema": _SCHEMA}
_NAME_SCHEMA: Final[dict[str, JsonValue]] = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "tier": {"type": "integer"}},
    "required": ["name", "tier"],
    "additionalProperties": False,
}
_NAME_RESPONSE_FORMAT: Final[dict[str, JsonValue]] = {
    "type": "json_schema",
    "json_schema": {"name": "named", "schema": _NAME_SCHEMA},
}
_OPTIONAL_SCHEMA: Final[dict[str, JsonValue]] = {"type": "object", "properties": _SCHEMA["properties"]}
_VALUES_SCHEMA: Final[dict[str, JsonValue]] = {
    "type": "object",
    "properties": {"values": {"type": "object"}},
    "required": ["values"],
}
_ADDITIONAL_OBJECT_SCHEMA: Final[dict[str, JsonValue]] = {
    **_OPTIONAL_SCHEMA,
    "additionalProperties": {"type": "object"},
}
_PATTERN_SCHEMA: Final[dict[str, JsonValue]] = {
    **_OPTIONAL_SCHEMA,
    "patternProperties": {"^[a-z_]+$": {"type": "object"}},
}
_ADDITIONAL_TRUE_SCHEMA: Final[dict[str, JsonValue]] = {**_OPTIONAL_SCHEMA, "additionalProperties": True}
_CLOSED_OPTIONAL_SCHEMA: Final[dict[str, JsonValue]] = {**_OPTIONAL_SCHEMA, "additionalProperties": False}
_CALLERS_FORMAT: Final[dict[str, JsonValue]] = {"type": "json_schema", "schema": {**_SCHEMA, "title": "callers"}}
_ANTHROPIC_JSON_TOOL: Final[dict[str, JsonValue]] = {"name": _JSON_TOOL, "input_schema": _SCHEMA}
_FORCED_JSON_TOOL: Final[dict[str, JsonValue]] = {"type": "tool", "name": _JSON_TOOL}
_REASONING_FIELDS: Final = ("thinking", "output_config", "max_tokens")
_USER_TOOL_SPEC: Final[dict[str, JsonValue]] = {
    "type": "function",
    "function": {
        "name": _USER_TOOL,
        "parameters": {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]},
    },
}
_MARKER: Final = re.compile(r"\b([a-z_]+) ([0-9a-f]{32})\b")
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_STARTED_WORKER: Final = re.compile(r"Started server process \[(\d+)\]")
_OWNED_PROXY_CELL_SECONDS: Final = 2 * graceful_stop_seconds() + 120
_BURST_SHAPES: Final = ("wrapped", "plain", "named", "values")
_STREAM_SHAPES: Final = (
    "plain",
    "wrapped",
    "named",
    "values",
    "large",
    "two_candidates",
    "inner_incomplete",
    "self_shaped",
    "scalar",
    "empty",
    "list_input",
    "deep",
    "other_tool",
    "named_partial",
    "parameter_label",
    "function_label",
    "label_unshared",
    "label_two_objects",
    "label_encoded",
    "named_encoded",
    "undeclared_encoded",
    "encoded_deep",
    "number_first",
    "bool_first",
    "null_first",
)
_HELD_STREAM_SHAPES: Final = frozenset(
    {
        "wrapped",
        "named",
        "values",
        "large",
        "two_candidates",
        "inner_incomplete",
        "deep",
        "other_tool",
        "named_partial",
        "parameter_label",
        "function_label",
        "label_unshared",
        "label_two_objects",
        "label_encoded",
        "named_encoded",
        "undeclared_encoded",
        "encoded_deep",
    }
)
_AS_WRITTEN_SHAPES: Final = ("label_non_ascii", "label_two_objects_non_ascii")
_TEXT_ANSWER_SHAPES: Final = frozenset({"undeclared_wrapper"})
_LABEL: Final = "RiskAssessment"
_NON_ASCII_LABEL: Final = "Évaluation des risques 风险"
_ENCODED_DEPTH: Final = 200_000
_FAMILY: Final = (
    "anthropic_unmapped",
    "azure_ai_unmapped",
    "bedrock_invoke",
    "bedrock_mantle",
    "claude_platform_unmapped",
    "vertex_haiku",
)
_STREAM_FAMILY: Final = ("anthropic_unmapped", "azure_ai_unmapped", "bedrock_invoke", "vertex_haiku")
_FRAGMENT_SIZE: Final = 20
_TIMING_BYTES: Final = 512 * 1024
_TIMING_FRAGMENT: Final = 4096
_TIMING_SMALL_BYTES: Final = 128 * 1024
_TIMING_GROWTH_LIMIT: Final = 6.0
_HEALTH_LATENCY_SECONDS: Final = 1.0
_SSE: Final = "text/event-stream"
_EVENT_STREAM: Final = "application/vnd.amazon.eventstream"
_RELAY_CHUNKS: Final = (
    b'data: {"choices":[{"delta":{"content":"relay one"}}]}\n\n',
    b'data: {"choices":[{"delta":{"content":"relay two"}}]}\n\n',
    b"data: [DONE]\n\n",
)
_PASS_THROUGH_ROUTES: Final = (
    "gemini",
    "cohere",
    "vllm",
    "mistral",
    "typesafe",
    "openrouter",
    "azure",
    "openai",
    "gigachat",
    "watsonx",
    "tinyfish",
    "nvidia_nim",
)


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


def _expected(shape: str, marker_hex: str) -> JsonValue:
    match shape:
        case (
            "plain"
            | "wrapped"
            | "named"
            | "values"
            | "named_input"
            | "deep"
            | "other_tool"
            | "parameter_label"
            | "function_label"
            | "named_encoded"
        ):
            return _payload(marker_hex)
        case "large":
            return {**_payload(marker_hex), "summary": marker_hex + "x" * 5000}
        case "named_partial" | "inner_incomplete":
            return {"tier": 2}
        case "name_property":
            return {"name": marker_hex, "tier": 1}
    return _tool_input(shape, marker_hex)


def _fragments(text: str, size: int) -> tuple[str, ...]:
    return tuple(text[start : start + size] for start in range(0, len(text), size))


def _fragment_size(shape: str) -> int:
    return _TIMING_FRAGMENT if shape == "encoded_deep" or _timing_text(shape) is not None else _FRAGMENT_SIZE


def _fragment_count(shape: str, marker_hex: str) -> int:
    return len(_fragments(_tool_text(shape, marker_hex), _fragment_size(shape)))


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
    if vertex and "output_format" in body:
        return _DEPRECATED_OUTPUT_FORMAT
    output_config: Final = body.get("output_config")
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


def _relay_peer(request: Request) -> Reply:
    assert request.body, request.target
    return Reply(content_type=_SSE, chunks=_RELAY_CHUNKS)


def _failing(status: int) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        _, marker_hex = _marker_of(_JSON_OBJECT.validate_json(request.body))
        return _error(status, f"scripted {status} {marker_hex}")

    return respond


def _holding(held: SimpleQueue[str], release: threading.Event) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        held.put(request.target)
        assert release.wait(timeout=60), "Held request was never released"
        return _peer(request)

    return respond


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


def _folded_format(delivery: _Delivery) -> dict[str, JsonValue]:
    assert "output_format" not in delivery.body, delivery.body
    return object_value(object_value(delivery.body["output_config"])["format"])


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


def _effort_body(model: str, marker: str) -> dict[str, JsonValue]:
    return {"model": model, "messages": [{"role": "user", "content": marker}], "reasoning_effort": "low"}


def _reasoning_fields(delivery: _Delivery) -> dict[str, JsonValue]:
    return {field: delivery.body.get(field) for field in _REASONING_FIELDS}


def _messages_body(model: str, marker: str, *, stream: bool = False, **extra: JsonValue) -> dict[str, JsonValue]:
    return {
        "model": model,
        "max_tokens": 64,
        "messages": [{"role": "user", "content": marker}],
        "stream": stream,
        **extra,
    }


def _responses_body(model: str, marker: str) -> dict[str, JsonValue]:
    return {"model": model, "input": marker, "text": _TEXT_FORMAT}


def _chat_message(response: httpx.Response) -> dict[str, JsonValue]:
    assert response.status_code == 200, response.text
    choice: Final = object_value(list_value(_JSON_OBJECT.validate_json(response.content)["choices"])[0])
    assert choice["finish_reason"] == "stop", response.text
    return object_value(choice["message"])


def _chat_content(response: httpx.Response) -> JsonValue:
    return json.loads(string_value(_chat_message(response)["content"]))


def _response_id(response: httpx.Response) -> str:
    return string_value(_JSON_OBJECT.validate_json(response.content)["id"])


def _delta_content(frame: str) -> str:
    choices: Final = list_value(_JSON_OBJECT.validate_json(frame).get("choices", []))
    if not choices:
        return ""
    content: Final = object_value(object_value(choices[0])["delta"]).get("content")
    return "" if content is None else string_value(content)


def _content_frames(response: httpx.Response) -> tuple[str, ...]:
    assert response.status_code == 200, response.text
    assert response.text.rstrip().endswith("data: [DONE]"), response.text
    contents: Final = tuple(
        _delta_content(line.removeprefix("data: ")) for line in response.text.splitlines() if line.startswith("data: {")
    )
    return tuple(content for content in contents if content)


def _timing_frames(response: httpx.Response) -> tuple[str, int]:
    assert response.status_code == 200, response.text
    frames: Final = tuple(
        line.removeprefix("data: ") for line in response.text.splitlines() if line.startswith("data: {")
    )
    error_frames: Final = sum(1 for frame in frames if "error" in _JSON_OBJECT.validate_json(frame))
    return "".join(_delta_content(frame) for frame in frames), error_frames


def _timed_stream(
    client: httpx.Client,
    key: str,
    wire: Wire,
    model: str,
    shape: str,
    response_format: dict[str, JsonValue],
    record_property: Callable[[str, object], None],
) -> float:
    marker, marker_hex = _marker(shape)
    raw: Final = _timing_text(shape)
    assert raw is not None
    started: Final = time.perf_counter()
    response: Final = _post(
        client, key, "/v1/chat/completions", _chat_body(model, marker, stream=True, response_format=response_format)
    )
    wall: Final = time.perf_counter() - started
    text, error_frames = _timing_frames(response)
    record_property(f"{shape}_seconds", round(wall, 3))
    record_property(f"{shape}_error_frames", error_frames)
    assert text == raw, (shape, len(text), len(raw), text[:80])
    _one_delivery(wire, marker_hex)
    return wall


def _stream_text(response: httpx.Response) -> str:
    return "".join(_content_frames(response))


def _stream_content(response: httpx.Response) -> JsonValue:
    return json.loads(_stream_text(response))


def _responses_stream_text(response: httpx.Response) -> JsonValue:
    assert response.status_code == 200, response.text
    frames: Final = tuple(
        _JSON_OBJECT.validate_json(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: {")
    )
    assert any(frame["type"] == "response.completed" for frame in frames), response.text
    return json.loads(
        "".join(string_value(frame["delta"]) for frame in frames if frame["type"] == "response.output_text.delta")
    )


def _responses_text(response: httpx.Response) -> JsonValue:
    assert response.status_code == 200, response.text
    items: Final = tuple(
        object_value(item) for item in list_value(_JSON_OBJECT.validate_json(response.content)["output"])
    )
    (message,) = tuple(item for item in items if item["type"] == "message")
    return json.loads(string_value(object_value(list_value(message["content"])[0])["text"]))


def _text_block_json(message: Mapping[str, JsonValue]) -> JsonValue:
    block: Final = object_value(list_value(message["content"])[0])
    return json.loads(string_value(block["text"]))


def _messages_text(response: httpx.Response) -> JsonValue:
    assert response.status_code == 200, response.text
    return _text_block_json(_JSON_OBJECT.validate_json(response.content))


def _messages_stream_text(response: httpx.Response) -> JsonValue:
    assert response.status_code == 200, response.text
    assert "event: message_stop" in response.text, response.text
    frames: Final = tuple(
        _JSON_OBJECT.validate_json(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: {")
    )
    deltas: Final = tuple(object_value(frame["delta"]) for frame in frames if frame["type"] == "content_block_delta")
    return json.loads("".join(string_value(delta["text"]) for delta in deltas if delta["type"] == "text_delta"))


def _vertex_deployment(gateway: Gateway, scenario: Scenario, api_base: str, backend: str) -> str:
    return scenario.model(
        model=f"vertex_ai/{backend}",
        api_base=api_base,
        api_key=None,
        vertex_project=_PROJECT,
        vertex_location=_LOCATION,
        vertex_credentials=service_account_json(_PROJECT, gateway.upstream_url.rstrip("/")),
    )


def _bedrock_invoke_deployment(scenario: Scenario, api_base: str, backend: str) -> str:
    return scenario.model(
        model=f"bedrock/invoke/{backend}",
        api_base=api_base,
        api_key=_BEDROCK_TOKEN,
        aws_region_name="us-east-1",
    )


def _rejects_forced_tools_without_native_output(row: Mapping[str, JsonValue]) -> bool:
    return row.get("supports_forced_tool_use") is False and row.get("supports_native_structured_output") is not True


def _claude_rejecting_forced_tools(gateway: Gateway, namespace: str, family: str) -> str:
    rows: Final = gateway.get("/public/litellm_model_cost_map")
    names: Final = sorted(
        name.removeprefix(namespace)
        for name, row in rows.items()
        if name.startswith(namespace + family) and _rejects_forced_tools_without_native_output(object_value(row))
    )
    assert names, f"The cost map has no {namespace}{family} row rejecting forced tools without native output"
    return names[0]


def _family_deployment(gateway: Gateway, scenario: Scenario, api_base: str, member: str) -> str:
    match member:
        case "anthropic_unmapped":
            return scenario.model(model=f"anthropic/{_UNMAPPED_MODEL}", api_base=api_base, api_key=_ANTHROPIC_KEY)
        case "azure_ai_unmapped":
            return scenario.model(model=f"azure_ai/{_UNMAPPED_MODEL}", api_base=api_base, api_key=_AZURE_KEY)
        case "bedrock_invoke":
            return _bedrock_invoke_deployment(scenario, api_base, _BEDROCK_INVOKE_MODEL)
        case "bedrock_mantle":
            return scenario.model(
                model=f"bedrock_mantle/{_BEDROCK_INVOKE_MODEL}",
                api_base=api_base,
                api_key=_MANTLE_KEY,
                aws_region_name="us-east-1",
            )
        case "claude_platform_unmapped":
            return scenario.model(
                model=f"bedrock/claude_platform/{_UNMAPPED_MODEL}",
                api_base=api_base,
                api_key=_PLATFORM_KEY,
                aws_region_name="us-east-1",
                workspace_id="scripted-workspace",
            )
        case "vertex_haiku":
            return _vertex_deployment(gateway, scenario, api_base, _EFFORT_GATED_MODEL)
    raise AssertionError(member)


def _owned_config(path: Path, gateway: Gateway, api_base: str) -> Path:
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    path.write_text(
        yaml.safe_dump(
            {
                **config,
                "model_list": [
                    {
                        "model_name": _OWNED_MODEL,
                        "litellm_params": {
                            "model": f"vertex_ai/{_TOOL_MODEL}",
                            "api_base": api_base,
                            "vertex_project": _PROJECT,
                            "vertex_location": _LOCATION,
                            "vertex_credentials": service_account_json(_PROJECT, gateway.upstream_url.rstrip("/")),
                        },
                    },
                    {
                        "model_name": _NIM_GROUP,
                        "litellm_params": {
                            "model": f"nvidia_nim/{_NIM_GROUP}",
                            "api_base": api_base,
                            "api_key": _SCRIPTED_KEY,
                        },
                    },
                ],
            }
        )
    )
    return path


def _pass_through_environment(api_base: str) -> dict[str, str]:
    bases: Final = (
        "GEMINI_API_BASE",
        "COHERE_API_BASE",
        "VLLM_API_BASE",
        "MISTRAL_API_BASE",
        "TYPESAFE_API_BASE",
        "OPENROUTER_API_BASE",
        "AZURE_API_BASE",
        "OPENAI_API_BASE",
        "GIGACHAT_API_BASE",
        "WATSONX_API_BASE",
        "TINYFISH_AGENT_API_BASE",
    )
    keys: Final = (
        "GEMINI_API_KEY",
        "COHERE_API_KEY",
        "VLLM_API_KEY",
        "MISTRAL_API_KEY",
        "TYPESAFE_API_KEY",
        "OPENROUTER_API_KEY",
        "AZURE_API_KEY",
        "OPENAI_API_KEY",
        "GIGACHAT_ACCESS_TOKEN",
        "WATSONX_TOKEN",
        "TINYFISH_API_KEY",
    )
    return {**{name: api_base for name in bases}, **{name: _SCRIPTED_KEY for name in keys}}


@dataclass(frozen=True, slots=True)
class _PassThroughCall:
    path: str
    body: dict[str, JsonValue]
    upstream_target: str
    headers: dict[str, str]


def _pass_through_call(route: str, marker: str, key: str) -> _PassThroughCall:
    chat: Final[dict[str, JsonValue]] = {
        "model": "scripted-chat",
        "messages": [{"role": "user", "content": marker}],
        "stream": True,
    }
    bearer: Final = {"Authorization": f"Bearer {key}"}
    match route:
        case "gemini":
            return _PassThroughCall(
                "/gemini/v1beta/models/scripted-model:streamGenerateContent?alt=sse",
                {"contents": [{"parts": [{"text": marker}]}]},
                "/v1beta/models/scripted-model:streamGenerateContent?alt=sse&key=scripted",
                {"x-goog-api-key": key},
            )
        case "cohere":
            return _PassThroughCall("/cohere/v2/chat", chat, "/v2/chat", bearer)
        case "vllm":
            return _PassThroughCall("/vllm/v1/chat/completions", chat, "/v1/chat/completions", bearer)
        case "mistral":
            return _PassThroughCall("/mistral/v1/chat/completions", chat, "/v1/chat/completions", bearer)
        case "typesafe":
            return _PassThroughCall("/typesafe/v1/chat/completions", chat, "/v1/chat/completions", bearer)
        case "openrouter":
            return _PassThroughCall("/openrouter/v1/chat/completions", chat, "/v1/chat/completions", bearer)
        case "azure":
            return _PassThroughCall(
                "/azure/openai/deployments/scripted-deployment/chat/completions?api-version=2024-10-21",
                chat,
                "/openai/deployments/scripted-deployment/chat/completions?api-version=2024-10-21",
                bearer,
            )
        case "openai":
            return _PassThroughCall("/openai/v1/chat/completions", chat, "/v1/chat/completions", bearer)
        case "gigachat":
            return _PassThroughCall("/gigachat/chat/completions", chat, "/chat/completions", bearer)
        case "watsonx":
            return _PassThroughCall(
                "/watsonx/ml/v1/text/chat_stream",
                {**chat, "model_id": "scripted-chat", "project_id": "scripted-project"},
                f"/ml/v1/text/chat_stream?version={_WATSONX_VERSION}",
                bearer,
            )
        case "tinyfish":
            return _PassThroughCall(
                "/tinyfish/v1/automation/run-sse",
                {"url": "https://example.com", "goal": marker},
                "/v1/automation/run-sse",
                bearer,
            )
        case "nvidia_nim":
            return _PassThroughCall(
                f"/nvidia_nim/{_NIM_GROUP}/v1/chat/completions", chat, "/v1/chat/completions", bearer
            )
    raise AssertionError(route)


def _clients(stack: ExitStack, base_url: str, count: int) -> tuple[httpx.Client, ...]:
    return tuple(
        stack.enter_context(httpx.Client(base_url=base_url, timeout=60, trust_env=False)) for _ in range(count)
    )


def _post(client: httpx.Client, key: str, path: str, body: Mapping[str, JsonValue]) -> httpx.Response:
    return client.post(path, json=dict(body), headers={"Authorization": f"Bearer {key}"})


def _post_or_dropped(client: httpx.Client, key: str, path: str, body: Mapping[str, JsonValue]) -> httpx.Response | None:
    try:
        return _post(client, key, path, body)
    except httpx.TransportError:
        return None


def _local_port(client: httpx.Client) -> int:
    with client.stream("GET", "/health/liveliness") as response:
        port: Final = int(response.extensions["network_stream"].get_extra_info("client_addr")[1])
        response.read()
    assert response.status_code == 200, response.text
    return port


def _accepted_client_ports(pid: int, proxy_port: int) -> frozenset[int]:
    return frozenset(
        connection.raddr.port
        for connection in psutil.Process(pid).net_connections(kind="tcp")
        if connection.raddr and connection.laddr.port == proxy_port
    )


def _seen_once(delivered: tuple[_Delivery, ...], marker_hexes: tuple[str, ...]) -> None:
    bodies: Final = tuple(json.dumps(sent.body) for sent in delivered)
    seen: Final = tuple(sum(marker_hex in body for body in bodies) for marker_hex in marker_hexes)
    assert seen == (1,) * len(marker_hexes), seen


def _pass_through_spend_rows(key: str) -> list[dict[str, JsonValue]]:
    return eventually(
        lambda: read_rows(
            'SELECT request_id, model, call_type FROM "LiteLLM_SpendLogs" WHERE api_key = %s AND call_type = %s',
            (sha256(key.encode()).hexdigest(), "pass_through_endpoint"),
        ),
        lambda rows: len(rows) == 1,
        seconds=90,
    )


def _spend_rows(request_id: str) -> list[dict[str, JsonValue]]:
    return eventually(
        lambda: read_rows('SELECT request_id, model FROM "LiteLLM_SpendLogs" WHERE request_id = %s', (request_id,)),
        lambda rows: len(rows) == 1,
        seconds=70,
    )


def _openai_client(gateway: Gateway) -> openai.OpenAI:
    return openai.OpenAI(
        base_url=f"{gateway.client.base_url}/v1",
        api_key=gateway.key,
        max_retries=0,
        http_client=httpx.Client(trust_env=False, timeout=60),
    )


@pytest.fixture(scope="module")
def pass_through_proxy(
    session_gateway: Gateway, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[tuple[OwnedProxy, Wire]]:
    directory: Final = tmp_path_factory.mktemp("structured-output-pass-through")
    with wire_server(_relay_peer) as wire:
        config: Final = _owned_config(directory / "pass-through.yaml", session_gateway, wire.url)
        with owned_proxy_process(
            session_gateway, directory, _pass_through_environment(wire.url), config=config
        ) as owned:
            yield owned, wire


def _anthropic_client(gateway: Gateway) -> anthropic.Anthropic:
    return anthropic.Anthropic(
        base_url=str(gateway.client.base_url),
        api_key=gateway.key,
        max_retries=0,
        http_client=httpx.Client(trust_env=False, timeout=60),
    )


@pytest.mark.parametrize(
    "shape",
    [
        "plain",
        "wrapped",
        "named",
        "values",
        "large",
        "two_candidates",
        "inner_incomplete",
        "self_shaped",
        "scalar",
        "empty",
        "list_input",
        "deep",
        "parameter_label",
        "function_label",
        "label_unshared",
        "label_two_objects",
        "label_encoded",
        "label_then_declared",
        "named_encoded",
        "undeclared_encoded",
        "encoded_deep",
        "number_first",
        "bool_first",
        "null_first",
        "label_non_ascii",
        "label_two_objects_non_ascii",
    ],
)
def test_vertex_chat_json_schema_content_is_the_schema_object(gateway: Gateway, shape: str) -> None:
    marker, marker_hex = _marker(shape)
    with wire_server(_peer) as wire, gateway.scenario() as scenario:
        model: Final = _vertex_deployment(gateway, scenario, wire.url, _TOOL_MODEL)
        response: Final = gateway.request("POST", "/v1/chat/completions", _chat_body(model, marker))
        assert _chat_content(response) == _expected(shape, marker_hex), response.text
        assert "tool_calls" not in _chat_message(response) or _chat_message(response)["tool_calls"] is None
        sent: Final = _one_delivery(wire, marker_hex)
        assert sent.target == f"{_VERTEX_MODELS}{_TOOL_MODEL}:rawPredict", sent.target
        assert _json_tool_schema(sent)["properties"] == _SCHEMA["properties"], sent.body
        assert "output_config" not in sent.body and "output_format" not in sent.body, sent.body


def test_vertex_chat_through_the_openai_sdk_unwraps_the_envelope(gateway: Gateway) -> None:
    marker, marker_hex = _marker("wrapped")
    with wire_server(_peer) as wire, gateway.scenario() as scenario:
        model: Final = _vertex_deployment(gateway, scenario, wire.url, _TOOL_MODEL)
        with _openai_client(gateway) as client:
            reply: Final = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": marker}],
                response_format={"type": "json_schema", "json_schema": {"name": "risk", "schema": _SCHEMA}},
            )
        assert json.loads(reply.choices[0].message.content or "") == _payload(marker_hex), reply
        _one_delivery(wire, marker_hex)


def test_vertex_chat_through_the_async_openai_sdk_unwraps_the_envelope(gateway: Gateway) -> None:
    marker, marker_hex = _marker("wrapped")

    async def generate(model: str) -> str | None:
        async with openai.AsyncOpenAI(
            base_url=f"{gateway.client.base_url}/v1",
            api_key=gateway.key,
            max_retries=0,
            http_client=httpx.AsyncClient(trust_env=False, timeout=60),
        ) as client:
            reply: Final = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": marker}],
                response_format={"type": "json_schema", "json_schema": {"name": "risk", "schema": _SCHEMA}},
            )
        return reply.choices[0].message.content

    with wire_server(_peer) as wire, gateway.scenario() as scenario:
        model: Final = _vertex_deployment(gateway, scenario, wire.url, _TOOL_MODEL)
        content: Final = asyncio.run(generate(model))
        assert json.loads(content or "") == _payload(marker_hex), content
        _one_delivery(wire, marker_hex)


def test_vertex_responses_json_schema_output_text_is_the_schema_object(gateway: Gateway) -> None:
    marker, marker_hex = _marker("wrapped")
    with wire_server(_peer) as wire, gateway.scenario() as scenario:
        model: Final = _vertex_deployment(gateway, scenario, wire.url, _TOOL_MODEL)
        response: Final = gateway.request("POST", "/v1/responses", _responses_body(model, marker))
        assert _responses_text(response) == _payload(marker_hex), response.text
        sent: Final = _one_delivery(wire, marker_hex)
        assert _json_tool_schema(sent)["properties"] == _SCHEMA["properties"], sent.body


@pytest.mark.parametrize("shape", _STREAM_SHAPES)
def test_vertex_chat_stream_json_schema_content_is_the_schema_object(gateway: Gateway, shape: str) -> None:
    marker, marker_hex = _marker(shape)
    with wire_server(_peer) as wire, gateway.scenario() as scenario:
        model: Final = _vertex_deployment(gateway, scenario, wire.url, _TOOL_MODEL)
        response: Final = gateway.request("POST", "/v1/chat/completions", _chat_body(model, marker, stream=True))
        frames: Final = _content_frames(response)
        assert json.loads("".join(frames)) == _expected(shape, marker_hex), response.text
        expected_frames: Final = 1 if shape in _HELD_STREAM_SHAPES else _fragment_count(shape, marker_hex)
        assert len(frames) == expected_frames, (len(frames), expected_frames, frames)
        sent: Final = _one_delivery(wire, marker_hex)
        assert sent.target == f"{_VERTEX_MODELS}{_TOOL_MODEL}{_VERTEX_STREAM_SUFFIX}", sent.target
        assert _json_tool_schema(sent)["properties"] == _SCHEMA["properties"], sent.body


def test_vertex_chat_stream_label_before_a_declared_key_keeps_streaming(gateway: Gateway) -> None:
    marker, marker_hex = _marker("label_then_declared")
    with wire_server(_peer) as wire, gateway.scenario() as scenario:
        model: Final = _vertex_deployment(gateway, scenario, wire.url, _TOOL_MODEL)
        response: Final = gateway.request("POST", "/v1/chat/completions", _chat_body(model, marker, stream=True))
        frames: Final = _content_frames(response)
        assert json.loads("".join(frames)) == _tool_input("label_then_declared", marker_hex), response.text
        assert len(frames) > 1, frames
        _one_delivery(wire, marker_hex)


@pytest.mark.parametrize("shape", _AS_WRITTEN_SHAPES)
def test_vertex_chat_stream_held_answer_without_an_envelope_is_sent_as_the_model_wrote_it(
    gateway: Gateway, shape: str
) -> None:
    marker, marker_hex = _marker(shape)
    with wire_server(_peer) as wire, gateway.scenario() as scenario:
        model: Final = _vertex_deployment(gateway, scenario, wire.url, _TOOL_MODEL)
        response: Final = gateway.request("POST", "/v1/chat/completions", _chat_body(model, marker, stream=True))
        assert _content_frames(response) == (_tool_text(shape, marker_hex),), response.text
        _one_delivery(wire, marker_hex)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
@pytest.mark.parametrize("schema", [_OPTIONAL_SCHEMA, {"type": "object"}], ids=["no_required", "no_properties"])
@pytest.mark.parametrize(
    ("shape", "unwrapped"), [("wrapped", True), ("parameter_label", False)], ids=["wrapped", "label"]
)
def test_vertex_chat_json_schema_without_required_keys_unwraps_only_the_tool_named_envelope(
    gateway: Gateway, schema: dict[str, JsonValue], shape: str, unwrapped: bool, stream: bool
) -> None:
    marker, marker_hex = _marker(shape)
    response_format: Final[dict[str, JsonValue]] = {
        "type": "json_schema",
        "json_schema": {"name": "risk", "schema": schema},
    }
    with wire_server(_peer) as wire, gateway.scenario() as scenario:
        model: Final = _vertex_deployment(gateway, scenario, wire.url, _TOOL_MODEL)
        response: Final = gateway.request(
            "POST", "/v1/chat/completions", _chat_body(model, marker, stream=stream, response_format=response_format)
        )
        content: Final = _stream_content(response) if stream else _chat_content(response)
        assert content == (_payload(marker_hex) if unwrapped else _tool_input(shape, marker_hex)), response.text
        if stream:
            expected_frames: Final = 1 if unwrapped else _fragment_count(shape, marker_hex)
            assert len(_content_frames(response)) == expected_frames, response.text
        _one_delivery(wire, marker_hex)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_vertex_chat_json_schema_that_declares_the_wrapper_key_keeps_the_answer_whole(
    gateway: Gateway, stream: bool
) -> None:
    marker, marker_hex = _marker("values")
    response_format: Final[dict[str, JsonValue]] = {
        "type": "json_schema",
        "json_schema": {"name": "valued", "schema": _VALUES_SCHEMA},
    }
    with wire_server(_peer) as wire, gateway.scenario() as scenario:
        model: Final = _vertex_deployment(gateway, scenario, wire.url, _TOOL_MODEL)
        response: Final = gateway.request(
            "POST", "/v1/chat/completions", _chat_body(model, marker, stream=stream, response_format=response_format)
        )
        content: Final = _stream_content(response) if stream else _chat_content(response)
        assert content == _tool_input("values", marker_hex), response.text
        if stream:
            assert len(_content_frames(response)) == _fragment_count("values", marker_hex), response.text
        _one_delivery(wire, marker_hex)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
@pytest.mark.parametrize(
    ("schema", "wrapper_unwrapped"),
    [
        (_ADDITIONAL_OBJECT_SCHEMA, False),
        (_PATTERN_SCHEMA, False),
        (_ADDITIONAL_TRUE_SCHEMA, False),
        (_CLOSED_OPTIONAL_SCHEMA, True),
    ],
    ids=["additional_object", "pattern_properties", "additional_true", "additional_false"],
)
@pytest.mark.parametrize("shape", ["undeclared_wrapper", "wrapped"])
def test_vertex_chat_json_schema_that_allows_undeclared_keys_keeps_an_undeclared_wrapper_whole(
    gateway: Gateway, shape: str, schema: dict[str, JsonValue], wrapper_unwrapped: bool, stream: bool
) -> None:
    marker, marker_hex = _marker(shape)
    unwrapped: Final = shape == "wrapped" or wrapper_unwrapped
    response_format: Final[dict[str, JsonValue]] = {
        "type": "json_schema",
        "json_schema": {"name": "dynamic", "schema": schema},
    }
    with wire_server(_peer) as wire, gateway.scenario() as scenario:
        model: Final = _vertex_deployment(gateway, scenario, wire.url, _TOOL_MODEL)
        response: Final = gateway.request(
            "POST", "/v1/chat/completions", _chat_body(model, marker, stream=stream, response_format=response_format)
        )
        content: Final = _stream_content(response) if stream else _chat_content(response)
        assert content == (_payload(marker_hex) if unwrapped else _tool_input(shape, marker_hex)), response.text
        if stream:
            expected_frames: Final = 1 if unwrapped else _fragment_count(shape, marker_hex)
            assert len(_content_frames(response)) == expected_frames, response.text
        _one_delivery(wire, marker_hex)


def test_vertex_responses_stream_json_schema_output_text_is_the_schema_object(gateway: Gateway) -> None:
    marker, marker_hex = _marker("wrapped")
    with wire_server(_peer) as wire, gateway.scenario() as scenario:
        model: Final = _vertex_deployment(gateway, scenario, wire.url, _TOOL_MODEL)
        response: Final = gateway.request("POST", "/v1/responses", {**_responses_body(model, marker), "stream": True})
        assert _responses_stream_text(response) == _payload(marker_hex), response.text
        sent: Final = _one_delivery(wire, marker_hex)
        assert sent.target == f"{_VERTEX_MODELS}{_TOOL_MODEL}{_VERTEX_STREAM_SUFFIX}", sent.target


@pytest.mark.parametrize("shape", ["other_tool", "named_input", "named_two_objects", "named_string", "named_partial"])
def test_vertex_chat_json_schema_unwraps_a_tool_named_envelope_only_around_one_object(
    gateway: Gateway, shape: str
) -> None:
    marker, marker_hex = _marker(shape)
    with wire_server(_peer) as wire, gateway.scenario() as scenario:
        model: Final = _vertex_deployment(gateway, scenario, wire.url, _TOOL_MODEL)
        response: Final = gateway.request("POST", "/v1/chat/completions", _chat_body(model, marker))
        assert _chat_content(response) == _expected(shape, marker_hex), response.text
        _one_delivery(wire, marker_hex)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
@pytest.mark.parametrize("shape", ["name_property", "name_value_plain"])
def test_vertex_chat_json_schema_with_a_name_property_unwraps_only_the_tool_named_wrapper(
    gateway: Gateway, shape: str, stream: bool
) -> None:
    marker, marker_hex = _marker(shape)
    with wire_server(_peer) as wire, gateway.scenario() as scenario:
        model: Final = _vertex_deployment(gateway, scenario, wire.url, _TOOL_MODEL)
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            _chat_body(model, marker, stream=stream, response_format=_NAME_RESPONSE_FORMAT),
        )
        content: Final = _stream_content(response) if stream else _chat_content(response)
        assert content == _expected(shape, marker_hex), response.text
        if stream and shape == "name_property":
            assert len(_content_frames(response)) == 1, response.text
        sent: Final = _one_delivery(wire, marker_hex)
        assert _json_tool_schema(sent)["properties"] == _NAME_SCHEMA["properties"], sent.body


@pytest.mark.parametrize("shape", ["wrapped", "inner_incomplete"])
def test_vertex_chat_json_schema_validation_runs_on_the_unwrapped_object(gateway: Gateway, shape: str) -> None:
    marker, marker_hex = _marker(shape)
    with wire_server(_peer) as wire, gateway.scenario() as scenario:
        model: Final = _vertex_deployment(gateway, scenario, wire.url, _TOOL_MODEL)
        response: Final = gateway.request(
            "POST", "/v1/chat/completions", {**_chat_body(model, marker), "enable_json_schema_validation": True}
        )
        if shape == "wrapped":
            assert _chat_content(response) == _payload(marker_hex), response.text
        else:
            assert response.status_code >= 400, response.text
            assert "JSONSchemaValidationError" in response.text, response.text
        _one_delivery(wire, marker_hex)
        assert gateway.request("GET", "/health/liveliness").status_code == 200


def test_vertex_chat_json_schema_next_to_a_user_tool_unwraps_the_envelope_and_keeps_the_tool_call(
    gateway: Gateway,
) -> None:
    marker, marker_hex = _marker("wrapped")
    with wire_server(_peer) as wire, gateway.scenario() as scenario:
        model: Final = _vertex_deployment(gateway, scenario, wire.url, _TOOL_MODEL)
        response: Final = gateway.request(
            "POST", "/v1/chat/completions", _chat_body(model, marker, tools=[_USER_TOOL_SPEC])
        )
        assert response.status_code == 200, response.text
        message: Final = object_value(
            object_value(list_value(_JSON_OBJECT.validate_json(response.content)["choices"])[0])["message"]
        )
        assert json.loads(string_value(message["content"])) == _payload(marker_hex), response.text
        (tool_call,) = tuple(object_value(call) for call in list_value(message["tool_calls"]))
        function: Final = object_value(tool_call["function"])
        assert function["name"] == _USER_TOOL, response.text
        assert json.loads(string_value(function["arguments"])) == {"q": marker_hex}, response.text
        sent: Final = _one_delivery(wire, marker_hex)
        assert _tool_names(sent.body) == frozenset({_USER_TOOL, _JSON_TOOL}), sent.body


@pytest.mark.parametrize("member", _FAMILY)
def test_tool_path_family_unwraps_the_envelope(gateway: Gateway, member: str) -> None:
    marker, marker_hex = _marker("wrapped")
    with wire_server(_peer) as wire, gateway.scenario() as scenario:
        model: Final = _family_deployment(gateway, scenario, wire.url, member)
        response: Final = gateway.request("POST", "/v1/chat/completions", _chat_body(model, marker))
        assert _chat_content(response) == _payload(marker_hex), response.text
        sent: Final = _one_delivery(wire, marker_hex)
        assert _json_tool_schema(sent)["properties"] == _SCHEMA["properties"], sent.body


@pytest.mark.parametrize("member", _STREAM_FAMILY)
def test_tool_path_family_stream_unwraps_the_envelope(gateway: Gateway, member: str) -> None:
    marker, marker_hex = _marker("wrapped")
    with wire_server(_peer) as wire, gateway.scenario() as scenario:
        model: Final = _family_deployment(gateway, scenario, wire.url, member)
        response: Final = gateway.request("POST", "/v1/chat/completions", _chat_body(model, marker, stream=True))
        frames: Final = _content_frames(response)
        assert frames, response.text
        assert json.loads("".join(frames)) == _payload(marker_hex), response.text
        assert len(frames) == 1, frames
        sent: Final = _one_delivery(wire, marker_hex)
        assert _json_tool_schema(sent)["properties"] == _SCHEMA["properties"], sent.body


@pytest.mark.parametrize("accepts_forced_tools", [True, False], ids=["accepts_forced_tools", "rejects_forced_tools"])
def test_vertex_chat_json_schema_forces_the_json_tool_only_when_the_model_accepts_forced_tools(
    gateway: Gateway, accepts_forced_tools: bool
) -> None:
    marker, marker_hex = _marker("plain")
    backend: Final = (
        _TOOL_MODEL if accepts_forced_tools else _claude_rejecting_forced_tools(gateway, "vertex_ai/", "claude")
    )
    with wire_server(_peer) as wire, gateway.scenario() as scenario:
        model: Final = _vertex_deployment(gateway, scenario, wire.url, backend)
        response: Final = gateway.request("POST", "/v1/chat/completions", _chat_body(model, marker))
        assert _chat_content(response) == _payload(marker_hex), response.text
        sent: Final = _one_delivery(wire, marker_hex)
        assert _json_tool_schema(sent)["properties"] == _SCHEMA["properties"], sent.body
        assert sent.body.get("tool_choice") == (_FORCED_JSON_TOOL if accepts_forced_tools else None), sent.body
        assert "output_config" not in sent.body and "output_format" not in sent.body, sent.body


def test_vertex_chat_json_schema_on_a_model_rejecting_forced_tools_maps_reasoning_effort_like_plain_chat(
    gateway: Gateway,
) -> None:
    backend: Final = _claude_rejecting_forced_tools(gateway, "vertex_ai/", "claude")
    structured_marker, structured_hex = _marker("plain")
    plain_marker, plain_hex = _marker("plain")
    with wire_server(_peer) as wire, gateway.scenario() as scenario:
        model: Final = _vertex_deployment(gateway, scenario, wire.url, backend)
        structured: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {**_effort_body(model, structured_marker), "response_format": _RESPONSE_FORMAT},
        )
        assert _chat_content(structured) == _payload(structured_hex), structured.text
        structured_sent: Final = _one_delivery(wire, structured_hex)
        plain: Final = gateway.request("POST", "/v1/chat/completions", _effort_body(model, plain_marker))
        assert plain.status_code == 200, plain.text
        plain_sent: Final = _one_delivery(wire, plain_hex)
    assert "thinking" in plain_sent.body, plain_sent.body
    assert _reasoning_fields(structured_sent) == _reasoning_fields(plain_sent), (structured_sent.body, plain_sent.body)


def test_bedrock_invoke_json_schema_sends_the_json_tool_unforced_when_the_model_rejects_forced_tools(
    gateway: Gateway,
) -> None:
    marker, marker_hex = _marker("plain")
    backend: Final = _claude_rejecting_forced_tools(gateway, "", "anthropic.claude")
    with wire_server(_peer) as wire, gateway.scenario() as scenario:
        model: Final = _bedrock_invoke_deployment(scenario, wire.url, backend)
        response: Final = gateway.request("POST", "/v1/chat/completions", _chat_body(model, marker))
        assert _chat_content(response) == _payload(marker_hex), response.text
        sent: Final = _one_delivery(wire, marker_hex)
        assert sent.target == f"/model/{backend}/invoke", sent.target
        assert _json_tool_schema(sent)["properties"] == _SCHEMA["properties"], sent.body
        assert "tool_choice" not in sent.body, sent.body


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_vertex_chat_json_schema_native_folds_output_format_into_output_config(gateway: Gateway, stream: bool) -> None:
    marker, marker_hex = _marker("native")
    with wire_server(_peer) as wire, gateway.scenario() as scenario:
        model: Final = _vertex_deployment(gateway, scenario, wire.url, _NATIVE_MODEL)
        response: Final = gateway.request("POST", "/v1/chat/completions", _chat_body(model, marker, stream=stream))
        content: Final = _stream_content(response) if stream else _chat_content(response)
        assert content == _payload(marker_hex), response.text
        sent: Final = _one_delivery(wire, marker_hex)
        assert _folded_format(sent) == _OUTPUT_FORMAT, sent.body
        assert "tools" not in sent.body, sent.body


def test_vertex_chat_native_through_the_openai_sdk_folds_output_format(gateway: Gateway) -> None:
    marker, marker_hex = _marker("native")
    with wire_server(_peer) as wire, gateway.scenario() as scenario:
        model: Final = _vertex_deployment(gateway, scenario, wire.url, _NATIVE_MODEL)
        with _openai_client(gateway) as client:
            reply: Final = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": marker}],
                response_format={"type": "json_schema", "json_schema": {"name": "risk", "schema": _SCHEMA}},
            )
        assert json.loads(reply.choices[0].message.content or "") == _payload(marker_hex), reply
        assert _folded_format(_one_delivery(wire, marker_hex)) == _OUTPUT_FORMAT


def test_vertex_responses_native_folds_output_format(gateway: Gateway) -> None:
    marker, marker_hex = _marker("native")
    with wire_server(_peer) as wire, gateway.scenario() as scenario:
        model: Final = _vertex_deployment(gateway, scenario, wire.url, _NATIVE_MODEL)
        response: Final = gateway.request("POST", "/v1/responses", _responses_body(model, marker))
        assert _responses_text(response) == _payload(marker_hex), response.text
        assert _folded_format(_one_delivery(wire, marker_hex)) == _OUTPUT_FORMAT


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_vertex_messages_folds_output_format_into_output_config(gateway: Gateway, stream: bool) -> None:
    marker, marker_hex = _marker("native")
    with wire_server(_peer) as wire, gateway.scenario() as scenario:
        model: Final = _vertex_deployment(gateway, scenario, wire.url, _NATIVE_MODEL)
        response: Final = gateway.request(
            "POST",
            "/v1/messages",
            _messages_body(model, marker, stream=stream, output_format=_OUTPUT_FORMAT),
            headers={"anthropic-beta": _STRUCTURED_OUTPUTS_BETA},
        )
        text: Final = _messages_stream_text(response) if stream else _messages_text(response)
        assert text == _payload(marker_hex), response.text
        assert f"msg_{marker_hex}" in response.text, response.text
        sent: Final = _one_delivery(wire, marker_hex)
        assert _folded_format(sent) == _OUTPUT_FORMAT, sent.body
        assert sent.target.endswith(_VERTEX_STREAM_SUFFIX if stream else ":rawPredict"), sent.target


def test_vertex_messages_through_the_anthropic_sdk_folds_output_format(gateway: Gateway) -> None:
    marker, marker_hex = _marker("native")
    with wire_server(_peer) as wire, gateway.scenario() as scenario:
        model: Final = _vertex_deployment(gateway, scenario, wire.url, _NATIVE_MODEL)
        with _anthropic_client(gateway) as client:
            reply: Final = client.beta.messages.create(
                model=model,
                max_tokens=64,
                messages=[{"role": "user", "content": marker}],
                output_format={"type": "json_schema", "schema": _SCHEMA},
                betas=[_STRUCTURED_OUTPUTS_BETA],
            )
        assert reply.id == f"msg_{marker_hex}", reply
        assert _text_block_json(_JSON_OBJECT.validate_json(reply.model_dump_json())) == _payload(marker_hex), reply
        assert _folded_format(_one_delivery(wire, marker_hex)) == _OUTPUT_FORMAT


def test_vertex_messages_through_the_async_anthropic_sdk_folds_output_format(gateway: Gateway) -> None:
    marker, marker_hex = _marker("native")

    async def generate(model: str) -> tuple[str, str]:
        async with anthropic.AsyncAnthropic(
            base_url=str(gateway.client.base_url),
            api_key=gateway.key,
            max_retries=0,
            http_client=httpx.AsyncClient(trust_env=False, timeout=60),
        ) as client:
            reply: Final = await client.beta.messages.create(
                model=model,
                max_tokens=64,
                messages=[{"role": "user", "content": marker}],
                output_format={"type": "json_schema", "schema": _SCHEMA},
                betas=[_STRUCTURED_OUTPUTS_BETA],
            )
        return reply.id, reply.model_dump_json()

    with wire_server(_peer) as wire, gateway.scenario() as scenario:
        model: Final = _vertex_deployment(gateway, scenario, wire.url, _NATIVE_MODEL)
        reply_id, dumped = asyncio.run(generate(model))
        assert reply_id == f"msg_{marker_hex}", dumped
        assert _text_block_json(_JSON_OBJECT.validate_json(dumped)) == _payload(marker_hex), dumped
        assert _folded_format(_one_delivery(wire, marker_hex)) == _OUTPUT_FORMAT


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
@pytest.mark.parametrize(
    ("callers_config", "expected_output_config"),
    [
        ({"format": _CALLERS_FORMAT}, {"format": _OUTPUT_FORMAT}),
        ({"effort": "high", "format": _CALLERS_FORMAT}, {"effort": "high", "format": _OUTPUT_FORMAT}),
    ],
    ids=["format", "effort_and_format"],
)
def test_vertex_messages_output_format_wins_over_the_callers_output_config_format(
    gateway: Gateway, callers_config: dict[str, JsonValue], expected_output_config: dict[str, JsonValue], stream: bool
) -> None:
    marker, marker_hex = _marker("native")
    with wire_server(_peer) as wire, gateway.scenario() as scenario:
        model: Final = _vertex_deployment(gateway, scenario, wire.url, _NATIVE_MODEL)
        response: Final = gateway.request(
            "POST",
            "/v1/messages",
            _messages_body(model, marker, stream=stream, output_format=_OUTPUT_FORMAT, output_config=callers_config),
        )
        text: Final = _messages_stream_text(response) if stream else _messages_text(response)
        assert text == _payload(marker_hex), response.text
        sent: Final = _one_delivery(wire, marker_hex)
        assert "output_format" not in sent.body, sent.body
        assert sent.body["output_config"] == expected_output_config, sent.body


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
@pytest.mark.parametrize(
    "schema",
    [_ADDITIONAL_OBJECT_SCHEMA, _PATTERN_SCHEMA, _ADDITIONAL_TRUE_SCHEMA],
    ids=["additional_object", "pattern_properties", "additional_true"],
)
def test_vertex_messages_json_schema_that_allows_undeclared_keys_keeps_the_answer_whole(
    gateway: Gateway, schema: dict[str, JsonValue], stream: bool
) -> None:
    marker, marker_hex = _marker("undeclared_wrapper")
    output_format: Final[dict[str, JsonValue]] = {"type": "json_schema", "schema": schema}
    with wire_server(_peer) as wire, gateway.scenario() as scenario:
        model: Final = _vertex_deployment(gateway, scenario, wire.url, _NATIVE_MODEL)
        response: Final = gateway.request(
            "POST",
            "/v1/messages",
            _messages_body(model, marker, stream=stream, output_format=output_format),
            headers={"anthropic-beta": _STRUCTURED_OUTPUTS_BETA},
        )
        text: Final = _messages_stream_text(response) if stream else _messages_text(response)
        assert text == _tool_input("undeclared_wrapper", marker_hex), response.text
        assert _folded_format(_one_delivery(wire, marker_hex)) == output_format


def test_vertex_messages_replaces_a_non_dict_output_config_with_the_folded_format(gateway: Gateway) -> None:
    marker, marker_hex = _marker("native")
    with wire_server(_peer) as wire, gateway.scenario() as scenario:
        model: Final = _vertex_deployment(gateway, scenario, wire.url, _NATIVE_MODEL)
        response: Final = gateway.request(
            "POST",
            "/v1/messages",
            _messages_body(model, marker, output_format=_OUTPUT_FORMAT, output_config="bogus"),
        )
        assert _messages_text(response) == _payload(marker_hex), response.text
        sent: Final = _one_delivery(wire, marker_hex)
        assert sent.body["output_config"] == {"format": _OUTPUT_FORMAT}, sent.body


def test_vertex_messages_drops_effort_but_keeps_the_folded_format_on_haiku(gateway: Gateway) -> None:
    marker, marker_hex = _marker("native")
    with wire_server(_peer) as wire, gateway.scenario() as scenario:
        model: Final = _vertex_deployment(gateway, scenario, wire.url, _EFFORT_GATED_MODEL)
        response: Final = gateway.request(
            "POST",
            "/v1/messages",
            _messages_body(model, marker, output_format=_OUTPUT_FORMAT, output_config={"effort": "high"}),
        )
        assert _messages_text(response) == _payload(marker_hex), response.text
        sent: Final = _one_delivery(wire, marker_hex)
        assert sent.body["output_config"] == {"format": _OUTPUT_FORMAT}, sent.body


def test_vertex_messages_forwards_an_explicit_output_config_format_unchanged(gateway: Gateway) -> None:
    marker, marker_hex = _marker("native")
    with wire_server(_peer) as wire, gateway.scenario() as scenario:
        model: Final = _vertex_deployment(gateway, scenario, wire.url, _NATIVE_MODEL)
        response: Final = gateway.request(
            "POST", "/v1/messages", _messages_body(model, marker, output_config={"format": _OUTPUT_FORMAT})
        )
        assert _messages_text(response) == _payload(marker_hex), response.text
        sent: Final = _one_delivery(wire, marker_hex)
        assert sent.body["output_config"] == {"format": _OUTPUT_FORMAT}, sent.body


@pytest.mark.parametrize(
    ("extra", "expected_output_config"),
    [
        ({"output_format": _OUTPUT_FORMAT, "output_config": None}, {"format": _OUTPUT_FORMAT}),
        ({"output_format": _OUTPUT_FORMAT, "output_config": {"format": None}}, {"format": _OUTPUT_FORMAT}),
        ({"output_format": _OUTPUT_FORMAT, "output_config": {}}, {"format": _OUTPUT_FORMAT}),
        (
            {"output_format": _OUTPUT_FORMAT, "output_config": {"effort": "high"}},
            {"effort": "high", "format": _OUTPUT_FORMAT},
        ),
        ({"output_format": None, "output_config": {"format": _OUTPUT_FORMAT}}, {"format": _OUTPUT_FORMAT}),
        ({"output_format": None}, None),
        ({"output_config": None}, None),
    ],
    ids=[
        "null_config",
        "null_format_key",
        "empty_config",
        "effort_config",
        "null_output_format_beside_callers",
        "null_output_format_alone",
        "null_config_alone",
    ],
)
def test_vertex_messages_output_format_beside_an_explicit_output_config(
    gateway: Gateway, extra: dict[str, JsonValue], expected_output_config: JsonValue
) -> None:
    marker, marker_hex = _marker("native")
    with wire_server(_peer) as wire, gateway.scenario() as scenario:
        model: Final = _vertex_deployment(gateway, scenario, wire.url, _NATIVE_MODEL)
        response: Final = gateway.request("POST", "/v1/messages", _messages_body(model, marker, **extra))
        if expected_output_config is None:
            assert response.status_code == 200, response.text
            assert f"plain text {marker_hex}" in response.text, response.text
        else:
            assert _messages_text(response) == _payload(marker_hex), response.text
        sent: Final = _one_delivery(wire, marker_hex)
        assert "output_format" not in sent.body, sent.body
        if expected_output_config is None:
            assert "output_config" not in sent.body, sent.body
        else:
            assert sent.body["output_config"] == expected_output_config, sent.body


def test_anthropic_messages_forwards_top_level_output_format_verbatim(gateway: Gateway) -> None:
    marker, marker_hex = _marker("native")
    with wire_server(_peer) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"anthropic/{_NATIVE_MODEL}", api_base=wire.url, api_key=_ANTHROPIC_KEY)
        response: Final = gateway.request(
            "POST",
            "/v1/messages",
            _messages_body(model, marker, output_format=_OUTPUT_FORMAT),
            headers={"anthropic-beta": _STRUCTURED_OUTPUTS_BETA},
        )
        assert _messages_text(response) == _payload(marker_hex), response.text
        sent: Final = _one_delivery(wire, marker_hex)
        assert sent.target == "/v1/messages", sent.target
        assert sent.body["output_format"] == _OUTPUT_FORMAT, sent.body
        assert "output_config" not in sent.body, sent.body


def test_vertex_chat_without_response_format_sends_no_tool_and_no_output_config(gateway: Gateway) -> None:
    marker, marker_hex = _marker("plain_text")
    with wire_server(_peer) as wire, gateway.scenario() as scenario:
        model: Final = _vertex_deployment(gateway, scenario, wire.url, _TOOL_MODEL)
        response: Final = gateway.request(
            "POST", "/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": marker}]}
        )
        assert response.status_code == 200, response.text
        assert f"plain text {marker_hex}" in response.text, response.text
        sent: Final = _one_delivery(wire, marker_hex)
        assert not {"tools", "output_config", "output_format"} & sent.body.keys(), sent.body


@pytest.mark.parametrize(
    "output_format",
    [5, [], {}, "x" * 5000],
    ids=["int", "list", "empty", "5kb"],
)
def test_vertex_messages_malformed_output_format_reaches_the_caller_as_the_peer_400(
    gateway: Gateway, output_format: JsonValue
) -> None:
    marker, marker_hex = _marker("native")
    with wire_server(_peer) as wire, gateway.scenario() as scenario:
        model: Final = _vertex_deployment(gateway, scenario, wire.url, _NATIVE_MODEL)
        response: Final = gateway.request(
            "POST", "/v1/messages", _messages_body(model, marker, output_format=output_format)
        )
        assert response.status_code == 400, response.text
        assert _BAD_FORMAT in response.text, response.text
        sent: Final = _one_delivery(wire, marker_hex)
        assert "output_format" not in sent.body, sent.body
        assert object_value(sent.body["output_config"])["format"] == output_format, sent.body
        assert gateway.request("GET", "/health/liveliness").status_code == 200


def test_vertex_messages_duplicate_output_format_keys_fold_the_last_value(gateway: Gateway) -> None:
    marker, marker_hex = _marker("native")
    with wire_server(_peer) as wire, gateway.scenario() as scenario:
        model: Final = _vertex_deployment(gateway, scenario, wire.url, _NATIVE_MODEL)
        raw: Final = (
            json.dumps(_messages_body(model, marker, output_format=5))[:-1]
            + f', "output_format": {json.dumps(_OUTPUT_FORMAT)}}}'
        )
        response: Final = gateway.client.post(
            "/v1/messages",
            content=raw,
            headers={"Authorization": f"Bearer {gateway.key}", "Content-Type": "application/json"},
        )
        assert _messages_text(response) == _payload(marker_hex), response.text
        assert _folded_format(_one_delivery(wire, marker_hex)) == _OUTPUT_FORMAT


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "object", "properties": ["tier", "summary"]},
        {"type": "object", "properties": "tier"},
        {"type": "object", "properties": None},
    ],
    ids=["properties_list", "properties_string", "properties_null"],
)
@pytest.mark.parametrize("shape", ["wrapped", "other_tool"])
@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_vertex_chat_malformed_response_format_schema_keeps_the_tool_input(
    gateway: Gateway, schema: JsonValue, shape: str, stream: bool
) -> None:
    marker, marker_hex = _marker(shape)
    with wire_server(_peer) as wire, gateway.scenario() as scenario:
        model: Final = _vertex_deployment(gateway, scenario, wire.url, _TOOL_MODEL)
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            _chat_body(
                model,
                marker,
                stream=stream,
                response_format={"type": "json_schema", "json_schema": {"name": "risk", "schema": schema}},
            ),
        )
        content: Final = _stream_content(response) if stream else _chat_content(response)
        assert content == _tool_input(shape, marker_hex), response.text
        _one_delivery(wire, marker_hex)
        assert gateway.request("GET", "/health/liveliness").status_code == 200


@pytest.mark.parametrize("status", [400, 403, 404])
def test_vertex_chat_peer_error_reaches_the_caller_and_the_deployment_keeps_serving(
    gateway: Gateway, status: int
) -> None:
    marker, marker_hex = _marker("wrapped")
    after, after_hex = _marker("wrapped")
    with gateway.scenario() as scenario:
        with wire_server(_failing(status)) as failing:
            port: Final = int(failing.url.rsplit(":", 1)[1])
            model: Final = _vertex_deployment(gateway, scenario, failing.url, _TOOL_MODEL)
            response: Final = gateway.request("POST", "/v1/chat/completions", _chat_body(model, marker))
            assert response.status_code == status, response.text
            assert f"scripted {status} {marker_hex}" in response.text, response.text
            _one_delivery(failing, marker_hex)
        with wire_server(_peer, port=port) as wire:
            recovered: Final = gateway.request("POST", "/v1/chat/completions", _chat_body(model, after))
            assert _chat_content(recovered) == _payload(after_hex), recovered.text
            _one_delivery(wire, after_hex)


def test_vertex_chat_unauthenticated_request_never_reaches_the_peer(gateway: Gateway) -> None:
    marker, _ = _marker("wrapped")
    with wire_server(_peer) as wire, gateway.scenario() as scenario:
        model: Final = _vertex_deployment(gateway, scenario, wire.url, _TOOL_MODEL)
        response: Final = gateway.request(
            "POST", "/v1/chat/completions", _chat_body(model, marker), key="sk-not-a-key-this-proxy-issued"
        )
        assert response.status_code == 401, response.text
        assert wire.drain() == ()


def test_vertex_chat_repeated_requests_each_reach_the_peer_and_log_spend_once(gateway: Gateway) -> None:
    markers: Final = tuple(_marker("wrapped") for _ in range(2))
    with wire_server(_peer) as wire, gateway.scenario() as scenario:
        model: Final = _vertex_deployment(gateway, scenario, wire.url, _TOOL_MODEL)
        responses: Final = tuple(
            gateway.request("POST", "/v1/chat/completions", _chat_body(model, marker)) for marker, _ in markers
        )
        for response, (_, marker_hex) in zip(responses, markers, strict=True):
            assert _chat_content(response) == _payload(marker_hex), response.text
        delivered: Final = _delivered(wire)
        assert len(delivered) == 2, delivered
        _seen_once(delivered, tuple(marker_hex for _, marker_hex in markers))
        request_ids: Final = tuple(_response_id(response) for response in responses)
        assert len(set(request_ids)) == 2, request_ids
        for request_id in request_ids:
            assert _spend_rows(request_id)[0]["request_id"] == request_id


def test_concurrent_burst_across_both_workers_unwraps_every_envelope(gateway: Gateway) -> None:
    with ExitStack() as stack:
        wire: Final = stack.enter_context(wire_server(_peer))
        scenario: Final = stack.enter_context(gateway.scenario())
        model: Final = _vertex_deployment(gateway, scenario, wire.url, _TOOL_MODEL)
        shapes: Final = tuple(_BURST_SHAPES[index % len(_BURST_SHAPES)] for index in range(12))
        markers: Final = tuple(_marker(shape) for shape in shapes)
        streams: Final = tuple(index % 3 == 2 for index in range(len(markers)))
        bodies: Final = tuple(
            _chat_body(model, marker, stream=stream) for (marker, _), stream in zip(markers, streams, strict=True)
        )
        clients: Final = _clients(stack, str(gateway.client.base_url), len(bodies))
        pool: Final = stack.enter_context(ThreadPoolExecutor(max_workers=len(bodies)))
        responses: Final = tuple(
            pool.map(
                lambda pair: _post(pair[0], gateway.key, "/v1/chat/completions", pair[1]),
                zip(clients, bodies, strict=True),
            )
        )
        for response, shape, (_, marker_hex), stream in zip(responses, shapes, markers, streams, strict=True):
            observed: Final = _stream_content(response) if stream else _chat_content(response)
            expected: Final = _expected(shape, marker_hex)
            assert observed == expected, (shape, stream, response.text)
        _seen_once(_delivered(wire), tuple(marker_hex for _, marker_hex in markers))


def test_upstream_outage_mid_run_recovers_with_envelopes_still_unwrapped(gateway: Gateway) -> None:
    with ExitStack() as stack:
        clients: Final = _clients(stack, str(gateway.client.base_url), 8)
        pool: Final = stack.enter_context(ThreadPoolExecutor(max_workers=len(clients)))
        scenario: Final = stack.enter_context(gateway.scenario())

        streams: Final = tuple(index % 2 == 1 for index in range(len(clients)))

        def wave() -> tuple[tuple[httpx.Response | None, str, bool], ...]:
            markers: Final = tuple(_marker("wrapped") for _ in clients)
            responses: Final = tuple(
                pool.map(
                    lambda item: _post_or_dropped(
                        item[0], gateway.key, "/v1/chat/completions", _chat_body(model, item[1][0], stream=item[2])
                    ),
                    zip(clients, markers, streams, strict=True),
                )
            )
            return tuple(zip(responses, tuple(marker_hex for _, marker_hex in markers), streams, strict=True))

        def unwrapped(wire: Wire, results: tuple[tuple[httpx.Response | None, str, bool], ...]) -> None:
            for response, marker_hex, stream in results:
                assert response is not None, results
                observed: Final = _stream_content(response) if stream else _chat_content(response)
                assert observed == _payload(marker_hex), response.text
            _seen_once(_delivered(wire), tuple(marker_hex for _, marker_hex, _ in results))

        with wire_server(_peer) as wire:
            port: Final = int(wire.url.rsplit(":", 1)[1])
            model: Final = _vertex_deployment(gateway, scenario, wire.url, _TOOL_MODEL)
            unwrapped(wire, wave())
        outage: Final = wave()
        assert all(response is not None and response.status_code >= 500 for response, _, _ in outage), outage
        assert gateway.request("GET", "/health/liveliness").status_code == 200
        with wire_server(_peer, port=port) as revived:
            unwrapped(revived, wave())


@pytest.mark.timeout(_OWNED_PROXY_CELL_SECONDS)
def test_worker_sigkill_mid_burst_leaves_the_replacement_unwrapping_envelopes(gateway: Gateway, tmp_path: Path) -> None:
    held: Final[SimpleQueue[str]] = SimpleQueue()
    release: Final = threading.Event()
    with ExitStack() as stack:
        wire: Final = stack.enter_context(wire_server(_holding(held, release)))
        stack.callback(release.set)
        config: Final = _owned_config(tmp_path / "structured-output-worker-kill.yaml", gateway, wire.url)
        owned: Final = stack.enter_context(owned_proxy_process(gateway, tmp_path, {}, config=config, workers=2))
        proxy_url: Final = owned.gateway.client.base_url
        workers: Final = eventually(
            lambda: tuple(int(pid) for pid in _STARTED_WORKER.findall(owned.log.read_text())),
            lambda pids: len(pids) == 2,
            seconds=30,
        )
        clients: Final = _clients(stack, str(proxy_url), 12)
        pool: Final = stack.enter_context(ThreadPoolExecutor(max_workers=len(clients)))
        ports: Final = tuple(_local_port(client) for client in clients)
        markers: Final = tuple(_marker("wrapped") for _ in clients)
        futures: Final = tuple(
            pool.submit(_post_or_dropped, client, gateway.key, "/v1/chat/completions", _chat_body(_OWNED_MODEL, marker))
            for client, (marker, _) in zip(clients, markers, strict=True)
        )
        eventually(held.qsize, lambda size: size == len(clients), seconds=30)
        shares: Final = {pid: _accepted_client_ports(pid, proxy_url.port or 0) & frozenset(ports) for pid in workers}
        assert sum(map(len, shares.values())) == len(clients), shares
        victim: Final = min((pid for pid in workers if shares[pid]), key=lambda pid: len(shares[pid]))
        psutil.Process(victim).send_signal(signal.SIGKILL)
        release.set()
        results: Final = tuple(future.result(timeout=60) for future in futures)
        for port, result, (_, marker_hex) in zip(ports, results, markers, strict=True):
            if port in shares[victim]:
                assert result is None, (port, result, shares)
            else:
                assert result is not None and _chat_content(result) == _payload(marker_hex), (port, result, shares)
        eventually(lambda: len(_STARTED_WORKER.findall(owned.log.read_text())), lambda started: started >= 3, 120)
        second_wave: Final = tuple(_marker("wrapped") for _ in range(6))
        for marker, marker_hex in second_wave:
            after: Final = _post(
                owned.gateway.client, gateway.key, "/v1/chat/completions", _chat_body(_OWNED_MODEL, marker)
            )
            assert _chat_content(after) == _payload(marker_hex), after.text
        _seen_once(_delivered(wire), tuple(marker_hex for _, marker_hex in markers + second_wave))
        assert owned.process.poll() is None


@pytest.mark.parametrize(
    ("shape", "response_format"),
    [
        ("unclosed_key", _RESPONSE_FORMAT),
        ("unclosed_name", _NAME_RESPONSE_FORMAT),
        ("unclosed_label", _RESPONSE_FORMAT),
        ("unclosed_label_key", _RESPONSE_FORMAT),
    ],
    ids=["unclosed_key", "unclosed_name", "unclosed_label", "unclosed_label_key"],
)
def test_vertex_chat_stream_never_closing_tool_input_stays_linear_and_keeps_health_responsive(
    gateway: Gateway, shape: str, response_format: dict[str, JsonValue], record_property: Callable[[str, object], None]
) -> None:
    latencies: Final[SimpleQueue[float]] = SimpleQueue()
    done: Final = threading.Event()
    with ExitStack() as stack:
        wire: Final = stack.enter_context(wire_server(_peer))
        scenario: Final = stack.enter_context(gateway.scenario())
        model: Final = _vertex_deployment(gateway, scenario, wire.url, _TOOL_MODEL)
        streaming, probing = _clients(stack, str(gateway.client.base_url), 2)
        _local_port(probing)

        def probe() -> None:
            while not done.wait(0.05):
                started: Final = time.perf_counter()
                assert probing.get("/health/liveliness").status_code == 200
                latencies.put(time.perf_counter() - started)

        prober: Final = threading.Thread(target=probe)
        prober.start()
        stack.callback(prober.join, 30)
        stack.callback(done.set)
        small, large = (
            _timed_stream(streaming, gateway.key, wire, model, size_shape, response_format, record_property)
            for size_shape in (f"{shape}_small", shape)
        )
        done.set()
        prober.join(timeout=30)
        probes: Final = tuple(latencies.get_nowait() for _ in range(latencies.qsize()))
        slowest: Final = max(probes, default=0.0)
        record_property("slowest_health_seconds", round(slowest, 3))
        record_property("health_probes", len(probes))
        assert large <= _TIMING_GROWTH_LIMIT * small, (shape, small, large)
        assert slowest <= _HEALTH_LATENCY_SECONDS, (shape, slowest, len(probes))


@pytest.mark.timeout(_OWNED_PROXY_CELL_SECONDS)
def test_anthropic_pass_through_stream_relays_the_tool_stream_verbatim_and_logs_spend(
    gateway: Gateway, tmp_path: Path
) -> None:
    marker, marker_hex = _marker("wrapped")
    overrides: Final = {"ANTHROPIC_API_BASE": "", "ANTHROPIC_API_KEY": _ANTHROPIC_KEY}
    with wire_server(_peer) as wire, gateway.scenario() as scenario:
        key: Final = scenario.key()
        with owned_proxy_process(gateway, tmp_path, {**overrides, "ANTHROPIC_API_BASE": wire.url}) as owned:
            response: Final = _post(
                owned.gateway.client,
                key,
                "/anthropic/v1/messages",
                _messages_body(_UNMAPPED_MODEL, marker, stream=True, tools=[_ANTHROPIC_JSON_TOOL]),
            )
            assert response.status_code == 200, response.text
            assert response.headers["content-type"].startswith(_SSE), response.headers
            expected: Final = _tool_reply(f"msg_{marker_hex}", "wrapped", marker_hex, False, "sse").chunks
            assert expected is not None
            assert response.content == b"".join(expected), response.text
            sent: Final = _one_delivery(wire, marker_hex)
            assert sent.target == "/v1/messages", sent.target
            assert _json_tool_schema(sent) == _SCHEMA, sent.body
            (row,) = _pass_through_spend_rows(key)
            assert row["model"] == _UNMAPPED_MODEL, row


def test_bedrock_pass_through_invoke_stream_relays_the_event_stream_verbatim(gateway: Gateway) -> None:
    marker, marker_hex = _marker("wrapped")
    with wire_server(_peer) as wire, gateway.scenario() as scenario:
        deployment: Final = scenario.model(
            model=f"bedrock/{_BEDROCK_INVOKE_MODEL}",
            api_base=wire.url,
            aws_access_key_id="AKIASCRIPTEDPROVIDER",
            aws_secret_access_key="scripted-secret",
            aws_region_name="us-east-1",
        )
        response: Final = gateway.request(
            "POST",
            f"/bedrock/model/{deployment}/invoke-with-response-stream",
            {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": marker}],
                "tools": [_ANTHROPIC_JSON_TOOL],
            },
        )
        assert response.status_code == 200, response.text
        assert response.headers["content-type"] == _EVENT_STREAM, response.headers
        expected: Final = _tool_reply(f"msg_{marker_hex}", "wrapped", marker_hex, False, "aws").chunks
        assert expected is not None
        assert response.content == b"".join(expected), response.content[:200]
        sent: Final = _one_delivery(wire, marker_hex)
        assert sent.target == f"/model/{_BEDROCK_INVOKE_MODEL}/invoke-with-response-stream", sent.target


@pytest.mark.timeout(_OWNED_PROXY_CELL_SECONDS)
@pytest.mark.parametrize("route", _PASS_THROUGH_ROUTES)
def test_pass_through_stream_relays_the_scripted_frames_verbatim(
    pass_through_proxy: tuple[OwnedProxy, Wire], session_gateway: Gateway, route: str
) -> None:
    owned, wire = pass_through_proxy
    marker_hex: Final = uuid.uuid4().hex
    call: Final = _pass_through_call(route, f"relay {marker_hex}", session_gateway.key)
    response: Final = owned.gateway.client.post(call.path, json=call.body, headers=call.headers)
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith(_SSE), response.headers
    assert response.content == b"".join(_RELAY_CHUNKS), response.text
    (seen,) = tuple(request for request in wire.drain() if marker_hex in request.body.decode())
    assert seen.target == call.upstream_target, seen.target
