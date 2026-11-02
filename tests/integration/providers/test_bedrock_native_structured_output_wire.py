import base64
import json
import re
import uuid
from collections.abc import Callable, Iterable, Iterator, Mapping
from itertools import chain
from typing import Final
from urllib.parse import unquote

import httpx
import pytest
from integration._support.client import Gateway, Scenario, held, list_value, object_value, string_value
from integration._support.database import read_rows
from integration._support.upstream import aws_event_stream_frame
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

_HAIKU: Final = "us.anthropic.claude-haiku-5-5"
_FABLE: Final = "us.anthropic.claude-fable-5"
_SONNET: Final = "us.anthropic.claude-sonnet-4-6"
_OPUS: Final = "us.anthropic.claude-opus-5-5"
_OPT_IN: Final = _OPUS
_OPUS_4_8: Final = "us.anthropic.claude-opus-4-8"
_DEEPSEEK: Final = "deepseek.v3.2"
_AWS: Final[dict[str, JsonValue]] = {
    "aws_access_key_id": "AKIASCRIPTEDPROVIDER",
    "aws_secret_access_key": "scripted-secret",
    "aws_region_name": "us-east-1",
}
_NO_CACHE: Final[dict[str, JsonValue]] = {"cache": {"no-cache": True}, "num_retries": 0}
_JSON_TOOL: Final = "json_tool_call"
_USER_TOOL: Final = "lookup"
_SIGNATURE: Final = "c2NyaXB0ZWQtc2lnbmF0dXJl"
_ADAPTIVE: Final[dict[str, JsonValue]] = {"type": "adaptive"}
_THINKING_TYPES: Final = frozenset({"enabled", "adaptive"})
_EVENT_STREAM: Final = "application/vnd.amazon.eventstream"
_USAGE: Final[dict[str, JsonValue]] = {"inputTokens": 11, "outputTokens": 7, "totalTokens": 18}
_MARKER: Final = re.compile(r"probe ([0-9a-f]{32})")
_JSON: Final = TypeAdapter(dict[str, JsonValue])
_JSON_VALUE: Final[TypeAdapter[JsonValue]] = TypeAdapter(JsonValue)
_FABLE_REJECTION: Final = (
    "The model returned the following errors: output_config.format: Extra inputs are not permitted"
)
_SCHEMA_TOO_COMPLEX: Final = (
    "The model returned the following errors: compilation exceeded limits: Schema is too complex."
)
_PATTERN_TOO_COMPLEX: Final = (
    "The model returned the following errors: output_config.format.schema: Unsupported regex feature in pattern "
    "field: pattern is too complex for structured output: reduce the {n,m} upper bound, narrow the character class "
    "range, or avoid nesting quantified groups"
)
_MAX_TOKENS_400: Final = (
    "The model returned the following errors: max_tokens: 2048 > 1024, the maximum for this request"
)
_SIGNATURE_400: Final = (
    "The model returned the following errors: messages.1.content.0: Invalid `signature` in `thinking` block"
)
_UNSUPPORTED_KEYWORDS: Final = (
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
    "maxItems",
    "uniqueItems",
    "minProperties",
    "maxProperties",
    "patternProperties",
    "oneOf",
)
_SCHEMA_MAPS: Final = ("properties", "$defs", "definitions")
_SCHEMA_LISTS: Final = ("anyOf", "allOf", "oneOf")
_SCHEMA: Final[dict[str, JsonValue]] = {
    "type": "object",
    "properties": {"tier": {"type": "integer"}, "summary": {"type": "string"}},
    "required": ["tier", "summary"],
    "additionalProperties": False,
}
_LIMITS_SCHEMA: Final[dict[str, JsonValue]] = {
    "type": "object",
    "properties": {
        "tier": {"type": "integer", "minimum": 1, "maximum": 5},
        "summary": {"type": "string"},
        "tags": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        "owner": {"anyOf": [{"$ref": "#/$defs/Person"}, {"type": "null"}]},
        "backup": {"type": ["object", "null"], "properties": {"name": {"type": "string"}}, "required": ["name"]},
    },
    "required": ["tier", "summary", "tags", "owner", "backup"],
    "$defs": {"Person": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}},
}
_OPEN_COUNTS: Final[dict[str, JsonValue]] = {"type": "object", "additionalProperties": {"type": "integer"}}
_DICT_SCHEMA: Final[dict[str, JsonValue]] = {
    "type": "object",
    "properties": {"store": {"type": "string"}, "counts": _OPEN_COUNTS},
    "required": ["store", "counts"],
    "additionalProperties": False,
}
_NULLABLE_SCHEMA: Final[dict[str, JsonValue]] = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "backup": {"type": ["object", "null"], "properties": {"name": {"type": "string"}}, "required": ["name"]},
    },
    "required": ["summary", "backup"],
    "additionalProperties": False,
}
_CLOSED_NULLABLE_SCHEMA: Final[dict[str, JsonValue]] = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "backup": {
            "type": ["object", "null"],
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
            "additionalProperties": False,
        },
    },
    "required": ["summary", "backup"],
    "additionalProperties": False,
}
_UNCLEANED_SCHEMA: Final[dict[str, JsonValue]] = {
    "type": "object",
    "properties": {"tier": {"type": "integer", "minimum": 1, "maximum": 5}, "summary": {"type": "string"}},
    "required": ["tier", "summary"],
    "additionalProperties": False,
}
_NUMERIC_FORMATS_SCHEMA: Final[dict[str, JsonValue]] = {
    "type": "object",
    "properties": {
        "tier": {"type": "integer", "format": "int32"},
        "score": {"type": "number", "format": "double"},
        "summary": {"type": "string"},
    },
    "required": ["tier", "score", "summary"],
    "additionalProperties": False,
}
_WIDE_FIELDS: Final = tuple(f"field_{index:02d}" for index in range(65))
_WIDE_SCHEMA: Final[dict[str, JsonValue]] = {
    "type": "object",
    "properties": {name: {"type": "string"} for name in _WIDE_FIELDS},
    "required": list(_WIDE_FIELDS),
    "additionalProperties": False,
}
_USER_TOOL_SPEC: Final[dict[str, JsonValue]] = {
    "type": "function",
    "function": {
        "name": _USER_TOOL,
        "description": "Look up a record",
        "parameters": {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]},
    },
}


def _payload(marker_hex: str) -> dict[str, JsonValue]:
    return {"tier": 2, "summary": marker_hex}


def _marker() -> tuple[str, str]:
    marker_hex: Final = uuid.uuid4().hex
    return f"probe {marker_hex}", marker_hex


def _response_format(schema: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    return {"type": "json_schema", "json_schema": {"name": "risk", "schema": dict(schema)}}


def _model_id(request: Request) -> str:
    return unquote(request.target).split("/model/", 1)[1].rsplit("/", 1)[0]


def _is_invoke(request: Request) -> bool:
    return unquote(request.target).rsplit("/", 1)[1].startswith("invoke")


def _is_stream(request: Request) -> bool:
    return unquote(request.target).endswith(("/converse-stream", "/invoke-with-response-stream"))


def _thinking_of(body: Mapping[str, JsonValue], invoke: bool) -> JsonValue:
    if invoke:
        return body.get("thinking")
    return object_value(body.get("additionalModelRequestFields") or {}).get("thinking")


def _thinking_requested(body: Mapping[str, JsonValue], invoke: bool) -> bool:
    thinking: Final = _thinking_of(body, invoke)
    return isinstance(thinking, dict) and thinking.get("type") in _THINKING_TYPES


def _native_schema(body: Mapping[str, JsonValue], invoke: bool) -> dict[str, JsonValue] | None:
    if invoke:
        output_config: Final = body.get("output_config")
        if not isinstance(output_config, dict) or "format" not in output_config:
            return None
        return object_value(object_value(object_value(output_config)["format"])["schema"])
    text_format: Final = object_value(body.get("outputConfig") or {}).get("textFormat")
    if text_format is None:
        return None
    json_schema: Final = object_value(object_value(object_value(text_format)["structure"])["jsonSchema"])
    return _JSON.validate_json(string_value(json_schema["schema"]))


def _tools(body: Mapping[str, JsonValue], invoke: bool) -> dict[str, dict[str, JsonValue]]:
    if invoke:
        return {
            string_value(object_value(tool)["name"]): object_value(object_value(tool)["input_schema"])
            for tool in list_value(body.get("tools", []))
        }
    specs: Final = tuple(
        object_value(object_value(tool)["toolSpec"])
        for tool in list_value(object_value(body.get("toolConfig") or {"tools": []})["tools"])
    )
    return {string_value(spec["name"]): object_value(object_value(spec["inputSchema"])["json"]) for spec in specs}


def _is_object_type(node: Mapping[str, JsonValue]) -> bool:
    node_type: Final = node.get("type")
    return node_type == "object" or (isinstance(node_type, list) and "object" in node_type)


def _child_schemas(node: Mapping[str, JsonValue]) -> Iterable[JsonValue]:
    mapped: Final = (node.get(key) for key in _SCHEMA_MAPS)
    listed: Final = (node.get(key) for key in _SCHEMA_LISTS)
    single: Final = (node.get(key) for key in ("items", "additionalProperties"))
    return chain(
        chain.from_iterable(children.values() for children in mapped if isinstance(children, dict)),
        chain.from_iterable(variants for variants in listed if isinstance(variants, list)),
        (child for child in single if isinstance(child, dict)),
    )


def _schema_nodes(schema: JsonValue) -> Iterator[dict[str, JsonValue]]:
    if not isinstance(schema, dict):
        return
    yield schema
    yield from chain.from_iterable(_schema_nodes(child) for child in _child_schemas(schema))


def _node_fault(node: Mapping[str, JsonValue]) -> str | None:
    node_type: Final = node.get("type")
    unsupported: Final = tuple(key for key in _UNSUPPORTED_KEYWORDS if key in node)
    if unsupported:
        return f"For '{node_type}' type, property '{unsupported[0]}' is not supported"
    min_items: Final = node.get("minItems")
    if isinstance(min_items, int) and min_items > 1:
        return "For 'array' type, property 'minItems' is not supported"
    if _is_object_type(node) and node.get("additionalProperties") is not False:
        return "For 'object' type, 'additionalProperties' must be explicitly set to false"
    return None


def _bedrock_schema_fault(schema: JsonValue) -> str | None:
    faults: Final = tuple(fault for node in _schema_nodes(schema) if (fault := _node_fault(node)) is not None)
    if not faults:
        return None
    return f"The model returned the following errors: output_config.format.schema: {faults[0]}"


def _rejection(model_id: str, native: dict[str, JsonValue] | None) -> str | None:
    if native is None:
        return None
    if model_id == _FABLE:
        return _FABLE_REJECTION
    if "anthropic." not in model_id:
        return None
    return _bedrock_schema_fault(native)


def _reasoning_block(marker_hex: str, invoke: bool) -> dict[str, JsonValue]:
    if invoke:
        return {"type": "thinking", "thinking": f"reasoning {marker_hex}", "signature": _SIGNATURE}
    return {"reasoningContent": {"reasoningText": {"text": f"reasoning {marker_hex}", "signature": _SIGNATURE}}}


def _answer_block(name: str | None, marker_hex: str, invoke: bool) -> dict[str, JsonValue]:
    tool_input: Final[dict[str, JsonValue]] = {"q": marker_hex} if name == _USER_TOOL else _payload(marker_hex)
    if name is None and invoke:
        return {"type": "text", "text": json.dumps(_payload(marker_hex))}
    if name is None:
        return {"text": json.dumps(_payload(marker_hex))}
    if invoke:
        return {"type": "tool_use", "id": f"toolu_{marker_hex}", "name": name, "input": tool_input}
    return {"toolUse": {"toolUseId": f"tooluse_{marker_hex}", "name": name, "input": tool_input}}


def _converse_frames(blocks: tuple[dict[str, JsonValue], ...], stop_reason: str) -> bytes:
    def frame(event_type: str, payload: Mapping[str, JsonValue]) -> bytes:
        return aws_event_stream_frame(
            {":event-type": event_type, ":content-type": "application/json", ":message-type": "event"},
            json.dumps(payload).encode(),
        )

    def block_frames(index: int, block: dict[str, JsonValue]) -> bytes:
        if "reasoningContent" in block:
            reasoning: Final = object_value(object_value(block["reasoningContent"])["reasoningText"])
            return b"".join(
                (
                    frame(
                        "contentBlockDelta",
                        {"delta": {"reasoningContent": {"text": reasoning["text"]}}, "contentBlockIndex": index},
                    ),
                    frame(
                        "contentBlockDelta",
                        {
                            "delta": {"reasoningContent": {"signature": reasoning["signature"]}},
                            "contentBlockIndex": index,
                        },
                    ),
                    frame("contentBlockStop", {"contentBlockIndex": index}),
                )
            )
        if "toolUse" in block:
            tool_use: Final = object_value(block["toolUse"])
            return b"".join(
                (
                    frame(
                        "contentBlockStart",
                        {
                            "start": {"toolUse": {"toolUseId": tool_use["toolUseId"], "name": tool_use["name"]}},
                            "contentBlockIndex": index,
                        },
                    ),
                    frame(
                        "contentBlockDelta",
                        {"delta": {"toolUse": {"input": json.dumps(tool_use["input"])}}, "contentBlockIndex": index},
                    ),
                    frame("contentBlockStop", {"contentBlockIndex": index}),
                )
            )
        return b"".join(
            (
                frame("contentBlockDelta", {"delta": {"text": block["text"]}, "contentBlockIndex": index}),
                frame("contentBlockStop", {"contentBlockIndex": index}),
            )
        )

    return b"".join(
        (
            frame("messageStart", {"role": "assistant"}),
            *(block_frames(index, block) for index, block in enumerate(blocks)),
            frame("messageStop", {"stopReason": stop_reason}),
            frame("metadata", {"usage": _USAGE, "metrics": {"latencyMs": 1}}),
        )
    )


def _invoke_chunk(event: Mapping[str, JsonValue]) -> bytes:
    return aws_event_stream_frame(
        {":event-type": "chunk", ":content-type": "application/json", ":message-type": "event"},
        json.dumps({"bytes": base64.b64encode(json.dumps(event).encode()).decode()}).encode(),
    )


def _invoke_block_events(index: int, block: Mapping[str, JsonValue]) -> tuple[dict[str, JsonValue], ...]:
    def start(content_block: dict[str, JsonValue]) -> dict[str, JsonValue]:
        return {"type": "content_block_start", "index": index, "content_block": content_block}

    def delta(payload: dict[str, JsonValue]) -> dict[str, JsonValue]:
        return {"type": "content_block_delta", "index": index, "delta": payload}

    stop: Final[dict[str, JsonValue]] = {"type": "content_block_stop", "index": index}
    match block.get("type"):
        case "thinking":
            return (
                start({"type": "thinking", "thinking": ""}),
                delta({"type": "thinking_delta", "thinking": block["thinking"]}),
                delta({"type": "signature_delta", "signature": block["signature"]}),
                stop,
            )
        case "tool_use":
            return (
                start({"type": "tool_use", "id": block["id"], "name": block["name"], "input": {}}),
                delta({"type": "input_json_delta", "partial_json": json.dumps(block["input"])}),
                stop,
            )
        case _:
            return (start({"type": "text", "text": ""}), delta({"type": "text_delta", "text": block["text"]}), stop)


def _invoke_frames(message: Mapping[str, JsonValue]) -> bytes:
    blocks: Final = tuple(object_value(block) for block in list_value(message["content"]))
    events: Final[tuple[Mapping[str, JsonValue], ...]] = (
        {"type": "message_start", "message": {**message, "content": [], "stop_reason": None}},
        *chain.from_iterable(_invoke_block_events(index, block) for index, block in enumerate(blocks)),
        {
            "type": "message_delta",
            "delta": {"stop_reason": message["stop_reason"], "stop_sequence": None},
            "usage": {"output_tokens": 7},
        },
        {"type": "message_stop"},
    )
    return b"".join(_invoke_chunk(event) for event in events)


def _called_tool(tools: Iterable[str], native: bool) -> str | None:
    names: Final = frozenset(tools)
    if _USER_TOOL in names:
        return _USER_TOOL
    if native or _JSON_TOOL not in names:
        return None
    return _JSON_TOOL


def _error_reply(message: str) -> Reply:
    return Reply(status=400, body=json.dumps({"message": message}).encode())


def _bedrock(request: Request) -> Reply:
    body: Final = _JSON.validate_json(request.body)
    invoke: Final = _is_invoke(request)
    found: Final = _MARKER.search(json.dumps(body))
    assert found is not None, body
    marker_hex: Final = found.group(1)
    native: Final = _native_schema(body, invoke)
    rejection: Final = _rejection(_model_id(request), native)
    if rejection is not None:
        return _error_reply(rejection)
    called: Final = _called_tool(_tools(body, invoke).keys(), native is not None)
    reasoning: Final = (_reasoning_block(marker_hex, invoke),) if _thinking_requested(body, invoke) else ()
    blocks: Final = (*reasoning, _answer_block(called, marker_hex, invoke))
    stop_reason: Final = "tool_use" if called is not None else "end_turn"
    if invoke:
        message: Final[dict[str, JsonValue]] = {
            "id": f"msg_{marker_hex}",
            "type": "message",
            "role": "assistant",
            "model": "claude-scripted",
            "content": list(blocks),
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {"input_tokens": 11, "output_tokens": 7},
        }
        if _is_stream(request):
            return Reply(body=_invoke_frames(message), content_type=_EVENT_STREAM)
        return Reply(body=json.dumps(message).encode())
    if _is_stream(request):
        return Reply(body=_converse_frames(blocks, stop_reason), content_type=_EVENT_STREAM)
    converse: Final[dict[str, JsonValue]] = {
        "output": {"message": {"role": "assistant", "content": list(blocks)}},
        "stopReason": stop_reason,
        "usage": _USAGE,
        "metrics": {"latencyMs": 1},
    }
    return Reply(body=json.dumps(converse).encode())


def _deployment(
    scenario: Scenario, wire: Wire, route: str, backend: str, model_info: Mapping[str, JsonValue] | None = None
) -> str:
    return scenario.model(model=f"bedrock/{route}/{backend}", api_base=wire.url, model_info=model_info, **_AWS)


def _chat_body(model: str, marker: str, **extra: JsonValue) -> dict[str, JsonValue]:
    return {"model": model, "messages": [{"role": "user", "content": marker}], "max_tokens": 2048, **_NO_CACHE, **extra}


def _messages_body(model: str, marker: str, **extra: JsonValue) -> dict[str, JsonValue]:
    return {"model": model, "max_tokens": 2048, "messages": [{"role": "user", "content": marker}], **_NO_CACHE, **extra}


def _one_call(wire: Wire, marker_hex: str) -> tuple[Request, dict[str, JsonValue]]:
    (request,) = wire.drain()
    body: Final = _JSON.validate_json(request.body)
    assert marker_hex in json.dumps(body), body
    return request, body


def _chat_message(response: httpx.Response) -> dict[str, JsonValue]:
    assert response.status_code == 200, response.text
    choices: Final = list_value(_JSON.validate_json(response.content)["choices"])
    return object_value(object_value(choices[0])["message"])


def _stream_message(response: httpx.Response) -> dict[str, JsonValue]:
    assert response.status_code == 200, response.text
    deltas: Final = tuple(
        object_value(object_value(list_value(_JSON.validate_json(line[6:])["choices"])[0])["delta"])
        for line in response.text.splitlines()
        if line.startswith("data: {")
    )
    return {
        "content": "".join(string_value(delta["content"]) for delta in deltas if delta.get("content")),
        "reasoning_content": "".join(
            string_value(delta["reasoning_content"]) for delta in deltas if delta.get("reasoning_content")
        ),
    }


def _content_json(message: Mapping[str, JsonValue]) -> JsonValue:
    return _JSON_VALUE.validate_json(string_value(message["content"]))


def _responses_items(response: httpx.Response) -> tuple[dict[str, JsonValue], ...]:
    assert response.status_code == 200, response.text
    return tuple(object_value(item) for item in list_value(_JSON.validate_json(response.content)["output"]))


def _responses_text(items: tuple[dict[str, JsonValue], ...]) -> JsonValue:
    (message,) = tuple(item for item in items if item.get("type") == "message")
    (text,) = tuple(
        object_value(part) for part in list_value(message["content"]) if object_value(part).get("type") == "output_text"
    )
    return _JSON_VALUE.validate_json(string_value(text["text"]))


def _messages_blocks(response: httpx.Response) -> tuple[dict[str, JsonValue], ...]:
    assert response.status_code == 200, response.text
    return tuple(object_value(block) for block in list_value(_JSON.validate_json(response.content)["content"]))


def _messages_text(blocks: tuple[dict[str, JsonValue], ...]) -> JsonValue:
    (text,) = tuple(block for block in blocks if block.get("type") == "text")
    return _JSON_VALUE.validate_json(string_value(text["text"]))


def _messages_stream_text(response: httpx.Response) -> JsonValue:
    assert response.status_code == 200, response.text
    events: Final = tuple(
        _JSON.validate_json(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: {")
    )
    deltas: Final = tuple(
        object_value(event["delta"]) for event in events if event.get("type") == "content_block_delta"
    )
    text: Final = "".join(string_value(delta["text"]) for delta in deltas if delta.get("type") == "text_delta")
    return _JSON_VALUE.validate_json(text)


def _refs(schema: JsonValue) -> frozenset[str]:
    return frozenset(string_value(node["$ref"]) for node in _schema_nodes(schema) if "$ref" in node)


def _resolves(schema: Mapping[str, JsonValue], ref: str) -> bool:
    prefix: Final = "#/$defs/"
    return ref.startswith(prefix) and ref.removeprefix(prefix) in object_value(schema.get("$defs") or {})


def _assert_native_without_tools(body: Mapping[str, JsonValue], invoke: bool, schema: Mapping[str, JsonValue]) -> None:
    assert _native_schema(body, invoke) == schema, body
    assert _JSON_TOOL not in _tools(body, invoke), body


@pytest.mark.parametrize(
    ("route", "stream"),
    [("converse", False), ("converse", True), ("invoke", False)],
    ids=["converse", "converse_stream", "invoke"],
)
def test_thinking_request_on_a_bedrock_native_model_gets_native_output_and_keeps_reasoning(
    gateway: Gateway, route: str, stream: bool
) -> None:
    marker, marker_hex = _marker()
    with wire_server(_bedrock) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, route, _HAIKU)
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            _chat_body(model, marker, stream=stream, thinking=_ADAPTIVE, response_format=_response_format(_SCHEMA)),
        )
        message: Final = _stream_message(response) if stream else _chat_message(response)
        assert _content_json(message) == _payload(marker_hex), response.text
        assert message["reasoning_content"] == f"reasoning {marker_hex}", response.text
        request, body = _one_call(wire, marker_hex)
        invoke: Final = route == "invoke"
        assert _is_stream(request) is stream, request.target
        _assert_native_without_tools(body, invoke, _SCHEMA)
        assert _thinking_requested(body, invoke), body


def test_responses_reasoning_effort_on_a_bedrock_native_model_gets_native_output_and_keeps_reasoning(
    gateway: Gateway,
) -> None:
    marker, marker_hex = _marker()
    with wire_server(_bedrock) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, "converse", _HAIKU)
        response: Final = gateway.request(
            "POST",
            "/v1/responses",
            {
                "model": model,
                "input": marker,
                "max_output_tokens": 2048,
                "reasoning": {"effort": "low"},
                "text": {"format": {"type": "json_schema", "name": "risk", "schema": _SCHEMA}},
                **_NO_CACHE,
            },
        )
        items: Final = _responses_items(response)
        assert _responses_text(items) == _payload(marker_hex), response.text
        (reasoning,) = tuple(item for item in items if item.get("type") == "reasoning")
        assert f"reasoning {marker_hex}" in json.dumps(reasoning), response.text
        _, body = _one_call(wire, marker_hex)
        _assert_native_without_tools(body, False, _SCHEMA)
        assert _thinking_requested(body, False), body


def test_messages_thinking_on_a_bedrock_converse_native_model_gets_native_output_and_keeps_reasoning(
    gateway: Gateway,
) -> None:
    marker, marker_hex = _marker()
    with wire_server(_bedrock) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, "converse", _HAIKU)
        response: Final = gateway.request(
            "POST",
            "/v1/messages",
            _messages_body(model, marker, thinking=_ADAPTIVE, output_format={"type": "json_schema", "schema": _SCHEMA}),
        )
        blocks: Final = _messages_blocks(response)
        assert _messages_text(blocks) == _payload(marker_hex), response.text
        (thinking,) = tuple(block for block in blocks if block.get("type") == "thinking")
        assert thinking["thinking"] == f"reasoning {marker_hex}", response.text
        _, body = _one_call(wire, marker_hex)
        _assert_native_without_tools(body, False, _SCHEMA)
        assert _thinking_requested(body, False), body


@pytest.mark.parametrize("route", ["converse", "invoke"])
def test_bedrock_native_model_without_thinking_keeps_the_forced_json_tool(gateway: Gateway, route: str) -> None:
    marker, marker_hex = _marker()
    with wire_server(_bedrock) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, route, _HAIKU)
        response: Final = gateway.request(
            "POST", "/v1/chat/completions", _chat_body(model, marker, response_format=_response_format(_SCHEMA))
        )
        message: Final = _chat_message(response)
        assert _content_json(message) == _payload(marker_hex), response.text
        assert not message.get("tool_calls"), response.text
        _, body = _one_call(wire, marker_hex)
        invoke: Final = route == "invoke"
        assert _native_schema(body, invoke) is None, body
        assert _tools(body, invoke)[_JSON_TOOL]["properties"] == _SCHEMA["properties"], body


@pytest.mark.parametrize(
    ("backend", "model_info", "native"),
    [
        (_OPUS, None, False),
        (_OPT_IN, {"supports_bedrock_native_structured_output": True}, True),
        (_OPUS_4_8, {"supports_bedrock_native_structured_output": False}, False),
    ],
    ids=["neither_key", "deployment_opt_in", "deployment_opt_out"],
)
def test_thinking_request_follows_the_bedrock_native_key_the_model_resolves_to(
    gateway: Gateway, backend: str, model_info: Mapping[str, JsonValue] | None, native: bool
) -> None:
    marker, marker_hex = _marker()
    with wire_server(_bedrock) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, "converse", backend, model_info)
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            _chat_body(model, marker, thinking=_ADAPTIVE, response_format=_response_format(_SCHEMA)),
        )
        assert _content_json(_chat_message(response)) == _payload(marker_hex), response.text
        _, body = _one_call(wire, marker_hex)
        assert (_native_schema(body, False) == _SCHEMA) is native, body
        assert (_JSON_TOOL in _tools(body, False)) is not native, body


def _opt_in_answer(gateway: Gateway, entry: str, model: str, marker: str) -> JsonValue:
    output_format: Final[dict[str, JsonValue]] = {"type": "json_schema", "schema": _SCHEMA}
    match entry:
        case "converse_chat" | "invoke_chat":
            chat: Final = _chat_body(model, marker, thinking=_ADAPTIVE, response_format=_response_format(_SCHEMA))
            return _content_json(_chat_message(gateway.request("POST", "/v1/chat/completions", chat)))
        case "converse_responses":
            responses: Final[dict[str, JsonValue]] = {
                "model": model,
                "input": marker,
                "max_output_tokens": 2048,
                "reasoning": {"effort": "low"},
                "text": {"format": {**output_format, "name": "risk"}},
                **_NO_CACHE,
            }
            return _responses_text(_responses_items(gateway.request("POST", "/v1/responses", responses)))
        case "converse_messages":
            adapted: Final = _messages_body(model, marker, thinking=_ADAPTIVE, output_format=output_format)
            return _messages_text(_messages_blocks(gateway.request("POST", "/v1/messages", adapted)))
        case _:
            invoked: Final = _messages_body(model, marker, output_config={"format": output_format})
            return _messages_text(_messages_blocks(gateway.request("POST", "/v1/messages", invoked)))


def _sent_native_schema(gateway: Gateway, wire: Wire, entry: str, model: str) -> dict[str, JsonValue] | None:
    marker, marker_hex = _marker()
    assert _opt_in_answer(gateway, entry, model, marker) == _payload(marker_hex)
    _, body = _one_call(wire, marker_hex)
    invoke: Final = entry.startswith("invoke")
    native: Final = _native_schema(body, invoke)
    assert native is None or _JSON_TOOL not in _tools(body, invoke), body
    return native


@pytest.mark.parametrize(
    "entry", ["converse_chat", "converse_responses", "converse_messages", "invoke_chat", "invoke_messages"]
)
def test_a_deployment_native_output_opt_in_stays_on_the_deployment_that_sets_it(gateway: Gateway, entry: str) -> None:
    route: Final = entry.split("_", 1)[0]
    with wire_server(_bedrock) as wire, gateway.scenario() as scenario:
        opted_in: Final = _deployment(
            scenario, wire, route, _OPT_IN, {"supports_bedrock_native_structured_output": True}
        )
        sibling: Final = _deployment(scenario, wire, route, _OPT_IN)
        assert _sent_native_schema(gateway, wire, entry, sibling) is None
        assert _sent_native_schema(gateway, wire, entry, opted_in) == _SCHEMA


def test_fable_5_json_schema_takes_the_json_tool_instead_of_the_native_output_bedrock_rejects(
    gateway: Gateway,
) -> None:
    marker, marker_hex = _marker()
    with wire_server(_bedrock) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, "converse", _FABLE)
        response: Final = gateway.request(
            "POST", "/v1/chat/completions", _chat_body(model, marker, response_format=_response_format(_SCHEMA))
        )
        assert _content_json(_chat_message(response)) == _payload(marker_hex), response.text
        _, body = _one_call(wire, marker_hex)
        assert "outputConfig" not in body, body
        assert _tools(body, False)[_JSON_TOOL]["properties"] == _SCHEMA["properties"], body


def _limits_answer(gateway: Gateway, entry: str, model: str, marker: str) -> JsonValue:
    output_config: Final[dict[str, JsonValue]] = {"format": {"type": "json_schema", "schema": _LIMITS_SCHEMA}}
    match entry:
        case "converse_chat" | "invoke_chat":
            body: Final = _chat_body(model, marker, response_format=_response_format(_LIMITS_SCHEMA))
            return _content_json(_chat_message(gateway.request("POST", "/v1/chat/completions", body)))
        case "converse_output_config":
            chat: Final = _chat_body(model, marker, output_config=output_config)
            return _content_json(_chat_message(gateway.request("POST", "/v1/chat/completions", chat)))
        case _:
            messages: Final = _messages_body(model, marker, output_config=output_config)
            return _messages_text(_messages_blocks(gateway.request("POST", "/v1/messages", messages)))


@pytest.mark.parametrize("entry", ["converse_chat", "converse_output_config", "invoke_chat", "invoke_messages"])
def test_native_schema_reaches_bedrock_with_limits_in_descriptions_and_every_object_closed(
    gateway: Gateway, entry: str
) -> None:
    marker, marker_hex = _marker()
    route: Final = entry.split("_", 1)[0]
    with wire_server(_bedrock) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, route, _SONNET)
        assert _limits_answer(gateway, entry, model, marker) == _payload(marker_hex)
        _, body = _one_call(wire, marker_hex)
        schema: Final = _native_schema(body, route == "invoke")
        assert schema is not None, body
        assert _bedrock_schema_fault(schema) is None, schema
        properties: Final = object_value(schema["properties"])
        assert object_value(properties["tags"])["minItems"] == 1, schema
        assert {"1", "5"} <= set(re.findall(r"\d+", string_value(object_value(properties["tier"])["description"])))
        assert all(_resolves(schema, ref) for ref in _refs(schema)), schema
        assert _JSON_TOOL not in _tools(body, route == "invoke"), body


@pytest.mark.parametrize("route", ["converse", "invoke"])
def test_open_object_schema_takes_the_json_tool_on_a_native_model_and_keeps_the_dict_open(
    gateway: Gateway, route: str
) -> None:
    marker, marker_hex = _marker()
    with wire_server(_bedrock) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, route, _SONNET)
        response: Final = gateway.request(
            "POST", "/v1/chat/completions", _chat_body(model, marker, response_format=_response_format(_DICT_SCHEMA))
        )
        assert _content_json(_chat_message(response)) == _payload(marker_hex), response.text
        _, body = _one_call(wire, marker_hex)
        invoke: Final = route == "invoke"
        assert _native_schema(body, invoke) is None, body
        tool_schema: Final = _tools(body, invoke)[_JSON_TOOL]
        assert object_value(tool_schema["properties"])["counts"] == _OPEN_COUNTS, tool_schema


def test_thinking_with_a_caller_tool_on_a_bedrock_native_model_sends_only_the_callers_tool(gateway: Gateway) -> None:
    marker, marker_hex = _marker()
    with wire_server(_bedrock) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, "converse", _HAIKU)
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            _chat_body(
                model,
                marker,
                thinking=_ADAPTIVE,
                tools=[_USER_TOOL_SPEC],
                response_format=_response_format(_SCHEMA),
            ),
        )
        (call,) = list_value(_chat_message(response)["tool_calls"])
        function: Final = object_value(object_value(call)["function"])
        assert function["name"] == _USER_TOOL, response.text
        assert json.loads(string_value(function["arguments"])) == {"q": marker_hex}, response.text
        _, body = _one_call(wire, marker_hex)
        assert _tools(body, False).keys() == {_USER_TOOL}, body
        assert _native_schema(body, False) == _SCHEMA, body
        assert _thinking_requested(body, False), body


@pytest.mark.parametrize(
    ("schema", "expected"),
    [(_UNCLEANED_SCHEMA, _UNCLEANED_SCHEMA), (_NULLABLE_SCHEMA, _CLOSED_NULLABLE_SCHEMA)],
    ids=["limits_kept", "nullable_object_closed"],
)
def test_non_anthropic_native_model_gets_the_callers_schema_with_objects_closed(
    gateway: Gateway, schema: dict[str, JsonValue], expected: dict[str, JsonValue]
) -> None:
    marker, marker_hex = _marker()
    with wire_server(_bedrock) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, "converse", _DEEPSEEK)
        response: Final = gateway.request(
            "POST", "/v1/chat/completions", _chat_body(model, marker, response_format=_response_format(schema))
        )
        assert _content_json(_chat_message(response)) == _payload(marker_hex), response.text
        _, body = _one_call(wire, marker_hex)
        assert _native_schema(body, False) == expected, body


@pytest.mark.parametrize(
    ("schema", "native"),
    [
        (_SCHEMA, True),
        (_DICT_SCHEMA, False),
    ],
    ids=["closed_schema_native", "open_object_inlined"],
)
def test_messages_on_bedrock_invoke_follows_the_bedrock_native_key_without_thinking(
    gateway: Gateway, schema: dict[str, JsonValue], native: bool
) -> None:
    marker, marker_hex = _marker()
    with wire_server(_bedrock) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, "invoke", _HAIKU)
        response: Final = gateway.request(
            "POST",
            "/v1/messages",
            _messages_body(model, marker, output_config={"format": {"type": "json_schema", "schema": schema}}),
        )
        assert _messages_text(_messages_blocks(response)) == _payload(marker_hex), response.text
        _, body = _one_call(wire, marker_hex)
        sent: Final = _native_schema(body, True)
        assert (sent is not None) is native, body
        prompt: Final = json.dumps(body["messages"])
        assert all((name in prompt) is not native for name in object_value(schema["properties"])), body


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "object", "properties": ["tier", "summary"]},
        {"type": "object", "properties": {"tier": {"$ref": 5}}, "additionalProperties": False},
        {"type": "object", "$defs": [1, 2], "properties": {"tier": {"$ref": "#/$defs/Tier"}}},
        {"type": ["object", "null"]},
        {"type": "object", "properties": {"tier": {"type": "integer"}}, "required": "tier"},
    ],
    ids=["properties_list", "ref_not_string", "defs_list", "type_list_without_properties", "required_string"],
)
def test_malformed_schema_on_a_native_model_reaches_bedrock_once_without_a_proxy_error(
    gateway: Gateway, schema: dict[str, JsonValue]
) -> None:
    marker, marker_hex = _marker()
    with wire_server(_bedrock) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, "converse", _SONNET)
        response: Final = gateway.request(
            "POST", "/v1/chat/completions", _chat_body(model, marker, response_format=_response_format(schema))
        )
        assert response.status_code < 500, response.text
        _, body = _one_call(wire, marker_hex)
        native: Final = _native_schema(body, False)
        fault: Final = None if native is None else _bedrock_schema_fault(native)
        assert response.status_code == (200 if fault is None else 400), response.text
        assert fault is None or fault.split(": ", 1)[1] in response.text, response.text
        assert (native is None) is (_JSON_TOOL in _tools(body, False)), body
        assert gateway.request("GET", "/health/liveliness").status_code == 200


@pytest.mark.parametrize(
    ("route", "backend", "thinking", "schema", "native"),
    [
        ("converse", _SONNET, False, _NUMERIC_FORMATS_SCHEMA, True),
        ("invoke", _HAIKU, True, _NUMERIC_FORMATS_SCHEMA, True),
        ("converse", _SONNET, False, _WIDE_SCHEMA, True),
        ("converse", _HAIKU, True, _WIDE_SCHEMA, False),
    ],
    ids=["numeric_formats_old_key", "numeric_formats_new_key_invoke", "wide_old_key", "wide_new_key"],
)
def test_numeric_formats_and_wide_schemas_follow_the_bedrock_native_key(
    gateway: Gateway, route: str, backend: str, thinking: bool, schema: dict[str, JsonValue], native: bool
) -> None:
    marker, marker_hex = _marker()
    extra: Final[dict[str, JsonValue]] = {"thinking": _ADAPTIVE} if thinking else {}
    with wire_server(_bedrock) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, route, backend)
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            _chat_body(model, marker, response_format=_response_format(schema), **extra),
        )
        assert _content_json(_chat_message(response)) == _payload(marker_hex), response.text
        _, body = _one_call(wire, marker_hex)
        invoke: Final = route == "invoke"
        assert (_native_schema(body, invoke) == schema) is native, body
        tools: Final = _tools(body, invoke)
        assert (_JSON_TOOL in tools) is not native, body
        assert native or tools[_JSON_TOOL]["properties"] == schema["properties"], body


def _rejecting_native_output(message: str) -> Callable[[Request], Reply]:
    def reply(request: Request) -> Reply:
        native: Final = _native_schema(_JSON.validate_json(request.body), _is_invoke(request))
        return _bedrock(request) if native is None else _error_reply(message)

    return reply


def _fallback_answer(gateway: Gateway, entry: str, model: str, marker: str) -> JsonValue:
    match entry:
        case "converse_chat_stream" | "invoke_chat_stream":
            streamed: Final = _chat_body(
                model, marker, stream=True, thinking=_ADAPTIVE, response_format=_response_format(_SCHEMA)
            )
            return _content_json(_stream_message(gateway.request("POST", "/v1/chat/completions", streamed)))
        case "invoke_messages_stream":
            streamed_messages: Final = _messages_body(
                model, marker, stream=True, output_config={"format": {"type": "json_schema", "schema": _SCHEMA}}
            )
            return _messages_stream_text(gateway.request("POST", "/v1/messages", streamed_messages))
        case "converse_output_config":
            chat: Final = _chat_body(
                model, marker, output_config={"format": {"type": "json_schema", "schema": _SCHEMA}}
            )
            return _content_json(_chat_message(gateway.request("POST", "/v1/chat/completions", chat)))
        case _:
            return _opt_in_answer(gateway, entry, model, marker)


@pytest.mark.parametrize(
    ("entry", "rejection"),
    [
        ("converse_chat", _FABLE_REJECTION),
        ("converse_chat", _PATTERN_TOO_COMPLEX),
        ("converse_chat_stream", _SCHEMA_TOO_COMPLEX),
        ("converse_responses", _FABLE_REJECTION),
        ("converse_messages", _FABLE_REJECTION),
        ("converse_output_config", _FABLE_REJECTION),
        ("invoke_chat", _FABLE_REJECTION),
        ("invoke_chat_stream", _FABLE_REJECTION),
        ("invoke_messages", _SCHEMA_TOO_COMPLEX),
        ("invoke_messages_stream", _FABLE_REJECTION),
    ],
)
def test_bedrock_rejecting_opt_in_native_output_gets_one_retry_without_it(
    gateway: Gateway, entry: str, rejection: str
) -> None:
    marker, marker_hex = _marker()
    route: Final = entry.split("_", 1)[0]
    invoke: Final = route == "invoke"
    with wire_server(_rejecting_native_output(rejection)) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, route, _OPUS_4_8)
        assert _fallback_answer(gateway, entry, model, marker) == _payload(marker_hex)
        rejected, retried = (_JSON.validate_json(request.body) for request in wire.drain())
        assert _native_schema(rejected, invoke) is not None, rejected
        assert _native_schema(retried, invoke) is None, retried
        tools: Final = _tools(retried, invoke)
        match entry:
            case "converse_output_config":
                assert not tools, retried
            case "invoke_messages" | "invoke_messages_stream":
                assert not tools, retried
                prompt: Final = json.dumps(retried["messages"])
                assert all(name in prompt for name in object_value(_SCHEMA["properties"])), retried
            case _:
                assert tools[_JSON_TOOL]["properties"] == _SCHEMA["properties"], retried


@pytest.mark.parametrize(
    ("backend", "rejection"),
    [(_OPUS_4_8, _MAX_TOKENS_400), (_SONNET, _FABLE_REJECTION)],
    ids=["other_400", "native_model"],
)
@pytest.mark.parametrize("route", ["converse", "invoke"])
def test_bedrock_400_is_not_retried_unless_it_rejects_opt_in_native_output(
    gateway: Gateway, route: str, backend: str, rejection: str
) -> None:
    marker, marker_hex = _marker()
    with wire_server(_rejecting_native_output(rejection)) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, route, backend)
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            _chat_body(model, marker, thinking=_ADAPTIVE, response_format=_response_format(_SCHEMA)),
        )
        assert response.status_code == 400, response.text
        assert rejection.split(": ", 1)[1] in response.text, response.text
        _, body = _one_call(wire, marker_hex)
        assert _native_schema(body, route == "invoke") == _SCHEMA, body


def _always_rejecting(message: str) -> Callable[[Request], Reply]:
    def reply(_: Request) -> Reply:
        return _error_reply(message)

    return reply


@pytest.mark.parametrize(
    ("backend", "rejection"),
    [(_OPUS_4_8, _MAX_TOKENS_400), (_SONNET, _FABLE_REJECTION)],
    ids=["other_400", "native_model"],
)
def test_messages_400_on_bedrock_invoke_reaches_the_caller_after_one_call_unless_it_rejects_opt_in_native_output(
    gateway: Gateway, backend: str, rejection: str
) -> None:
    marker, marker_hex = _marker()
    with wire_server(_always_rejecting(rejection)) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, "invoke", backend)
        response: Final = gateway.request(
            "POST",
            "/v1/messages",
            _messages_body(model, marker, output_config={"format": {"type": "json_schema", "schema": _SCHEMA}}),
        )
        assert response.status_code == 400, response.text
        assert rejection.split(": ", 1)[1] in response.text, response.text
        _one_call(wire, marker_hex)


def test_a_messages_retry_without_native_output_still_omits_the_deployments_dropped_params(gateway: Gateway) -> None:
    marker, marker_hex = _marker()
    with wire_server(_rejecting_native_output(_FABLE_REJECTION)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"bedrock/invoke/{_OPUS_4_8}",
            api_base=wire.url,
            model_info=None,
            additional_drop_params=["stop_sequences"],
            **_AWS,
        )
        request: Final = _messages_body(
            model, marker, stop_sequences=["END"], output_config={"format": {"type": "json_schema", "schema": _SCHEMA}}
        )
        assert _messages_text(_messages_blocks(gateway.request("POST", "/v1/messages", request))) == _payload(
            marker_hex
        )
        rejected, retried = (_JSON.validate_json(call.body) for call in wire.drain())
        assert _native_schema(rejected, True) is not None, rejected
        assert _native_schema(retried, True) is None, retried
        assert "stop_sequences" not in rejected, rejected
        assert "stop_sequences" not in retried, retried


def _carries_thinking_block(message: JsonValue) -> bool:
    content: Final = object_value(message).get("content")
    return isinstance(content, list) and any(object_value(block).get("type") == "thinking" for block in content)


def _replays_thinking(body: Mapping[str, JsonValue]) -> bool:
    return any(_carries_thinking_block(message) for message in list_value(body["messages"]))


def _rejecting_native_output_and_foreign_signatures(signature_first: bool) -> Callable[[Request], Reply]:
    def reply(request: Request) -> Reply:
        body: Final = _JSON.validate_json(request.body)
        checks: Final = (
            (_replays_thinking(body), _SIGNATURE_400),
            (_native_schema(body, True) is not None, _FABLE_REJECTION),
        )
        failing: Final = tuple(message for failed, message in (checks if signature_first else checks[::-1]) if failed)
        return _error_reply(failing[0]) if failing else _bedrock(request)

    return reply


@pytest.mark.parametrize("signature_first", [False, True], ids=["rejection_first", "signature_first"])
def test_messages_retry_without_native_output_keeps_the_thinking_signature_fix(
    gateway: Gateway, signature_first: bool
) -> None:
    marker, marker_hex = _marker()
    history: Final[list[JsonValue]] = [
        {"role": "user", "content": [{"type": "text", "text": f"first question {marker}"}]},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "from another deployment", "signature": _SIGNATURE},
                {"type": "text", "text": "first answer"},
            ],
        },
        {"role": "user", "content": [{"type": "text", "text": "second question"}]},
    ]
    with (
        wire_server(_rejecting_native_output_and_foreign_signatures(signature_first)) as wire,
        gateway.scenario() as scenario,
    ):
        model: Final = _deployment(scenario, wire, "invoke", _OPUS_4_8)
        request: Final = {
            **_messages_body(
                model, marker, thinking=_ADAPTIVE, output_config={"format": {"type": "json_schema", "schema": _SCHEMA}}
            ),
            "messages": history,
        }
        assert _messages_text(_messages_blocks(gateway.request("POST", "/v1/messages", request))) == _payload(
            marker_hex
        )
        sent: Final = tuple(_JSON.validate_json(call.body) for call in wire.drain())
        assert len(sent) == 3, sent
        assert _native_schema(sent[0], True) is not None and _replays_thinking(sent[0]), sent[0]
        assert _native_schema(sent[-1], True) is None and not _replays_thinking(sent[-1]), sent[-1]
        assert all(name in json.dumps(sent[-1]["messages"]) for name in object_value(_SCHEMA["properties"])), sent[-1]


@pytest.mark.parametrize("entry", ["converse_chat", "invoke_messages"])
def test_a_request_retried_without_native_output_lands_one_success_spend_row(gateway: Gateway, entry: str) -> None:
    marker, marker_hex = _marker()
    invoke: Final = entry.startswith("invoke")
    with wire_server(_rejecting_native_output(_FABLE_REJECTION)) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, entry.split("_", 1)[0], _OPUS_4_8)
        assert _fallback_answer(gateway, entry, model, marker) == _payload(marker_hex)
        sent: Final = tuple(_JSON.validate_json(call.body) for call in wire.drain())
        assert len(sent) == (1 if _native_schema(sent[0], invoke) is None else 2), sent
        rows: Final = held(
            lambda: read_rows('SELECT status FROM "LiteLLM_SpendLogs" WHERE model_group=%s', (model,)),
            lambda found: len(found) == 1,
            holding=2,
            seconds=70,
        )
        assert rows[0]["status"] == "success", rows


_CALLER_OUTPUT_CONFIG_WITH_A_SCHEMA_OBJECT: Final[dict[str, JsonValue]] = {
    "textFormat": {"type": "json_schema", "structure": {"jsonSchema": {"name": "risk", "schema": {"type": "object"}}}}
}
_BEDROCK_SCHEMA_OBJECT_400: Final = "Start of structure or map found where not expected"


def _converse_json_schema_field(body: Mapping[str, JsonValue]) -> JsonValue:
    text_format: Final = object_value(body.get("outputConfig") or {}).get("textFormat") or {}
    structure: Final = object_value(text_format).get("structure") or {}
    return object_value(object_value(structure).get("jsonSchema") or {}).get("schema")


def _deserializing_like_bedrock(request: Request) -> Reply:
    schema: Final = _converse_json_schema_field(_JSON.validate_json(request.body))
    if schema is None or isinstance(schema, str):
        return _bedrock(request)
    return Reply(status=400, body=json.dumps({"Message": _BEDROCK_SCHEMA_OBJECT_400}).encode())


def test_a_malformed_caller_output_config_on_an_opt_in_model_gets_bedrocks_own_400_after_one_call(
    gateway: Gateway,
) -> None:
    marker, marker_hex = _marker()
    with wire_server(_deserializing_like_bedrock) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, "converse", _OPUS_4_8)
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            _chat_body(
                model,
                marker,
                thinking=_ADAPTIVE,
                response_format=_response_format(_SCHEMA),
                outputConfig=_CALLER_OUTPUT_CONFIG_WITH_A_SCHEMA_OBJECT,
            ),
        )
        assert response.status_code == 400, response.text
        assert _BEDROCK_SCHEMA_OBJECT_400 in response.text, response.text
        _, body = _one_call(wire, marker_hex)
        assert body["outputConfig"] == _CALLER_OUTPUT_CONFIG_WITH_A_SCHEMA_OBJECT, body


_CALLER_OUTPUT_CONFIG: Final[dict[str, JsonValue]] = {
    "textFormat": {"type": "json_schema", "structure": {"jsonSchema": {"name": "risk", "schema": json.dumps(_SCHEMA)}}}
}


@pytest.mark.parametrize("thinking", [None, _ADAPTIVE], ids=["no_thinking", "adaptive_thinking"])
def test_a_caller_output_config_bedrock_refuses_is_sent_once_on_an_opt_in_model(
    gateway: Gateway, thinking: dict[str, JsonValue] | None
) -> None:
    marker, marker_hex = _marker()
    with wire_server(_rejecting_native_output(_PATTERN_TOO_COMPLEX)) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, "converse", _OPUS_4_8)
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            _chat_body(
                model,
                marker,
                response_format=_response_format(_SCHEMA),
                outputConfig=_CALLER_OUTPUT_CONFIG,
                **({} if thinking is None else {"thinking": thinking}),
            ),
        )
        assert response.status_code == 400, response.text
        assert "pattern is too complex for structured output" in response.text, response.text
        _, body = _one_call(wire, marker_hex)
        assert body["outputConfig"] == _CALLER_OUTPUT_CONFIG, body
