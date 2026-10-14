import json
import re
import uuid
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final

import yaml
from integration._support.client import Gateway, eventually, object_value
from integration._support.database import read_rows
from integration._support.wire import Reply, Request
from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter

# https://platform.claude.com/docs/en/build-with-claude/prompt-caching (read 2026-09-28): at most 4 blocks with cache_control
ANTHROPIC_CACHE_CONTROL_CAP: Final = 4
# https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html (read 2026-09-28): at most 4 cache checkpoints
BEDROCK_CACHE_CHECKPOINT_CAP: Final = 4

ANTHROPIC_MODEL: Final = "claude-opus-5-5"
BEDROCK_MODEL: Final = "anthropic.claude-opus-5-5"
PROVIDER_KEY: Final = "synthetic-provider-key"
SYSTEM: Final = "Use the provided tool results to answer the user."
ASK: Final = "Look up the weather in London, Paris, and Tokyo."
CITIES: Final = ("London", "Paris", "Tokyo")
EPHEMERAL: Final[dict[str, JsonValue]] = {"type": "ephemeral"}
POINTS: Final[list[JsonValue]] = [
    {"location": "message", "role": "system"},
    {"location": "message", "index": -1},
]
TOOL: Final[dict[str, JsonValue]] = {
    "type": "function",
    "function": {
        "name": "lookup_weather",
        "description": "Look up the weather in a city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
}
SYSTEM_LABEL: Final = f"system:{SYSTEM}"
ASK_LABEL: Final = f"user:text:{ASK}"
_MARKER: Final = re.compile(rb"marker-([0-9a-f]{32})")
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])


@dataclass(frozen=True, slots=True)
class Mark:
    label: str
    ttl: str | None


def new_marker() -> str:
    return uuid.uuid4().hex


def final_text(marker: str) -> str:
    return f"Summarize the results in one word. marker-{marker}"


def final_label(marker: str) -> str:
    return f"user:text:{final_text(marker)}"


def call_id(city: str) -> str:
    return f"call_weather_{city.lower()}"


def tool_use_label(city: str) -> str:
    return f"assistant:tool_use:{call_id(city)}"


def tool_call(city: str, **fields: JsonValue) -> dict[str, JsonValue]:
    return {
        "id": call_id(city),
        "type": "function",
        "function": {"name": "lookup_weather", "arguments": json.dumps({"city": city})},
        **fields,
    }


def marked_calls(mark: JsonValue = EPHEMERAL, cities: Sequence[str] = CITIES) -> list[JsonValue]:
    return [tool_call(city, cache_control=mark) for city in cities]


def client_marked() -> list[str]:
    return [ASK_LABEL, *(tool_use_label(city) for city in CITIES)]


def ask(*, marked: bool) -> dict[str, JsonValue]:
    block: Final[dict[str, JsonValue]] = {"type": "text", "text": ASK}
    return {"role": "user", "content": [{**block, "cache_control": EPHEMERAL} if marked else block]}


def tool_results(calls: Sequence[JsonValue]) -> list[JsonValue]:
    return [
        {"role": "tool", "tool_call_id": call["id"], "content": "sunny."}
        for call in calls
        if isinstance(call, dict) and str(call.get("id", "")).startswith("call_")
    ]


def conversation(
    marker: str,
    calls: Sequence[JsonValue],
    *,
    ask_marked: bool = True,
    assistant: dict[str, JsonValue] | None = None,
) -> list[JsonValue]:
    turn: Final[dict[str, JsonValue]] = {
        "role": "assistant",
        "content": "",
        "tool_calls": list(calls),
        **(assistant or {}),
    }
    return [
        {"role": "system", "content": SYSTEM},
        ask(marked=ask_marked),
        turn,
        *tool_results(calls),
        {"role": "user", "content": final_text(marker)},
    ]


def chat_body(model: str, messages: Sequence[JsonValue], **fields: JsonValue) -> dict[str, JsonValue]:
    return {"model": model, "messages": list(messages), "tools": [TOOL], "max_tokens": 64, **fields}


def messages_body(model: str, marker: str, *, stream: bool) -> dict[str, JsonValue]:
    return {
        "model": model,
        "max_tokens": 64,
        "stream": stream,
        "system": [{"type": "text", "text": SYSTEM}],
        "tools": [
            {
                "name": "lookup_weather",
                "description": "Look up the weather in a city.",
                "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
            }
        ],
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": ASK, "cache_control": EPHEMERAL}]},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": call_id(city),
                        "name": "lookup_weather",
                        "input": {"city": city},
                        "cache_control": EPHEMERAL,
                    }
                    for city in CITIES
                ],
            },
            {
                "role": "user",
                "content": [
                    *({"type": "tool_result", "tool_use_id": call_id(city), "content": "sunny."} for city in CITIES),
                    {"type": "text", "text": final_text(marker)},
                ],
            },
        ],
    }


def responses_body(model: str, marker: str) -> dict[str, JsonValue]:
    items: Final[list[JsonValue]] = [
        {"role": "user", "content": [{"type": "input_text", "text": ASK, "cache_control": EPHEMERAL}]},
        *(
            {
                "type": "function_call",
                "call_id": call_id(city),
                "name": "lookup_weather",
                "arguments": json.dumps({"city": city}),
                "cache_control": EPHEMERAL,
            }
            for city in CITIES
        ),
        *({"type": "function_call_output", "call_id": call_id(city), "output": "sunny."} for city in CITIES),
        {"role": "user", "content": final_text(marker)},
    ]
    return {
        "model": model,
        "instructions": SYSTEM,
        "max_output_tokens": 64,
        "tools": [{"type": "function", "name": "lookup_weather", "parameters": {"type": "object"}}],
        "input": items,
    }


def marker_of(request: Request) -> str:
    found: Final = _MARKER.findall(request.body)
    return found[-1].decode() if found else "unmarked"


class _AnthropicBlock(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)
    type: str
    text: str | None = None
    id: str | None = None
    tool_use_id: str | None = None
    name: str | None = None
    cache_control: JsonValue = None


class _AnthropicMessage(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)
    role: str
    content: str | tuple[_AnthropicBlock, ...]


class _AnthropicTool(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)
    name: str = ""
    cache_control: JsonValue = None


class _AnthropicBody(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)
    system: str | tuple[_AnthropicBlock, ...] = ()
    messages: tuple[_AnthropicMessage, ...] = ()
    tools: tuple[_AnthropicTool, ...] = ()
    cache_control: JsonValue = None
    stream: bool = False


def _ttl(cache_control: JsonValue) -> str | None:
    if not isinstance(cache_control, dict):
        return None
    ttl: Final = cache_control.get("ttl")
    return ttl if isinstance(ttl, str) else None


def _block_label(role: str, block: _AnthropicBlock) -> str:
    detail: Final = block.text if block.type == "text" else block.id or block.tool_use_id or ""
    return f"{role}:{block.type}:{detail}"


def _message_blocks(body: _AnthropicBody) -> Iterator[tuple[str, _AnthropicBlock]]:
    for message in body.messages:
        if isinstance(message.content, tuple):
            yield from ((message.role, block) for block in message.content)


def _anthropic_body(request: Request) -> _AnthropicBody:
    return _AnthropicBody.model_validate_json(request.body)


def anthropic_marks(request: Request) -> tuple[Mark, ...]:
    body: Final = _anthropic_body(request)
    system: Final = body.system if isinstance(body.system, tuple) else ()
    return (
        *(Mark(f"tool:{tool.name}", _ttl(tool.cache_control)) for tool in body.tools if tool.cache_control is not None),
        *(
            Mark(f"system:{block.text}", _ttl(block.cache_control))
            for block in system
            if block.cache_control is not None
        ),
        *(
            Mark(_block_label(role, block), _ttl(block.cache_control))
            for role, block in _message_blocks(body)
            if block.cache_control is not None
        ),
        *((Mark("request", _ttl(body.cache_control)),) if body.cache_control is not None else ()),
    )


def anthropic_labels(request: Request) -> list[str]:
    return [mark.label for mark in anthropic_marks(request)]


def _anthropic_error(message: str) -> Reply:
    return Reply(
        status=400,
        body=json.dumps({"type": "error", "error": {"type": "invalid_request_error", "message": message}}).encode(),
    )


def _ttl_out_of_order(marks: Sequence[Mark]) -> bool:
    first_short: Final = next((index for index, mark in enumerate(marks) if mark.ttl != "1h"), len(marks))
    return any(mark.ttl == "1h" for mark in marks[first_short:])


def _usage() -> dict[str, JsonValue]:
    return {"input_tokens": 12, "output_tokens": 1, "cache_creation_input_tokens": 12, "cache_read_input_tokens": 0}


def _anthropic_message(identity: str) -> dict[str, JsonValue]:
    return {
        "id": identity,
        "type": "message",
        "role": "assistant",
        "model": ANTHROPIC_MODEL,
        "content": [{"type": "text", "text": "sunny"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": _usage(),
    }


def _event(name: str, data: dict[str, JsonValue]) -> bytes:
    return f"event: {name}\ndata: {json.dumps(data)}\n\n".encode()


def _anthropic_stream(identity: str) -> tuple[bytes, ...]:
    return (
        _event(
            "message_start",
            {"type": "message_start", "message": {**_anthropic_message(identity), "content": [], "stop_reason": None}},
        ),
        _event(
            "content_block_start",
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        ),
        _event(
            "content_block_delta",
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "sunny"}},
        ),
        _event("content_block_stop", {"type": "content_block_stop", "index": 0}),
        _event(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 1},
            },
        ),
        _event("message_stop", {"type": "message_stop"}),
    )


def anthropic_peer(request: Request) -> Reply:
    marks: Final = anthropic_marks(request)
    if len(marks) > ANTHROPIC_CACHE_CONTROL_CAP:
        return _anthropic_error(
            f"A maximum of {ANTHROPIC_CACHE_CONTROL_CAP} blocks with cache_control may be provided. Found {len(marks)}."
        )
    if _ttl_out_of_order(marks):
        return _anthropic_error("a ttl='1h' cache_control block must not come after a ttl='5m' cache_control block")
    identity: Final = f"msg_{marker_of(request)}"
    if _anthropic_body(request).stream:
        return Reply(content_type="text/event-stream", chunks=_anthropic_stream(identity))
    return Reply(body=json.dumps(_anthropic_message(identity)).encode())


def _bedrock_label(role: str, block: JsonValue) -> str:
    if not isinstance(block, dict):
        return f"{role}:start"
    if isinstance(block.get("text"), str):
        return f"{role}:text:{block['text']}"
    for kind in ("toolUse", "toolResult"):
        inner = block.get(kind)
        if isinstance(inner, dict):
            return f"{role}:{kind}:{inner.get('toolUseId')}"
    spec: Final = block.get("toolSpec")
    return f"tool:{spec.get('name')}" if isinstance(spec, dict) else f"{role}:other"


def _cache_points(role: str, blocks: JsonValue) -> Iterator[str]:
    listed: Final = blocks if isinstance(blocks, list) else []
    for previous, block in zip([None, *listed], listed):
        if isinstance(block, dict) and "cachePoint" in block:
            yield _bedrock_label(role, previous)


def _bedrock_sections(body: dict[str, JsonValue]) -> Iterator[tuple[str, JsonValue]]:
    tool_config: Final = body.get("toolConfig")
    yield ("tool", tool_config.get("tools") if isinstance(tool_config, dict) else None)
    yield ("system", body.get("system"))
    messages: Final = body.get("messages")
    for message in messages if isinstance(messages, list) else []:
        if isinstance(message, dict):
            yield (str(message.get("role")), message.get("content"))


def bedrock_labels(request: Request) -> list[str]:
    body: Final = _JSON_OBJECT.validate_json(request.body)
    return [label for role, blocks in _bedrock_sections(body) for label in _cache_points(role, blocks)]


def bedrock_peer(request: Request) -> Reply:
    found: Final = len(bedrock_labels(request))
    if found > BEDROCK_CACHE_CHECKPOINT_CAP:
        return Reply(
            status=400,
            headers={"x-amzn-errortype": "ValidationException"},
            body=json.dumps(
                {
                    "message": f"A maximum of {BEDROCK_CACHE_CHECKPOINT_CAP} cache checkpoints may be provided. Found {found}."
                }
            ).encode(),
        )
    return Reply(
        body=json.dumps(
            {
                "output": {"message": {"role": "assistant", "content": [{"text": "sunny"}]}},
                "stopReason": "end_turn",
                "usage": {
                    "inputTokens": 12,
                    "outputTokens": 1,
                    "totalTokens": 13,
                    "cacheWriteInputTokens": 12,
                    "cacheReadInputTokens": 0,
                },
                "metrics": {"latencyMs": 1},
            }
        ).encode()
    )


def gateway_injected(response_id: str) -> bool:
    rows: Final = eventually(
        lambda: read_rows(
            """SELECT metadata->>'litellm_gateway_injected_cache' AS injected FROM "LiteLLM_SpendLogs" """
            "WHERE request_id=%s",
            (response_id,),
        ),
        lambda values: len(values) == 1,
        seconds=70,
    )
    return rows[0]["injected"] is not None


def post_chat(gateway: Gateway, body: dict[str, JsonValue], *, key: str | None = None) -> tuple[int, str, str]:
    response: Final = gateway.request("POST", "/v1/chat/completions", body, key=key)
    identity: Final = _JSON_OBJECT.validate_json(response.content).get("id") if response.status_code == 200 else None
    return response.status_code, str(identity), response.text


def anthropic_deployment(name: str, api_base: str, **fields: JsonValue) -> dict[str, JsonValue]:
    return {
        "model_name": name,
        "litellm_params": {
            "model": f"anthropic/{ANTHROPIC_MODEL}",
            "api_base": api_base,
            "api_key": PROVIDER_KEY,
            **fields,
        },
    }


def owned_config(
    directory: Path,
    model_list: Sequence[JsonValue],
    *,
    litellm_settings: Mapping[str, JsonValue] = MappingProxyType({}),
    router_settings: Mapping[str, JsonValue] = MappingProxyType({}),
) -> Path:
    config: Final = _JSON_OBJECT.validate_python(
        yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    )
    merged: Final = {
        **config,
        "model_list": list(model_list),
        "litellm_settings": {**object_value(config["litellm_settings"]), **litellm_settings},
        "router_settings": {**object_value(config["router_settings"]), "num_retries": 0, **router_settings},
    }
    path: Final = directory / f"cache-control-marks-{uuid.uuid4().hex}.yaml"
    path.write_text(yaml.safe_dump(merged))
    return path
