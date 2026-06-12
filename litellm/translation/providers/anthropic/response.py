"""Anthropic ``/v1/messages`` response JSON -> IR ``ChatResponse``.

The envelope is lenient (unknown metadata keys are ignored, exactly like v1)
but content blocks are fail-closed: a block type the v2 request surface
cannot trigger (server tools, citations, compaction) is a loud error value,
never silently dropped. The json_tool_call rewrite (structured outputs on the
json-tool strategy) happens here because the tool is an anthropic-side
artifact the caller never asked for by name.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from types import MappingProxyType

from expression import Error, Nothing, Ok, Option, Result, Some
from expression.collections import Block

from ...errors import BoundaryError, TranslationError
from ...ir import (
    CacheCreationDetails,
    ChatRequest,
    ChatResponse,
    ContentBlock,
    FinishReason,
    JsonBlob,
    PlainJson,
    RedactedThinking,
    ResponseUsage,
    Text,
    Thinking,
    ToolUse,
)
from . import params as p
from .tools import request_name_maps

_ParseResult = Result[ChatResponse, TranslationError]

# v1 map_finish_reason, anthropic rows; unknown reasons default to "stop"
# with a warning in v1, so the same default applies here.
FINISH_MAP: Mapping[str, FinishReason] = MappingProxyType(
    {
        "stop_sequence": "stop",
        "end_turn": "stop",
        "max_tokens": "length",
        "tool_use": "tool_calls",
        "refusal": "content_filter",
        "compaction": "length",
    }
)


def parse_response(raw: PlainJson, request: ChatRequest) -> _ParseResult:
    if not isinstance(raw, dict):
        return Error(_boundary("response body is not a JSON object"))
    if "error" in raw:
        return Error(_boundary(f"provider error payload: {raw['error']!r}"))
    content = raw.get("content")
    if not isinstance(content, list):
        return Error(_boundary("response 'content' is not an array"))
    blocks: list[ContentBlock] = []
    for block in content:
        parsed = _parse_block(block, request)
        if isinstance(parsed, TranslationError):
            return Error(parsed)
        blocks.append(parsed)  # nosemgrep: translation-no-mutation
    stop_reason = raw.get("stop_reason")
    finish = (
        FINISH_MAP.get(stop_reason, "stop") if isinstance(stop_reason, str) else "stop"
    )
    usage = parse_usage(raw.get("usage"))
    if isinstance(usage, TranslationError):
        return Error(usage)
    model = raw.get("model")
    response_id = raw.get("id")
    response = ChatResponse(
        id=response_id if isinstance(response_id, str) else "",
        model=model if isinstance(model, str) else request.model,
        content=Block.of_seq(blocks),
        finish=finish,
        usage=usage,
        synthesized_json_content=False,
    )
    return Ok(_resolve_json_tool(response, request))


def _boundary(reason: str) -> TranslationError:
    return TranslationError.of_boundary(BoundaryError.of(Block.of_seq([reason])))


def _parse_block(
    block: PlainJson, request: ChatRequest
) -> ContentBlock | TranslationError:
    if not isinstance(block, dict):
        return _boundary("response content block is not an object")
    if block.get("citations") is not None:
        return TranslationError.of_unsupported(
            "response citations blocks; unreachable for v2-sent requests"
        )
    kind = block.get("type")
    if kind == "text":
        text = block.get("text")
        if not isinstance(text, str):
            return _boundary("text block is missing 'text'")
        return ContentBlock.of_text(Text(text=text, cache=Nothing))
    if kind == "tool_use":
        return _parse_tool_use(block, request)
    if kind == "thinking":
        thinking = block.get("thinking")
        signature = block.get("signature")
        return ContentBlock.of_thinking(
            Thinking(
                thinking=thinking if isinstance(thinking, str) else "",
                signature=Some(signature) if isinstance(signature, str) else Nothing,
                cache=Nothing,
            )
        )
    if kind == "redacted_thinking":
        data = block.get("data")
        return ContentBlock.of_redacted_thinking(
            RedactedThinking(data=data if isinstance(data, str) else "")
        )
    return TranslationError.of_unsupported(
        f"response content block type {kind!r}; unreachable for v2-sent requests"
    )


def _parse_tool_use(
    block: dict[str, PlainJson], request: ChatRequest
) -> ContentBlock | TranslationError:
    if "caller" in block:
        return TranslationError.of_unsupported(
            "programmatic tool calling (caller); unreachable for v2-sent requests"
        )
    identifier = block.get("id")
    name = block.get("name")
    if not isinstance(identifier, str) or not isinstance(name, str):
        return _boundary("tool_use block is missing 'id'/'name'")
    _, reverse = request_name_maps(request.tools)
    return ContentBlock.of_tool_use(
        ToolUse(
            id=identifier,
            name=reverse.get(name, name),
            arguments=JsonBlob(value=block.get("input")),
            cache=Nothing,
        )
    )


def parse_usage(raw: PlainJson) -> ResponseUsage | TranslationError:
    if not isinstance(raw, dict):
        return _boundary("response 'usage' is not an object")
    creation = raw.get("cache_creation")
    details = (
        Some(_cache_creation_details(creation))
        if isinstance(creation, dict)
        else Nothing
    )
    return ResponseUsage(
        input_tokens=_int_of(raw.get("input_tokens")),
        output_tokens=_int_of(raw.get("output_tokens")),
        cache_creation_input_tokens=_int_of(raw.get("cache_creation_input_tokens")),
        cache_read_input_tokens=_int_of(raw.get("cache_read_input_tokens")),
        cache_creation=details,
        total_tokens=Nothing,
    )


def _cache_creation_details(creation: dict[str, PlainJson]) -> CacheCreationDetails:
    five = creation.get("ephemeral_5m_input_tokens")
    hour = creation.get("ephemeral_1h_input_tokens")
    return CacheCreationDetails(
        five_minute=Some(five) if isinstance(five, int) else Nothing,
        one_hour=Some(hour) if isinstance(hour, int) else Nothing,
    )


def _int_of(value: PlainJson) -> int:
    """v1 tolerates explicit nulls and non-numerics in usage counts."""
    if isinstance(value, bool):
        return 0
    if isinstance(value, (int, float)):
        return int(value)
    return 0


def _uses_json_tool(request: ChatRequest) -> bool:
    match request.response_format:
        case Option(tag="some", some=response_format):
            return response_format.tag == "json_schema" and not p.uses_output_format(
                request.model
            )
        case _:
            return False


def _resolve_json_tool(response: ChatResponse, request: ChatRequest) -> ChatResponse:
    """v1 ``_resolve_json_mode_non_streaming``: when structured outputs ride
    the json_tool_call strategy, the forced tool call comes back as plain
    content and the stop reason is rewritten to a clean stop. The mixed
    user-tools case is unreachable (the request side refuses it)."""
    if not _uses_json_tool(request):
        return response
    tool_uses = [
        block.tool_use for block in response.content if block.tag == "tool_use"
    ]
    json_calls = [tool for tool in tool_uses if tool.name == "json_tool_call"]
    if not json_calls or len(json_calls) != len(tool_uses):
        return response
    arguments = json_calls[0].arguments.value
    if arguments is None:
        return response
    payload: PlainJson = (
        arguments["values"]
        if isinstance(arguments, dict) and arguments.get("values") is not None
        else arguments
    )
    # v1 replaces the whole message with the JSON content (any text or
    # thinking blocks in the raw response are discarded with it).
    kept = Block.of_seq(
        [ContentBlock.of_text(Text(text=json.dumps(payload), cache=Nothing))]
    )
    return ChatResponse(
        id=response.id,
        model=response.model,
        content=kept,
        finish="stop",
        usage=response.usage,
        synthesized_json_content=True,
    )
