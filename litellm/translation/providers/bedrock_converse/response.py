"""Bedrock Converse response JSON -> IR ``ChatResponse``.

The envelope is lenient like v1 (``ConverseResponseBlock`` is a TypedDict
cast); content blocks are fail-closed: a block the v2 request surface cannot
trigger (citations, guardrail traces, reasoning embedded in text) is a loud
error value. The json_tool_call rewrite (structured outputs ride the
synthetic-tool strategy on every ported converse model) happens here,
including v1's single-"properties"-key unwrap.
"""

from __future__ import annotations

import json

from expression import Error, Nothing, Ok, Option, Result, Some
from expression.collections import Block

from ...errors import BoundaryError, TranslationError
from ...ir import (
    ChatRequest,
    ChatResponse,
    ContentBlock,
    JsonBlob,
    PlainJson,
    RedactedThinking,
    ResponseUsage,
    Text,
    Thinking,
    ToolUse,
)
from .params import FINISH_MAP
from .tools import reverse_name_map

_ParseResult = Result[ChatResponse, TranslationError]


def parse_response(raw: PlainJson, request: ChatRequest) -> _ParseResult:
    if not isinstance(raw, dict):
        return Error(_boundary("response body is not a JSON object"))
    if "trace" in raw:
        return Error(
            TranslationError.of_unsupported(
                "guardrail trace responses; unreachable for v2-sent requests"
            )
        )
    output = raw.get("output")
    message = output.get("message") if isinstance(output, dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, list):
        return Error(_boundary("response output.message.content is not an array"))
    reverse = reverse_name_map(request.tools)
    blocks: list[ContentBlock] = []
    for block in content:
        parsed = _parse_block(block, reverse)
        if isinstance(parsed, TranslationError):
            return Error(parsed)
        if parsed is not None:
            blocks.append(parsed)  # nosemgrep: translation-no-mutation
    stop_reason = raw.get("stopReason")
    finish = (
        FINISH_MAP.get(stop_reason, "stop") if isinstance(stop_reason, str) else "stop"
    )
    usage = _parse_usage(raw.get("usage"))
    if isinstance(usage, TranslationError):
        return Error(usage)
    response = ChatResponse(
        id="",
        model=request.model,
        content=Block.of_seq(blocks),
        finish=finish,
        usage=usage,
        synthesized_json_content=False,
    )
    return Ok(_resolve_json_tool(response, request))


def _boundary(reason: str) -> TranslationError:
    return TranslationError.of_boundary(BoundaryError.of(Block.of_seq([reason])))


def _parse_block(
    block: PlainJson, reverse: dict[str, str]
) -> ContentBlock | None | TranslationError:
    if not isinstance(block, dict):
        return _boundary("response content block is not an object")
    if "citationsContent" in block:
        return TranslationError.of_unsupported(
            "citationsContent blocks; unreachable for v2-sent requests"
        )
    if "text" in block:
        text = block["text"]
        if not isinstance(text, str):
            return _boundary("text block is not a string")
        if "<thinking>" in text:
            return TranslationError.of_unsupported(
                "reasoning embedded in text takes v1's _parse_content_for_reasoning"
            )
        return ContentBlock.of_text(Text(text=text, cache=Nothing))
    if "toolUse" in block:
        return _parse_tool_use(block["toolUse"], reverse)
    if "reasoningContent" in block:
        return _parse_reasoning(block["reasoningContent"])
    return TranslationError.of_unsupported(
        f"response content block keys {sorted(block)!r}; unreachable for v2-sent requests"
    )


def _parse_tool_use(
    payload: PlainJson, reverse: dict[str, str]
) -> ContentBlock | TranslationError:
    if not isinstance(payload, dict):
        return _boundary("toolUse block is not an object")
    identifier = payload.get("toolUseId")
    name = payload.get("name")
    if not isinstance(identifier, str) or not isinstance(name, str):
        return _boundary("toolUse block is missing 'toolUseId'/'name'")
    return ContentBlock.of_tool_use(
        ToolUse(
            id=identifier,
            name=reverse.get(name, name),
            arguments=JsonBlob(value=payload.get("input")),
            cache=Nothing,
        )
    )


def _parse_reasoning(payload: PlainJson) -> ContentBlock | None | TranslationError:
    if not isinstance(payload, dict):
        return _boundary("reasoningContent block is not an object")
    if "redactedContent" in payload:
        data = payload["redactedContent"]
        return ContentBlock.of_redacted_thinking(
            RedactedThinking(data=data if isinstance(data, str) else "")
        )
    text_block = payload.get("reasoningText")
    if not isinstance(text_block, dict):
        return _boundary("reasoningContent block is missing 'reasoningText'")
    text = text_block.get("text")
    signature = text_block.get("signature")
    return ContentBlock.of_thinking(
        Thinking(
            thinking=text if isinstance(text, str) else "",
            signature=Some(signature) if isinstance(signature, str) else Nothing,
            cache=Nothing,
        )
    )


def _parse_usage(raw: PlainJson) -> ResponseUsage | TranslationError:
    if not isinstance(raw, dict):
        return _boundary("response 'usage' is not an object")
    input_tokens = _int_of(raw.get("inputTokens"))
    output_tokens = _int_of(raw.get("outputTokens"))
    total = raw.get("totalTokens")
    return ResponseUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_creation_input_tokens=_int_of(raw.get("cacheWriteInputTokens")),
        cache_read_input_tokens=_int_of(raw.get("cacheReadInputTokens")),
        cache_creation=Nothing,
        total_tokens=Some(_int_of(total)) if total is not None else Some(0),
    )


def _int_of(value: PlainJson) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, (int, float)):
        return int(value)
    return 0


def _uses_json_tool(request: ChatRequest) -> bool:
    match request.response_format:
        case Option(tag="some", some=response_format):
            return response_format.tag == "json_schema"
        case _:
            return False


def _resolve_json_tool(response: ChatResponse, request: ChatRequest) -> ChatResponse:
    """v1 ``_filter_json_mode_tools`` for the only reachable shape: every tool
    call is the synthetic json_tool_call (the request side refuses mixing
    user tools with response_format). The message content is REPLACED by the
    unwrapped arguments JSON; reasoning blocks survive; finish becomes stop."""
    if not _uses_json_tool(request):
        return response
    tool_uses = [
        block.tool_use for block in response.content if block.tag == "tool_use"
    ]
    json_calls = [tool for tool in tool_uses if tool.name == "json_tool_call"]
    if not json_calls or len(json_calls) != len(tool_uses):
        return response
    arguments = json_calls[0].arguments.value
    payload: PlainJson = (
        arguments["properties"]
        if (
            isinstance(arguments, dict)
            and len(arguments) == 1
            and "properties" in arguments
        )
        else arguments
    )
    kept = [
        block
        for block in response.content
        if block.tag in ("thinking", "redacted_thinking")
    ]
    text = ContentBlock.of_text(Text(text=json.dumps(payload), cache=Nothing))
    return ChatResponse(
        id=response.id,
        model=response.model,
        content=Block.of_seq([*kept, text]),
        finish="stop",
        usage=response.usage,
        synthesized_json_content=True,
    )
