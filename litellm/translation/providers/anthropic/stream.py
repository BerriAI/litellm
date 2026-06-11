"""Anthropic SSE stream -> IR stream events.

One SSE ``data:`` payload maps to at most one ``StreamEvent``; keep-alives
and block-stop bookkeeping map to none. Event types the v2 request surface
cannot trigger (server tools, citations) are loud error values. Tool names
are reverse-mapped through the per-request map, mirroring the non-streaming
path.
"""

from __future__ import annotations

import json
from collections.abc import Mapping

from expression import Error, Ok, Result
from expression.collections import Block

from ...errors import BoundaryError, TranslationError
from ...ir import (
    ChatRequest,
    FinishReason,
    PlainJson,
    SignatureDelta,
    StreamEvent,
    StreamFinish,
    StreamStart,
    TextDelta,
    ThinkingDelta,
    ToolArgsDelta,
    ToolUseStart,
)
from .response import FINISH_MAP, parse_usage
from .tools import request_name_maps

_EventResult = Result[StreamEvent | None, TranslationError]


def reverse_names(request: ChatRequest) -> Mapping[str, str]:
    _, reverse = request_name_maps(request.tools)
    return reverse


def parse_sse_line(line: str, reverse: Mapping[str, str]) -> _EventResult:
    """One raw SSE line -> at most one event. Non-data lines are framing."""
    stripped = line.strip()
    if not stripped.startswith("data:"):
        return Ok(None)
    payload = stripped[len("data:") :].strip()
    try:
        event: PlainJson = json.loads(payload)
    except ValueError:
        return Error(_boundary(f"stream payload is not JSON: {payload[:120]!r}"))
    return parse_event(event, reverse)


def parse_event(event: PlainJson, reverse: Mapping[str, str]) -> _EventResult:
    if not isinstance(event, dict):
        return Error(_boundary("stream event is not an object"))
    kind = event.get("type")
    if kind in ("ping", "content_block_stop"):
        return Ok(None)
    if kind == "message_start":
        return _start_event(event)
    if kind == "content_block_start":
        return _block_start_event(event, reverse)
    if kind == "content_block_delta":
        return _delta_event(event)
    if kind == "message_delta":
        return _finish_event(event)
    if kind == "message_stop":
        return Ok(StreamEvent.of_stop())
    if kind == "error":
        return Error(_boundary(f"provider stream error: {event.get('error')!r}"))
    return Error(
        TranslationError.of_unsupported(
            f"stream event type {kind!r}; unreachable for v2-sent requests"
        )
    )


def _boundary(reason: str) -> TranslationError:
    return TranslationError.of_boundary(BoundaryError.of(Block.of_seq([reason])))


def _start_event(event: dict[str, PlainJson]) -> _EventResult:
    message = event.get("message")
    if not isinstance(message, dict):
        return Error(_boundary("message_start is missing 'message'"))
    usage = parse_usage(message.get("usage") or {})
    if isinstance(usage, TranslationError):
        return Error(usage)
    identifier = message.get("id")
    model = message.get("model")
    return Ok(
        StreamEvent.of_start(
            StreamStart(
                id=identifier if isinstance(identifier, str) else "",
                model=model if isinstance(model, str) else "",
                usage=usage,
            )
        )
    )


def _block_start_event(
    event: dict[str, PlainJson], reverse: Mapping[str, str]
) -> _EventResult:
    block = event.get("content_block")
    index = event.get("index")
    if not isinstance(block, dict) or not isinstance(index, int):
        return Error(_boundary("content_block_start is malformed"))
    kind = block.get("type")
    if kind == "text":
        text = block.get("text")
        if isinstance(text, str) and text:
            return Ok(StreamEvent.of_text_delta(TextDelta(index=index, text=text)))
        return Ok(None)
    if kind == "thinking":
        return Ok(None)  # thinking starts empty; deltas carry the content
    if kind == "tool_use":
        identifier = block.get("id")
        name = block.get("name")
        if not isinstance(identifier, str) or not isinstance(name, str):
            return Error(_boundary("tool_use block start is missing 'id'/'name'"))
        return Ok(
            StreamEvent.of_tool_use_start(
                ToolUseStart(index=index, id=identifier, name=reverse.get(name, name))
            )
        )
    return Error(
        TranslationError.of_unsupported(
            f"stream content block type {kind!r}; unreachable for v2-sent requests"
        )
    )


def _delta_event(event: dict[str, PlainJson]) -> _EventResult:
    delta = event.get("delta")
    index = event.get("index")
    if not isinstance(delta, dict) or not isinstance(index, int):
        return Error(_boundary("content_block_delta is malformed"))
    kind = delta.get("type")
    if kind == "text_delta":
        text = delta.get("text")
        return Ok(
            StreamEvent.of_text_delta(
                TextDelta(index=index, text=text if isinstance(text, str) else "")
            )
        )
    if kind == "input_json_delta":
        partial = delta.get("partial_json")
        return Ok(
            StreamEvent.of_tool_args_delta(
                ToolArgsDelta(
                    index=index,
                    partial_json=partial if isinstance(partial, str) else "",
                )
            )
        )
    if kind == "thinking_delta":
        thinking = delta.get("thinking")
        return Ok(
            StreamEvent.of_thinking_delta(
                ThinkingDelta(
                    index=index,
                    thinking=thinking if isinstance(thinking, str) else "",
                )
            )
        )
    if kind == "signature_delta":
        signature = delta.get("signature")
        return Ok(
            StreamEvent.of_signature_delta(
                SignatureDelta(
                    index=index,
                    signature=signature if isinstance(signature, str) else "",
                )
            )
        )
    return Error(
        TranslationError.of_unsupported(
            f"stream delta type {kind!r}; unreachable for v2-sent requests"
        )
    )


def _finish_event(event: dict[str, PlainJson]) -> _EventResult:
    delta = event.get("delta")
    usage = event.get("usage")
    stop_reason = delta.get("stop_reason") if isinstance(delta, dict) else None
    finish: FinishReason = (
        FINISH_MAP.get(stop_reason, "stop") if isinstance(stop_reason, str) else "stop"
    )
    output_tokens = 0
    if isinstance(usage, dict):
        raw = usage.get("output_tokens")
        output_tokens = int(raw) if isinstance(raw, (int, float)) else 0
    return Ok(
        StreamEvent.of_finish(StreamFinish(finish=finish, output_tokens=output_tokens))
    )
