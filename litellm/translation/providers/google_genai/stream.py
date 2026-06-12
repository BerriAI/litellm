"""Parsed generateContent stream events -> IR composite chunks.

Pinned at the parsed-event seam: SSE ``data:`` line splitting and v1's
accumulated-json fragmentation fallback are transport plumbing in front of
``ModelResponseIterator.chunk_parser``. One ``GenerateContentResponse`` event
maps onto one ``CompositeChunk`` (gemini streams COMPLETE functionCall args
per chunk; there are no partial deltas to decompose). Mid-stream HTTP-200
``error`` objects are loud errors exactly like v1 raising ``VertexAIError``;
the stop->tool_calls finish rewrite and cumulative tool index live in the
inbound fold's ``StreamState``.
"""

from __future__ import annotations

import json

from expression import Error, Nothing, Ok, Option, Result, Some
from expression.collections import Block

from ...errors import BoundaryError, TranslationError
from ...ir import (
    CompositeChunk,
    FinishReason,
    PlainJson,
    ResponseUsage,
    StreamEvent,
    StreamToolCall,
)
from . import params as p
from .response import parse_usage_metadata

_EventResult = Result[StreamEvent | None, TranslationError]


def parse_event(event: PlainJson) -> _EventResult:
    if not isinstance(event, dict):
        return Error(_boundary("stream event is not an object"))
    if "error" in event:
        return Error(
            _boundary(f"mid-stream error payload (v1 raises): {event['error']!r}")
        )
    feedback = event.get("promptFeedback")
    if isinstance(feedback, dict) and "blockReason" in feedback:
        return Error(
            TranslationError.of_unsupported(
                "promptFeedback.blockReason stream chunks take v1's content-filter path"
            )
        )
    candidates = event.get("candidates")
    usage = _usage_of(event)
    if isinstance(usage, TranslationError):
        return Error(usage)
    if not isinstance(candidates, list) or len(candidates) == 0:
        if usage.is_none():
            return Ok(None)
        return Ok(StreamEvent.of_chunk(_empty_chunk(event, usage)))
    if len(candidates) > 1:
        return Error(
            TranslationError.of_unsupported(
                "multiple stream candidates (n > 1); v1 emits multiple choices"
            )
        )
    candidate = candidates[0]
    if not isinstance(candidate, dict):
        return Error(_boundary("stream candidate is not an object"))
    return _candidate_chunk(event, candidate, usage)


def _boundary(reason: str) -> TranslationError:
    return TranslationError.of_boundary(BoundaryError.of(Block.of_seq([reason])))


def _event_id(event: dict[str, PlainJson]) -> str:
    response_id = event.get("responseId")
    return response_id if isinstance(response_id, str) else ""


def _usage_of(
    event: dict[str, PlainJson],
) -> Option[ResponseUsage] | TranslationError:
    if "usageMetadata" not in event:
        return Nothing
    parsed = parse_usage_metadata(event.get("usageMetadata"))
    if isinstance(parsed, TranslationError):
        return parsed
    return Some(parsed)


def _empty_chunk(
    event: dict[str, PlainJson], usage: Option[ResponseUsage]
) -> CompositeChunk:
    return CompositeChunk(
        id=_event_id(event),
        text=Nothing,
        reasoning=Nothing,
        signatures=Block.empty(),
        tool_calls=Block.empty(),
        finish=Nothing,
        usage=usage,
    )


def _candidate_chunk(
    event: dict[str, PlainJson],
    candidate: dict[str, PlainJson],
    usage: Option[ResponseUsage],
) -> _EventResult:
    for key in ("groundingMetadata", "urlContextMetadata", "logprobsResult"):
        if key in candidate:
            return Error(
                TranslationError.of_unsupported(
                    f"stream candidate {key}; v1's metadata extraction handles it"
                )
            )
    content = candidate.get("content")
    parts = content.get("parts") if isinstance(content, dict) else None
    collected = _collect_parts(parts if isinstance(parts, list) else [])
    if isinstance(collected, TranslationError):
        return Error(collected)
    text, reasoning, signatures, tool_calls = collected
    finish_raw = candidate.get("finishReason")
    finish: Option[FinishReason] = Nothing
    if isinstance(finish_raw, str):
        finish = Some(p.map_finish(finish_raw, len(tool_calls) > 0))
    return Ok(
        StreamEvent.of_chunk(
            CompositeChunk(
                id=_event_id(event),
                text=Some(text) if text is not None else Nothing,
                reasoning=Some(reasoning) if reasoning is not None else Nothing,
                signatures=Block.of_seq(signatures),
                tool_calls=Block.of_seq(tool_calls),
                finish=finish,
                usage=usage,
            )
        )
    )


_Collected = tuple[str | None, str | None, list[str], list[StreamToolCall]]


def _collect_parts(parts: list[PlainJson]) -> _Collected | TranslationError:
    text: str | None = None
    reasoning: str | None = None
    signatures: list[str] = []
    tool_calls: list[StreamToolCall] = []
    for part in parts:
        if not isinstance(part, dict):
            return _boundary("stream part is not an object")
        signature = part.get("thoughtSignature")
        if isinstance(signature, str):
            signatures = [*signatures, signature]
        if "functionCall" in part:
            call = _tool_call(part)
            if isinstance(call, TranslationError):
                return call
            tool_calls = [*tool_calls, call]
            continue
        if "inlineData" in part:
            return TranslationError.of_unsupported(
                "inlineData stream parts (media output); v1 handles them"
            )
        part_text = part.get("text")
        if not isinstance(part_text, str):
            return TranslationError.of_unsupported(
                f"stream part keys {sorted(part)!r}; v1 handles them"
            )
        if len(part_text) == 0:
            continue  # v1 get_assistant_content_message skips empty strings
        if part.get("thought") is True:
            reasoning = (reasoning or "") + part_text
        else:
            text = (text or "") + part_text
    return text, reasoning, signatures, tool_calls


def _tool_call(part: dict[str, PlainJson]) -> StreamToolCall | TranslationError:
    call = part.get("functionCall")
    if not isinstance(call, dict):
        return _boundary("stream functionCall part is malformed")
    name = call.get("name")
    args = call.get("args")
    if not isinstance(name, str) or not isinstance(args, dict):
        return _boundary("stream functionCall is missing name/args")
    native_id = call.get("id")
    identifier = native_id if isinstance(native_id, str) else ""
    signature = part.get("thoughtSignature")
    if isinstance(signature, str) and signature:
        identifier = f"{identifier}{p.THOUGHT_SIGNATURE_SEPARATOR}{signature}"
    return StreamToolCall(
        id=identifier,
        name=name,
        arguments_json=json.dumps(args, ensure_ascii=False),
    )
