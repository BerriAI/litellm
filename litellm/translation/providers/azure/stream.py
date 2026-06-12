"""Azure OpenAI SSE chunk payloads -> IR stream events.

The chunk family is openai's (same SDK models, same
``handle_openai_chat_completion_chunk`` branch), so the decode delegates to
the openai_compat parser. Azure's two wire extras: every chunk carries a
``model`` the wrapper re-reads onto itself (streaming_handler.py:1448-1454),
re-attached here so the ``azure`` chunk dialect can adopt it; and
content-filter annotations (choice-level ``content_filter_results``,
empty-choices ``prompt_filter_results`` chunks) ride the widened
openai_compat normalization -- v1 keeps the choice-level results on content
chunks via the ``StreamingChoices(**choice_json)`` rebuild and swallows the
empty-choices chunk entirely, which the fold reproduces.
"""

from __future__ import annotations

import json

from expression import Error, Ok, Result
from expression.collections import Block

from ...errors import BoundaryError, TranslationError
from ...ir import JsonBlob, PlainJson, StreamEvent
from ..openai_compat.stream import parse_event as openai_parse_event

_EventResult = Result[StreamEvent | None, TranslationError]


def parse_line(line: str) -> _EventResult:
    stripped = line.strip()
    if not stripped.startswith("data:"):
        return Ok(None)
    payload = stripped[len("data:") :].strip()
    if payload == "[DONE]":
        return Ok(StreamEvent.of_stop())
    try:
        event: PlainJson = json.loads(payload)
    except ValueError:
        return Error(
            TranslationError.of_boundary(
                BoundaryError.of(
                    Block.of_seq([f"stream payload is not JSON: {payload[:120]!r}"])
                )
            )
        )
    return parse_event(event)


def parse_event(event: PlainJson) -> _EventResult:
    base = openai_parse_event(event)
    match base:
        case Result(tag="ok", ok=parsed):
            pass
        case _:
            return base
    if parsed is None or parsed.tag != "wire_chunk" or not isinstance(event, dict):
        return base
    model = event.get("model")
    chunk = parsed.wire_chunk.value
    if not isinstance(model, str) or not isinstance(chunk, dict):
        return base
    # The azure decode seam is the VALIDATED SDK ChatCompletionChunk, which
    # materializes service_tier=None even when the wire JSON omits it; v1's
    # preserve_upstream_non_openai_attributes then copies it onto every
    # emitted chunk.
    return Ok(
        StreamEvent.of_wire_chunk(
            JsonBlob(value={"service_tier": None, **chunk, "model": model})
        )
    )
