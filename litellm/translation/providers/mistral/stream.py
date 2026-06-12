"""Mistral SSE ``chat.completion.chunk`` payloads -> IR stream events.

v1's decode is ``MistralChatResponseIterator(
OpenAIChatCompletionStreamingHandler)`` over standard ``data:``/[DONE] SSE
lines: a pre-step normalizes magistral content-LIST deltas into
``content`` (None when no text segments) + ``thinking_blocks`` (signature
``"mistral"``) + ``reasoning_content`` (set only when a thinking text
exists), then the BASE handler rebuild runs — which is exactly the shared
httpx_chunk factory's behavior (``reasoning="rename"``: the base handler
pops ``delta.reasoning`` into ``reasoning_content``). The factory's new
``passthrough_delta_keys`` axis admits the pre-step's ``thinking_blocks``
(wave-2b-beta; mistral is the consumer, rows in
test_differential_mistral_stream.py).

v1's pre-step wraps itself in ``except Exception: return super().
chunk_parser(chunk)`` — but the base rebuild then raises on the still-list
content (Delta validation), so malformed content lists are LOUD here too
(boundary errors naming the v1 raise).
"""

from __future__ import annotations

from expression import Error, Result
from expression.collections import Block

from ...errors import BoundaryError, TranslationError
from ...ir import PlainJson, StreamEvent
from ..openai_compat.httpx_chunk import HttpxChunkPolicy, make_parse_event
from ..openai_compat.stream import make_parse_line

_EventResult = Result[StreamEvent | None, TranslationError]

_POLICY = HttpxChunkPolicy(
    reasoning="rename", passthrough_delta_keys=("thinking_blocks",)
)
_family_parse_event = make_parse_event(_POLICY)


def _boundary(reason: str) -> TranslationError:
    return TranslationError.of_boundary(
        BoundaryError.of(Block.of_seq([f"mistral {reason} (v1 raises)"]))
    )


def parse_event(event: PlainJson) -> _EventResult:
    pre_stepped = _with_content_list_pre_step(event)
    if isinstance(pre_stepped, TranslationError):
        return Error(pre_stepped)
    return _family_parse_event(pre_stepped)


def _with_content_list_pre_step(event: PlainJson) -> PlainJson | TranslationError:
    if not isinstance(event, dict):
        return event  # the factory owns the non-object boundary error
    choices = event.get("choices")
    if not isinstance(choices, list):
        return event
    reshaped: list[PlainJson] = []
    for choice in choices:
        stepped = _pre_step_choice(choice)
        if isinstance(stepped, TranslationError):
            return stepped
        reshaped = [*reshaped, stepped]
    return {**event, "choices": reshaped}


def _pre_step_choice(choice: PlainJson) -> PlainJson | TranslationError:
    if not isinstance(choice, dict):
        return choice
    delta = choice.get("delta")
    if not isinstance(delta, dict):
        return choice
    content = delta.get("content")
    if not isinstance(content, list):
        return choice
    normalized = _normalize_content_blocks(content)
    if isinstance(normalized, TranslationError):
        return normalized
    text, thinking_blocks, reasoning = normalized
    stepped: dict[str, PlainJson] = {
        key: value
        for key, value in delta.items()
        if key not in ("thinking_blocks", "reasoning_content")
    }
    stepped = {**stepped, "content": text}
    if thinking_blocks:
        stepped = {
            **stepped,
            "thinking_blocks": thinking_blocks,
            "reasoning_content": reasoning,
        }
    return {**choice, "delta": stepped}


def _normalize_content_blocks(
    content: list[PlainJson],
) -> tuple[PlainJson, list[PlainJson], PlainJson] | TranslationError:
    """Mirror ``MistralChatResponseIterator._normalize_content_blocks``:
    text segments JOIN (None when none exist), thinking texts join with
    ``""`` per block and ``"\\n"`` across blocks."""
    text_segments: list[str] = []
    thinking_blocks: list[PlainJson] = []
    reasoning_segments: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            return _boundary("stream content block is not an object")
        block_type = block.get("type")
        if block_type == "thinking":
            thinking = block.get("thinking", [])
            if not isinstance(thinking, list):
                return _boundary("stream thinking value is not a list")
            collected = _thinking_text(thinking)
            if isinstance(collected, TranslationError):
                return collected
            if collected:
                reasoning_segments = [*reasoning_segments, collected]
                thinking_blocks = [
                    *thinking_blocks,
                    {
                        "type": "thinking",
                        "thinking": collected,
                        "signature": "mistral",
                    },
                ]
        elif block_type == "text":
            text = block.get("text", "")
            if not isinstance(text, str):
                return _boundary("stream text block is not a string")
            text_segments = [*text_segments, text]
    normalized_text: PlainJson = "".join(text_segments) if text_segments else None
    reasoning: PlainJson = "\n".join(reasoning_segments) if reasoning_segments else None
    return normalized_text, thinking_blocks, reasoning


def _thinking_text(thinking: list[PlainJson]) -> str | TranslationError:
    parts: list[str] = []
    for item in thinking:
        if not isinstance(item, dict):
            return _boundary("stream thinking item is not an object")
        if item.get("type") == "text":
            text = item.get("text", "")
            if not isinstance(text, str):
                # the old arm coerced to "" and SERVED a stripped chunk;
                # v1's pre-step except-arm replays the still-list content
                # and the wrapper raises MidStreamFallbackError
                return _boundary("stream thinking text is not a string")
            parts = [*parts, text]
    return "".join(parts)


parse_line = make_parse_line(parse_event)
