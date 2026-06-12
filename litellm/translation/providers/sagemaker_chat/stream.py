"""sagemaker_chat stream events, pinned at the AWS event-stream
PARSED-EVENT seam (the bedrock precedent: botocore framing is transport).

v1 decodes AWS event-stream BYTES through ``AWSEventStreamDecoder(model="",
is_messages_api=True)`` and each decoded JSON event becomes a VALIDATED
``StreamingChatCompletionChunk`` riding ``CustomStreamWrapper``'s default
openai branch. The validated-chunk materialization is the SDK-dump shape
the shared openai parser normalizes — PLUS two litellm-Delta validation
effects the SDK wire never shows (probed in-process at HEAD):

- every emitted content delta carries ``provider_specific_fields: None``
  (the litellm Delta rebuild materializes it on the FIRST chunk too, where
  the SDK path's role-bearing delta lacks it);
- a tool_call ``type`` that is missing OR explicit-null validates to
  ``"function"`` (the dict-path default; the SDK parser keeps None).

So ``parse_event`` composes the shared openai parser with a post-step
applying exactly those two normalizations. A chunk failing validation
RAISES out of v1's decoder (not swallowed — unlike watsonx's iterator),
matching the parser's loud errors. There is no SSE line form (no
``parse_line``).
"""

from __future__ import annotations

from expression import Error, Result
from expression.collections import Block

from ...errors import BoundaryError, TranslationError
from ...ir import JsonBlob, PlainJson, StreamEvent
from ..openai_compat.stream import parse_event as openai_parse_event

_EventResult = Result[StreamEvent | None, TranslationError]


def parse_event(event: PlainJson) -> _EventResult:
    reason = _validation_reason(event)
    if reason is not None:
        return Error(
            TranslationError.of_boundary(
                BoundaryError.of(
                    Block.of_seq(
                        [
                            f"sagemaker_chat {reason} (v1's "
                            "StreamingChatCompletionChunk validation raises)"
                        ]
                    )
                )
            )
        )
    return openai_parse_event(event).map(_with_validation_defaults)


def _validation_reason(event: PlainJson) -> str | None:
    """The shared openai parser leniently coerces non-string delta fields to
    None (safe on the SDK path, where upstream validation guarantees
    strings), but v1's decoder VALIDATES each event here and raises — so
    wrong-typed delta strings must be loud, never coerced-and-served."""
    if not isinstance(event, dict):
        return None  # the parser owns the non-object error
    choices = event.get("choices")
    if not isinstance(choices, list):
        return None
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        delta = choice.get("delta")
        if not isinstance(delta, dict):
            continue
        for field in ("content", "role", "refusal"):
            value = delta.get(field)
            if value is not None and not isinstance(value, str):
                return f"stream delta {field} is not a string"
    return None


def _with_validation_defaults(event: StreamEvent | None) -> StreamEvent | None:
    if event is None or event.tag != "wire_chunk":
        return event
    chunk = event.wire_chunk.value
    if not isinstance(chunk, dict):
        return event
    choices = chunk.get("choices")
    if not isinstance(choices, list):
        return event
    reshaped = [_choice_with_defaults(choice) for choice in choices]
    return StreamEvent.of_wire_chunk(JsonBlob(value={**chunk, "choices": reshaped}))


def _choice_with_defaults(choice: PlainJson) -> PlainJson:
    if not isinstance(choice, dict):
        return choice
    delta = choice.get("delta")
    if not isinstance(delta, dict):
        return choice
    stepped: dict[str, PlainJson] = {"provider_specific_fields": None, **delta}
    tool_calls = stepped.get("tool_calls")
    if isinstance(tool_calls, list):
        stepped = {
            **stepped,
            "tool_calls": [_tool_call_with_type(call) for call in tool_calls],
        }
    return {**choice, "delta": stepped}


def _tool_call_with_type(call: PlainJson) -> PlainJson:
    if not isinstance(call, dict):
        return call
    if call.get("type") is None:
        return {**call, "type": "function"}
    return call
