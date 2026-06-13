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

``parse_event`` therefore composes THREE steps mirroring that validation
(verifier-wave2b-beta F5 widened the original content/role/refusal-only
pre-check):

1. ``_validation_reason`` — every shape the pydantic construction REJECTS
   is a loud typed error (v1 raises out of the decoder, no watsonx-style
   swallow): non-str delta content/role/refusal, a present non-dict delta
   (the validated model requires the field), non-list ``tool_calls``,
   tool entries with non-str id/name/arguments/type or non-coercible
   index, non-lax-int choice indexes (None included — pydantic rejects
   it), non-dict ``usage``, and usage dicts missing any of the three
   token keys or carrying values the lax coercion rejects (None and
   fractional floats RAISE; digit strings, bools and integral floats
   coerce — all probed);
2. ``_normalized_event`` — the SERVING lax coercions v1's validation
   applies (choice/tool index bool -> int, digit-str -> int, integral
   float -> int) plus v1's finish_reason semantics at the wrapper
   (verifier F7): a FALSY finish ("", 0, {}) fails the truthy gate — no
   finish rides and the synthesized trailing stop is seam scope; a truthy
   non-str hashable finish maps to "stop" (map_finish_reason's .get
   default — v1 SERVES it, the old parser erred); a truthy UNHASHABLE
   finish is loud in step 1 (v1's map raises TypeError); truthy strings
   ride verbatim (the envelope runs v1's live map on both sides);
3. the shared openai parser + the litellm-validation post-step.

There is no SSE line form (no ``parse_line``).
"""

from __future__ import annotations

from expression import Error, Result
from expression.collections import Block

from ...errors import BoundaryError, TranslationError
from ...ir import JsonBlob, PlainJson, StreamEvent
from ..openai_compat.stream import parse_event as openai_parse_event

_EventResult = Result[StreamEvent | None, TranslationError]

_USAGE_KEYS = ("prompt_tokens", "completion_tokens", "total_tokens")


def parse_event(event: PlainJson) -> _EventResult:
    reason = _validation_reason(event)
    if reason is not None:
        return Error(
            TranslationError.of_boundary(
                BoundaryError.of(Block.of_seq([f"sagemaker_chat {reason}"]))
            )
        )
    return openai_parse_event(_normalized_event(event)).map(_with_validation_defaults)


def _validation_reason(event: PlainJson) -> str | None:
    if not isinstance(event, dict):
        return None  # the parser owns the non-object error
    usage_reason = _usage_reason(event.get("usage"), "usage" in event)
    if usage_reason is not None:
        return usage_reason
    choices = event.get("choices")
    if not isinstance(choices, list):
        return None  # the parser owns the missing-choices error (v1 KeyErrors)
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        reason = _choice_reason(choice)
        if reason is not None:
            return reason
    return None


def _choice_reason(choice: dict[str, PlainJson]) -> str | None:
    index = choice.get("index")
    if "index" in choice and _lax_int(index) is None:
        return (
            "stream choice index is not an integer (v1's "
            "StreamingChatCompletionChunk validation raises — None included)"
        )
    finish = choice.get("finish_reason")
    if finish and not isinstance(finish, (str, bool, int, float)):
        return (
            "stream finish_reason is a truthy unhashable value (v1's "
            "map_finish_reason raises TypeError)"
        )
    delta = choice.get("delta")
    if delta is not None and not isinstance(delta, dict):
        return (
            "stream delta is not an object (v1's StreamingChatCompletionChunk "
            "validation raises — delta Field required)"
        )
    if not isinstance(delta, dict):
        return None
    return _delta_reason(delta)


def _delta_reason(delta: dict[str, PlainJson]) -> str | None:
    for field in ("content", "role", "refusal"):
        value = delta.get(field)
        if value is not None and not isinstance(value, str):
            return (
                f"stream delta {field} is not a string (v1's "
                "StreamingChatCompletionChunk validation raises)"
            )
    tool_calls = delta.get("tool_calls")
    if tool_calls is None:
        return None
    if not isinstance(tool_calls, list):
        return (
            "stream delta tool_calls is not a list (v1's "
            "StreamingChatCompletionChunk validation raises)"
        )
    for call in tool_calls:
        if not isinstance(call, dict):
            continue  # v1's validation DROPS non-dict entries and serves
        reason = _tool_call_reason(call)
        if reason is not None:
            return reason
    return None


def _tool_call_reason(call: dict[str, PlainJson]) -> str | None:
    function = call.get("function")
    if not isinstance(function, dict):
        return (
            "stream tool_call function is missing or not an object (v1's "
            "ChatCompletionDeltaToolCall validation raises)"
        )
    if (
        "index" in call
        and call.get("index") is not None
        and _lax_int(call.get("index")) is None
    ):
        return (
            "stream tool_call index is not an integer (v1's "
            "ChatCompletionDeltaToolCall validation raises)"
        )
    for field, value in (
        ("id", call.get("id")),
        ("type", call.get("type")),
        ("name", function.get("name")),
        ("arguments", function.get("arguments")),
    ):
        if value is not None and not isinstance(value, str):
            return (
                f"stream tool_call {field} is not a string (v1's "
                "ChatCompletionDeltaToolCall validation raises)"
            )
    return None


def _usage_reason(usage: PlainJson, present: bool) -> str | None:
    if not present or usage is None:
        return None
    if not isinstance(usage, dict):
        return (
            "stream usage is not an object (v1's StreamingChatCompletionChunk "
            "validation raises)"
        )
    for key in _USAGE_KEYS:
        if key not in usage or _lax_int(usage.get(key)) is None:
            return (
                f"stream usage {key} is missing or not an integer (v1's "
                "Usage validation raises — None and fractional values "
                "included; digit strings/bools/integral floats coerce)"
            )
    return None


def _lax_int(value: PlainJson) -> int | None:
    """pydantic's lax int: bool/int, integral floats, numeric strings."""
    if isinstance(value, (bool, int)):
        return int(value)
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    if not isinstance(value, str):
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        as_float = float(value)
    except ValueError:
        return None
    return int(as_float) if as_float.is_integer() else None


def _normalized_event(event: PlainJson) -> PlainJson:
    if not isinstance(event, dict):
        return event
    choices = event.get("choices")
    if not isinstance(choices, list):
        return event
    reshaped = [_normalized_choice(choice) for choice in choices]
    return {**event, "choices": reshaped}


def _normalized_choice(choice: PlainJson) -> PlainJson:
    if not isinstance(choice, dict):
        return choice
    stepped: dict[str, PlainJson] = {**choice, "finish_reason": _finish(choice)}
    if "index" in choice:
        coerced = _lax_int(choice.get("index"))
        if coerced is not None:
            stepped = {**stepped, "index": coerced}
    delta = choice.get("delta")
    if isinstance(delta, dict):
        tool_calls = delta.get("tool_calls")
        if isinstance(tool_calls, list):
            calls = [_normalized_tool_call(call) for call in tool_calls]
            stepped = {**stepped, "delta": {**delta, "tool_calls": calls}}
    return stepped


def _finish(choice: dict[str, PlainJson]) -> PlainJson:
    finish = choice.get("finish_reason")
    if not finish:
        # v1's wrapper truthy-gates the finish: falsy values ride as NO
        # finish and the synthesized trailing stop is seam scope (F7)
        return None
    if not isinstance(finish, str):
        # truthy non-str hashable: v1 serves map_finish_reason's default
        return "stop"
    return finish


def _normalized_tool_call(call: PlainJson) -> PlainJson:
    if not isinstance(call, dict) or "index" not in call:
        return call
    coerced = _lax_int(call.get("index"))
    return call if coerced is None else {**call, "index": coerced}


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
