"""OpenAI SSE ``chat.completion.chunk`` payloads -> IR stream events.

v1 never decodes these itself: the OpenAI SDK parses SSE into
``ChatCompletionChunk`` models and ``CustomStreamWrapper``'s default branch
(``handle_openai_chat_completion_chunk``, streaming_handler.py:536-583,
1447-1509) re-emits them near-verbatim. Because the outbound chunk IS the
inbound family, each chunk maps to one ``wire_chunk`` event carrying the
chunk normalized to the SDK-dump shape (every delta key present, ``None``
when absent), and the ``openai`` chunk dialect in the inbound fold owns the
wrapper's stateful behavior (first-chunk role, empty-chunk suppression, the
choices=[] usage passthrough). Shapes a v2-sent request cannot trigger
(multiple choices, ``function_call`` deltas, logprobs, audio) are loud error
values.
"""

from __future__ import annotations

import json

from expression import Error, Ok, Result
from expression.collections import Block

from ...errors import BoundaryError, TranslationError
from ...ir import JsonBlob, PlainJson, StreamEvent

_EventResult = Result[StreamEvent | None, TranslationError]

_DELTA_KEYS = ("content", "function_call", "refusal", "role", "tool_calls")

_ENVELOPE_KEYS = frozenset(
    # ModelResponseStream.model_fields | {"usage", "error"}: the fields v1's
    # preserve_upstream_non_openai_attributes does NOT copy across.
    {
        "choices",
        "created",
        "error",
        "id",
        "model",
        "object",
        "provider_specific_fields",
        "system_fingerprint",
        "usage",
    }
)

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
        return Error(_boundary(f"stream payload is not JSON: {payload[:120]!r}"))
    return parse_event(event)


def parse_event(event: PlainJson) -> _EventResult:
    if not isinstance(event, dict):
        return Error(_boundary("stream chunk is not an object"))
    if event.get("error") is not None:
        return Error(_boundary(f"provider stream error: {event.get('error')!r}"))
    choices = event.get("choices")
    if not isinstance(choices, list):
        return Error(_boundary("stream chunk 'choices' is missing"))
    if len(choices) > 1:
        return Error(
            TranslationError.of_unsupported(
                "multiple stream choices (n > 1); unreachable for v2-sent requests"
            )
        )
    normalized_choices: list[PlainJson] = []
    if len(choices) == 1:
        normalized = _normalize_choice(choices[0])
        if isinstance(normalized, TranslationError):
            return Error(normalized)
        normalized_choices = [normalized]
    identifier = event.get("id")
    chunk: dict[str, PlainJson] = {
        # Keys outside ModelResponseStream's field set ride along verbatim:
        # v1's wrapper setattrs them onto every emitted chunk
        # (preserve_upstream_non_openai_attributes, e.g. service_tier).
        **{key: value for key, value in event.items() if key not in _ENVELOPE_KEYS},
        "id": identifier if isinstance(identifier, str) else None,
        "system_fingerprint": _string_or_none(event.get("system_fingerprint")),
        "choices": normalized_choices,
        "usage": event.get("usage") if isinstance(event.get("usage"), dict) else None,
    }
    return Ok(StreamEvent.of_wire_chunk(JsonBlob(value=chunk)))


def _boundary(reason: str) -> TranslationError:
    return TranslationError.of_boundary(BoundaryError.of(Block.of_seq([reason])))


def _string_or_none(value: PlainJson) -> PlainJson:
    return value if isinstance(value, str) else None


def _normalize_choice(choice: PlainJson) -> PlainJson | TranslationError:
    if not isinstance(choice, dict):
        return _boundary("stream choice is not an object")
    extra_keys = set(choice.keys()) - {
        "index",
        "delta",
        "logprobs",
        "finish_reason",
        # azure's content-filter annotation: v1's wrapper keeps it on content
        # chunks via the StreamingChoices(**choice_json) rebuild (the same,
        # provider-agnostic code path), so it rides the normalized choice and
        # the fold decides emission (kept on content, dropped on finish).
        "content_filter_results",
    }
    if extra_keys:
        return TranslationError.of_unsupported(
            f"stream choice keys {sorted(extra_keys)!r}; unreachable for v2-sent requests"
        )
    if choice.get("logprobs") is not None:
        return TranslationError.of_unsupported(
            "stream logprobs; unreachable for v2-sent requests"
        )
    finish = choice.get("finish_reason")
    if finish is not None and not isinstance(finish, str):
        return _boundary("stream finish_reason is not a string")
    if finish == "function_call":
        return TranslationError.of_unsupported(
            "legacy function_call stream finish; the v2 surface cannot send 'functions'"
        )
    # Any other finish string rides through verbatim (post-send leniency, the
    # PR #30138 boundary): the fold emits it raw and the shared
    # StreamingChoices envelope runs v1's live map_finish_reason on BOTH
    # sides, so quirky compat finishes normalize identically instead of
    # erroring after the request was billed.
    raw_delta = choice.get("delta")
    delta = raw_delta if isinstance(raw_delta, dict) else {}
    normalized_delta = _normalize_delta(delta)
    if isinstance(normalized_delta, TranslationError):
        return normalized_delta
    if finish is not None and _delta_bears_content(normalized_delta):
        return TranslationError.of_unsupported(
            "finish chunk with a non-empty delta; v1's wrapper interleaves it"
        )
    index = choice.get("index")
    normalized: dict[str, PlainJson] = {
        "index": index if isinstance(index, int) else 0,
        "delta": normalized_delta,
        "logprobs": None,
        "finish_reason": finish,
    }
    if "content_filter_results" in choice:
        normalized = {
            **normalized,
            "content_filter_results": choice["content_filter_results"],
        }
    return normalized


def _normalize_delta(
    delta: dict[str, PlainJson],
) -> dict[str, PlainJson] | TranslationError:
    extra_keys = set(delta.keys()) - set(_DELTA_KEYS)
    if extra_keys:
        return TranslationError.of_unsupported(
            f"stream delta keys {sorted(extra_keys)!r}; unreachable for v2-sent requests"
        )
    if delta.get("function_call") is not None:
        return TranslationError.of_unsupported(
            "legacy function_call stream delta; the v2 surface cannot send 'functions'"
        )
    tool_calls = delta.get("tool_calls")
    normalized_calls: PlainJson = None
    if tool_calls is not None:
        if not isinstance(tool_calls, list):
            return _boundary("stream delta 'tool_calls' is not an array")
        gathered: list[PlainJson] = []
        for call in tool_calls:
            normalized = _normalize_tool_call(call)
            if isinstance(normalized, TranslationError):
                return normalized
            gathered = [*gathered, normalized]
        normalized_calls = gathered
    return {
        "content": _string_or_none(delta.get("content")),
        "function_call": None,
        "refusal": _string_or_none(delta.get("refusal")),
        "role": _string_or_none(delta.get("role")),
        "tool_calls": normalized_calls,
    }


def _normalize_tool_call(call: PlainJson) -> PlainJson | TranslationError:
    if not isinstance(call, dict):
        return _boundary("stream tool_call is not an object")
    extra_keys = set(call.keys()) - {"index", "id", "function", "type"}
    if extra_keys:
        return TranslationError.of_unsupported(
            f"stream tool_call keys {sorted(extra_keys)!r}; unreachable for v2-sent requests"
        )
    raw_function = call.get("function")
    function = raw_function if isinstance(raw_function, dict) else {}
    index = call.get("index")
    return {
        "index": index if isinstance(index, int) else 0,
        "id": _string_or_none(call.get("id")),
        "function": {
            "arguments": _string_or_none(function.get("arguments")),
            "name": _string_or_none(function.get("name")),
        },
        "type": _string_or_none(call.get("type")),
    }


def _delta_bears_content(delta: dict[str, PlainJson]) -> bool:
    content = delta.get("content")
    return (isinstance(content, str) and len(content) > 0) or delta.get(
        "tool_calls"
    ) is not None
