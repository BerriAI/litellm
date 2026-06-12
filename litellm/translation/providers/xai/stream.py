"""xai (Grok) SSE ``chat.completion.chunk`` payloads -> IR stream events.

The v1 decode is the httpx dict path: ``BaseModelResponseIterator`` strips
``data:`` lines, ``XAIChatCompletionStreamingHandler.chunk_parser`` applies
the xai rewrites, and the rebuilt ``ModelResponseStream`` flows through
``CustomStreamWrapper``'s default openai branch. The xai deltas vs the
openai parser (all verified in-process at HEAD):

- the chunk_parser rebuild keeps ONLY id/created/model/choices/usage:
  ``system_fingerprint`` and every top-level extra (``citations``) are
  DROPPED — the opposite of the SDK path's verbatim extras passthrough.
- a ``choices: []`` chunk carrying ``usage`` gets a dummy choice injected in
  v1 purely so the wrapper machinery swallows it and re-synthesizes the
  final usage chunk; v2 keeps the openai passthrough shape (``choices: []``
  + the FOLDED usage) so the seam's synthesized-final-usage contract from
  the openai port applies unchanged.
- every usage-bearing chunk gets the reasoning fold + total normalize
  (the dict variants of the response post-steps), but the folded usage is
  ATTACHED only to the ``choices: []`` tail: v1's wrapper strips usage from
  every emitted content/finish chunk and only the synthesized final chunk
  carries it (verifier-grok F2).
- ``delta.reasoning`` is renamed to ``reasoning_content`` (the base
  handler's rename) and native ``reasoning_content`` deltas are admitted —
  real Grok reasoning traffic, not an unreachable shape.
- a tool_call entry WITHOUT a ``type`` key gains ``type: "function"``
  (litellm's ``ChatCompletionDeltaToolCall`` default fires on the dict
  path; the SDK path yields None there, so the openai parser must not).
"""

from __future__ import annotations

from expression import Error, Ok, Result
from expression.collections import Block

from ...errors import BoundaryError, TranslationError
from ...ir import JsonBlob, PlainJson, StreamEvent
from ..openai_compat.stream import make_parse_line
from .response import fold_reasoning_tokens, normalize_usage_totals

_EventResult = Result[StreamEvent | None, TranslationError]

_DELTA_KEYS = (
    "content",
    "function_call",
    "refusal",
    "role",
    "tool_calls",
    "reasoning",
    "reasoning_content",
)


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
    folded = _folded_usage(event.get("usage"))
    if isinstance(folded, TranslationError):
        return Error(folded)
    normalized_choices: list[PlainJson] = []
    if len(choices) == 1:
        normalized = _normalize_choice(choices[0])
        if isinstance(normalized, TranslationError):
            return Error(normalized)
        normalized_choices = [normalized]
    identifier = event.get("id")
    chunk: dict[str, PlainJson] = {
        # No extras passthrough and no system_fingerprint: the v1 chunk_parser
        # rebuild keeps only id/created/model/choices/usage. Usage rides ONLY
        # the choices:[] tail: v1's wrapper strips usage from every emitted
        # content/finish chunk and re-synthesizes the final usage chunk
        # (verifier-grok F2; the fold above still runs so uncoercible values
        # stay as loud as v1's chunk_parser).
        "id": identifier if isinstance(identifier, str) else None,
        "system_fingerprint": None,
        "choices": normalized_choices,
        "usage": folded if len(normalized_choices) == 0 else None,
    }
    return Ok(StreamEvent.of_wire_chunk(JsonBlob(value=chunk)))


def _folded_usage(raw_usage: PlainJson) -> PlainJson | TranslationError:
    """v1 folds + normalizes every usage-bearing chunk inside chunk_parser;
    an uncoercible token value raises out of the iterator there, so the
    coercion errors from the shared arithmetic stay loud here too."""
    if not isinstance(raw_usage, dict):
        return None
    folded = fold_reasoning_tokens(raw_usage)
    if isinstance(folded, TranslationError):
        return folded
    return normalize_usage_totals(folded)


def _boundary(reason: str) -> TranslationError:
    return TranslationError.of_boundary(BoundaryError.of(Block.of_seq([reason])))


def _string_or_none(value: PlainJson) -> PlainJson:
    return value if isinstance(value, str) else None


def _normalize_choice(choice: PlainJson) -> PlainJson | TranslationError:
    if not isinstance(choice, dict):
        return _boundary("stream choice is not an object")
    extra_keys = set(choice.keys()) - {"index", "delta", "logprobs", "finish_reason"}
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
    return {
        "index": index if isinstance(index, int) else 0,
        "delta": normalized_delta,
        "logprobs": None,
        "finish_reason": finish,
    }


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
    # provider_specific_fields: None always (the dict-path wrapper stamps the
    # null provider field on content-bearing deltas); refusal and
    # reasoning_content ride VERBATIM but only when the wire carried the key,
    # mirroring Delta's set-only serialization — v1 FORWARDS a refusal that
    # rides a role/content delta and swallows refusal-only deltas
    # (verifier-grok F1 corrected the earlier drop claim; replay-verified).
    base: dict[str, PlainJson] = {
        "content": _string_or_none(delta.get("content")),
        "function_call": None,
        "provider_specific_fields": None,
        "role": _string_or_none(delta.get("role")),
        "tool_calls": normalized_calls,
    }
    if "refusal" in delta:
        base = {**base, "refusal": _string_or_none(delta.get("refusal"))}
    # the base handler renames delta.reasoning unconditionally
    # (_map_reasoning_to_reasoning_content)
    reasoning = (
        delta.get("reasoning")
        if "reasoning" in delta
        else (delta.get("reasoning_content") if "reasoning_content" in delta else None)
    )
    if "reasoning" in delta or "reasoning_content" in delta:
        return {**base, "reasoning_content": _string_or_none(reasoning)}
    return base


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
        # dict-path default: ChatCompletionDeltaToolCall fills "function" when
        # the wire omits the key (the SDK path keeps None there)
        "type": _string_or_none(call.get("type")) if "type" in call else "function",
    }


def _delta_bears_content(delta: dict[str, PlainJson]) -> bool:
    content = delta.get("content")
    return (isinstance(content, str) and len(content) > 0) or delta.get(
        "tool_calls"
    ) is not None


parse_line = make_parse_line(parse_event)
