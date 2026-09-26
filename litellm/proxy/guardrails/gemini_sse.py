"""Gemini SSE frame masking for guardrail streaming hooks.

The Google ``:streamGenerateContent`` route relays the upstream ``data:`` frames as raw bytes, so a
guardrail's ``async_post_call_streaming_iterator_hook`` cannot read them as chunk objects. These
helpers fold such a stream into the one response body a non-streaming call would have returned,
rewrite its text and function-call arguments in place, and re-emit it as one frame. Every other
field (thought signatures, function-call ids, model version, response id, safety ratings, usage)
is carried through untouched, because a client echoes the model turn back on its next request and
Gemini 3 rejects a function call whose thought signature is missing.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import accumulate, chain, groupby
from types import MappingProxyType
from typing import Final, TypeAlias

from pydantic import TypeAdapter, ValidationError

from litellm.proxy.guardrails.anthropic_sse import joined_sse_stream, parsed_sse_events

_GEMINI_RESPONSE_KEYS: Final = frozenset({"candidates", "usageMetadata", "promptFeedback"})
_TEXT_RUN_PART_KEYS: Final = frozenset({"text", "thought", "thoughtSignature"})
_EMPTY_UNSIGNED_TEXT_PART: Final = MappingProxyType({"text": ""})
_UNREADABLE_ARGUMENTS: Final = "left the function call arguments unreadable as JSON after masking"

TextMasker: TypeAlias = Callable[[str], Awaitable[str]]  # mutable-ok: Callable params
_JsonObject: TypeAlias = Mapping[str, object]
_JSON_OBJECT: Final = TypeAdapter(_JsonObject)
_JSON_ARRAY: Final = TypeAdapter(tuple[object, ...])


@dataclass(frozen=True, slots=True)
class GeminiStreamUnchanged:
    """Masking rewrote nothing, so the original frames are replayed as they arrived."""


@dataclass(frozen=True, slots=True)
class GeminiStreamMasked:
    frames: tuple[bytes, ...]


@dataclass(frozen=True, slots=True)
class GeminiStreamUnreadable:
    reason: str


GeminiStreamMaskResult: TypeAlias = GeminiStreamUnchanged | GeminiStreamMasked | GeminiStreamUnreadable


def is_gemini_sse_stream(all_chunks: Sequence[object]) -> bool:
    sse_stream: Final = joined_sse_stream(all_chunks)
    if sse_stream is None:
        return False
    return any(not _GEMINI_RESPONSE_KEYS.isdisjoint(event) for event in parsed_sse_events(sse_stream))


async def mask_gemini_sse_stream(all_chunks: Sequence[object], mask_text: TextMasker) -> GeminiStreamMaskResult:
    """The stream's frames folded into one response, with every text and function-call argument masked.

    Text arrives split across frames, so the fragments of one text part are joined before they are
    scanned; PII that only exists once the fragments meet is otherwise forwarded in halves. Top-level
    keys and candidate keys take the last frame's value, and candidates are merged by index.
    """
    sse_stream: Final = joined_sse_stream(all_chunks)
    if sse_stream is None:
        return GeminiStreamUnreadable("could not decode the streaming response as UTF-8")
    events: Final = parsed_sse_events(sse_stream)
    candidates: Final = _merged_candidates(tuple(_candidates_of(events)))
    masked_candidates: Final = tuple([await _masked_candidate(candidate, mask_text) for candidate in candidates])
    unreadable: Final = next((item for item in masked_candidates if isinstance(item, GeminiStreamUnreadable)), None)
    if unreadable is not None:
        return unreadable
    if masked_candidates == candidates:
        return GeminiStreamUnchanged()
    response: Final = MappingProxyType({**_later_wins(events), "candidates": masked_candidates})
    return GeminiStreamMasked((f"data: {_json_text(response)}\n\n".encode(),))


def _json_text(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, default=dict)


def _json_object(value: object) -> _JsonObject | None:
    try:
        return _JSON_OBJECT.validate_python(value)
    except ValidationError:
        return None


def _json_objects(values: Iterable[object]) -> Iterator[_JsonObject]:
    return (parsed for parsed in map(_json_object, values) if parsed is not None)


def _json_array(value: object) -> tuple[object, ...]:
    try:
        return _JSON_ARRAY.validate_python(value)
    except ValidationError:
        return ()


def _later_wins(mappings: Sequence[_JsonObject]) -> _JsonObject:
    return MappingProxyType(dict(chain.from_iterable(mapping.items() for mapping in mappings)))


def _candidates_of(events: Sequence[_JsonObject]) -> Iterator[_JsonObject]:
    for event in events:
        yield from _json_objects(_json_array(event.get("candidates")))


def _merged_candidates(candidates: Sequence[_JsonObject]) -> tuple[_JsonObject, ...]:
    indexes: Final = tuple(dict.fromkeys(_candidate_index(candidate) for candidate in candidates))
    return tuple(
        _merged_candidate(tuple(candidate for candidate in candidates if _candidate_index(candidate) == index))
        for index in indexes
    )


def _candidate_index(candidate: _JsonObject) -> int:
    index: Final = candidate.get("index")
    return index if isinstance(index, int) else 0


def _merged_candidate(fragments: Sequence[_JsonObject]) -> _JsonObject:
    merged: Final = _later_wins(fragments)
    contents: Final = tuple(_json_objects(fragment.get("content") for fragment in fragments))
    if not contents:
        return merged
    content: Final = MappingProxyType({**_later_wins(contents), "parts": _merged_parts(tuple(_parts_of(contents)))})
    return MappingProxyType({**merged, "content": content})


def _parts_of(contents: Sequence[_JsonObject]) -> Iterator[object]:
    for content in contents:
        yield from _json_array(content.get("parts"))


def _merged_parts(parts: Sequence[object]) -> tuple[object, ...]:
    """Adjacent fragments of one text part joined into that part; every other part kept as it came.

    The empty unsigned text part Gemini streams on its final frame is dropped, since the
    non-streaming body carries none and Gemini rejects it when a client echoes it back.
    """
    run_ids: Final = tuple(run_id for run_id, _ in accumulate(parts, _next_run, initial=(0, None)))[1:]
    merged: Final = tuple(
        _merged_run(tuple(part for part, _ in run)) for _, run in groupby(zip(parts, run_ids), key=lambda pair: pair[1])
    )
    return tuple(part for part in merged if part != _EMPTY_UNSIGNED_TEXT_PART)


def _next_run(state: tuple[int, object], part: object) -> tuple[int, object]:
    run_id, previous = state
    return (run_id if _continues_text_run(previous, part) else run_id + 1, part)


def _continues_text_run(previous: object, part: object) -> bool:
    """Whether ``part`` is the next fragment of the text part ``previous`` started.

    A fragment carrying a thought signature closes its run, since the signature belongs to the
    part it arrived on and one merged part can only carry one.
    """
    previous_fragment: Final = _text_fragment(previous)
    fragment: Final = _text_fragment(part)
    if previous_fragment is None or fragment is None:
        return False
    return (
        bool(previous_fragment.get("thought")) == bool(fragment.get("thought"))
        and "thoughtSignature" not in previous_fragment
    )


def _text_fragment(part: object) -> _JsonObject | None:
    fragment: Final = _json_object(part)
    if fragment is None or not _TEXT_RUN_PART_KEYS.issuperset(fragment) or not isinstance(fragment.get("text"), str):
        return None
    return fragment


def _merged_run(run: Sequence[object]) -> object:
    """A run longer than one part holds text fragments only, since nothing else continues a run."""
    if len(run) == 1:
        return run[0]
    fragments: Final = tuple(_json_objects(run))
    text: Final = "".join(str(fragment["text"]) for fragment in fragments)
    return MappingProxyType({**fragments[0], **fragments[-1], "text": text})


async def _masked_candidate(candidate: _JsonObject, mask_text: TextMasker) -> _JsonObject | GeminiStreamUnreadable:
    content: Final = _json_object(candidate.get("content"))
    if content is None:
        return candidate
    parts: Final = _json_array(content.get("parts"))
    masked_parts: Final = tuple([await _masked_part(part, mask_text) for part in parts])
    unreadable: Final = next((item for item in masked_parts if isinstance(item, GeminiStreamUnreadable)), None)
    if unreadable is not None:
        return unreadable
    return MappingProxyType({**candidate, "content": MappingProxyType({**content, "parts": masked_parts})})


async def _masked_part(part: object, mask_text: TextMasker) -> object | GeminiStreamUnreadable:
    fragment: Final = _json_object(part)
    if fragment is None:
        return part
    text: Final = fragment.get("text")
    if isinstance(text, str):
        return MappingProxyType({**fragment, "text": await mask_text(text)}) if text else part
    function_call: Final = _json_object(fragment.get("functionCall"))
    if function_call is None:
        return part
    arguments: Final = _json_object(function_call.get("args"))
    if arguments is None:
        return part
    masked_arguments: Final = await mask_text(_json_text(arguments))
    try:
        parsed_arguments: Final = _JSON_OBJECT.validate_json(masked_arguments)
    except ValidationError:
        return GeminiStreamUnreadable(_UNREADABLE_ARGUMENTS)
    return MappingProxyType({**fragment, "functionCall": MappingProxyType({**function_call, "args": parsed_arguments})})
