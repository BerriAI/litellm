"""watsonx SSE stream lines -> IR stream events (the GENERIC dialect).

researcher-4 classified the watsonx stream as "openai dialect over SSE" —
DRIFT: the openai_like handler decodes through the databricks
``ModelResponseIterator`` (llms/databricks/streaming_utils.py), which
yields ``GenericStreamingChunk`` dicts that ride ``CustomStreamWrapper``'s
GENERIC arm — the same arm cohere uses, so the fold dialect is the shared
``generic`` one. Mirrored per chunk (probed in-process at HEAD):

- line decode = ``_strip_sse_data_from_chunk``: the ``data:`` prefix is
  OPTIONAL (raw JSON lines parse too), empty lines yield empty chunks the
  wrapper drops, ``[DONE]`` is a stop marker;
- ``choices: []`` -> a usage-only payload (``or 0`` coercions);
- delta content -> text; role-only and name-only-tool-call deltas yield
  EMPTY chunks v1 swallows (a tool_call without non-None ``arguments``
  never reaches the wire chunk — pinned); a PRESENT non-dict delta is
  LOUD — v1's StreamingChoices sets no ``.delta`` attribute and the
  iterator AttributeErrors -> MidStreamFallbackError (verifier F4);
- only the FIRST tool call rides, as id/type/function/INDEX (the index key
  is part of v1's GChunk here, unlike cohere's); tool-call value types
  mirror the validated Delta — lax index coercion serves (str "3" -> 3,
  bool -> int, integral float; v1's pydantic does the same), everything
  the validation rejects (non-str id/name/type, non-coercible index) is a
  LOUD pinned divergence: v1 swallows the whole chunk and v2 never
  invents id/name None or index 0 in its place (critic B1.4);
- finish_reason: a truthy STRING rides VERBATIM — the chunk envelope
  (``ModelResponseStream`` -> ``StreamingChoices``) runs v1's live
  ``map_finish_reason`` on BOTH sides, so the OpenAI four, IBM's
  time_limit/cancelled/error -> "stop", max_tokens -> "length" AND every
  unknown string -> "stop" normalize identically (verifier F7 refuted the
  old conservative loud arm: v1 SERVES "stop"); a truthy non-str hashable
  value -> "stop" (v1's map .get default); a FALSY value -> no finish
  (StreamingChoices' truthy gate; the synthesized trailing finish is seam
  scope); a truthy unhashable value is LOUD (v1's map .get raises
  TypeError -> MidStreamFallbackError);
- usage rides ONLY on the ``choices: []`` tail payload (v1's wrapper
  strips mid-stream usage from emitted chunks; the include_usage final
  chunk is seam scope — NOTE the wrapper OVERRIDES null/zero tail values
  with token-counter ESTIMATES there, so the future streaming seam must
  reproduce the estimate arm, not just pass the tail through); a PRESENT
  non-dict usage is LOUD on any chunk (v1 reads ``.prompt_tokens`` off it
  raw and AttributeErrors -> MidStreamFallbackError); usage VALUES mirror
  litellm ``Usage``'s lax coercion (None -> 0, digit strings, bools,
  integral floats — probed) and non-coercible values are LOUD pinned
  divergences (validation fails, v1 swallows the WHOLE chunk).

PINNED DIVERGENCES (fail-closed on failure paths, the compat_httpx
error-chunk precedent): v1 SILENTLY SWALLOWS non-JSON lines, chunks that
fail ``ModelResponseStream`` validation (pydantic ValidationError is a
ValueError, caught by the iterator's except arm), and non-str delta
content (the wrapper's generic arm drops it) — v2 errors loudly on each,
named report rows.
"""

from __future__ import annotations

import json

from expression import Error, Ok, Result
from expression.collections import Block

from ...errors import BoundaryError, TranslationError
from ...ir import JsonBlob, PlainJson, StreamEvent

_EventResult = Result[StreamEvent | None, TranslationError]


def _boundary(reason: str) -> TranslationError:
    return TranslationError.of_boundary(BoundaryError.of(Block.of_seq([reason])))


def parse_line(line: str) -> _EventResult:
    stripped = line.strip()
    if not stripped:
        return Ok(None)  # v1 yields an empty GenericStreamingChunk
    if stripped.startswith("data:"):
        stripped = stripped[len("data:") :].strip()
    if not stripped:
        return Ok(None)
    if stripped == "[DONE]":
        return Ok(StreamEvent.of_stop())
    try:
        event: PlainJson = json.loads(stripped)
    except ValueError:
        return Error(
            _boundary(
                f"non-JSON watsonx stream line {stripped[:80]!r}: v1's "
                "iterator SWALLOWS it silently (pinned divergence — "
                "fail-closed on a failure path)"
            )
        )
    return parse_event(event)


def parse_event(event: PlainJson) -> _EventResult:
    if not isinstance(event, dict):
        return Error(_boundary("watsonx stream chunk is not an object"))
    choices = event.get("choices")
    if not isinstance(choices, list):
        return Error(
            _boundary(
                "watsonx stream chunk has no choices list (v1's "
                "ModelResponseStream validation swallows it — pinned "
                "divergence)"
            )
        )
    if len(choices) == 0:
        return _usage_only_payload(event)
    if len(choices) > 1:
        return Error(
            TranslationError.of_unsupported(
                "multiple stream choices (n > 1); unreachable for v2-sent requests"
            )
        )
    choice = choices[0]
    if not isinstance(choice, dict):
        return Error(_boundary("watsonx stream choice is not an object (v1 swallows)"))
    mid_usage = _mid_usage_check(event)
    if isinstance(mid_usage, TranslationError):
        return Error(mid_usage)
    return _choice_payload(choice)


def _choice_payload(choice: dict[str, PlainJson]) -> _EventResult:
    delta = _delta_value(choice)
    if isinstance(delta, TranslationError):
        return Error(delta)
    text = _text_value(delta)
    if isinstance(text, TranslationError):
        return Error(text)
    tool_call = _first_tool_call(delta)
    if isinstance(tool_call, TranslationError):
        return Error(tool_call)
    finish = _finish_value(choice)
    if isinstance(finish, TranslationError):
        return Error(finish)
    if not text and tool_call is None and finish is None:
        return Ok(None)  # empty GenericStreamingChunk; the wrapper drops it
    payload: dict[str, PlainJson] = {
        "text": text,
        "tool_call": tool_call,
        "finish": finish,
        "usage": None,  # valid mid-stream usage is STRIPPED (v1's wrapper)
        "provider_fields": None,
    }
    return Ok(StreamEvent.of_wire_chunk(JsonBlob(value=payload)))


def _delta_value(
    choice: dict[str, PlainJson],
) -> dict[str, PlainJson] | TranslationError:
    raw_delta = choice.get("delta")
    if raw_delta is None:
        return {}  # StreamingChoices defaults a missing/null delta to Delta()
    if not isinstance(raw_delta, dict):
        return _boundary(
            "watsonx delta is present but not an object (v1's "
            "StreamingChoices sets no .delta and the iterator "
            "AttributeErrors -> MidStreamFallbackError)"
        )
    return raw_delta


def _mid_usage_check(event: dict[str, PlainJson]) -> None | TranslationError:
    """Mirror v1 on a content/tool chunk carrying ``usage``: a present
    non-dict raises out of the iterator (raw ``.prompt_tokens`` read ->
    AttributeError -> MidStreamFallbackError); a dict failing ``Usage``'s
    lax validation fails ModelResponseStream construction and v1 SWALLOWS
    the whole chunk (pinned divergence); a valid dict is stripped from the
    emitted chunk (the wrapper withholds mid-stream usage)."""
    if "usage" not in event:
        return None
    usage = event.get("usage")
    if usage is None:
        return None
    if not isinstance(usage, dict):
        return _boundary(
            "watsonx mid-stream usage is not an object (v1 reads "
            ".prompt_tokens off it and AttributeErrors -> "
            "MidStreamFallbackError)"
        )
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = _int_or_zero(usage.get(key), key)
        if isinstance(value, TranslationError):
            return value
    return None


def _usage_only_payload(event: dict[str, PlainJson]) -> _EventResult:
    usage = event.get("usage")
    if usage is None:
        return Ok(None)
    if not isinstance(usage, dict):
        return Error(
            _boundary(
                "watsonx usage tail is not an object (v1 reads "
                ".prompt_tokens off it and AttributeErrors -> "
                "MidStreamFallbackError)"
            )
        )
    coerced: dict[str, PlainJson] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = _int_or_zero(usage.get(key), key)
        if isinstance(value, TranslationError):
            return Error(value)
        coerced = {**coerced, key: value}
    payload: dict[str, PlainJson] = {
        "text": "",
        "tool_call": None,
        "finish": None,
        "usage": coerced,
        "provider_fields": None,
    }
    return Ok(StreamEvent.of_wire_chunk(JsonBlob(value=payload)))


def _int_or_zero(value: PlainJson, key: str) -> int | TranslationError:
    """v1's ``final_usage.prompt_tokens or 0`` over the VALIDATED Usage —
    litellm ``Usage`` lax-coerces None -> 0, bools, integral floats and
    numeric strings (probed); anything it rejects fails ModelResponseStream
    construction and v1 swallows the whole chunk."""
    if value is None:
        return 0
    if isinstance(value, (bool, int)):
        return int(value) or 0
    if isinstance(value, float) and value.is_integer():
        return int(value) or 0
    if isinstance(value, str):
        coerced = _lax_int(value)
        if coerced is not None:
            return coerced or 0
    return _boundary(
        f"watsonx usage {key} is not an integer (v1's Usage validation "
        "swallows the chunk — pinned divergence)"
    )


def _lax_int(value: str) -> int | None:
    """pydantic's lax str -> int (digit strings, integral float strings)."""
    try:
        return int(value)
    except ValueError:
        pass
    try:
        as_float = float(value)
    except ValueError:
        return None
    return int(as_float) if as_float.is_integer() else None


def _text_value(delta: dict[str, PlainJson]) -> str | TranslationError:
    content = delta.get("content")
    if content is None:
        return ""
    if not isinstance(content, str):
        return _boundary(
            "watsonx delta content is not a string (v1 silently swallows "
            "the chunk — probed: the non-str text never reaches an emitted "
            "chunk — pinned divergence)"
        )
    return content


def _first_tool_call(delta: dict[str, PlainJson]) -> PlainJson | TranslationError:
    """Mirror the iterator over the VALIDATED Delta: ``function`` is a
    REQUIRED field (a tool_call without it fails ModelResponseStream
    validation and v1 swallows the whole chunk — pinned divergence) and
    missing/null ``arguments`` validates to ``""``, so the iterator's
    ``arguments is not None`` check never fails post-validation: name-only
    tool starts ride with arguments ``""``."""
    tool_calls = delta.get("tool_calls")
    if tool_calls is None:
        return None
    if not isinstance(tool_calls, list):
        return _boundary(
            "watsonx delta tool_calls is not a list (the validated Delta "
            "keeps it verbatim and v1's iterator subscript raises -> "
            "MidStreamFallbackError)"
        )
    if len(tool_calls) == 0:
        return None
    first = tool_calls[0]
    if not isinstance(first, dict):
        return _boundary(
            "watsonx tool_calls entry is not an object (v1's Delta "
            "validation drops it and swallows the tool content — pinned "
            "divergence)"
        )
    function = first.get("function")
    if not isinstance(function, dict):
        return _boundary(
            "watsonx tool_call without a function object (v1's validation "
            "swallows the chunk — pinned divergence)"
        )
    arguments = function.get("arguments")
    if arguments is not None and not isinstance(arguments, str):
        return _boundary(
            "watsonx tool_call arguments is not a string (v1's validation "
            "swallows the chunk — pinned divergence)"
        )
    field_reason = _tool_field_reason(first, function)
    if field_reason is not None:
        return _boundary(field_reason)
    index = _tool_index(first.get("index"))
    if isinstance(index, TranslationError):
        return index
    identifier = first.get("id")
    name = function.get("name")
    return {
        "id": identifier if isinstance(identifier, str) else None,
        "type": "function",  # v1's GChunk hardcodes the type
        "function": {
            "name": name if isinstance(name, str) else None,
            "arguments": arguments if isinstance(arguments, str) else "",
        },
        "index": index,
    }


def _tool_field_reason(
    first: dict[str, PlainJson], function: dict[str, PlainJson]
) -> str | None:
    """Non-str id/name/type fail v1's Delta validation and the iterator
    SWALLOWS the whole chunk — the old tail coerced id/name to None and
    SERVED an invented tool call on a chunk v1 drops (critic B1.4)."""
    for field, value in (
        ("id", first.get("id")),
        ("name", function.get("name")),
        ("type", first.get("type")),
    ):
        if value is not None and not isinstance(value, str):
            return (
                f"watsonx tool_call {field} is not a string (v1's Delta "
                "validation swallows the chunk — pinned divergence; v2 "
                "never invents a value in its place)"
            )
    return None


def _tool_index(index: PlainJson) -> int | TranslationError:
    """The validated Delta lax-coerces the index (str "3" -> 3, bool -> int,
    integral float -> int; absent/None -> 0 — all probed served by v1); a
    non-coercible value fails validation and v1 swallows the whole chunk.
    The old arm rewrote every non-int to 0 — a value invention."""
    if index is None:
        return 0
    if isinstance(index, (bool, int)):
        return int(index)
    if isinstance(index, float) and index.is_integer():
        return int(index)
    if isinstance(index, str):
        coerced = _lax_int(index)
        if coerced is not None:
            return coerced
    return _boundary(
        "watsonx tool_call index is not an integer (v1's Delta validation "
        "swallows the chunk — pinned divergence; v2 never rewrites it to 0)"
    )


def _finish_value(choice: dict[str, PlainJson]) -> PlainJson | TranslationError:
    """Mirror StreamingChoices' finish handling inside v1's
    ``ModelResponseStream(**chunk)`` validation (probed at HEAD, verifier
    F7): a FALSY value (None/""/0/{}/[]/false) fails the truthy gate — no
    finish rides, the wrapper's synthesized trailing finish is seam scope;
    a truthy STRING rides VERBATIM (the chunk envelope runs v1's live
    ``map_finish_reason`` on BOTH sides — known members map, unknown
    strings default to "stop", identically); a truthy non-str hashable
    value maps to "stop" (the same ``.get`` default); a truthy UNHASHABLE
    value raises TypeError out of v1's map -> MidStreamFallbackError."""
    finish = choice.get("finish_reason")
    if not finish:
        return None
    if isinstance(finish, str):
        return finish
    if isinstance(finish, (bool, int, float)):
        return "stop"
    return _boundary(
        "watsonx finish_reason is a truthy unhashable value (v1's "
        "map_finish_reason raises TypeError -> MidStreamFallbackError)"
    )
