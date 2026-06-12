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
  never reaches the wire chunk — pinned);
- only the FIRST tool call rides, as id/type/function/INDEX (the index key
  is part of v1's GChunk here, unlike cohere's);
- finish_reason maps through the wrapper's ``map_finish_reason``: the
  OpenAI four ride verbatim, IBM's time_limit/cancelled/error -> "stop",
  max_tokens -> "length"; anything else is a LOUD error (conservative —
  v1 defaults unknowns to "stop");
- usage rides ONLY on the ``choices: []`` tail payload (v1's wrapper
  strips mid-stream usage from emitted chunks; the include_usage final
  chunk is seam scope).

PINNED DIVERGENCES (fail-closed on failure paths, the compat_httpx
error-chunk precedent): v1's iterator SILENTLY SWALLOWS non-JSON lines and
chunks that fail ``ModelResponseStream`` validation (pydantic
ValidationError is a ValueError, caught by the iterator's except arm) —
v2 errors loudly on both, named report rows.
"""

from __future__ import annotations

import json
from types import MappingProxyType

from expression import Error, Ok, Result
from expression.collections import Block

from ...errors import BoundaryError, TranslationError
from ...ir import JsonBlob, PlainJson, StreamEvent

_EventResult = Result[StreamEvent | None, TranslationError]

_FINISH_MAP = MappingProxyType(
    {
        "stop": "stop",
        "length": "length",
        "tool_calls": "tool_calls",
        "content_filter": "content_filter",
        "time_limit": "stop",
        "cancelled": "stop",
        "error": "stop",
        "max_tokens": "length",
    }
)


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
    raw_delta = choice.get("delta", {})
    delta = raw_delta if isinstance(raw_delta, dict) else {}
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
        "usage": None,
        "provider_fields": None,
    }
    return Ok(StreamEvent.of_wire_chunk(JsonBlob(value=payload)))


def _usage_only_payload(event: dict[str, PlainJson]) -> _EventResult:
    usage = event.get("usage")
    if usage is None:
        return Ok(None)
    if not isinstance(usage, dict):
        return Error(
            _boundary(
                "watsonx usage tail is not an object (v1's validation "
                "swallows the chunk — pinned divergence)"
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
    """v1's ``final_usage.prompt_tokens or 0`` over the VALIDATED Usage."""
    if value is None:
        return 0
    if isinstance(value, (bool, int)):
        return int(value) or 0
    if isinstance(value, float) and value.is_integer():
        return int(value) or 0
    return _boundary(
        f"watsonx usage {key} is not an integer (v1's Usage validation "
        "swallows the chunk — pinned divergence)"
    )


def _text_value(delta: dict[str, PlainJson]) -> str | TranslationError:
    content = delta.get("content")
    if content is None:
        return ""
    if not isinstance(content, str):
        return _boundary(
            "watsonx delta content is not a string (v1's validation "
            "swallows the chunk — pinned divergence)"
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
            "watsonx delta tool_calls is malformed (v1's validation "
            "swallows the chunk — pinned divergence)"
        )
    if len(tool_calls) == 0:
        return None
    first = tool_calls[0]
    if not isinstance(first, dict):
        return _boundary(
            "watsonx delta tool_calls is malformed (v1's validation "
            "swallows the chunk — pinned divergence)"
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
    identifier = first.get("id")
    name = function.get("name")
    index = first.get("index")
    return {
        "id": identifier if isinstance(identifier, str) else None,
        "type": "function",
        "function": {
            "name": name if isinstance(name, str) else None,
            "arguments": arguments if isinstance(arguments, str) else "",
        },
        "index": index if isinstance(index, int) else 0,
    }


def _finish_value(choice: dict[str, PlainJson]) -> PlainJson | TranslationError:
    finish = choice.get("finish_reason")
    if finish is None:
        return None
    if not isinstance(finish, str):
        return _boundary(
            "watsonx finish_reason is not a string (v1's validation "
            "swallows the chunk — pinned divergence)"
        )
    mapped = _FINISH_MAP.get(finish)
    if mapped is None:
        return _boundary(
            f"watsonx finish_reason {finish!r} outside the pinned map "
            "(conservative: v1's map_finish_reason defaults unknowns to "
            "'stop'; re-derive before serving it)"
        )
    return mapped
