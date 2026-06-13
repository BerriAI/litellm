"""databricks SSE stream chunks -> IR stream events.

v1 decodes through ``DatabricksChatResponseIterator`` (a
``BaseModelResponseIterator``): each line rides ``_handle_string_chunk`` (an
SSE ``data:`` prefix stripped if present, ``[DONE]`` ends the stream, a line
that fails ``json.loads`` is SILENTLY SWALLOWED). v2 errors loudly on the
swallow cases — the watsonx/ollama line-seam PINNED DIVERGENCE (fail-closed on
a failure path, named report row).

``parse_event`` normalizes one wire chunk into the payload the ``databricks``
chunk dialect folds into ONE ``ModelResponseStream`` body (v1 yields one out
chunk per wire chunk, NO split). Probed chunk-by-chunk at HEAD:

- the wire chunk ``usage`` is DROPPED ENTIRELY (DB-R5: the iterator never
  passes usage to ``ModelResponseStream``); the seam must NOT attach a parsed
  usage tail;
- ``{}`` tool arguments are rewritten to ``""`` (non-json_mode chunks);
- a content LIST is flattened to a content str (``extract_content_str``),
  reasoning/summary blocks become reasoning_content + thinking_blocks, and a
  ``citations`` list on the FIRST content item becomes
  ``provider_specific_fields.citation = citations[0]``;
- json_mode (the STATEFUL ``_last_function_name`` machine) converts a
  ``json_tool_call`` tool delta to content with the json.loads+dumps byte
  REFORMAT (``{"a":1}`` -> ``{"a": 1}``, DB-R8) and ``{}`` -> ``""``; the
  ``_last_function_name`` rides ``StreamState.last_function_name``. json_mode
  is a REQUEST-side fallback (v2 never sends a json_mode request), so this arm
  is dormant from v2's own flow but pinned by the stream gate's REAL-iterator
  replay;
- missing ``id``/``created``/``model``/``choices`` -> loud (v1 raises
  DatabricksException 400).
"""

from __future__ import annotations

import json

from expression import Error, Ok, Result
from expression.collections import Block

from ...errors import BoundaryError, TranslationError
from ...ir import JsonBlob, PlainJson, StreamEvent

_EventResult = Result[StreamEvent | None, TranslationError]

RESPONSE_FORMAT_TOOL_NAME = "json_tool_call"
"""The ONE in-package mirror of ``litellm.constants.RESPONSE_FORMAT_TOOL_NAME``;
the stream gate pins it against the constant at HEAD."""


def _boundary(reason: str) -> TranslationError:
    return TranslationError.of_boundary(BoundaryError.of(Block.of_seq([reason])))


def parse_line(line: str) -> _EventResult:
    stripped = line.strip()
    if stripped.startswith("data:"):
        stripped = stripped[len("data:") :].strip()
    if "[DONE]" in line:
        return Ok(StreamEvent.of_stop())
    if not stripped:
        return Error(
            _boundary(
                "empty databricks stream line: v1's base iterator silently "
                "swallows it — deliberate fail-closed divergence on a failure "
                "path (named report row)"
            )
        )
    try:
        event: PlainJson = json.loads(stripped)
    except ValueError:
        return Error(
            _boundary(
                f"non-JSON databricks stream line {stripped[:80]!r}: v1's base "
                "iterator silently swallows it — deliberate fail-closed "
                "divergence on a failure path (named report row)"
            )
        )
    return parse_event(event)


def parse_event(event: PlainJson) -> _EventResult:
    if not isinstance(event, dict):
        return Error(_boundary("databricks stream chunk is not an object (v1 raises)"))
    for key in ("id", "created", "model", "choices"):
        if key not in event:
            return Error(
                _boundary(
                    f"databricks stream chunk missing {key!r} (v1 raises "
                    "DatabricksException 400)"
                )
            )
    choices = event["choices"]
    if not isinstance(choices, list):
        return Error(_boundary("databricks stream choices is not a list (v1 raises)"))
    normalized = _choices(choices)
    if isinstance(normalized, TranslationError):
        return Error(normalized)
    payload: dict[str, PlainJson] = {
        "id": event["id"],
        "created": event["created"],
        "model": event["model"],
        "choices": normalized,
    }
    return Ok(StreamEvent.of_wire_chunk(JsonBlob(value=payload)))


def _choices(choices: list[PlainJson]) -> list[PlainJson] | TranslationError:
    out: list[PlainJson] = []
    for choice in choices:
        normalized = _choice(choice)
        if isinstance(normalized, TranslationError):
            return normalized
        out = [*out, normalized]
    return out


def _choice(choice: PlainJson) -> PlainJson | TranslationError:
    if not isinstance(choice, dict):
        return _boundary("databricks stream choice is not an object (v1 raises)")
    if "delta" not in choice:
        return _boundary(
            "databricks stream choice missing 'delta' (v1's KeyError -> "
            "DatabricksException 400)"
        )
    delta = choice["delta"]
    if not isinstance(delta, dict):
        return _boundary("databricks stream delta is not an object (v1 raises)")
    return {
        "index": choice.get("index", 0),
        "finish_reason": choice.get("finish_reason"),
        "delta": dict(delta),
    }
