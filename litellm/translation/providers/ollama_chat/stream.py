"""Ollama NDJSON stream lines -> IR stream events.

v1 decodes through ``OllamaChatCompletionResponseIterator`` (a
``BaseModelResponseIterator``): each line rides ``_handle_string_chunk`` —
an SSE ``data:`` prefix is stripped if present, ``[DONE]`` ends the stream,
and a line that fails ``json.loads`` (or parses falsy: ``""``, ``{}``,
``null``) is SILENTLY SWALLOWED. v2 errors loudly on those — the watsonx
line-seam PINNED DIVERGENCE (fail-closed on a failure path, named report
row).

``parse_event`` normalizes one wire chunk into the payload the
``ollama_chat`` chunk dialect folds (the fold owns the think-tag state
machine because it is STATEFUL — ``StreamState.started_reasoning``/
``finished_reasoning``):

- missing ``message``/``done`` keys -> loud (v1 raises OllamaError 400);
  null/non-object ``message`` -> loud (v1 AttributeErrors);
- tool_calls entries -> the validated delta shape (``index: 0``, ``type:
  "function"``, dict arguments re-dumped with v1's ``", "``/``": "``
  separators). The wire ``id`` is KEPT only while arguments are incomplete;
  COMPLETE arguments (a dict, or a string json.loads accepts, length > 0)
  get a fresh uuid4 in v1 — CLOBBERING any wire-carried id (probed). The
  mint is envelope nondeterminism: the payload carries the ``""`` sentinel
  and the seam (or the gate adapter) mints the bare uuid;
- ``finish`` rides the RAW ``done_reason or "stop"`` only when ``done is
  True`` (v1's identity check: a truthy non-True ``done`` is a mid chunk),
  overridden to ``"tool_calls"`` when the chunk carries a tool_calls key
  (v1: ``is not None`` — an EMPTY list still overrides); the live
  ``map_finish_reason`` runs inside StreamingChoices on BOTH sides, so the
  raw value needs no in-package map here (the watsonx rule);
- usage: v1's iterator stamps eval counts on EVERY chunk but the WRAPPER
  strips them from every emitted chunk (probed — the dossier's OL-R2
  "per-chunk usage" claim holds only below the wrapper). The payload
  carries usage ONLY for a done chunk that names at least one count; the
  fold emits it as the trailing ``choices: []`` tail (the family seam
  contract — v1's include_usage synthesis carries exactly the wire values
  then; with NO counts v1 synthesizes token-counter ESTIMATES the seam
  owns, pinned in the gate).
"""

from __future__ import annotations

import json

from expression import Error, Ok, Result
from expression.collections import Block

from ...errors import BoundaryError, TranslationError
from ...ir import JsonBlob, PlainJson, StreamEvent

_EventResult = Result[StreamEvent | None, TranslationError]

MINT_TOOL_ID = ""
"""Sentinel for "v1 mints a bare uuid4 here" (the StreamToolCall empty-id
convention); the seam/gate adapter replaces it before chunk construction."""


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
                "empty ollama stream line: v1's base iterator silently "
                "swallows it — deliberate fail-closed divergence on a "
                "failure path (named report row)"
            )
        )
    try:
        event: PlainJson = json.loads(stripped)
    except ValueError:
        return Error(
            _boundary(
                f"non-JSON ollama stream line {stripped[:80]!r}: v1's base "
                "iterator silently swallows it — deliberate fail-closed "
                "divergence on a failure path (named report row)"
            )
        )
    if not event:
        return Error(
            _boundary(
                "falsy-JSON ollama stream line (empty object/null/zero): "
                "v1's base iterator silently swallows it — deliberate "
                "fail-closed divergence on a failure path (named report row)"
            )
        )
    return parse_event(event)


def parse_event(event: PlainJson) -> _EventResult:
    if not isinstance(event, dict):
        return Error(
            _boundary("ollama stream chunk is not an object (v1 raises)")
        )
    if "message" not in event or "done" not in event:
        return Error(
            _boundary(
                "ollama stream chunk missing 'message'/'done' (v1 raises "
                "OllamaError 400)"
            )
        )
    message = event.get("message")
    if not isinstance(message, dict):
        return Error(
            _boundary(
                "ollama stream chunk 'message' is not an object (v1 "
                "AttributeErrors)"
            )
        )
    tool_calls = _delta_tool_calls(message.get("tool_calls"))
    if isinstance(tool_calls, TranslationError):
        return Error(tool_calls)
    thinking = message.get("thinking")
    if thinking and not isinstance(thinking, str):
        return Error(
            _boundary(
                "ollama stream chunk thinking is truthy but not a string "
                "(v1's Delta validation raises)"
            )
        )
    content = message.get("content")
    if content and not isinstance(content, str):
        return Error(
            _boundary(
                "ollama stream chunk content is truthy but not a string "
                "(v1's tag scan raises TypeError)"
            )
        )
    done = event.get("done")
    finish = _finish(event, done, message)
    if isinstance(finish, TranslationError):
        return Error(finish)
    payload: dict[str, PlainJson] = {
        "thinking": thinking if isinstance(thinking, str) else None,
        "content": content if isinstance(content, str) else None,
        "tool_calls": tool_calls,
        "finish": finish,
        "usage": _done_usage(event) if done is True else None,
    }
    return Ok(StreamEvent.of_wire_chunk(JsonBlob(value=payload)))


def _finish(
    event: dict[str, PlainJson], done: PlainJson, message: dict[str, PlainJson]
) -> PlainJson | TranslationError:
    if done is not True:
        return None
    if message.get("tool_calls") is not None:
        # v1 overrides on key PRESENCE (`is not None`), empty list included
        return "tool_calls"
    done_reason = event.get("done_reason")
    if isinstance(done_reason, (list, dict)):
        return _boundary(
            "unhashable ollama done_reason (the live map_finish_reason "
            "lookup raises TypeError in v1's StreamingChoices)"
        )
    return done_reason if done_reason else "stop"


def _done_usage(event: dict[str, PlainJson]) -> PlainJson:
    if "prompt_eval_count" not in event and "eval_count" not in event:
        return None
    prompt = event.get("prompt_eval_count", 0)
    completion = event.get("eval_count", 0)
    if not isinstance(prompt, int) or not isinstance(completion, int):
        return None  # non-int counts never reach v1's synthesized chunk cleanly
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }


def _delta_tool_calls(
    tool_calls: PlainJson,
) -> list[PlainJson] | None | TranslationError:
    if tool_calls is None:
        return None
    if not isinstance(tool_calls, list):
        return _boundary(
            "ollama stream tool_calls is not a list (v1's id loop raises)"
        )
    out: list[PlainJson] = []
    for call in tool_calls:
        if not isinstance(call, dict):
            return _boundary(
                "ollama stream tool_call entry is not an object (v1 raises)"
            )
        function = call.get("function")
        if not isinstance(function, dict):
            return _boundary(
                "ollama stream tool_call has no 'function' object (v1's "
                ".get chain AttributeErrors)"
            )
        arguments = function.get("arguments")
        entry = _delta_entry(call, function, arguments)
        if isinstance(entry, TranslationError):
            return entry
        out = [*out, entry]
    return out


def _delta_entry(
    call: dict[str, PlainJson],
    function: dict[str, PlainJson],
    arguments: PlainJson,
) -> PlainJson | TranslationError:
    if arguments is not None and not isinstance(arguments, (str, dict)):
        return _boundary(
            "ollama stream tool_call arguments are neither a string nor an "
            "object (v1's completeness probe / Delta validation raises)"
        )
    complete = _arguments_complete(arguments)
    wire_id = call.get("id")
    identifier: PlainJson
    if complete:
        identifier = MINT_TOOL_ID  # v1 mints uuid4, clobbering any wire id
    else:
        identifier = wire_id if isinstance(wire_id, str) else None
    name = function.get("name")
    if name is not None and not isinstance(name, str):
        return _boundary(
            "ollama stream tool_call name is not a string (v1's Delta "
            "validation raises)"
        )
    arguments_str = (
        json.dumps(arguments) if isinstance(arguments, dict) else arguments
    )
    return {
        "id": identifier,
        "type": "function",
        "index": 0,
        "function": {"name": name, "arguments": arguments_str},
    }


def _arguments_complete(arguments: PlainJson) -> bool:
    """v1's gate: arguments non-None, ``len > 0``, and a dict or a string
    json.loads accepts (``_is_function_call_complete``)."""
    if arguments is None or not isinstance(arguments, (str, dict)):
        return False
    if len(arguments) == 0:
        return False
    if isinstance(arguments, dict):
        return True
    try:
        json.loads(arguments)
    except ValueError:
        return False
    return True
