"""Cohere v2 chat stream lines -> IR stream events.

v1 decodes through ``CohereV2ModelResponseIterator`` (llms/cohere/
common_utils.py): each wire LINE is ``json.loads``-ed directly — there is no
``data:`` strip for str lines and no ``[DONE]`` sentinel, so ``parse_line``
here mirrors that bare-JSON line seam (a non-JSON line is a loud error where
v1 raises RuntimeError out of the iterator). ``chunk_parser`` yields
``GenericStreamingChunk`` dicts that ride ``CustomStreamWrapper``'s generic
arm; each cohere event maps to one ``wire_chunk`` carrying the generic
payload the shared ``generic`` chunk dialect folds:

- ``type == "content-delta"`` -> text (v1's dict/str content fork; a non-str
  ``text`` value is loud — v1's Delta validation raises on it);
- ``type == "tool-call-delta"`` -> one tool_call entry, id/name/arguments
  defaulting to ``""`` (v1's ``.get(..., "")``); ``tool-call-start`` events
  match NO v1 arm and are swallowed;
- ``event == "tool-plan-delta"`` / ``event == "citation-start"`` ->
  provider fields (v1 setattrs them top-level AND the delta carries them);
- ``event == "message-end"`` -> finish (mapped exactly like v1's
  ``map_finish_reason``: COMPLETE -> stop, MAX_TOKENS -> length,
  ERROR_TOXIC -> content_filter, anything else -> stop, probed in-process)
  + usage from ``usage.tokens``;
- any other event -> swallowed (v1 yields an empty GenericStreamingChunk
  the wrapper drops).

NOTE the v1 quirk this mirrors verbatim: chunk_parser reads the
``message-end``/tool-plan/citation shapes off the ``event`` key, while the
Cohere v2 wire sends ``type`` for every event — so on REAL traffic those
arms never fire, the message-end is swallowed, and v1's WRAPPER synthesizes
the trailing finish chunk at StopIteration (seam scope; the differential
replays pin both regimes). Wrong-TYPED intermediates (non-dict ``delta``,
non-dict ``tool_calls[0]``, null ``usage.tokens``…) are loud errors —
v1's chunk_parser raises ValueError out of the iterator on each.
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
        "COMPLETE": "stop",
        "MAX_TOKENS": "length",
        "ERROR_TOXIC": "content_filter",
    }
)


def _boundary(reason: str) -> TranslationError:
    return TranslationError.of_boundary(BoundaryError.of(Block.of_seq([reason])))


def parse_line(line: str) -> _EventResult:
    stripped = line.strip()
    if not stripped:
        return Error(
            _boundary(
                "empty cohere stream line: v1's iterator json.loads every "
                "line and raises RuntimeError on it"
            )
        )
    try:
        event: PlainJson = json.loads(stripped)
    except ValueError:
        return Error(
            _boundary(
                f"non-JSON cohere stream line {stripped[:80]!r}: v1's "
                "iterator raises RuntimeError (no data:/[DONE] framing on "
                "this wire)"
            )
        )
    return parse_event(event)


_Fields = tuple[str, PlainJson, PlainJson, PlainJson, PlainJson]
"""(text, tool_call, finish, usage, provider_fields) — one GenericStreamingChunk."""


def parse_event(event: PlainJson) -> _EventResult:
    if not isinstance(event, dict):
        return Error(_boundary("cohere stream event is not an object (v1 raises)"))
    index_reason = _index_reason(event)
    if index_reason is not None:
        return Error(_boundary(index_reason))
    fields = _event_fields(event)
    if isinstance(fields, TranslationError):
        return Error(fields)
    text, tool_call, finish, usage, provider_fields = fields
    if "citations" in event:
        seeded = provider_fields if isinstance(provider_fields, dict) else {}
        provider_fields = {**seeded, "citations": event.get("citations")}
    if not text and tool_call is None and provider_fields is None and finish is None:
        return Ok(None)  # v1 yields an empty GenericStreamingChunk the wrapper drops
    payload: dict[str, PlainJson] = {
        "text": text,
        "tool_call": tool_call,
        "finish": finish,
        "usage": usage,
        "provider_fields": provider_fields,
    }
    return Ok(StreamEvent.of_wire_chunk(JsonBlob(value=payload)))


def _event_fields(event: dict[str, PlainJson]) -> _Fields | TranslationError:
    """Mirror chunk_parser's arm order: the ``type``-keyed content/tool arms
    first, then the ``event``-keyed plan/citation/message-end arms."""
    chunk_type = event.get("type", "")
    if chunk_type == "content-delta":
        text = _content_delta_text(event)
        if isinstance(text, TranslationError):
            return text
        return (text, None, None, None, None)
    if chunk_type == "tool-call-delta":
        tool_call = _tool_call_delta(event)
        if isinstance(tool_call, TranslationError):
            return tool_call
        return ("", tool_call, None, None, None)
    return _keyed_event_fields(event)


def _keyed_event_fields(event: dict[str, PlainJson]) -> _Fields | TranslationError:
    event_type = event.get("event", "")
    if event_type == "tool-plan-delta":
        plan = _message_field(event, "tool_plan")
        if isinstance(plan, TranslationError):
            return plan
        return ("", None, None, None, plan)
    if event_type == "citation-start":
        citation = _citation_fields(event)
        if isinstance(citation, TranslationError):
            return citation
        return ("", None, None, None, citation)
    if event_type == "message-end":
        ended = _message_end(event)
        if isinstance(ended, TranslationError):
            return ended
        finish, usage = ended
        return ("", None, finish, usage, None)
    return ("", None, None, None, None)


def _index_reason(event: dict[str, PlainJson]) -> str | None:
    """v1 runs ``int(chunk.get("index", 0))`` first; a non-coercible value
    raises ValueError out of the iterator (the value itself is never used —
    the emitted choice index is always 0)."""
    if "index" not in event:
        return None
    index = event.get("index")
    if isinstance(index, (bool, int, float)):
        return None
    if isinstance(index, str):
        try:
            int(index)
        except ValueError:
            return f"non-numeric cohere chunk index {index!r} (v1 raises)"
        return None
    return f"non-numeric cohere chunk index {index!r} (v1 raises)"


def _delta_message(
    event: dict[str, PlainJson], *, nested_data: bool
) -> dict[str, PlainJson] | TranslationError:
    """Walk ``[data.]delta.message`` exactly like v1's ``.get(key, {})``
    chains: a MISSING level is ``{}``; a PRESENT non-dict level is an
    AttributeError -> ValueError raise in v1 (loud here)."""
    source: PlainJson = event
    keys = ("data", "delta", "message") if nested_data else ("delta", "message")
    for key in keys:
        if not isinstance(source, dict):
            return _boundary(
                f"cohere stream {key!r} parent is not an object (v1 raises)"
            )
        source = source.get(key, {})
    if not isinstance(source, dict):
        return _boundary("cohere stream delta.message is not an object (v1 raises)")
    return source


def _content_delta_text(event: dict[str, PlainJson]) -> str | TranslationError:
    message = _delta_message(event, nested_data=False)
    if isinstance(message, TranslationError):
        return message
    content = message.get("content", {})
    if isinstance(content, dict):
        if "text" not in content:
            return ""
        text = content.get("text")
        if not isinstance(text, str):
            return _boundary(
                "cohere content-delta text is not a string (v1's Delta "
                "validation raises on it)"
            )
        return text
    if isinstance(content, str):
        return content
    return ""  # v1's fork returns "" for any other content shape


def _tool_call_delta(event: dict[str, PlainJson]) -> PlainJson | TranslationError:
    delta = event.get("delta", {})
    if not isinstance(delta, dict):
        return _boundary("cohere tool-call-delta delta is not an object (v1 raises)")
    tool_calls = delta.get("tool_calls", [])
    if not tool_calls:
        return None  # v1's truthiness gate
    if not isinstance(tool_calls, list):
        return _boundary("cohere delta.tool_calls is not a list (v1 raises)")
    first = tool_calls[0]
    if not isinstance(first, dict):
        return _boundary("cohere tool_calls[0] is not an object (v1 raises)")
    identifier = first.get("id", "")
    name = first.get("name", "")
    arguments = first.get("arguments", "")
    if (
        not isinstance(identifier, str)
        or not isinstance(name, str)
        or not isinstance(arguments, str)
    ):
        return _boundary(
            "cohere tool-call-delta id/name/arguments is not a string "
            "(v1's Delta tool-call validation raises)"
        )
    return {
        "id": identifier,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def _message_field(
    event: dict[str, PlainJson], key: str
) -> PlainJson | TranslationError:
    message = _delta_message(event, nested_data=True)
    if isinstance(message, TranslationError):
        return message
    value = message.get(key, "")
    if not value:
        return None  # v1's truthiness gate
    return {key: value}


def _citation_fields(event: dict[str, PlainJson]) -> PlainJson | TranslationError:
    message = _delta_message(event, nested_data=True)
    if isinstance(message, TranslationError):
        return message
    citations = message.get("citations", {})
    if not citations:
        return None  # v1's truthiness gate
    if not isinstance(citations, dict):
        return _boundary("cohere citation-start citations is not an object (v1 raises)")
    return {
        "citations": [
            {
                "start": citations.get("start", 0),
                "end": citations.get("end", 0),
                "text": citations.get("text", ""),
                "sources": citations.get("sources", []),
                "type": citations.get("type", "TEXT_CONTENT"),
            }
        ]
    }


def _message_end(
    event: dict[str, PlainJson],
) -> tuple[PlainJson, PlainJson] | TranslationError:
    data = event.get("data", {})
    if not isinstance(data, dict):
        return _boundary("cohere message-end data is not an object (v1 raises)")
    delta = data.get("delta", {})
    if not isinstance(delta, dict):
        return _boundary("cohere message-end delta is not an object (v1 raises)")
    raw_finish = delta.get("finish_reason", "stop")
    # v1's map_finish_reason defaults EVERY unmapped value — non-strings,
    # null, "" included — to "stop" (probed in-process).
    finish = (
        _FINISH_MAP.get(raw_finish, "stop") if isinstance(raw_finish, str) else "stop"
    )
    usage_data = delta.get("usage", {})
    if not usage_data:
        return finish, None  # v1's truthiness gate
    if not isinstance(usage_data, dict):
        return _boundary("cohere message-end usage is not an object (v1 raises)")
    tokens = usage_data.get("tokens", {})
    if not isinstance(tokens, dict):
        return _boundary("cohere message-end usage.tokens is not an object (v1 raises)")
    prompt = tokens.get("input_tokens", 0)
    completion = tokens.get("output_tokens", 0)
    if not isinstance(prompt, (bool, int, float)) or not isinstance(
        completion, (bool, int, float)
    ):
        return _boundary(
            "cohere message-end token counts are not numeric (v1 raises "
            "building ChatCompletionUsageBlock)"
        )
    usage: PlainJson = {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }
    return finish, usage
