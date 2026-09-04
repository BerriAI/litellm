"""
Mid-task stall detection for the Complexity Router.

Looks at the assistant's own recent tool calls -- visible on every request an agentic
client resends, since each turn carries the whole conversation so far -- for a tight loop
of identical calls or repeated tool errors. No LLM call, no state: the same fixed-size
window is rescanned on every classified turn, so a stall reads the same way whether it
started one turn ago or ten, and stops reading as a stall the moment the recent calls
change.

Assistant tool calls appear in two shapes depending on the API surface, and this module
reads both without translating one into the other:
- Anthropic Messages: assistant `content` blocks of type "tool_use" (id, name, input),
  answered by a later user-turn `content` block of type "tool_result" (tool_use_id,
  is_error).
- Chat completions: assistant `tool_calls` entries (id, function.name, function.arguments
  as a JSON string), answered by a later `role: "tool"` message. Chat completions has no
  standard error flag on that message, so those calls are judged on repetition alone.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from itertools import islice
from typing import Final, NamedTuple

_ARGUMENTS_PARSE_FAILED: Final = object()


class _ToolCallEvent(NamedTuple):
    signature: tuple[str, str]
    is_error: bool | None
    """None when the surface carries no structured error signal for this call. Never
    treated as an error: a call this module cannot judge must not count toward the tally."""


def _json_arguments(raw: str) -> object:
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return _ARGUMENTS_PARSE_FAILED


def _tool_call_signature(name: str, raw_arguments: object) -> tuple[str, str]:
    """A (name, canonical-arguments) pair that compares equal across both surfaces'
    argument shapes: a dict (Anthropic `input`) and a JSON-encoded string (chat
    completions `function.arguments`) representing the same call must match."""
    parsed: Final = _json_arguments(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
    arguments: Final = raw_arguments if parsed is _ARGUMENTS_PARSE_FAILED else parsed
    try:
        return name, json.dumps(arguments, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return name, str(arguments)


def _iter_tool_result_error_pairs(messages: Sequence[Mapping[str, object]]) -> Iterator[tuple[str, bool]]:
    """(call id, whether that call's result was an error), read only where the surface
    reports one: an Anthropic Messages `tool_result` content block's `is_error`."""
    for msg in messages:
        content = msg.get("content")
        if msg.get("role") != "user" or not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, Mapping) and part.get("type") == "tool_result":
                call_id = part.get("tool_use_id")
                if isinstance(call_id, str):
                    yield call_id, bool(part.get("is_error", False))


def _iter_tool_call_events_newest_first(messages: Sequence[Mapping[str, object]]) -> Iterator[_ToolCallEvent]:
    """Every tool call the assistant made, newest first, paired with its result's error
    status where the surface reports one."""
    error_by_call_id: Final = dict(_iter_tool_result_error_pairs(messages))
    for msg in reversed(messages):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        if isinstance(content, list):
            for part in reversed(content):
                if not (isinstance(part, Mapping) and part.get("type") == "tool_use"):
                    continue
                name = part.get("name")
                if isinstance(name, str):
                    call_id = part.get("id")
                    yield _ToolCallEvent(
                        signature=_tool_call_signature(name, part.get("input")),
                        is_error=error_by_call_id.get(call_id) if isinstance(call_id, str) else None,
                    )
        tool_calls = msg.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for call in reversed(tool_calls):
            function = call.get("function") if isinstance(call, Mapping) else None
            name = function.get("name") if isinstance(function, Mapping) else None
            if isinstance(name, str):
                yield _ToolCallEvent(
                    signature=_tool_call_signature(name, function.get("arguments") if function else None),
                    is_error=None,
                )


def detect_stalled_task(
    messages: Sequence[Mapping[str, object]] | None,
    *,
    window: int,
    repeat_threshold: int,
) -> bool:
    """Whether the assistant's recent tool-call activity looks stuck: repeat_threshold or
    more of the last `window` tool calls share an identical signature, or resolved to an
    error on a surface that reports one.

    Reads the whole message list rather than only the turns since the newest human ask,
    so a follow-up like "try again" does not discard the evidence that came before it.
    """
    if not messages or repeat_threshold <= 0:
        return False
    recent: Final = tuple(islice(_iter_tool_call_events_newest_first(messages), window))
    if len(recent) < repeat_threshold:
        return False
    _, most_common_count = Counter(event.signature for event in recent).most_common(1)[0]
    if most_common_count >= repeat_threshold:
        return True
    error_count: Final = sum(1 for event in recent if event.is_error)
    return error_count >= repeat_threshold
