"""
Mid-task stall detection for the Complexity Router.

Reads the assistant's own recent tool calls, which every agentic client resends on each
turn, and reports whether the task currently looks stuck. No LLM call and no stored state:
the same window is rescanned per classified turn, so the verdict follows the conversation
rather than latching.

Tool calls arrive in two shapes and are read in place rather than translated:
- Anthropic Messages: assistant `tool_use` content blocks, answered by a user-turn
  `tool_result` block carrying `is_error`
- Chat completions: assistant `tool_calls` entries, answered by a `role: "tool"` message,
  which has no standard error flag, so those calls are judged on repetition alone
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from itertools import islice
from typing import Final, NamedTuple

_ARGUMENTS_PARSE_FAILED: Final = object()


class _ToolCallEvent(NamedTuple):
    signature: tuple[str, str]
    is_error: bool | None
    """None where the surface reports no error status, and never counted as an error."""


def _json_arguments(raw: str) -> object:
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return _ARGUMENTS_PARSE_FAILED


def _tool_call_signature(name: str, raw_arguments: object) -> tuple[str, str]:
    """Canonicalized so the same call compares equal across both surfaces, which carry
    arguments as a dict and as a JSON string respectively."""
    parsed: Final = _json_arguments(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
    arguments: Final = raw_arguments if parsed is _ARGUMENTS_PARSE_FAILED else parsed
    try:
        return name, json.dumps(arguments, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return name, str(arguments)


def _iter_tool_result_error_pairs(messages: Sequence[Mapping[str, object]]) -> Iterator[tuple[str, bool]]:
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
    """Whether the newest tool call is still part of a stuck pattern: it repeats, or it
    errored, at least repeat_threshold times across the last `window` calls.

    Both tests are anchored on the newest call rather than counting whichever pattern is
    most common in the window. A task that tried the same thing three times and then moved
    on has those three calls in the window for a while yet, and counting them alone would
    escalate a request that already recovered. Anchoring also leaves room between the
    matches, so a retry loop broken up by an unrelated lookup still reads as stuck.
    """
    if not messages or repeat_threshold <= 0:
        return False
    recent: Final = tuple(islice(_iter_tool_call_events_newest_first(messages), window))
    if len(recent) < repeat_threshold:
        return False
    newest: Final = recent[0]
    repeats: Final = sum(1 for event in recent if event.signature == newest.signature)
    if repeats >= repeat_threshold:
        return True
    if not newest.is_error:
        return False
    return sum(1 for event in recent if event.is_error) >= repeat_threshold
