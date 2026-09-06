"""
Tool-trajectory signals for the Complexity Router.

Reads the assistant's own recent tool calls, which every agentic client resends on each turn,
and reports where the task currently sits: erroring, repeating itself, reading around, or
producing. No LLM call and no stored state, so the same window is rescanned per classified turn
and the reading follows the conversation rather than latching.

Tool calls arrive in two shapes and are read in place rather than translated:
- Anthropic Messages: assistant `tool_use` content blocks, answered by a user-turn
  `tool_result` block carrying `is_error`
- Chat completions: assistant `tool_calls` entries, answered by a `role: "tool"` message,
  which has no standard error flag, so `error_severity` is always 0.0 on that surface and a
  reading taken there rests on the other three signals

This module also owns the tool-call parsing both this and `stall_detector` read from, so the
two agree on what counts as the same call across surfaces.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import islice
from typing import Final, Literal, NamedTuple

ToolIntent = Literal["read", "write", "execute", "unknown"]

_ARGUMENTS_PARSE_FAILED: Final = object()

_TOKEN_PATTERN: Final = re.compile(r"[A-Z]+(?![a-z])|[A-Za-z][a-z0-9]*")

# Matched against name tokens rather than the raw name so "thread_get" reads as a get and not
# as a read. Ordered: a tool that both edits and runs is producing, and one that both reads and
# writes has already stopped exploring. Deliberately generic verbs, since a vendor's tool names
# are theirs to change; anything unrecognized stays "unknown" and counts toward no signal.
_INTENT_VERBS: Final[tuple[tuple[ToolIntent, frozenset[str]], ...]] = (
    (
        "write",
        frozenset(
            {
                "write",
                "edit",
                "create",
                "update",
                "patch",
                "apply",
                "insert",
                "replace",
                "save",
                "delete",
                "remove",
                "rename",
                "move",
                "mkdir",
                "commit",
            }
        ),
    ),
    ("execute", frozenset({"bash", "shell", "run", "exec", "execute", "command", "terminal"})),
    (
        "read",
        frozenset(
            {
                "read",
                "get",
                "list",
                "ls",
                "search",
                "grep",
                "glob",
                "find",
                "view",
                "cat",
                "fetch",
                "query",
                "lookup",
                "describe",
                "inspect",
            }
        ),
    ),
)


class ToolCallEvent(NamedTuple):
    signature: tuple[str, str]
    is_error: bool | None
    """None where the surface reports no error status, and never counted as an error."""


@dataclass(frozen=True, slots=True)
class TrajectorySignals:
    """Where the task sits, as four independent fractions over the same window of tool calls.

    Every field is in [0, 1] and none of them quotes the prompt. `observed_calls` is the
    denominator they share, and is 0 exactly when the turn carries no tool-call history, which
    is the "no evidence" case a caller must not read as "nothing happening".
    """

    error_severity: float
    spinning: float
    exploring: float
    production_intensity: float
    observed_calls: int


NO_TRAJECTORY: Final = TrajectorySignals(
    error_severity=0.0,
    spinning=0.0,
    exploring=0.0,
    production_intensity=0.0,
    observed_calls=0,
)


def _json_arguments(raw: str) -> object:
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return _ARGUMENTS_PARSE_FAILED


def tool_call_signature(name: str, raw_arguments: object) -> tuple[str, str]:
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


def iter_tool_call_events_newest_first(messages: Sequence[Mapping[str, object]]) -> Iterator[ToolCallEvent]:
    """Newest tool call first, each carrying the error status of the result that answered it.

    A result always sits after the call it answers, so walking backward reaches it first and the
    error map fills in as the walk goes. Callers bound this with the window they want, and a long
    conversation then costs only the messages that window actually reaches, rather than a full
    scan for results the walk will never ask about.
    """
    error_by_call_id: Final[dict[str, bool]] = {}  # mutable-ok: fills in as the walk reaches results; never escapes
    for msg in reversed(messages):
        if msg.get("role") != "assistant":
            error_by_call_id.update(_iter_tool_result_error_pairs((msg,)))
            continue
        content = msg.get("content")
        if isinstance(content, list):
            for part in reversed(content):
                if not (isinstance(part, Mapping) and part.get("type") == "tool_use"):
                    continue
                name = part.get("name")
                if isinstance(name, str):
                    call_id = part.get("id")
                    yield ToolCallEvent(
                        signature=tool_call_signature(name, part.get("input")),
                        is_error=error_by_call_id.get(call_id) if isinstance(call_id, str) else None,
                    )
        tool_calls = msg.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for call in reversed(tool_calls):
            function = call.get("function") if isinstance(call, Mapping) else None
            name = function.get("name") if isinstance(function, Mapping) else None
            if isinstance(name, str):
                yield ToolCallEvent(
                    signature=tool_call_signature(name, function.get("arguments") if function else None),
                    is_error=None,
                )


def resolve_tool_intent(name: str, overrides: Mapping[str, ToolIntent] | None = None) -> ToolIntent:
    """What a tool call does, by name. An operator's override wins over the verbs.

    Overrides are matched case-insensitively, so an operator writing them the way their tool
    prints them does not have to know how the client capitalizes it on the wire.
    """
    lowered: Final = name.lower()
    override: Final = (
        next((intent for key, intent in overrides.items() if key.lower() == lowered), None) if overrides else None
    )
    if override is not None:
        return override
    tokens: Final[frozenset[str]] = (
        frozenset(  # mutable-ok: wrapped in frozenset immediately; no mutable state escapes this function
            str(token).lower() for token in _TOKEN_PATTERN.findall(name)
        )
    )
    for intent, verbs in _INTENT_VERBS:
        if tokens & verbs:
            return intent
    return "unknown"


def compute_trajectory_signals(
    messages: Sequence[Mapping[str, object]] | None,
    *,
    window: int,
    tool_intents: Mapping[str, ToolIntent] | None = None,
) -> TrajectorySignals:
    """Read the last `window` tool calls and report the four signals over them.

    Fractions are unweighted: `window` is the only recency control, so the reading stays a
    plain count a reader can reproduce from the same messages rather than a curve with a
    constant in it that nothing has yet calibrated.
    """
    if not messages or window <= 0:
        return NO_TRAJECTORY
    recent: Final = tuple(islice(iter_tool_call_events_newest_first(messages), window))
    if not recent:
        return NO_TRAJECTORY
    total: Final = len(recent)
    intents: Final = tuple(resolve_tool_intent(event.signature[0], tool_intents) for event in recent)
    return TrajectorySignals(
        error_severity=sum(1 for event in recent if event.is_error) / total,
        spinning=1.0 - len({event.signature for event in recent}) / total,
        exploring=intents.count("read") / total,
        production_intensity=intents.count("write") / total,
        observed_calls=total,
    )
