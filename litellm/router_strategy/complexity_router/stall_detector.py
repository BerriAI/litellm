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

Parsing is shared with `trajectory_signals`, so the two agree on what counts as the same call.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import islice
from typing import Final

from litellm.router_strategy.complexity_router.trajectory_signals import iter_tool_call_events_newest_first


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
    recent: Final = tuple(islice(iter_tool_call_events_newest_first(messages), window))
    if len(recent) < repeat_threshold:
        return False
    newest: Final = recent[0]
    repeats: Final = sum(1 for event in recent if event.signature == newest.signature)
    if repeats >= repeat_threshold:
        return True
    if not newest.is_error:
        return False
    return sum(1 for event in recent if event.is_error) >= repeat_threshold
