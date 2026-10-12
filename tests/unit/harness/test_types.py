"""Tests for litellm/harness/types.py."""

from __future__ import annotations

import asyncio

import pytest

from litellm.harness.errors import StateIncompatible
from litellm.harness.types import (
    Approval,
    Done,
    Harness,
    Result,
    State,
    Usage,
    require_harness,
)


def test_harness_is_plain_enum():
    assert Harness.CODEX.value == "codex"
    assert Harness.TOOL_LOOP.value == "tool_loop"
    assert not isinstance(Harness.CODEX, str)


@pytest.mark.parametrize(
    "given,hint",
    [
        ("codex", "Harness.CODEX"),
        ("claude-code", "Harness.CLAUDE_CODE"),
        ("OPENCODE", "Harness.OPENCODE"),
    ],
)
def test_require_harness_hint(given, hint):
    with pytest.raises(TypeError, match=hint):
        require_harness(given)


def test_require_harness_no_hint_for_unknown():
    with pytest.raises(TypeError) as info:
        require_harness(42)
    assert "Did you mean" not in str(info.value)
    assert require_harness(Harness.DEEPAGENTS) is Harness.DEEPAGENTS


def test_usage_total_tokens():
    assert Usage(input_tokens=3, output_tokens=4, calls=1).total_tokens == 7


def test_done_exposes_result_fields():
    result = Result(
        text="t",
        output=None,
        files=[],
        events=[],
        usage=Usage(1, 2, 1),
        cost=0.5,
        stop_reason="done",
        session_id="s",
    )
    done = Done(result)
    assert done.usage.total_tokens == 3
    assert done.cost == 0.5
    assert done.stop_reason == "done"


def test_state_round_trip_and_errors():
    state = State(Harness.CODEX, "thread-1", "/work", model="gpt")
    assert State.loads(state.dumps()) == state
    with pytest.raises(StateIncompatible):
        State.loads(b"not json")
    with pytest.raises(StateIncompatible):
        State.loads(b'{"harness": "nope", "version": 1, "workdir": "/"}')
    with pytest.raises(StateIncompatible, match="version"):
        State.loads(b'{"harness": "codex", "version": 99, "workdir": "/"}')


async def test_approval_allow_deny_once():
    approval = Approval(tool="bash", input={})
    assert not approval.answered
    approval.allow()
    approval.deny("late")
    assert await approval.wait() == (True, "")
    assert approval.answered


async def test_approval_resolved_from_other_thread():
    approval = Approval(tool="bash", input={})
    await asyncio.to_thread(approval.deny, "nope")
    assert await approval.wait() == (False, "nope")
