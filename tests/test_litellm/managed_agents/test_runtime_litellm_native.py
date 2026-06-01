"""
Unit tests for ``LiteLLMAgentRuntime``.

These tests exercise the manual tool loop without hitting a real LLM.
``litellm.acompletion`` is monkeypatched to a scripted async function
that returns a pre-baked sequence of completions, letting us assert:
  * tool calls flow into ``sandbox.execute_tool``
  * the loop terminates on a no-tool-call assistant message
  * tool results are appended to the message history correctly
"""

import json
from typing import Any, Dict, List

import pytest

from litellm.managed_agents.agent_runtime.base import AgentConfig, SessionState
from litellm.managed_agents.agent_runtime.litellm_native import LiteLLMAgentRuntime
from litellm.managed_agents.events import (
    EVENT_TYPE_ASSISTANT_MESSAGE,
    EVENT_TYPE_RUN_FINISHED,
    EVENT_TYPE_TOOL_RESULT,
    EVENT_TYPE_TOOL_USE,
)
from litellm.managed_agents.sandbox.base import Sandbox, ToolResult


class _RecordingSandbox(Sandbox):
    """Sandbox that records every tool call and returns a configurable result."""

    def __init__(self, response: str = "ok"):
        self.calls: List[Dict[str, Any]] = []
        self.response = response

    async def execute_tool(self, tool_name: str, tool_input: Dict[str, Any]):
        self.calls.append({"name": tool_name, "input": tool_input})
        return ToolResult(output=self.response)


def _completion(content=None, tool_calls=None, finish_reason="stop"):
    """Build an OpenAI-shaped completion dict the runtime knows how to parse."""
    msg: Dict[str, Any] = {"role": "assistant", "content": content or ""}
    if tool_calls is not None:
        msg["tool_calls"] = tool_calls
    return {"choices": [{"index": 0, "finish_reason": finish_reason, "message": msg}]}


def _make_acompletion_stub(scripted: List[Dict[str, Any]]):
    """Return an async stub that yields the next scripted response per call.

    Accepts arbitrary kwargs (matching litellm.acompletion's signature) so
    a stale stub doesn't fail with 'unexpected keyword argument' if the
    runtime starts passing more kwargs in the future.
    """
    calls = {"n": 0, "kwargs": []}

    async def fake_acompletion(*args, **kwargs):
        calls["kwargs"].append(kwargs)
        idx = calls["n"]
        calls["n"] += 1
        if idx >= len(scripted):
            raise AssertionError(
                f"acompletion called more times than scripted: {idx + 1} > {len(scripted)}"
            )
        return scripted[idx]

    return fake_acompletion, calls


@pytest.mark.asyncio
async def test_terminates_when_assistant_emits_text_only(monkeypatch):
    fake, _ = _make_acompletion_stub(
        [_completion(content="all done")],
    )
    monkeypatch.setattr("litellm.acompletion", fake)

    runtime = LiteLLMAgentRuntime()
    sandbox = _RecordingSandbox()
    events = []
    async for ev in runtime.run(
        prompt="say done",
        sandbox=sandbox,
        session_state=SessionState(session_id="sess"),
        agent_config=AgentConfig(name="x", model="gpt-4o-mini"),
    ):
        events.append(ev)

    types = [e.type for e in events]
    assert types == [EVENT_TYPE_ASSISTANT_MESSAGE, EVENT_TYPE_RUN_FINISHED]
    assert events[0].data["content"] == "all done"
    # No tool calls should have been routed to the sandbox.
    assert sandbox.calls == []


@pytest.mark.asyncio
async def test_routes_tool_call_to_sandbox(monkeypatch):
    tool_call = {
        "id": "call_123",
        "type": "function",
        "function": {
            "name": "Bash",
            "arguments": json.dumps({"command": "echo hi"}),
        },
    }
    fake, calls = _make_acompletion_stub(
        [
            _completion(tool_calls=[tool_call], finish_reason="tool_calls"),
            _completion(content="all done"),
        ]
    )
    monkeypatch.setattr("litellm.acompletion", fake)

    runtime = LiteLLMAgentRuntime()
    sandbox = _RecordingSandbox(response="hi\n")
    events = []
    async for ev in runtime.run(
        prompt="run echo hi",
        sandbox=sandbox,
        session_state=SessionState(session_id="sess"),
        agent_config=AgentConfig(name="x", model="gpt-4o-mini"),
    ):
        events.append(ev)

    types = [e.type for e in events]
    assert types == [
        EVENT_TYPE_TOOL_USE,
        EVENT_TYPE_TOOL_RESULT,
        EVENT_TYPE_ASSISTANT_MESSAGE,
        EVENT_TYPE_RUN_FINISHED,
    ]
    assert sandbox.calls == [{"name": "Bash", "input": {"command": "echo hi"}}]

    # Second acompletion call should have included the tool result message.
    second_kwargs = calls["kwargs"][1]
    msgs = second_kwargs["messages"]
    assert any(m.get("role") == "tool" and m.get("content") == "hi\n" for m in msgs)


@pytest.mark.asyncio
async def test_default_tools_passed_when_config_empty(monkeypatch):
    fake, calls = _make_acompletion_stub([_completion(content="bye")])
    monkeypatch.setattr("litellm.acompletion", fake)

    runtime = LiteLLMAgentRuntime()
    async for _ in runtime.run(
        prompt="hi",
        sandbox=_RecordingSandbox(),
        session_state=SessionState(session_id="sess"),
        agent_config=AgentConfig(name="x", model="gpt-4o-mini"),
    ):
        pass

    first_kwargs = calls["kwargs"][0]
    tool_names = {t["function"]["name"] for t in first_kwargs["tools"]}
    assert {"Bash", "Read", "Write", "Edit", "ls"} <= tool_names


@pytest.mark.asyncio
async def test_max_turns_terminates_with_max_turns_reason(monkeypatch):
    """When the LLM keeps asking for tool calls forever, we still stop cleanly."""
    forever_tool_call = {
        "id": "call_forever",
        "type": "function",
        "function": {"name": "Bash", "arguments": "{}"},
    }
    # Always return the same tool call so the loop runs to max_turns.
    scripted = [
        _completion(tool_calls=[forever_tool_call], finish_reason="tool_calls")
        for _ in range(3)
    ]
    fake, _ = _make_acompletion_stub(scripted)
    monkeypatch.setattr("litellm.acompletion", fake)

    runtime = LiteLLMAgentRuntime(max_turns=3)
    types = []
    async for ev in runtime.run(
        prompt="loop",
        sandbox=_RecordingSandbox(),
        session_state=SessionState(session_id="sess"),
        agent_config=AgentConfig(name="x", model="gpt-4o-mini"),
    ):
        types.append((ev.type, ev.data.get("stop_reason")))

    assert types[-1] == (EVENT_TYPE_RUN_FINISHED, "max_turns")
