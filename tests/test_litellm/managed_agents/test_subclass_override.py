"""
Tests for subclassing ``AgentRuntime`` to add ``before_tool_call`` /
``after_tool_call`` hooks.

This is the primary extension point for callers who need to audit,
redact, or rewrite tool calls without forking the entire runtime.
"""

import json
from typing import Any, Dict, List

import pytest

from litellm.managed_agents.agent_runtime.base import AgentConfig, SessionState
from litellm.managed_agents.agent_runtime.litellm_native import LiteLLMAgentRuntime
from litellm.managed_agents.events import (
    EVENT_TYPE_RUN_FINISHED,
    EVENT_TYPE_TOOL_RESULT,
)
from litellm.managed_agents.sandbox.base import Sandbox, ToolResult


class _PassthroughSandbox(Sandbox):
    """Sandbox that echoes the tool name + input back so we can see what hook saw."""

    def __init__(self):
        self.invocations: List[Dict[str, Any]] = []

    async def execute_tool(self, tool_name: str, tool_input: Dict[str, Any]):
        self.invocations.append({"name": tool_name, "input": tool_input})
        return ToolResult(output=f"executed {tool_name} with {tool_input!r}")


class _AuditingRuntime(LiteLLMAgentRuntime):
    """Subclass that records every hook fire and rewrites tool input."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.before_calls: List[Dict[str, Any]] = []
        self.after_calls: List[Dict[str, Any]] = []

    async def before_tool_call(self, tool_name, tool_input):
        self.before_calls.append({"name": tool_name, "input": dict(tool_input)})
        # Demonstrate rewriting: append " (audited)" to any command.
        if tool_name == "Bash" and "command" in tool_input:
            return {**tool_input, "command": tool_input["command"] + " (audited)"}
        return tool_input

    async def after_tool_call(self, tool_name, tool_input, result):
        self.after_calls.append(
            {
                "name": tool_name,
                "input": dict(tool_input),
                "is_error": result.is_error,
            }
        )
        return result


def _completion_with_call(call_id, name, args):
    return {
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps(args),
                            },
                        }
                    ],
                },
            }
        ]
    }


def _completion_text(text):
    return {
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": text},
            }
        ]
    }


@pytest.mark.asyncio
async def test_before_and_after_hooks_fire_with_rewrite(monkeypatch):
    scripted = [
        _completion_with_call("call_1", "Bash", {"command": "ls"}),
        _completion_text("done"),
    ]
    call_state = {"i": 0}

    async def fake_acompletion(**kwargs):
        i = call_state["i"]
        call_state["i"] += 1
        return scripted[i]

    monkeypatch.setattr("litellm.acompletion", fake_acompletion)

    runtime = _AuditingRuntime()
    sandbox = _PassthroughSandbox()

    events = []
    async for ev in runtime.run(
        prompt="run ls",
        sandbox=sandbox,
        session_state=SessionState(session_id="sess"),
        agent_config=AgentConfig(name="x", model="gpt-4o-mini"),
    ):
        events.append(ev)

    # both hooks fired exactly once (one tool call)
    assert len(runtime.before_calls) == 1
    assert runtime.before_calls[0]["name"] == "Bash"
    assert runtime.before_calls[0]["input"] == {"command": "ls"}

    assert len(runtime.after_calls) == 1
    assert runtime.after_calls[0]["name"] == "Bash"

    # rewrite is honoured: sandbox saw the augmented command, not the original
    assert sandbox.invocations == [
        {"name": "Bash", "input": {"command": "ls (audited)"}}
    ]

    # tool_result event in the stream reflects what the sandbox returned
    tool_results = [e for e in events if e.type == EVENT_TYPE_TOOL_RESULT]
    assert len(tool_results) == 1
    assert "ls (audited)" in tool_results[0].data["output"]

    # And we hit the terminal event at the end.
    assert events[-1].type == EVENT_TYPE_RUN_FINISHED


class _BeforeRaisesRuntime(LiteLLMAgentRuntime):
    async def before_tool_call(self, tool_name, tool_input):
        raise ValueError("nope")


@pytest.mark.asyncio
async def test_before_hook_raises_surfaces_as_tool_result_error(monkeypatch):
    scripted = [
        _completion_with_call("call_1", "Bash", {"command": "ls"}),
        _completion_text("ok"),
    ]
    call_state = {"i": 0}

    async def fake_acompletion(**kwargs):
        i = call_state["i"]
        call_state["i"] += 1
        return scripted[i]

    monkeypatch.setattr("litellm.acompletion", fake_acompletion)

    runtime = _BeforeRaisesRuntime()
    sandbox = _PassthroughSandbox()
    events = []
    async for ev in runtime.run(
        prompt="run ls",
        sandbox=sandbox,
        session_state=SessionState(session_id="sess"),
        agent_config=AgentConfig(name="x", model="gpt-4o-mini"),
    ):
        events.append(ev)

    # Sandbox was never reached — the hook short-circuited.
    assert sandbox.invocations == []
    tool_results = [e for e in events if e.type == EVENT_TYPE_TOOL_RESULT]
    assert len(tool_results) == 1
    assert tool_results[0].data["is_error"] is True
    assert "before_tool_call" in tool_results[0].data["output"]
