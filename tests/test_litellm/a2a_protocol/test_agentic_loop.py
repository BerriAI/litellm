"""Tests for the completion-bridge agentic loop and its completion routing.

Covers the helpers the loop is built on (turning MCP tool results into chat tool
messages, exposing other A2A agents as function tools), the loop driver itself
(MCP tool execution, agent delegation, and the non-empty-answer guarantee), and
the routing in ``_run_completion``.
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from litellm.a2a_protocol.litellm_completion_bridge import handler
from litellm.a2a_protocol.litellm_completion_bridge.agentic_loop import (
    AGENT_TOOL_PREFIX,
    _build_agent_tools,
    _mcp_results_to_tool_messages,
    run_agentic_loop,
)


def _tool_call(call_id: str, name: str, arguments: str = "{}"):
    return SimpleNamespace(
        id=call_id, function=SimpleNamespace(name=name, arguments=arguments)
    )


def _response(content=None, tool_calls=None):
    message = SimpleNamespace(
        content=content,
        tool_calls=tool_calls,
        model_dump=lambda: {"role": "assistant", "content": content},
    )
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_mcp_results_become_tool_role_messages():
    """Regression: ``_execute_tool_calls`` returns ``{tool_call_id, result,
    name}``, which is not a valid chat message. Feeding that back verbatim left
    the model blind to the result, so it re-called the tool until the loop was
    exhausted and produced an empty answer. The result must be reshaped into a
    ``tool`` role message whose ``content`` holds the result text."""
    raw = [{"tool_call_id": "call_1", "result": "sunny, 21C", "name": "get_weather"}]

    messages = _mcp_results_to_tool_messages(raw)

    assert messages == (
        {"role": "tool", "tool_call_id": "call_1", "content": "sunny, 21C"},
    )
    # The original keys must not survive: leaking "result"/"name" or dropping
    # "role" is exactly the bug.
    assert "result" not in messages[0]
    assert "name" not in messages[0]


def test_mcp_results_coerce_non_string_results_and_preserve_order():
    raw = [
        {"tool_call_id": "a", "result": {"city": "Paris", "temp_c": 21}},
        {"tool_call_id": "b", "result": None},
    ]

    messages = _mcp_results_to_tool_messages(raw)

    assert [m["tool_call_id"] for m in messages] == ["a", "b"]
    assert all(m["role"] == "tool" for m in messages)
    assert isinstance(messages[0]["content"], str)
    assert "Paris" in messages[0]["content"]


def test_build_agent_tools_one_function_tool_per_agent():
    tools = _build_agent_tools(
        [
            {
                "agent_id": "translator",
                "name": "Translator",
                "description": "Translates text",
            },
            {"agent_id": "summariser"},
        ]
    )

    assert len(tools) == 2
    first = tools[0]["function"]
    assert first["name"] == f"{AGENT_TOOL_PREFIX}translator"
    assert first["description"] == "Translates text"
    # The model must supply a self-contained message for the sub-agent.
    assert first["parameters"]["required"] == ["message"]
    assert first["parameters"]["properties"]["message"]["type"] == "string"
    # An agent without its own description still gets a usable default.
    assert "summariser" in tools[1]["function"]["description"]


def test_build_agent_tools_skips_agents_without_id():
    tools = _build_agent_tools([{"name": "no id here"}, {"id": "kept"}])

    names = [t["function"]["name"] for t in tools]
    assert names == [f"{AGENT_TOOL_PREFIX}kept"]


@pytest.mark.asyncio
async def test_loop_delegates_to_agent_then_answers():
    """An agent tool call routes through call_agent, the reply is fed back as a
    tool message, and the loop returns the model's final prose."""
    call_agent = AsyncMock(return_value="bonjour")
    with patch("litellm.acompletion", new_callable=AsyncMock) as acompletion:
        acompletion.side_effect = [
            _response(
                tool_calls=[
                    _tool_call(
                        "c1",
                        f"{AGENT_TOOL_PREFIX}translator",
                        json.dumps({"message": "say hi in french"}),
                    )
                ]
            ),
            _response(content="done"),
        ]
        result = await run_agentic_loop(
            model="gemini/gemini-flash-latest",
            messages=[{"role": "user", "content": "hi"}],
            completion_kwargs={},
            callable_agents=[{"agent_id": "translator", "name": "Translator"}],
            call_agent=call_agent,
        )

    assert result.choices[0].message.content == "done"
    call_agent.assert_awaited_once_with("translator", "say hi in french")
    fed_back = acompletion.await_args_list[1].kwargs["messages"]
    assert {"role": "tool", "tool_call_id": "c1", "content": "bonjour"} in fed_back


@pytest.mark.asyncio
async def test_loop_executes_mcp_tool_and_feeds_result_back():
    """Regression at the loop level: an MCP tool result is reshaped to a tool
    message so the model sees it and answers instead of re-calling the tool."""
    handler_path = (
        "litellm.responses.mcp.litellm_proxy_mcp_handler.LiteLLM_Proxy_MCP_Handler"
    )
    with (
        patch("litellm.acompletion", new_callable=AsyncMock) as acompletion,
        patch(
            "litellm.a2a_protocol.litellm_completion_bridge.agentic_loop._load_mcp_tools",
            new_callable=AsyncMock,
        ) as load_mcp,
        patch(
            f"{handler_path}._execute_tool_calls", new_callable=AsyncMock
        ) as execute_tool_calls,
    ):
        acompletion.side_effect = [
            _response(tool_calls=[_tool_call("t1", "get_weather")]),
            _response(content="It is sunny"),
        ]
        load_mcp.return_value = (
            [{"type": "function", "function": {"name": "get_weather"}}],
            {"get_weather": "server-1"},
        )
        execute_tool_calls.return_value = [
            {"tool_call_id": "t1", "result": "sunny", "name": "get_weather"}
        ]
        result = await run_agentic_loop(
            model="gemini/gemini-flash-latest",
            messages=[{"role": "user", "content": "weather?"}],
            completion_kwargs={},
            callable_agents=[],
            call_agent=AsyncMock(),
            enable_mcp_tools=True,
        )

    assert result.choices[0].message.content == "It is sunny"
    fed_back = acompletion.await_args_list[1].kwargs["messages"]
    assert {"role": "tool", "tool_call_id": "t1", "content": "sunny"} in fed_back


@pytest.mark.asyncio
async def test_loop_makes_tool_free_final_call_when_last_turn_has_no_prose():
    """If the loop is exhausted on a tool-calling turn, one tool-free call is made
    so the agent answers instead of returning an empty message."""
    with patch("litellm.acompletion", new_callable=AsyncMock) as acompletion:
        acompletion.side_effect = [
            _response(tool_calls=[_tool_call("a1", f"{AGENT_TOOL_PREFIX}t")]),
            _response(content="final summary"),
        ]
        result = await run_agentic_loop(
            model="m",
            messages=[{"role": "user", "content": "x"}],
            completion_kwargs={},
            callable_agents=[{"agent_id": "t"}],
            call_agent=AsyncMock(return_value="r"),
            max_iterations=1,
        )

    assert result.choices[0].message.content == "final summary"
    assert acompletion.await_count == 2
    # the final call is tool-free
    assert "tools" not in acompletion.await_args_list[1].kwargs


@pytest.mark.asyncio
async def test_loop_surfaces_sub_agent_failure_without_crashing():
    """If a delegated agent call raises, the loop records the error as the tool
    reply and still returns a final answer instead of propagating the exception."""
    call_agent = AsyncMock(side_effect=RuntimeError("boom"))
    with patch("litellm.acompletion", new_callable=AsyncMock) as acompletion:
        acompletion.side_effect = [
            _response(tool_calls=[_tool_call("c1", f"{AGENT_TOOL_PREFIX}flaky")]),
            _response(content="handled"),
        ]
        result = await run_agentic_loop(
            model="m",
            messages=[{"role": "user", "content": "x"}],
            completion_kwargs={},
            callable_agents=[{"agent_id": "flaky"}],
            call_agent=call_agent,
        )

    assert result.choices[0].message.content == "handled"
    fed_back = acompletion.await_args_list[1].kwargs["messages"]
    assert any(
        m.get("role") == "tool"
        and "Error calling agent 'flaky'" in str(m.get("content"))
        for m in fed_back
    )


@pytest.mark.asyncio
async def test_run_completion_routes_to_agentic_loop_when_agent_can_delegate():
    with patch(
        "litellm.a2a_protocol.litellm_completion_bridge.agentic_loop.run_agentic_loop",
        new_callable=AsyncMock,
    ) as loop:
        loop.return_value = "LOOP"
        result = await handler._run_completion(
            {"model": "m", "messages": [{"role": "user", "content": "x"}]},
            {"callable_agents": [{"agent_id": "a"}], "call_agent": AsyncMock()},
        )

    assert result == "LOOP"
    loop.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_completion_routes_to_mcp_when_only_tools_enabled():
    with patch(
        "litellm.responses.mcp.chat_completions_handler.acompletion_with_mcp",
        new_callable=AsyncMock,
    ) as acompletion_with_mcp:
        acompletion_with_mcp.return_value = "MCP"
        result = await handler._run_completion(
            {"model": "m", "messages": []}, {"enable_mcp_tools": True}
        )

    assert result == "MCP"
    acompletion_with_mcp.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_completion_plain_when_no_tools_or_agents():
    with patch("litellm.acompletion", new_callable=AsyncMock) as acompletion:
        acompletion.return_value = "PLAIN"
        result = await handler._run_completion({"model": "m", "messages": []}, {})

    assert result == "PLAIN"
    acompletion.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_completion_strips_agent_tools_before_agentic_loop():
    """Regression: an agent carrying `tools`/`tool_choice` in its litellm_params
    must not reach the loop's completion_kwargs, or they collide with the explicit
    tools= the loop passes (TypeError: got multiple values for 'tools')."""
    with patch(
        "litellm.a2a_protocol.litellm_completion_bridge.agentic_loop.run_agentic_loop",
        new_callable=AsyncMock,
    ) as loop:
        loop.return_value = "LOOP"
        await handler._run_completion(
            {
                "model": "m",
                "messages": [],
                "tools": [{"type": "function", "function": {"name": "x"}}],
                "tool_choice": "auto",
            },
            {"callable_agents": [{"agent_id": "a"}], "call_agent": AsyncMock()},
        )

    completion_kwargs = loop.await_args.kwargs["completion_kwargs"]
    assert "tools" not in completion_kwargs
    assert "tool_choice" not in completion_kwargs


@pytest.mark.asyncio
async def test_run_completion_strips_agent_tools_on_mcp_path():
    """Same collision guard on the MCP path: acompletion_with_mcp passes tools=
    explicitly, so a tools key from litellm_params must be dropped (else the call
    raises before the request is even made)."""
    with patch(
        "litellm.responses.mcp.chat_completions_handler.acompletion_with_mcp",
        new_callable=AsyncMock,
    ) as acompletion_with_mcp:
        acompletion_with_mcp.return_value = "MCP"
        result = await handler._run_completion(
            {
                "model": "m",
                "messages": [],
                "tools": [{"type": "function", "function": {"name": "x"}}],
            },
            {"enable_mcp_tools": True},
        )

    assert result == "MCP"
