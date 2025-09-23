import json
import pytest

import sys, types


@pytest.mark.smoke
def test_mini_agent_echo_tool_roundtrip(monkeypatch):
    # Stub optional heavy deps + MCP so we can import litellm without extras
    monkeypatch.setitem(sys.modules, "fastuuid", types.SimpleNamespace(uuid4=lambda: "0" * 32))
    mcp_stub = types.SimpleNamespace(ClientSession=object)
    mcp_types_stub = types.SimpleNamespace(
        CallToolRequestParams=type("CallToolRequestParams", (), {}),
        CallToolResult=type("CallToolResult", (), {}),
        Tool=type("Tool", (), {}),
    )
    monkeypatch.setitem(sys.modules, "mcp", mcp_stub)
    monkeypatch.setitem(sys.modules, "mcp.types", mcp_types_stub)
    # Import after stubbing
    from litellm.experimental_mcp_client.mini_agent.litellm_mcp_mini_agent import (
        AgentConfig,
        EchoMCP,
        arun_mcp_mini_agent,
    )
    import litellm.experimental_mcp_client.mini_agent.litellm_mcp_mini_agent as mini_mod


    calls = {"n": 0}

    async def fake_arouter_call(*, model, messages, stream=False, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            # Ask to call echo("hi") first
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Plan: call echo('hi')",
                            "tool_calls": [
                                {
                                    "id": "tc_1",
                                    "type": "function",
                                    "function": {
                                        "name": "echo",
                                        "arguments": json.dumps({"text": "hi"}),
                                    },
                                }
                            ],
                        }
                    }
                ]
            }
        else:
            # After tool returns, produce final content
            return {
                "choices": [
                    {"message": {"role": "assistant", "content": "Done: hi"}}
                ]
            }

    monkeypatch.setattr(mini_mod, "arouter_call", fake_arouter_call)

    cfg = AgentConfig(model="dummy", max_iterations=4)
    messages = [{"role": "user", "content": "Please echo and finish."}]
    import asyncio
    out = asyncio.run(arun_mcp_mini_agent(messages, mcp=EchoMCP(), cfg=cfg))

    assert out.stopped_reason == "success"
    assert "hi" in (out.final_answer or "")
    assert any(
        inv.get("name") == "echo" and (inv.get("ok") is True)
        for inv in (out.iterations[0].tool_invocations if out.iterations else [])
    )
