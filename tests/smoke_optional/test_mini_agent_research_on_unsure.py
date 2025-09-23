import json
import types
import sys
import pytest


@pytest.mark.smoke
@pytest.mark.asyncio
async def test_mini_agent_research_on_unsure(monkeypatch):
    # Stubs to import litellm
    monkeypatch.setitem(sys.modules, "fastuuid", types.SimpleNamespace(uuid4=lambda: "0" * 32))
    monkeypatch.setitem(sys.modules, "mcp", types.SimpleNamespace(ClientSession=object))
    monkeypatch.setitem(
        sys.modules,
        "mcp.types",
        types.SimpleNamespace(
            CallToolRequestParams=type("CallToolRequestParams", (), {}),
            CallToolResult=type("CallToolResult", (), {}),
            Tool=type("Tool", (), {}),
        ),
    )

    from litellm.experimental_mcp_client.mini_agent.litellm_mcp_mini_agent import (
        AgentConfig,
        LocalMCPInvoker,
        arun_mcp_mini_agent,
    )
    import litellm.experimental_mcp_client.mini_agent.litellm_mcp_mini_agent as mini_mod

    state = {"call": 0}

    async def fake_arouter_call(*, model, messages, stream=False, **kwargs):
        state["call"] += 1
        c = state["call"]
        if c == 1:
            # Model is unsure; agent should steer towards research and not finish yet
            return {"choices": [{"message": {"role": "assistant", "content": "I'm not sure."}}]}
        if c == 2:
            # After agent appends directive, model issues a research tool call
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "researching",
                            "tool_calls": [
                                {
                                    "id": "tc_r",
                                    "type": "function",
                                    "function": {
                                        "name": "research_echo",
                                        "arguments": json.dumps({"query": "what is X?"}),
                                    },
                                }
                            ],
                        }
                    }
                ]
            }
        return {"choices": [{"message": {"role": "assistant", "content": "Final answer."}}]}

    monkeypatch.setattr(mini_mod, "arouter_call", fake_arouter_call)

    cfg = AgentConfig(model="dummy", max_iterations=5, research_on_unsure=True, max_research_hops=1)
    messages = [{"role": "user", "content": "question"}]
    out = await arun_mcp_mini_agent(messages, mcp=LocalMCPInvoker(), cfg=cfg)
    assert out.stopped_reason == "success"
    # Ensure a research directive assistant message was injected
    assert any(
        m.get("role") == "assistant" and "Use available research tools" in (m.get("content") or "")
        for m in out.messages
    )
