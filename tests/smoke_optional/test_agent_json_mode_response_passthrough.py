import pytest
import sys, types

@pytest.mark.smoke
def test_agent_json_mode_response_passthrough(monkeypatch):
    """
    JSON-only assistant content is preserved through the agent message stream.
    """
    monkeypatch.setitem(sys.modules, "fastuuid", types.SimpleNamespace(uuid4=lambda: "0"*32))
    monkeypatch.setitem(sys.modules, "mcp", types.SimpleNamespace(ClientSession=object))
    monkeypatch.setitem(sys.modules, "mcp.types", types.SimpleNamespace(
        CallToolRequestParams=object, CallToolResult=object, Tool=object
    ))

    from litellm.experimental_mcp_client.mini_agent.litellm_mcp_mini_agent import (
        AgentConfig, LocalMCPInvoker, run_mcp_mini_agent
    )
    import litellm.experimental_mcp_client.mini_agent.litellm_mcp_mini_agent as mod

    payload = {"ok": True, "score": 0.99, "explain": {"why": "deterministic"}}

    async def fake_router(**kwargs):
        return {"choices": [{"message": {"role": "assistant", "content": payload}}]}

    monkeypatch.setattr(mod, "arouter_call", fake_router)

    cfg = AgentConfig(model="noop", max_iterations=1, enable_repair=False, use_tools=False)
    res = run_mcp_mini_agent([{"role": "user", "content": "json please"}], mcp=LocalMCPInvoker(), cfg=cfg)
    assert any(isinstance(m.get("content"), dict) and m["content"].get("ok") is True for m in res.messages)
