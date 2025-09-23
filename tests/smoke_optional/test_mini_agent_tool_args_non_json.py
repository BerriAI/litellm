import json, pytest, sys, types

@pytest.mark.smoke
@pytest.mark.asyncio
async def test_mini_agent_tool_args_non_json(monkeypatch):
    monkeypatch.setitem(sys.modules,"fastuuid", types.SimpleNamespace(uuid4=lambda:"0"*32))
    monkeypatch.setitem(sys.modules,"mcp", types.SimpleNamespace(ClientSession=object))
    monkeypatch.setitem(sys.modules,"mcp.types", types.SimpleNamespace(
        CallToolRequestParams=object, CallToolResult=object, Tool=object))
    from litellm.experimental_mcp_client.mini_agent.litellm_mcp_mini_agent import AgentConfig, LocalMCPInvoker, arun_mcp_mini_agent
    import litellm.experimental_mcp_client.mini_agent.litellm_mcp_mini_agent as mod

    async def fake_call(**kw):
        return {"choices":[{"message":{"role":"assistant","content":"tool","tool_calls":[
            {"id":"tc","type":"function","function":{"name":"echo","arguments":"{BAD JSON]"}}
        ]}}]}
    monkeypatch.setattr(mod,"arouter_call",fake_call)

    out = await arun_mcp_mini_agent([{"role":"user","content":"go"}], mcp=LocalMCPInvoker(), cfg=AgentConfig(model="m",max_iterations=2,enable_repair=True,use_tools=True))
    joined = "\n".join((m.get("content") or "") for m in out.messages if isinstance(m,dict))
    assert "Observation" in joined or out.stopped_reason in ("success","max_iterations")
