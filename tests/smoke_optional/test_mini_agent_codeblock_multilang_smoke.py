import json, pytest

@pytest.mark.smoke
@pytest.mark.asyncio
async def test_mini_agent_autorun_c_codeblock_smoke(monkeypatch):
    import litellm.experimental_mcp_client.mini_agent.litellm_mcp_mini_agent as agent

    # Assistant returns a C code block
    async def fake_arouter_call(*, model, messages, stream=False, **kwargs):
        return {"choices":[{"message":{"role":"assistant","content":"""
```c
int main(){return 0;}
```
"""}}]}

    calls=[]
    class StubMCP(agent.LocalMCPInvoker):
        async def call_openai_tool(self, openai_tool):  # type: ignore[override]
            calls.append(openai_tool)
            # Simulate success
            return json.dumps({"ok": True, "name": (openai_tool.get('function') or {}).get('name'), "rc": 0, "stdout": "ok\n", "stderr": ""})

    monkeypatch.setattr(agent, 'arouter_call', fake_arouter_call, raising=True)

    cfg = agent.AgentConfig(model='dummy', max_iterations=1, enable_repair=False, use_tools=True, auto_run_code_on_code_block=True)
    out = await agent.arun_mcp_mini_agent(messages=[{"role":"user","content":"do it"}], mcp=StubMCP(), cfg=cfg)

    # Expect exec_code tool invocation
    assert any((c.get('function') or {}).get('name')=='exec_code' for c in calls)
