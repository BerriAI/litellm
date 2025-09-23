import json
import pytest


@pytest.mark.smoke
@pytest.mark.asyncio
async def test_mini_agent_autorun_python_codeblock_smoke(monkeypatch):
    """Deterministic smoke: when the assistant replies with a ```python block,
    the agent auto-runs it via exec_python and appends an observation.

    This isolates the behavior with monkeypatches (no real code exec).
    """
    import litellm.experimental_mcp_client.mini_agent.litellm_mcp_mini_agent as agent

    # 1) Stub arouter_call to return an assistant message with a python block
    async def fake_arouter_call(*, model, messages, stream=False, **kwargs):
        return {
            "choices": [
                {"message": {"role": "assistant", "content": """
```python
print('42')
```"""}}
            ]
        }

    monkeypatch.setattr(agent, "arouter_call", fake_arouter_call, raising=True)

    # 2) Stub LocalMCPInvoker.call_openai_tool to capture exec_python
    calls = []

    class StubMCP(agent.LocalMCPInvoker):
        async def call_openai_tool(self, openai_tool):  # type: ignore[override]
            calls.append(openai_tool)
            # Simulate success run
            return json.dumps({
                "ok": True,
                "name": "exec_python",
                "rc": 0,
                "stdout": "42\n",
                "stderr": "",
                "result": "42\n",
            })

    # 3) Run the agent with auto_run_code_on_code_block enabled
    cfg = agent.AgentConfig(model="dummy", max_iterations=1, enable_repair=False, use_tools=True, auto_run_code_on_code_block=True)
    out = await agent.arun_mcp_mini_agent(
        messages=[{"role": "user", "content": "do it"}], mcp=StubMCP(), cfg=cfg
    )

    # 4) Assert tool call was made and observation appended
    assert any((c.get("function") or {}).get("name") == "exec_python" for c in calls)
    # Final answer may be empty (since we ran one iteration); messages should include tool output
    joined = "\n".join(m.get("content", "") for m in out.messages if isinstance(m.get("content"), str))
    assert "42" in joined
