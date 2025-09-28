import pytest, litellm
pytestmark = (pytest.mark.mini_agent,)

from litellm.experimental_mcp_client.mini_agent.litellm_mcp_mini_agent import AgentConfig, LocalMCPInvoker, arun_mcp_mini_agent

"""
Default iterations == 3 (finalize path).
"""

@pytest.mark.asyncio
async def test_default_iterations_three(monkeypatch):
    calls = {"n": 0}
    async def fake_acompletion(*, model, messages, **kw):
        calls["n"] += 1
        if calls["n"] < 3:
            return {"choices":[{"message":{"role":"assistant","content":"Budget exceeded"}}]}
        return {"choices":[{"message":{"role":"assistant","content":"done"}}]}
    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    cfg = AgentConfig(model="dummy", escalate_on_budget_exceeded=True, escalate_model="dummy")
    await arun_mcp_mini_agent([{"role":"user","content":"go"}], mcp=LocalMCPInvoker(), cfg=cfg)

    assert calls["n"] == 3
