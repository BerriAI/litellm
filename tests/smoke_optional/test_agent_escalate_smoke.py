
import pytest

@pytest.mark.smoke
@pytest.mark.asyncio
async def test_agent_escalate_on_last_step(monkeypatch):
    from litellm.experimental_mcp_client.mini_agent import litellm_mcp_mini_agent as agent

    called = []

    async def fake_call(*, model, messages, stream=False, **kwargs):
        called.append(model)
        # Return plain assistant message without tool calls so loop can consider final
        return {"choices":[{"message":{"role":"assistant","content":"ok"}}]}

    monkeypatch.setattr(agent, 'arouter_call', fake_call)

    cfg = agent.AgentConfig(
        model='base/x',
        max_iterations=1,
        use_tools=True,  # so the loop won't immediately finalize without a nudge
        enable_repair=False,
        auto_run_code_on_code_block=False,
        escalate_on_budget_exceeded=True,
        escalate_model='escalate/x',
    )
    messages=[{"role":"user","content":"hello"}]
    res = await agent.arun_mcp_mini_agent(messages, mcp=agent.EchoMCP(), cfg=cfg)
    # Ensure escalation model was used on that single iteration
    assert 'escalate/x' in called, f"models called: {called}"
    assert res.stopped_reason in ('success','max_iterations')
