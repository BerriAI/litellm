import pytest
from litellm.experimental_mcp_client.mini_agent.agent_proxy import AgentRunReq, run


@pytest.mark.asyncio
async def test_mini_agent_local_backend_returns_deterministic_answer():
    req = AgentRunReq(
        messages=[{"role": "user", "content": "hello"}],
        model="dummy-local",
        tool_backend="local",
        max_iterations=2,
    )
    resp = await run(req)

    assert resp["ok"] is True
    assert resp["final_answer"] == "hello"
    assert resp["metrics"]["iterations"] == 2
    assert resp["messages"] == [{"role": "assistant", "content": "hello"}]
