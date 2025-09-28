"""
Unsure → research hint → retry.

Why this matters:
- Confirms research_on_unsure adds an observation and loops.
- We simulate the LLM returning 'not sure' once, then a final answer.
"""

import pytest
import litellm

from litellm.experimental_mcp_client.mini_agent.litellm_mcp_mini_agent import (
    AgentConfig, LocalMCPInvoker, arun_mcp_mini_agent
)

# File-level marks
pytestmark = [pytest.mark.mini_agent]


@pytest.mark.asyncio
async def test_research_on_unsure_hints_and_retries(monkeypatch):
    calls = {"n": 0}

    async def fake_acompletion(*, model, messages, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"choices": [{"message": {"role":"assistant","content":"I'm not sure"}}]}
        return {"choices": [{"message": {"role":"assistant","content":"Final with citations"}}]}  # finalize

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    cfg = AgentConfig(model="dummy", max_iterations=3, research_on_unsure=True)
    res = await arun_mcp_mini_agent(
        [{"role":"user","content":"answer with sources"}],
        mcp=LocalMCPInvoker(),
        cfg=cfg
    )

    assert res.stopped_reason == "success"
    assert "Final" in (res.final_answer or "")
    # Ensure the "unsure" observation was appended
    assert any(
        m.get("role") == "assistant" and "Model is unsure" in (m.get("content") or "")
        for m in res.messages
    )
