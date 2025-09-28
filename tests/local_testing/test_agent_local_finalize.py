"""
Local agent: finalize without tools.

Why this matters:
- Proves the core loop returns a final answer when no tools are invoked.
- Uses a monkeypatched litellm.acompletion so we don't hit any provider.
"""

import pytest
import litellm

from litellm.experimental_mcp_client.mini_agent.litellm_mcp_mini_agent import (
    AgentConfig,
    LocalMCPInvoker,
    arun_mcp_mini_agent,
)

# File-level marks
pytestmark = [pytest.mark.mini_agent]


@pytest.mark.asyncio
async def test_local_finalize_without_tools(monkeypatch):
    async def fake_acompletion(*, model, messages, **kw):
        return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    cfg = AgentConfig(model="dummy", max_iterations=3, use_tools=False)
    msgs = [{"role": "user", "content": "say hi and stop"}]
    res = await arun_mcp_mini_agent(msgs, mcp=LocalMCPInvoker(), cfg=cfg)

    assert res.stopped_reason == "success"
    assert (res.final_answer or "").strip() == "ok"
    assert isinstance(res.messages, list) and len(res.messages) >= 2
