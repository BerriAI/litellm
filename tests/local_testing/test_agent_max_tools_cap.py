"""
Per-iteration tool cap.

Why this matters:
- Ensures the agent respects max_tools_per_iter and doesn't fan out endlessly.
"""

import json
import pytest
import litellm

from litellm.experimental_mcp_client.mini_agent.litellm_mcp_mini_agent import (
    AgentConfig, LocalMCPInvoker, arun_mcp_mini_agent
)

# File-level marks
pytestmark = [pytest.mark.mini_agent]


@pytest.mark.asyncio
async def test_max_tools_per_iter_is_respected(monkeypatch):
    step = {"n": 0}

    async def fake_acompletion(*, model, messages, **kw):
        if step["n"] == 0:
            step["n"] += 1
            return {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "run two python snippets",
                        "tool_calls": [
                            {"id":"t1","type":"function","function":{
                                "name":"exec_python","arguments":json.dumps({"code":"print('A')"})}},
                            {"id":"t2","type":"function","function":{
                                "name":"exec_python","arguments":json.dumps({"code":"print('B')"})}},
                        ]
                    }
                }]
            }
        return {"choices": [{"message": {"role": "assistant", "content": "done"}}]}

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    cfg = AgentConfig(model="dummy", max_iterations=2, max_tools_per_iter=1)
    res = await arun_mcp_mini_agent(
        [{"role":"user","content":"do two tools then finish"}],
        mcp=LocalMCPInvoker(),
        cfg=cfg
    )

    assert res.stopped_reason == "success"
    assert res.iterations and len(res.iterations[0].tool_invocations) == 1
