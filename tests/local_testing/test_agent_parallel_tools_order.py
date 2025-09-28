"""
Parallel tool execution preserves output order.

Why this matters:
- With parallelized tool execution, we still append tool outputs in ORIGINAL call order.
"""

import asyncio
import json
import pytest
import litellm

from litellm.experimental_mcp_client.mini_agent.litellm_mcp_mini_agent import (
    AgentConfig, LocalMCPInvoker, arun_mcp_mini_agent
)

# File-level marks
pytestmark = [pytest.mark.mini_agent]


class SlowLocal(LocalMCPInvoker):
    async def call_openai_tool(self, openai_tool):
        f = openai_tool.get("function",{}) or {}
        args = f.get("arguments","{}")
        try:
            obj = json.loads(args)
        except Exception:
            obj = {}
        code = (obj.get("code") or "")
        # Delay A so it completes after B
        if "print('A')" in code or 'print("A")' in code:
            await asyncio.sleep(0.05)
        return await super().call_openai_tool(openai_tool)


@pytest.mark.asyncio
async def test_parallel_tools_order_preserved(monkeypatch):
    step = {"n": 0}

    async def fake_acompletion(*, model, messages, **kw):
        if step["n"] == 0:
            step["n"] += 1
            return {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "call A and B",
                        "tool_calls": [
                            {"id":"a","type":"function","function":{
                                "name":"exec_python","arguments":json.dumps({"code":"print('A')"})}},
                            {"id":"b","type":"function","function":{
                                "name":"exec_python","arguments":json.dumps({"code":"print('B')"})}},
                        ]
                    }
                }]
            }
        return {"choices": [{"message": {"role": "assistant", "content": "done"}}]}

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    cfg = AgentConfig(model="dummy", max_iterations=2)
    res = await arun_mcp_mini_agent(
        [{"role":"user","content":"run two tools then stop"}],
        mcp=SlowLocal(),
        cfg=cfg
    )

    # Extract tool message contents in order
    tool_msgs = [m for m in res.messages if m.get("role") == "tool"]
    assert [m["content"] for m in tool_msgs] == ["A", "B"]
