"""
Auto-run code block -> exec_python.

Why this matters:
- Validates the fenced-code extraction and automatic tool execution.
- Ensures stdout from exec_python is captured and attached to the conversation.
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
async def test_auto_codeblock_exec_python_runs(monkeypatch):
    SENTINEL = "ND_SMOKE_OK"

    async def fake_acompletion(*, model, messages, **kw):
        # Return a message with a fenced python code block
        return {"choices": [{"message": {
            "role": "assistant",
            "content": f"```python\nprint('{SENTINEL}')\n```"
        }}]}

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    cfg = AgentConfig(model="dummy", max_iterations=2, auto_run_code_on_code_block=True)
    msgs = [{"role": "user", "content": "please print ND_SMOKE_OK in python"}]
    res = await arun_mcp_mini_agent(msgs, mcp=LocalMCPInvoker(), cfg=cfg)

    # The first (and only) iteration should contain one tool invocation on exec_python
    assert res.stopped_reason == "success"
    assert res.iterations, "no iteration record captured"
    invs = res.iterations[0].tool_invocations
    assert any(inv.get("name") == "exec_python" and inv.get("ok") for inv in invs)
    # Ensure stdout contains the sentinel
    payload = invs[0]
    out = payload.get("stdout") or payload.get("text") or payload.get("result") or ""
    assert SENTINEL in out
