"""
Shell allowlist enforcement (security).

Why this matters:
- Confirms disallowed commands (e.g., `uname -a`) are blocked by LocalMCPInvoker.
- Tests the tool directly, without LLM involvement.
"""

import json
import pytest

from litellm.experimental_mcp_client.mini_agent.litellm_mcp_mini_agent import LocalMCPInvoker

# File-level marks
pytestmark = [pytest.mark.mini_agent]


@pytest.mark.asyncio
async def test_exec_shell_allowlist_blocks_uname():
    inv = LocalMCPInvoker()  # default allowlist: ("echo",)
    call = {
        "id": "tc1",
        "type": "function",
        "function": {"name": "exec_shell", "arguments": json.dumps({"cmd": "uname -a"})}
    }
    out = await inv.call_openai_tool(call)
    obj = json.loads(out)
    assert obj.get("ok") is False
    assert "prefix not allowed" in (obj.get("error") or "")
