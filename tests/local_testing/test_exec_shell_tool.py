"""
Purpose
- Enforce shell allowlist and verify shell tool invocation capture is wired.

Scope
- DOES: stub arouter_call to request exec_shell; assert tool_invocation ok=True appears.
- DOES NOT: execute real shell outside the allowlist or use network.

Run
- `pytest tests/smoke -k test_exec_shell_tool_allowlist -q`
"""
import json
import types
import sys
import pytest


def test_exec_shell_tool_allowlist(monkeypatch):
    monkeypatch.setitem(sys.modules, "fastuuid", types.SimpleNamespace(uuid4=lambda: "0" * 32))
    monkeypatch.setitem(sys.modules, "mcp", types.SimpleNamespace(ClientSession=object))
    monkeypatch.setitem(
        sys.modules,
        "mcp.types",
        types.SimpleNamespace(
            CallToolRequestParams=type("CallToolRequestParams", (), {}),
            CallToolResult=type("CallToolResult", (), {}),
            Tool=type("Tool", (), {}),
        ),
    )

    from litellm.experimental_mcp_client.mini_agent.litellm_mcp_mini_agent import (
        AgentConfig,
        LocalMCPInvoker,
        arun_mcp_mini_agent,
    )
    import litellm.experimental_mcp_client.mini_agent.litellm_mcp_mini_agent as mini_mod

    # LLM asks to run a simple echo; then ends
    async def fake_arouter_call(*, model, messages, stream=False, **kwargs):
        if len([m for m in messages if m.get("role") == "assistant"]) == 0:
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "run shell",
                            "tool_calls": [
                                {
                                    "id": "tc1",
                                    "type": "function",
                                    "function": {"name": "exec_shell", "arguments": json.dumps({"cmd": "echo HI"})},
                                }
                            ],
                        }
                    }
                ]
            }
        return {"choices": [{"message": {"role": "assistant", "content": "Done."}}]}

    monkeypatch.setattr(mini_mod, "arouter_call", fake_arouter_call)

    cfg = AgentConfig(model="dummy", max_iterations=3)
    messages = [{"role": "user", "content": "use shell"}]
    import asyncio
    out = asyncio.run(arun_mcp_mini_agent(messages, mcp=LocalMCPInvoker(shell_allow_prefixes=["echo"]), cfg=cfg))
    assert out.stopped_reason == "success"
    # Ensure shell output captured in tool preview
    invs = out.iterations[0].tool_invocations
    assert any(i.get("ok") for i in invs)
