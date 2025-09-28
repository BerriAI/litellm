"""
Purpose
- Isolate the loop: failing tool → repaired tool → final answer; last tool is clean.

Scope
- DOES: stub arouter_call to return buggy then fixed tool_calls; assert last rc==0 and stderr==''.
- DOES NOT: run real code; tool outputs are controlled via stubs.

Run
- `pytest tests/smoke -k test_mini_agent_loop_isolation_smoke -q`
"""
import json
import types
import sys
import pytest


@pytest.mark.asyncio
async def test_mini_agent_loop_isolation_until_clean(monkeypatch):
    """
    Deterministic smoke that isolates the mini-agent's iterate->run->repair loop.

    We monkeypatch the LLM call (arouter_call) to return a sequence of assistant
    messages: (1) tool call with buggy code, (2) tool call with fixed code,
    (3) final direct answer. We assert the last tool run is clean (rc=0, no stderr).
    """
    # Make import of litellm safe (avoid optional deps blowing up)
    monkeypatch.setitem(sys.modules, "fastuuid", types.SimpleNamespace(uuid4=lambda: "0" * 32))
    mcp_stub = types.SimpleNamespace(ClientSession=object)
    mcp_types_stub = types.SimpleNamespace(
        CallToolRequestParams=type("CallToolRequestParams", (), {}),
        CallToolResult=type("CallToolResult", (), {}),
        Tool=type("Tool", (), {}),
    )
    monkeypatch.setitem(sys.modules, "mcp", mcp_stub)
    monkeypatch.setitem(sys.modules, "mcp.types", mcp_types_stub)

    from litellm.experimental_mcp_client.mini_agent.litellm_mcp_mini_agent import (
        AgentConfig,
        LocalMCPInvoker,
        arun_mcp_mini_agent,
    )
    import litellm.experimental_mcp_client.mini_agent.litellm_mcp_mini_agent as mini

    calls = {"n": 0}

    async def fake_arouter_call(*, model, messages, stream=False, **kwargs):
        calls["n"] += 1
        c = calls["n"]
        if c == 1:
            # Ask to run buggy python (non-zero rc, stderr)
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "tc_bug",
                                    "type": "function",
                                    "function": {
                                        "name": "exec_python",
                                        "arguments": json.dumps({
                                            "code": "import sys\nprint('bad', file=sys.stderr)\nraise SystemExit(1)\n",
                                        }),
                                    },
                                }
                            ],
                        }
                    }
                ]
            }
        if c == 2:
            # Ask to run fixed python
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "tc_ok",
                                    "type": "function",
                                    "function": {
                                        "name": "exec_python",
                                        "arguments": json.dumps({
                                            "code": "print('a3b2c1')\nprint('a2b2c2d2e2f2g2')\n",
                                        }),
                                    },
                                }
                            ],
                        }
                    }
                ]
            }
        # Final direct answer to allow agent to stop with success
        return {"choices": [{"message": {"role": "assistant", "content": "Done"}}]}

    monkeypatch.setattr(mini, "arouter_call", fake_arouter_call)

    cfg = AgentConfig(
        model="dummy",
        max_iterations=4,
        enable_repair=True,
        use_tools=True,
        auto_run_code_on_code_block=False,  # we explicitly use tool_calls path
        tool_timeout_sec=15.0,
    )
    messages = [{"role": "user", "content": "Please implement compress_runs and print two tests."}]

    out = await arun_mcp_mini_agent(messages, mcp=LocalMCPInvoker(), cfg=cfg)

    # Ensure we stopped successfully
    assert out.stopped_reason == "success"

    # Find the last tool invocation and assert it's clean
    all_invs = []
    for it in out.iterations:
        all_invs.extend(it.tool_invocations)
    assert any((i.get("ok") is False) for i in all_invs), "Expected at least one failing run"
    last = all_invs[-1]
    assert last.get("ok") is True and (last.get("rc") or 0) == 0
    assert (last.get("stderr") or "").strip() == ""
    # Verify expected outputs were produced at some point
    tool_texts = [i.get("stdout") or i.get("result") or i.get("answer") or "" for i in all_invs]
    joined = "\n".join(tool_texts).lower()
    assert "a3b2c1" in joined and "a2b2c2d2e2f2g2" in joined
