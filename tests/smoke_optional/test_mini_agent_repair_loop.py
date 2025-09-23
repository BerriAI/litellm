import json
import types
import sys
import pytest


@pytest.mark.smoke
@pytest.mark.asyncio
async def test_mini_agent_repair_loop_exec_python(monkeypatch):
    # Stub optional deps for importing litellm
    monkeypatch.setitem(sys.modules, "fastuuid", types.SimpleNamespace(uuid4=lambda: "0" * 32))

    # Stub MCP import used by litellm.experimental_mcp_client.__init__
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
    # Patch the Router call to simulate model steps across iterations
    state = {"call": 0}

    async def fake_arouter_call(*, model, messages, stream=False, **kwargs):
        state["call"] += 1
        c = state["call"]
        # 1) Ask to run failing python
        if c == 1:
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Attempt step",
                            "tool_calls": [
                                {
                                    "id": "tc_fail",
                                    "type": "function",
                                    "function": {
                                        "name": "exec_python",
                                        "arguments": json.dumps({"code": "import sys; print('ERR', file=sys.stderr); sys.exit(2)"}),
                                    },
                                }
                            ],
                        }
                    }
                ]
            }
        # 2) After observation is appended, propose corrected code
        if c == 2:
            # Ensure the last assistant message contains Observation
            last = messages[-1]
            assert last.get("role") == "assistant" and "Observation" in (last.get("content") or "")
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Retry with fix",
                            "tool_calls": [
                                {
                                    "id": "tc_ok",
                                    "type": "function",
                                    "function": {
                                        "name": "exec_python",
                                        "arguments": json.dumps({"code": "print('OK')"}),
                                    },
                                }
                            ],
                        }
                    }
                ]
            }
        # 3) Finish with direct answer
        return {"choices": [{"message": {"role": "assistant", "content": "Done: OK"}}]}

    import litellm.experimental_mcp_client.mini_agent.litellm_mcp_mini_agent as mini_mod

    monkeypatch.setattr(mini_mod, "arouter_call", fake_arouter_call)

    cfg = AgentConfig(model="dummy", max_iterations=5, enable_repair=True)
    messages = [{"role": "user", "content": "Please run python and finish."}]

    out = await arun_mcp_mini_agent(messages, mcp=LocalMCPInvoker(), cfg=cfg)
    assert out.stopped_reason == "success"
    assert "OK" in (out.final_answer or "")
    # Check the appended observation contains stderr preview
    obs_msgs = [m for m in out.messages if m.get("role") == "assistant" and "Observation from last tool run" in str(m.get("content") or "")]
    assert any("ERR" in m["content"] for m in obs_msgs)
    # Ensure we recorded one failing and later one passing invocation across iterations
    all_invs = []
    for it in out.iterations:
        all_invs.extend(it.tool_invocations)
    assert any((i.get("error") for i in all_invs))
    assert any((i.get("ok") for i in all_invs))
