import os
import shutil
import subprocess
import sys
import time
import pytest


@pytest.mark.ndsmoke
def test_node_gateway_optional(monkeypatch):
    node = shutil.which("node")
    if not node:
        pytest.skip("node not installed; skipping gateway smoke")

    # Start the gateway
    # Prefer local/ path if present; else fallback to in-tree example path
    cand_local = os.path.join(os.getcwd(), "local", "mini_agent", "node_tools_gateway")
    cand_tree = os.path.join(
        os.getcwd(),
        "litellm",
        "experimental_mcp_client",
        "mini_agent",
        "node_tools_gateway",
    )
    gw_dir = cand_local if os.path.isdir(cand_local) else cand_tree
    if not os.path.isdir(gw_dir):
        pytest.skip("node gateway directory not found")
    env = os.environ.copy()
    env.setdefault("PORT", "8788")
    proc = subprocess.Popen([node, "server.mjs"], cwd=gw_dir, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        # Readiness poll (up to ~3s)
        import json as _json
        import urllib.request as _req
        ready = False
        for _ in range(20):
            try:
                with _req.urlopen("http://127.0.0.1:8788/tools", timeout=0.15) as r:  # type: ignore
                    if r.status == 200:
                        _ = _json.loads(r.read().decode("utf-8"))
                        ready = True
                        break
            except Exception:
                pass
            time.sleep(0.15)
        if not ready:
            pytest.skip("node gateway not ready in time")
        # Stub fastuuid required by litellm imports
        monkeypatch.setitem(sys.modules, "fastuuid", type("_U", (), {"uuid4": staticmethod(lambda: "0"*32)})())
        # Stub MCP import used by experimental_mcp_client package init
        monkeypatch.setitem(sys.modules, "mcp", type("_M", (), {"ClientSession": object})())
        T = type("_T", (), {})
        setattr(T, "CallToolRequestParams", type("CallToolRequestParams", (), {}))
        setattr(T, "CallToolResult", type("CallToolResult", (), {}))
        setattr(T, "Tool", type("Tool", (), {}))
        monkeypatch.setitem(sys.modules, "mcp.types", T())
        from litellm.experimental_mcp_client.mini_agent.http_tools_invoker import HttpToolsInvoker
        from litellm.experimental_mcp_client.mini_agent.litellm_mcp_mini_agent import AgentConfig, run_mcp_mini_agent

        mcp = HttpToolsInvoker("http://127.0.0.1:8788")
        cfg = AgentConfig(model="dummy")
        messages = [{"role": "user", "content": "Use echo('hi') and finish."}]

        # Monkeypatch Router call to return a tool_call then final
        import litellm.experimental_mcp_client.mini_agent.litellm_mcp_mini_agent as mini_mod

        state = {"call": 0}

        async def fake_arouter_call(*, model, messages, stream=False, **kwargs):
            state["call"] += 1
            c = state["call"]
            if c == 1:
                return {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "call echo",
                                "tool_calls": [
                                    {
                                        "id": "tc",
                                        "type": "function",
                                        "function": {"name": "echo", "arguments": "{\"text\":\"hi\"}"},
                                    }
                                ],
                            }
                        }
                    ]
                }
            return {"choices": [{"message": {"role": "assistant", "content": "Done"}}]}

        monkeypatch.setattr(mini_mod, "arouter_call", fake_arouter_call)

        res = run_mcp_mini_agent(messages, mcp=mcp, cfg=cfg)
        assert res.stopped_reason == "success"
    finally:
        proc.terminate()
        try:
            proc.wait(1)
        except Exception:
            proc.kill()
