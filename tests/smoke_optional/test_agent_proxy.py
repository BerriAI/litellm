import json
import pytest
fastapi = pytest.importorskip("fastapi", reason="optional extra for mini_agent endpoint")
from fastapi.testclient import TestClient  # type: ignore
import sys, types


@pytest.mark.smoke
def test_agent_proxy_run_minimal(monkeypatch):
    # One-shot answer with no tool calls
    async def fake_arouter_call(*, model, messages, stream=False, **kwargs):
        return {"choices": [{"message": {"role": "assistant", "content": "Hello"}}]}

    # Stub optional heavy deps + MCP so we can import litellm without extras
    monkeypatch.setitem(sys.modules, "fastuuid", types.SimpleNamespace(uuid4=lambda: "0" * 32))
    mcp_stub = types.SimpleNamespace(ClientSession=object)
    mcp_types_stub = types.SimpleNamespace(
        CallToolRequestParams=type("CallToolRequestParams", (), {}),
        CallToolResult=type("CallToolResult", (), {}),
        Tool=type("Tool", (), {}),
    )
    monkeypatch.setitem(sys.modules, "mcp", mcp_stub)
    monkeypatch.setitem(sys.modules, "mcp.types", mcp_types_stub)
    # Import after stubbing
    from litellm.experimental_mcp_client.mini_agent.agent_proxy import app
    import litellm.experimental_mcp_client.mini_agent.litellm_mcp_mini_agent as mini_mod
    monkeypatch.setattr(mini_mod, "arouter_call", fake_arouter_call)

    client = TestClient(app)
    req = {
        "messages": [{"role": "user", "content": "say hi"}],
        "model": "dummy",
        "tool_backend": "echo",
    }
    r = client.post("/agent/run", json=req)
    assert r.status_code == 200
    data = r.json()
    assert data["stopped_reason"] == "success"
    assert data["final_answer"] == "Hello"
