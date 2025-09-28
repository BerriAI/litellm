import json
import sys
import types
import pytest


@pytest.mark.smoke
def test_agent_proxy_errors_and_headers(monkeypatch):
    fastapi = pytest.importorskip("fastapi")
    _ = pytest.importorskip("fastapi.testclient")
    from fastapi.testclient import TestClient

    # Stub optional deps pulled by litellm on import
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

    # Import app now
    from litellm.experimental_mcp_client.mini_agent.agent_proxy import app
    import litellm.experimental_mcp_client.mini_agent.agent_proxy as ap_mod
    import litellm.experimental_mcp_client.mini_agent.litellm_mcp_mini_agent as agent_mod

    # Monkeypatch arouter_call to short-circuit LLM call
    async def fake_arouter_call(*, model, messages, stream=False, **kwargs):
        return {"choices": [{"message": {"role": "assistant", "content": "Done"}}]}
    agent_mod.arouter_call = fake_arouter_call

    # Missing httpx -> 400
    class _FakeInvokerMissing:  # raises ImportError scenario is handled at call-site already
        pass

    # Replace HttpToolsInvoker with a dummy that records headers and tools
    recorded = {}

    class _FakeInvoker:
        def __init__(self, base_url, headers=None):
            recorded["base_url"] = base_url
            recorded["headers"] = dict(headers or {})

        async def list_openai_tools(self):
            return []

        async def call_openai_tool(self, openai_tool):
            return json.dumps({"ok": True, "text": "hi"})

    monkeypatch.setattr(ap_mod, "HttpToolsInvoker", _FakeInvoker, raising=True)

    client = TestClient(app)

    # 1) Missing tool_http_base_url → 400
    r = client.post("/agent/run", json={
        "messages": [{"role": "user", "content": "test"}],
        "model": "dummy",
        "tool_backend": "http"
    })
    assert r.status_code == 400

    # 2) Headers passthrough
    r2 = client.post("/agent/run", json={
        "messages": [{"role": "user", "content": "test"}],
        "model": "dummy",
        "tool_backend": "http",
        "tool_http_base_url": "http://127.0.0.1:9999",
        "tool_http_headers": {"Authorization": "Bearer X"}
    })
    assert r2.status_code == 200
    assert recorded.get("headers", {}).get("Authorization") == "Bearer X"
