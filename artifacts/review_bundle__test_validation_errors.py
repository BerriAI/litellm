# Source: tests/smoke_optional/test_agent_proxy_validation_errors.py (lines 1-120)
# tests/smoke/test_agent_proxy_validation_errors.py
import json
import pytest
fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

@pytest.mark.smoke
def test_agent_proxy_headers_precedence(monkeypatch):
    from litellm.experimental_mcp_client.mini_agent import http_tools_invoker as inv_mod
    from litellm.experimental_mcp_client.mini_agent import litellm_mcp_mini_agent as mini
    from litellm.experimental_mcp_client.mini_agent.agent_proxy import app

    recorded = {}

    class _Client:
        def __init__(self, *a, headers=None, **k):
            if headers is not None:
                recorded["headers"] = dict(headers)
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def get(self, url, headers=None):
            if headers is not None:
                recorded["headers"] = dict(headers)
            return type("R", (), {"json": lambda self: [], "raise_for_status": lambda self: None})()
        async def post(self, url, json=None, headers=None):
            if headers is not None:
                recorded["headers"] = dict(headers)
            return type("R", (), {"json": lambda self: {"text": "ok"}, "raise_for_status": lambda self: None})()

    # Use our tolerant httpx stub so we can capture headers
    import types
    inv_mod.httpx = types.SimpleNamespace(AsyncClient=_Client)

    # Short-circuit the LLM call; we only care that /tools (and/or /invoke) was called with the right headers
    async def _fake_router_call(*, model, messages, stream=False, **kwargs):
        return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
    monkeypatch.setattr(mini, "arouter_call", _fake_router_call, raising=True)

    # Set env headers; request headers should override these
    monkeypatch.setenv("MINI_AGENT_TOOL_HTTP_HEADERS", json.dumps({"X-Env": "A"}))

    c = TestClient(app)
    r = c.post(
        "/agent/run",
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "model": "m",
            "tool_backend": "http",
            "tool_http_base_url": "http://127.0.0.1:9",
            "tool_http_headers": {"X-Env": "B", "X-Req": "C"},
        },
    )
    assert r.status_code == 200
    # Normalize header keys to lowercase for case-insensitive compare
    hdrs = {str(k).lower(): v for k, v in (recorded.get("headers", {}) or {}).items()}
    assert hdrs.get("x-env") == "B" and hdrs.get("x-req") == "C"

