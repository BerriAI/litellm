# tests/smoke/test_agent_proxy_validation_errors.py
import importlib
import json
import types

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient


def _mk_response(status=200, data=None, text=""):
    class _Response:
        def __init__(self):
            self.status_code = status
            self._data = data if data is not None else []
            self.text = text
            self.headers = {}

        def json(self):
            return self._data

    return _Response()


@pytest.mark.smoke
def test_agent_proxy_headers_precedence(monkeypatch):
    from litellm.experimental_mcp_client.mini_agent import http_tools_invoker as inv_mod
    from litellm.experimental_mcp_client.mini_agent import litellm_mcp_mini_agent as mini
    from litellm.experimental_mcp_client.mini_agent import agent_proxy as ap_mod

    # Ensure we are working with a fresh agent_proxy (no lingering monkeypatches)
    ap_mod = importlib.reload(ap_mod)

    captured = {"get": None, "post": None}

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url, headers=None):
            captured["get"] = dict(headers or {})
            return _mk_response(200, [])

        async def post(self, url, json=None, headers=None):
            captured["post"] = dict(headers or {})
            return _mk_response(200, {"ok": True})

    inv_mod.httpx = types.SimpleNamespace(AsyncClient=_Client)

    async def _fake_router_call(*, model, messages, stream=False, **kwargs):
        return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

    monkeypatch.setattr(mini, "arouter_call", _fake_router_call, raising=True)
    monkeypatch.setenv("MINI_AGENT_TOOL_HTTP_HEADERS", json.dumps({"X-Env": "A"}))

    client = TestClient(ap_mod.app)
    response = client.post(
        "/agent/run",
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "model": "m",
            "tool_backend": "http",
            "tool_http_base_url": "http://127.0.0.1:9",
            "tool_http_headers": {"X-Env": "B", "X-Req": "C"},
        },
    )
    assert response.status_code == 200

    hdrs_source = captured["post"] or captured["get"] or {}
    hdrs = {str(k).lower(): v for k, v in hdrs_source.items()}
    assert hdrs.get("x-env") == "B" and hdrs.get("x-req") == "C"
