import types
import sys
import pytest


class _Resp:
    def __init__(self, json_obj, status_code=200):
        self._json = json_obj
        self.status_code = status_code
    def json(self):
        return self._json
    def raise_for_status(self):
        if not (200 <= self.status_code < 300):
            raise RuntimeError(f"HTTP {self.status_code}")


recorded = {}

class _Client:
    def __init__(self, *a, headers=None, **k):
        self._headers = headers or {}
        recorded["headers"] = self._headers
    async def __aenter__(self):
        return self
    async def __aexit__(self, *exc):
        return False
    async def get(self, url):
        return _Resp([])
    async def post(self, url, json=None):
        return _Resp({"text": "ok"})


@pytest.mark.smoke
@pytest.mark.asyncio
async def test_http_tools_invoker_headers_passthrough(monkeypatch):
    # Stub fastuuid to satisfy litellm imports
    monkeypatch.setitem(sys.modules, "fastuuid", types.SimpleNamespace(uuid4=lambda: "0" * 32))
    # Stub required modules
    monkeypatch.setitem(sys.modules, "mcp", types.SimpleNamespace(ClientSession=object))
    monkeypatch.setitem(sys.modules, "mcp.types", types.SimpleNamespace(
        CallToolRequestParams=type("CallToolRequestParams", (), {}),
        CallToolResult=type("CallToolResult", (), {}),
        Tool=type("Tool", (), {}),
    ))
    from litellm.experimental_mcp_client.mini_agent import http_tools_invoker as inv_mod
    inv_mod.httpx = types.SimpleNamespace(AsyncClient=_Client)
    inv = inv_mod.HttpToolsInvoker("http://fake", headers={"Authorization": "Bearer Z"})

    # list tools to trigger client
    await inv.list_openai_tools()
    # post invoke to trigger client
    await inv.call_openai_tool({"function": {"name": "echo", "arguments": "{}"}})
    # validate headers were recorded by the fake client
    assert recorded.get("headers", {}).get("Authorization") == "Bearer Z"
