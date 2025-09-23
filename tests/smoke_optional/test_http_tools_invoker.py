import json
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


class _FakeClient:
    def __init__(self, *a, **k):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get(self, url):
        if url.endswith("/tools"):
            return _Resp(
                [
                    {
                        "type": "function",
                        "function": {
                            "name": "echo",
                            "parameters": {
                                "type": "object",
                                "properties": {"text": {"type": "string"}},
                                "required": ["text"],
                            },
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "list_dir",
                            "parameters": {
                                "type": "object",
                                "properties": {"path": {"type": "string"}},
                                "required": ["path"],
                            },
                        },
                    },
                ]
            )
        return _Resp({}, 404)

    async def post(self, url, json=None):  # noqa: A002 - keep name for signature but avoid shadowing below
        if url.endswith("/invoke"):
            body = json or {}
            if body.get("name") == "echo":
                try:
                    import json as _json
                    args = body.get("arguments", "{}")
                    if not isinstance(args, str):
                        args = _json.dumps(args)
                    text = _json.loads(args).get("text", "")
                except Exception:
                    text = ""
                return _Resp({"text": text})
            return _Resp({"error": "tool_not_found"}, 404)
        return _Resp({}, 404)


@pytest.mark.smoke
@pytest.mark.asyncio
async def test_http_tools_invoker_monkeypatch(monkeypatch):
    pytest.importorskip('mcp', reason='optional; package-level import requires it')
    # Some litellm imports expect optional deps like fastuuid; stub to avoid env coupling
    monkeypatch.setitem(sys.modules, 'fastuuid', types.SimpleNamespace(uuid4=lambda: '0' * 32))
    import types as _types
    mcp_pkg = _types.ModuleType('mcp')
    mcp_types = _types.ModuleType('mcp.types')
    setattr(mcp_pkg, 'ClientSession', object)
    setattr(mcp_types, 'CallToolRequestParams', object)
    setattr(mcp_types, 'CallToolResult', object)
    monkeypatch.setitem(sys.modules, 'mcp', mcp_pkg)
    monkeypatch.setitem(sys.modules, 'mcp.types', mcp_types)
    from litellm.experimental_mcp_client.mini_agent import http_tools_invoker as inv_mod
    # Patch the module's httpx client directly to avoid import-order issues
    inv_mod.httpx = types.SimpleNamespace(AsyncClient=_FakeClient)
    HttpToolsInvoker = inv_mod.HttpToolsInvoker

    inv = HttpToolsInvoker('http://fake')
    tools = await inv.list_openai_tools()
    names = [t.get('function', {}).get('name') for t in tools]
    assert 'echo' in names and 'list_dir' in names

    openai_call = {
        'id': 'tc1',
        'type': 'function',
        'function': {'name': 'echo', 'arguments': json.dumps({'text': 'hello'})},
    }
    out = await inv.call_openai_tool(openai_call)
    assert out == 'hello'
