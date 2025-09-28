"""
HTTP Tools Invoker contract (monkeypatched httpx).

Why this matters:
- Validates the adapter lists tools and executes a function call via /invoke.
- No real HTTP server—uses a fake AsyncClient to keep it deterministic.
"""

import json as _json
import pytest

from litellm.experimental_mcp_client.mini_agent.http_tools_invoker import HttpToolsInvoker

# File-level marks
pytestmark = [pytest.mark.mini_agent]


class _Resp:
    def __init__(self, obj, status=200):
        self._obj = obj
        self.status_code = status
        self.headers = {}
        self.text = _json.dumps(obj)

    def json(self):
        return self._obj

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
                    }
                ]
            )
        return _Resp({}, 404)

    async def post(self, url, body=None, json=None):
        # Support either kw: body or json
        payload = body if body is not None else (json or {})
        if url.endswith("/invoke"):
            if payload.get("name") == "echo":
                try:
                    args = payload.get("arguments", "{}")
                    if not isinstance(args, str):
                        args = _json.dumps(args)
                    text = _json.loads(args).get("text", "")
                except Exception:
                    text = ""
                return _Resp({"text": text})
            return _Resp({"error": "tool_not_found"}, 404)
        return _Resp({}, 404)


@pytest.mark.asyncio
async def test_http_tools_invoker_contract(monkeypatch):
    import httpx

    # Monkeypatch AsyncClient with deterministic fake
    monkeypatch.setattr(httpx, "AsyncClient", _FakeClient)

    inv = HttpToolsInvoker("http://fake")
    tools = await inv.list_openai_tools()
    names = [t.get("function", {}).get("name") for t in tools]
    assert "echo" in names

    call = {
        "id": "1",
        "type": "function",
        "function": {"name": "echo", "arguments": _json.dumps({"text": "hello"})},
    }
    out = await inv.call_openai_tool(call)
    assert out == "hello"
