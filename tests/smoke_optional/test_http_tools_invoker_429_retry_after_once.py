import pytest, types, time

@pytest.mark.smoke
@pytest.mark.asyncio
async def test_http_tools_invoker_429_retry_after_once(monkeypatch):
    """
    One polite retry on 429 with Retry-After header.
    """
    from litellm.experimental_mcp_client.mini_agent import http_tools_invoker as inv_mod

    calls = {"post": 0}
    tmarks = []

    class _Resp429:
        status_code = 429
        text = "rate limited"
        def json(self): return {}
        def raise_for_status(self): raise Exception("429")

    class _RespOK:
        status_code = 200
        text = ""
        def json(self): return {"text": "ok"}
        def raise_for_status(self): return None

    class _Client:
        def __init__(self, *a, **k): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def get(self, url): return _RespOK()
        async def post(self, url, json=None):
            calls["post"] += 1
            tmarks.append(time.time())
            if calls["post"] == 1:
                resp = _Resp429()
                resp.headers = {"Retry-After": "0"}
                return resp
            return _RespOK()

    inv_mod.httpx = types.SimpleNamespace(AsyncClient=_Client)
    inv = inv_mod.HttpToolsInvoker("http://fake")
    out = await inv.call_openai_tool({"function": {"name": "echo", "arguments": "{}"}})
    assert out == "ok"
    assert calls["post"] == 2
