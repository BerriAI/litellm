import types
import pytest


@pytest.mark.smoke
@pytest.mark.asyncio
async def test_http_tools_invoker_error_tail(monkeypatch):
    # Patch httpx client used by HttpToolsInvoker to simulate 500 with body
    from litellm.experimental_mcp_client.mini_agent import http_tools_invoker as inv_mod

    class _Resp:
        def __init__(self, status_code=500, text='INTERNAL ERROR: something bad\nDetails...'):
            self.status_code = status_code
            self._text = text

        def json(self):
            return {}

        def raise_for_status(self):
            raise Exception("boom")

        @property
        def text(self):
            return self._text

    class _Client:
        def __init__(self, *a, **k):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, *a):
            return False
        async def get(self, url):
            return _Resp(200, '[]')
        async def post(self, url, json=None):
            return _Resp(500, 'INTERNAL ERROR: details lorem ipsum dolor sit amet')

    inv_mod.httpx = types.SimpleNamespace(AsyncClient=_Client)  # type: ignore
    inv = inv_mod.HttpToolsInvoker("http://fake")

    call = {
        "id": "tc1",
        "type": "function",
        "function": {"name": "x", "arguments": "{}"},
    }
    with pytest.raises(Exception) as ei:
        await inv.call_openai_tool(call)
    msg = str(ei.value)
    assert '500' in msg or 'INTERNAL' in msg or 'error' in msg.lower()

