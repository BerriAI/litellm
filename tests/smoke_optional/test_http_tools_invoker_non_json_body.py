import types, pytest

@pytest.mark.smoke
@pytest.mark.asyncio
async def test_http_tools_invoker_non_json_body():
    from litellm.experimental_mcp_client.mini_agent import http_tools_invoker as inv_mod
    class _Client:
        async def __aenter__(self): return self
        async def __aexit__(self, *_): return False
        async def get(self, _url): return type("R", (), {"json": lambda _ : [], "raise_for_status": lambda _ : None})()
        async def post(self, _url, _json=None):
            return type("R", (), {
                "raise_for_status": lambda _ : None,
                "json": lambda _ : (_ for _ in ()).throw(ValueError("not json")),
                "text": "plain text body",
                "status_code": 200
            })()
    inv_mod.httpx = types.SimpleNamespace(AsyncClient=_Client)
    inv = inv_mod.HttpToolsInvoker("http://fake")
    out = await inv.call_openai_tool({"function": {"name": "echo", "arguments": '{"text":"x"}'}})
    assert isinstance(out, str)