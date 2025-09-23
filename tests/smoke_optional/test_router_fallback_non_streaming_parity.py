import pytest
from litellm.router import Router

class _Resp:
    def __init__(self, text: str): self.choices=[type("C",(),{"message":type("M",(),{"content":text})()})()]

@pytest.mark.smoke
@pytest.mark.asyncio
async def test_router_fallback_non_streaming_parity(monkeypatch):
    r = Router(model_list=[{"model_name":"m","litellm_params":{"model":"openai/gpt-4o-mini","api_key":"sk"}}])

    calls = {"primary":0,"fb":0}
    async def primary(**kw):
        calls["primary"]+=1
        raise RuntimeError("primary boom")
    async def fb(*a,**k):
        calls["fb"]+=1
        return _Resp("ok-fallback")

    import litellm
    monkeypatch.setattr(litellm,"acompletion",primary)
    monkeypatch.setattr(Router,"async_function_with_fallbacks_common_utils",staticmethod(lambda *a,**k: fb()))

    out = await r.acompletion(model="m", messages=[{"role":"user","content":"hi"}], stream=False)
    try:
        txt = out.choices[0].message.content
    except Exception:
        txt = out.get("choices",[{}])[0].get("message",{}).get("content")
    assert txt == "ok-fallback"
    assert calls["primary"]>=1 and calls["fb"]==1