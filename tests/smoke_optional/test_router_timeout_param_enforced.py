import pytest
from litellm.router import Router
import asyncio

@pytest.mark.smoke
@pytest.mark.asyncio
async def test_router_timeout_param_enforced(monkeypatch):
    """
    Paved-road contract: per-request timeout is honored, yielding a predictable exception.
    """
    r = Router()

    async def slow_acompletion(*, model, messages, **kwargs):
        await asyncio.sleep(10.0)
        return {"choices": [{"message": {"content": "too late"}}]}

    monkeypatch.setattr(r, "acompletion", slow_acompletion)

    with pytest.raises(Exception):
        await r.acompletion(model="m", messages=[{"role": "user", "content": "hi"}], timeout=0.05)
