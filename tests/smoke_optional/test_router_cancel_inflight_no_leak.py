import pytest
from litellm.router import Router
import asyncio

@pytest.mark.smoke
@pytest.mark.asyncio
async def test_router_cancel_inflight_no_leak(monkeypatch):
    """
    Contract: closing during inflight calls should not crash or hang.
    """
    r = Router()

    started = asyncio.Event()
    async def never_returns(*, model, messages, **kwargs):
        started.set()
        while True:
            await asyncio.sleep(0.1)

    monkeypatch.setattr(r, "acompletion", never_returns)

    task = asyncio.create_task(r.acompletion(model="m", messages=[{"role": "user", "content": "hi"}]))
    await started.wait()
    try:
        await asyncio.wait_for(r.aclose(), timeout=0.5)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
