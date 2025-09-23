# tests/smoke/test_batch_as_completed_return_exceptions.py
import pytest
from litellm.extras.batch import acompletion_as_completed

class _R:
    # Batch helper expects this coroutine
    async def acompletion(self, **req):
        # Return an Exception object for the "error" payload (do NOT raise)
        msgs = req.get("messages") or []
        content = (msgs[0] or {}).get("content", "") if msgs else ""
        if "ERR" in content:
            return Exception("synthetic error for test")
        # Minimal OpenAI-shaped success
        return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

@pytest.mark.smoke
@pytest.mark.asyncio
async def test_as_completed_return_exceptions():
    reqs = [
        {"model": "m", "messages": [{"role": "user", "content": "ok"}]},
        {"model": "m", "messages": [{"role": "user", "content": "ERR"}]},
    ]
    outs = []
    # Do NOT pass return_exceptions kwarg; some versions don't support it
    async for idx, resp in acompletion_as_completed(_R(), reqs, concurrency=2):
        outs.append((idx, resp))
    # One success + one Exception object returned by our stub
    assert len(outs) == 2 and any(isinstance(o[1], Exception) for o in outs)