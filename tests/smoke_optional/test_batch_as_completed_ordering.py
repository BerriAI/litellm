import asyncio
import pytest


@pytest.mark.smoke
@pytest.mark.asyncio
async def test_batch_as_completed_ordering():
    from litellm.extras.batch import acompletion_as_completed

    class R:
        async def acompletion(self, *, model, messages, **kw):
            txt = messages[0]["content"]
            if "slow" in txt:
                await asyncio.sleep(0.05)
                return {"choices": [{"message": {"content": "S"}}]}
            await asyncio.sleep(0.01)
            return {"choices": [{"message": {"content": "F"}}]}

    async def run():
        reqs = [
            {"model": "m", "messages": [{"role": "user", "content": "slow"}]},
            {"model": "m", "messages": [{"role": "user", "content": "fast"}]},
        ]
        outs = []
        async for i, resp in acompletion_as_completed(R(), reqs, concurrency=2):
            outs.append((i, resp["choices"][0]["message"]["content"]))
        return outs

    outs = await run()
    # second finishes first
    assert outs[0][0] == 1
    assert outs[1][0] == 0

