import asyncio
import pytest
from litellm.router import Router
from litellm.router_utils.parallel_acompletion import RouterParallelRequest

@pytest.mark.smoke
@pytest.mark.asyncio
async def test_parallel_acompletions_exceptions_and_limits(monkeypatch):
    r = Router()

    async def stub(*, model, messages, **kw):
        txt = messages[0]["content"]
        await asyncio.sleep(0)
        if "ERR" in txt:
            raise RuntimeError("boom")
        return type("R", (), {"text": f"ok:{txt}"})

    monkeypatch.setattr(r, "acompletion", stub)

    reqs = [
        RouterParallelRequest("m", [{"role": "user", "content": "A"}], {}),
        RouterParallelRequest("m", [{"role": "user", "content": "ERR"}], {}),
        RouterParallelRequest("m", [{"role": "user", "content": "B"}], {}),
    ]
    out = await r.parallel_acompletions(
        reqs, preserve_order=True, return_exceptions=True, concurrency=10
    )
    assert [o.index for o in out] == [0, 1, 2]
    assert out[0].error is None and out[2].error is None
    assert isinstance(out[1].error, Exception)

@pytest.mark.smoke
@pytest.mark.asyncio
async def test_parallel_acompletions_empty_list():
    r = Router()
    out = await r.parallel_acompletions([], preserve_order=True)
    assert out == []