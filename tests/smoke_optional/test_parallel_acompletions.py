import asyncio
import pytest

from litellm import Router
from litellm.router_utils.parallel_acompletion import RouterParallelRequest


@pytest.mark.asyncio
async def test_parallel_acompletions_preserve_order_true(monkeypatch):
    """
    Validate minimal fan-out behavior:
    - preserve_order=True yields outputs sorted by original index
    - per-request responses are returned alongside index, error=None
    This uses a stubbed Router.acompletion to avoid any network/provider keys.
    """
    router = Router(model_list=[])  # no deployments needed; we stub acompletion
    calls = []

    async def stub_acompletion(model, messages, **kwargs):
        calls.append(messages[0]["content"])
        await asyncio.sleep(0)  # yield once for concurrency scheduling
        class R:
            def __init__(self, text): self.text = text
        return R(f"ok:{messages[0]['content']}")

    monkeypatch.setattr(router, "acompletion", stub_acompletion)

    requests = [
        RouterParallelRequest(model="any", messages=[{"role": "user", "content": "A"}], kwargs={}),
        RouterParallelRequest(model="any", messages=[{"role": "user", "content": "B"}], kwargs={}),
    ]

    out = await router.parallel_acompletions(requests, preserve_order=True, return_exceptions=True)
    assert len(out) == 2
    # parallel_acompletions returns small objects with attributes: index, response, error
    assert [o.index for o in out] == [0, 1]
    assert [getattr(o.response, "text", "") for o in out] == ["ok:A", "ok:B"]
    assert all(o.error is None for o in out)
    assert calls == ["A", "B"]


@pytest.mark.asyncio
async def test_parallel_acompletions_preserve_order_false(monkeypatch):
    """
    Validate ordering by arrival when preserve_order=False.
    We only assert set equality on responses; arrival order may vary.
    """
    router = Router(model_list=[])

    async def stub_acompletion(model, messages, **kwargs):
        await asyncio.sleep(0)
        return messages[0]["content"]

    monkeypatch.setattr(router, "acompletion", stub_acompletion)

    requests = [
        RouterParallelRequest(model="any", messages=[{"role": "user", "content": "X"}], kwargs={}),
        RouterParallelRequest(model="any", messages=[{"role": "user", "content": "Y"}], kwargs={}),
        RouterParallelRequest(model="any", messages=[{"role": "user", "content": "Z"}], kwargs={}),
    ]

    out = await router.parallel_acompletions(requests, preserve_order=False, return_exceptions=True)
    assert len(out) == 3
    assert set(o.response for o in out) == {"X", "Y", "Z"}
    assert all(o.error is None for o in out)