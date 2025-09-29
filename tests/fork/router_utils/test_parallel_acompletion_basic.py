import asyncio
import pytest

from litellm.router_utils.parallel_acompletion import (
    RouterParallelRequest,
    gather_parallel_acompletions,
    run_parallel_requests,
)


class _StubRouter:
    def __init__(self):
        self.calls = []

    async def acompletion(self, *, model, messages, **kwargs):
        await asyncio.sleep(kwargs.get("delay", 0))
        self.calls.append((model, messages))
        return {"model": model, "text": messages[0]["content"]}


@pytest.mark.asyncio
async def test_gather_parallel_acompletions_preserves_order():
    router = _StubRouter()
    reqs = [
        RouterParallelRequest(model="a", messages=[{"role": "user", "content": "one"}]),
        RouterParallelRequest(model="b", messages=[{"role": "user", "content": "two"}]),
        RouterParallelRequest(model="c", messages=[{"role": "user", "content": "three"}]),
    ]

    results = await gather_parallel_acompletions(router, reqs, preserve_order=True)

    assert [r.index for r in results] == [0, 1, 2]
    assert [r.response["text"] for r in results] == ["one", "two", "three"]
    assert all(r.exception is None for r in results)


@pytest.mark.asyncio
async def test_run_parallel_requests_surfaces_exceptions():
    router = _StubRouter()

    async def failing_completion(*, model, messages, **kwargs):
        raise RuntimeError("boom")

    router.acompletion = failing_completion  # type: ignore[attr-defined]

    req = RouterParallelRequest(model="a", messages=[{"role": "user", "content": "fail"}])
    out = await run_parallel_requests(router, [req], preserve_order=True, return_exceptions=True)
    assert out[0][0] == 0
    assert out[0][1] is None
    assert isinstance(out[0][2], RuntimeError)


@pytest.mark.asyncio
async def test_gather_parallel_acompletions_streams_are_aggregated():
    class _StreamRouter:
        async def acompletion(self, *, model, messages, stream=False, **kwargs):
            assert stream is True

            async def _gen():
                yield {"choices": [{"delta": {"content": "hello "}}]}
                yield {"choices": [{"delta": {"content": "world"}}]}

            return _gen()

    router = _StreamRouter()
    req = RouterParallelRequest(
        model="stream-model",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
    )

    results = await gather_parallel_acompletions(router, [req], preserve_order=True)

    assert results[0].exception is None
    assert results[0].response == {"choices": [{"message": {"content": "hello world"}}]}
