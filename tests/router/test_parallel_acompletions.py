import os
import asyncio
import pytest

# Ensure flag ON for these tests
os.environ["LITELLM_ENABLE_PARALLEL_ACOMPLETIONS"] = "1"

from litellm import Router  # noqa: E402
from litellm.experimental_flags import ENABLE_PARALLEL_ACOMPLETIONS  # noqa: E402
from litellm.router_utils.parallel_acompletion import (  # noqa: E402
    RouterParallelRequest,
)


@pytest.mark.asyncio
async def test_parallel_acompletions_basic(monkeypatch):
    assert ENABLE_PARALLEL_ACOMPLETIONS is True

    # Build a dummy router with a single pseudo deployment (use a harmless model alias)
    router = Router(
        model_list=[
            {
                "model_name": "dummy",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",  # replaced at runtime with your own credentials in real use
                    "api_key": "sk-FAKE",
                },
            }
        ]
    )

    # Monkeypatch router.acompletion to avoid real network
    async def fake_acompletion(model, messages, **kwargs):
        return {"model": model, "content": messages[-1]["content"].upper()}

    monkeypatch.setattr(router, "acompletion", fake_acompletion)

    requests = [
        RouterParallelRequest(
            model="dummy", messages=[{"role": "user", "content": "a"}]
        ),
        RouterParallelRequest(
            model="dummy", messages=[{"role": "user", "content": "b"}]
        ),
        RouterParallelRequest(
            model="dummy", messages=[{"role": "user", "content": "c"}]
        ),
    ]

    results = await router.parallel_acompletions(
        requests, concurrency=2, preserve_order=True
    )
    assert len(results) == 3
    assert all(r.exception is None for r in results)
    # Verify results map back to original requests
    for i, r in enumerate(results):
        assert r.index == i
        assert r.request.model == requests[i].model
        assert r.request.messages == requests[i].messages
    payloads = [r.response for r in results]
    assert [p["content"] for p in payloads] == ["A", "B", "C"]


@pytest.mark.asyncio
async def test_iter_parallel_acompletions(monkeypatch):
    router = Router(
        model_list=[
            {
                "model_name": "dummy2",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": "sk-FAKE2",
                },
            }
        ]
    )

    async def fake_acompletion(model, messages, **kwargs):
        await asyncio.sleep(0.01)
        return {"ok": True, "last": messages[-1]["content"]}

    monkeypatch.setattr(router, "acompletion", fake_acompletion)

    requests = [
        RouterParallelRequest(
            model="dummy2", messages=[{"role": "user", "content": "x"}]
        ),
        RouterParallelRequest(
            model="dummy2", messages=[{"role": "user", "content": "y"}]
        ),
    ]

    seen = []
    idx_seen = []
    async for result in router.iter_parallel_acompletions(requests, concurrency=2):
        assert result.exception is None
        # Validate mapping via index and attached request
        assert result.index in (0, 1)
        expected = {0: "x", 1: "y"}[result.index]
        assert result.request.messages[0]["content"] == expected
        idx_seen.append(result.index)
        seen.append(result.response["last"])
    assert set(seen) == {"x", "y"}
    assert set(idx_seen) == {0, 1}


@pytest.mark.asyncio
async def test_iter_parallel_acompletions_fail_fast(monkeypatch):
    router = Router(
        model_list=[
            {
                "model_name": "dummy3",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": "sk-FAKE3",
                },
            }
        ]
    )

    async def failing_acompletion(model, messages, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(router, "acompletion", failing_acompletion)

    requests = [
        RouterParallelRequest(
            model="dummy3", messages=[{"role": "user", "content": "x"}]
        ),
        RouterParallelRequest(
            model="dummy3", messages=[{"role": "user", "content": "y"}]
        ),
    ]

    with pytest.raises(RuntimeError):
        async for _ in router.iter_parallel_acompletions(
            requests, concurrency=2, return_exceptions=False
        ):
            pass


@pytest.mark.asyncio
async def test_flag_disabled_raises(monkeypatch):
    # Simulate flag off
    monkeypatch.delenv("LITELLM_ENABLE_PARALLEL_ACOMPLETIONS", raising=False)
    # Reload module not strictly needed if we rely on runtime check of env variable before instantiation,
    # but router methods raise based on ENABLE_PARALLEL_ACOMPLETIONS value (import-time).
    from importlib import reload
    import litellm.experimental_flags as flags

    reload(flags)

    # Reload router to re-evaluate imported flag at definition time
    import litellm.router as router_module

    reload(router_module)
    from litellm import Router

    router = Router(model_list=[])

    with pytest.raises(RuntimeError):
        # We pass empty list; early runtime flag check should still raise
        await router.parallel_acompletions([])  # type: ignore
