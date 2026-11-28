import asyncio
from typing import Final, Literal

import pytest
from prometheus_client import CollectorRegistry, Histogram

from litellm.litellm_core_utils import callback_timing


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ("success", "failure", "cancelled"))
async def test_callback_timing_observes_once_when_hook_exits(
    monkeypatch: pytest.MonkeyPatch,
    outcome: Literal["success", "failure", "cancelled"],
) -> None:
    registry: Final = CollectorRegistry()
    histogram: Final = Histogram(
        "litellm_callback_execution_seconds",
        "Callback hook duration",
        ("callback_name", "hook", "execution_path"),
        registry=registry,
    )
    labels: Final = {
        "callback_name": "callbacks.sink",
        "hook": "async_log_success_event",
        "execution_path": "background",
    }
    monkeypatch.setattr(callback_timing, "_callback_timing_enabled", lambda: True)
    monkeypatch.setattr(callback_timing, "_load_histogram", lambda: histogram)
    inside: Final = asyncio.Event()
    release: Final = asyncio.Event()
    failure: Final = RuntimeError("callback failed")

    async def run_hook() -> str:
        with callback_timing.measure_callback_execution(labels["callback_name"], labels["hook"], "background"):
            inside.set()
            await release.wait()
            if outcome == "failure":
                raise failure
            return "callback result"

    task: Final = asyncio.create_task(run_hook())
    try:
        await asyncio.wait_for(inside.wait(), timeout=5)
        assert registry.get_sample_value("litellm_callback_execution_seconds_count", labels) is None
        if outcome == "cancelled":
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        elif outcome == "failure":
            release.set()
            with pytest.raises(RuntimeError) as caught:
                await task
            assert caught.value is failure
        else:
            release.set()
            assert await task == "callback result"
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    assert registry.get_sample_value("litellm_callback_execution_seconds_count", labels) == 1
