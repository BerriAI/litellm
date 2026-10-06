import asyncio
from queue import SimpleQueue
from typing import Final

import pytest

from litellm.proxy.lens.activity import observe_operation, observed_model, track_activity
from litellm.proxy.lens.models import Activity, Coverage, InFlight, ModelRequest, ModelResult, Review, ToolCount


@pytest.mark.asyncio
async def test_concurrent_operations_keep_the_remaining_tool_visible_and_preserve_completed_counts() -> None:
    reports: Final = SimpleQueue[Activity]()
    python_started: Final = asyncio.Event()
    read_finished: Final = asyncio.Event()

    async def progress(
        stage: str | None,
        coverage: Coverage | None,
        review: Review | None = None,
        reading: tuple[InFlight, ...] | None = None,
        activity: Activity | None = None,
        /,
    ) -> None:
        assert (stage, coverage, review, reading) == (None, None, None, None)
        assert activity is not None
        reports.put(activity)

    async with track_activity(
        progress, identity="review:one", phase="review", label="Review", execution_ids=("one",)
    ) as tracker:

        async def read() -> None:
            async with observe_operation(tracker, "read"):
                await python_started.wait()
            read_finished.set()

        async def python() -> None:
            async with observe_operation(tracker, "python"):
                python_started.set()
                await read_finished.wait()
                assert tracker.activity.operations == ("python",)

        await asyncio.wait_for(asyncio.gather(read(), python()), timeout=1)
        assert tracker.activity.operations == ()
        assert tracker.activity.tool_calls == (ToolCount(name="read", calls=1), ToolCount(name="python", calls=1))
        async with observe_operation(tracker, "read"):
            assert tracker.activity.operations == ("read",)
        assert frozenset(tracker.activity.tool_calls) == frozenset(
            (ToolCount(name="read", calls=2), ToolCount(name="python", calls=1))
        )

    events: Final = tuple(reports.get_nowait() for _ in range(reports.qsize()))
    assert events[0].operations == () and not events[0].finished
    assert any(event.operations == ("read", "python") for event in events)
    assert events[-1].finished and events[-1].operations == ()
    assert events[-1].tool_calls == tracker.activity.tool_calls


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", (False, True))
async def test_model_error_or_cancellation_finishes_activity_without_exposing_prompt_or_response(cancel: bool) -> None:
    reports: Final = SimpleQueue[Activity]()
    entered: Final = asyncio.Event()
    release: Final = asyncio.Event()
    request: Final = ModelRequest(prompt="private trace payload", purpose="extract")

    async def progress(
        _stage: str | None,
        _coverage: Coverage | None,
        _review: Review | None = None,
        _reading: tuple[InFlight, ...] | None = None,
        activity: Activity | None = None,
        /,
    ) -> None:
        assert activity is not None
        reports.put(activity)

    async def model(body: ModelRequest) -> ModelResult:
        assert body is request
        entered.set()
        await release.wait()
        raise ValueError("private model diagnostic")

    async def work() -> None:
        async with track_activity(
            progress, identity="candidate:one", phase="investigate", label="Check candidate", execution_ids=("one",)
        ) as tracker:
            await observed_model(model, tracker)(request)

    task: Final = asyncio.create_task(work())
    await asyncio.wait_for(entered.wait(), timeout=1)
    if cancel:
        task.cancel()
    else:
        release.set()
    with pytest.raises(asyncio.CancelledError if cancel else ValueError):
        await task
    events: Final = tuple(reports.get_nowait() for _ in range(reports.qsize()))
    assert any(event.operations == ("model",) for event in events)
    assert events[-1].finished and events[-1].operations == ()
    assert all(event.tool_calls == () and "private" not in event.model_dump_json() for event in events)
