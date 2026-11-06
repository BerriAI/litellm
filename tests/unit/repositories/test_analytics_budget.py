import asyncio
from collections import deque
from collections.abc import AsyncGenerator, Mapping, Sequence
from contextlib import asynccontextmanager, nullcontext
from datetime import timedelta
from typing import Final, Literal

import pytest

from litellm.repositories.analytics_budget import (
    AnalyticsCapacityError,
    AnalyticsTransaction,
    BoundedAnalyticsDatabase,
    analytics_deadline,
)


class _ClockedLoop(asyncio.SelectorEventLoop):
    _ready: deque[asyncio.Handle]
    _scheduled: list[asyncio.TimerHandle]

    def __init__(self) -> None:
        super().__init__()
        self.now = 10000.0

    def time(self) -> float:
        return self.now

    def _run_once(self) -> None:
        if not self._ready and self._scheduled:
            self.now = max(self.now, self._scheduled[0].when())
        super()._run_once()  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]  # CPython loop hook lacks stubs


class _Driver:
    def __init__(self, stage: Literal["begin", "query", "commit", "rollback", "healthy"]) -> None:
        self.stage = stage
        self.cancelled = asyncio.Event()

    async def _stall(self) -> None:
        try:
            await asyncio.Event().wait()
        finally:
            self.cancelled.set()

    @asynccontextmanager
    async def tx(self, *, max_wait: timedelta, timeout: timedelta) -> AsyncGenerator[AnalyticsTransaction]:
        if self.stage == "begin":
            await self._stall()
        try:
            yield self
        finally:
            if self.stage in ("commit", "rollback"):
                await self._stall()

    async def query_raw(self, query: str, *args: object) -> Sequence[Mapping[str, object]]:
        if query == "SELECT 1 AS result":
            if self.stage in ("query", "rollback"):
                await self._stall()
            return ({"result": 1},)
        return ({"acquired": True},)


@pytest.mark.parametrize("stage", ("begin", "query", "commit", "rollback"))
@pytest.mark.parametrize("deadline_seconds", (None, 0.25, 10.0))
def test_stalled_driver_is_cancelled_without_retaining_admission_or_tasks(
    stage: Literal["begin", "query", "commit", "rollback"], deadline_seconds: float | None
) -> None:
    async def scenario() -> None:
        loop: Final = asyncio.get_running_loop()
        existing: Final = asyncio.all_tasks()
        driver: Final = _Driver(stage)
        start: Final = loop.time()
        context: Final = nullcontext() if deadline_seconds is None else analytics_deadline(start + deadline_seconds)
        with context:
            task: Final = asyncio.create_task(BoundedAnalyticsDatabase(driver).query_raw("SELECT 1 AS result"))
        done, _ = await asyncio.wait((task,), timeout=10)
        if not done:
            task.cancel()
            watchdog: Final = loop.call_later(1, task.cancel)
            try:
                with pytest.raises(asyncio.CancelledError):
                    await task
            finally:
                watchdog.cancel()
        assert done, "Analytics retained its task after the query and cleanup budget"
        with pytest.raises(AnalyticsCapacityError):
            await task
        assert driver.cancelled.is_set()
        assert loop.time() - start <= (7.25 if deadline_seconds != 0.25 else 1.25)
        healthy: Final = BoundedAnalyticsDatabase(_Driver("healthy"))
        assert await asyncio.gather(
            healthy.query_raw("SELECT 1 AS result"), healthy.query_raw("SELECT 1 AS result")
        ) == [({"result": 1},), ({"result": 1},)]
        assert asyncio.all_tasks() == existing

    loop: Final = _ClockedLoop()
    try:
        loop.run_until_complete(scenario())
    finally:
        loop.close()
