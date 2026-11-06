import asyncio
import heapq
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
        while self._scheduled and self._scheduled[0].cancelled():
            heapq.heappop(self._scheduled)
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
    async def tx(self, *, max_wait: timedelta, timeout: timedelta) -> AsyncGenerator[AnalyticsTransaction, None]:
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


class _Process:
    pid: object = None


class _HeldEngine:
    def __init__(self, stage: Literal["begin", "query", "commit", "rollback"], *, failure: bool = False) -> None:
        self.stage = stage
        self.failure = failure
        self.entered = asyncio.Event()
        self.released = asyncio.Event()
        self.rollback_entered = asyncio.Event()
        self.rollback_released = asyncio.Event()
        self.process = _Process()
        self.calls = 0
        self.starts = 0

    async def _request(self, stage: str) -> None:
        self.calls += 1
        if stage == self.stage:
            self.entered.set()
            await self.released.wait()
            if self.failure:
                raise RuntimeError("synthetic engine error")

    async def query(self, content: str, *, tx_id: str | None) -> object:
        if content == "SELECT 1 AS result":
            await self._request("query")
        return {}

    async def start_transaction(self, *, content: str) -> str:
        self.starts += 1
        identifier: Final = f"synthetic-transaction-{self.starts}"
        await self._request("begin")
        return identifier

    async def commit_transaction(self, tx_id: str) -> None:
        await self._request("commit")

    async def rollback_transaction(self, tx_id: str) -> None:
        await self._request("rollback")
        if self.stage == "begin":
            self.rollback_entered.set()
            await self.rollback_released.wait()


class _TrackedDriver:
    def __init__(self, engine: _HeldEngine) -> None:
        from litellm.proxy.db.prisma_client import (
            _PrismaDrainTracker,  # pyright: ignore[reportPrivateUsage]  # exercise real cancellation ownership
            _TrackedPrismaEngine,  # pyright: ignore[reportPrivateUsage]  # exercise real cancellation ownership
        )

        self.stage = engine.stage
        self.tracker = _PrismaDrainTracker()
        self.engine = _TrackedPrismaEngine(engine, self.tracker)

    @asynccontextmanager
    async def tx(self, *, max_wait: timedelta, timeout: timedelta) -> AsyncGenerator[AnalyticsTransaction, None]:
        identifier: Final = await self.engine.start_transaction(content="{}")
        try:
            yield self
            if self.stage == "rollback":
                raise RuntimeError("synthetic query failure")
        except BaseException:
            await self.engine.rollback_transaction(identifier)
            raise
        else:
            await self.engine.commit_transaction(identifier)

    async def query_raw(self, query: str, *args: object) -> Sequence[Mapping[str, object]]:
        from litellm.proxy.db.log_db_metrics import claim_db_io

        with claim_db_io():
            await self.engine.query(query, tx_id="synthetic-transaction")
        return ({"result": 1},) if query == "SELECT 1 AS result" else ({"acquired": True},)


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ("begin", "query", "commit", "rollback"))
@pytest.mark.parametrize("failure", (False, True))
async def test_cancelled_analytics_keeps_admission_until_detached_engine_work_and_rollback_finish(
    stage: Literal["begin", "query", "commit", "rollback"],
    failure: bool,
) -> None:
    engines: Final = (_HeldEngine(stage, failure=failure), _HeldEngine(stage, failure=failure))
    drivers: Final = tuple(_TrackedDriver(engine) for engine in engines)
    tasks: Final = tuple(
        asyncio.create_task(BoundedAnalyticsDatabase(driver).query_raw("SELECT 1 AS result")) for driver in drivers
    )
    try:
        await asyncio.gather(*(engine.entered.wait() for engine in engines))
        for task in tasks:
            task.cancel()
        for task in tasks:
            with pytest.raises(asyncio.CancelledError):
                await task
        calls: Final = tuple(engine.calls for engine in engines)
        for _ in range(64):
            with pytest.raises(AnalyticsCapacityError):
                await BoundedAnalyticsDatabase(drivers[0]).query_raw("SELECT 1 AS result")
            assert tuple(engine.calls for engine in engines) == calls
        assert tuple(len(driver.engine._detached) for driver in drivers) == (1, 1)  # pyright: ignore[reportPrivateUsage]  # bounded retained engine tasks are the behavior under test
        for engine in engines:
            engine.released.set()
        if stage == "begin" and not failure:
            await asyncio.gather(*(engine.rollback_entered.wait() for engine in engines))
            with pytest.raises(AnalyticsCapacityError):
                await BoundedAnalyticsDatabase(drivers[0]).query_raw("SELECT 1 AS result")
    finally:
        for engine in engines:
            engine.released.set()
            engine.rollback_released.set()
        await asyncio.gather(*(driver.tracker.wait_until_drained() for driver in drivers))
        await asyncio.sleep(0)
        await asyncio.sleep(0)
    assert not any(driver.engine._detached for driver in drivers)  # pyright: ignore[reportPrivateUsage]  # verify detached engine tasks drained
    healthy: Final = BoundedAnalyticsDatabase(_Driver("healthy"))
    assert await asyncio.gather(healthy.query_raw("SELECT 1 AS result"), healthy.query_raw("SELECT 1 AS result")) == [
        ({"result": 1},),
        ({"result": 1},),
    ]
