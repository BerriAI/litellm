import asyncio
import time
from collections.abc import AsyncGenerator, Callable, Coroutine
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Final

from apscheduler.schedulers.asyncio import (  # pyright: ignore[reportMissingTypeStubs]  # APScheduler has no stubs
    AsyncIOScheduler,
)

from litellm.proxy.shutdown.scheduled_jobs import AwaitableAsyncIOExecutor, stop_in_flight_scheduler_jobs
from tests.integration._support.client import eventually
from tests.integration._support.database import read_rows


async def wait_until(predicate: Callable[[], bool], description: str) -> None:
    deadline: Final = time.monotonic() + 15
    while not predicate():
        assert time.monotonic() < deadline, description
        await asyncio.sleep(0.01)


async def wait_for_blocked_update(database_url: str, blocker_pid: int, table: str) -> None:
    await asyncio.to_thread(
        eventually,
        lambda: read_rows(
            "SELECT pid FROM pg_stat_activity WHERE datname=current_database() "
            "AND state='active' AND wait_event_type='Lock' AND %s::int=ANY(pg_blocking_pids(pid)) "
            "AND query ILIKE '%%UPDATE%%' AND query LIKE %s",
            (str(blocker_pid), f'%"{table}"%'),
            database_url=database_url,
        ),
        lambda rows: len(rows) == 1,
        seconds=15,
    )


async def wait_for_database_idle(database_url: str) -> None:
    await asyncio.to_thread(
        eventually,
        lambda: read_rows(
            "SELECT pid, state, query FROM pg_stat_activity WHERE datname=current_database() "
            "AND pid<>pg_backend_pid() AND backend_type='client backend' "
            "AND (state<>'idle' OR xact_start IS NOT NULL)",
            (),
            database_url=database_url,
        ),
        lambda rows: not rows,
        seconds=15,
    )


@dataclass(frozen=True, slots=True)
class ScheduledWrite:
    scheduler: AsyncIOScheduler
    executor: AwaitableAsyncIOExecutor
    outcome: asyncio.Future[None]

    async def cancel(self, release: Callable[[], None]) -> None:
        jobs: Final = self.executor.in_flight_jobs()
        assert len(jobs) == 1, "Expected one blocked application job before shutdown"
        shutdown: Final = asyncio.create_task(
            stop_in_flight_scheduler_jobs(
                self.scheduler, self.executor, finish_timeout_seconds=0.01, cancel_timeout_seconds=15
            )
        )
        try:
            await wait_until(
                lambda: any(isinstance(job, asyncio.Task) and job.cancelling() > 0 for job in jobs),
                "Scheduler never requested cancellation of the blocked write",
            )
        finally:
            release()
            await asyncio.wait_for(shutdown, timeout=20)


@asynccontextmanager
async def scheduled_write(operation: Callable[[], Coroutine[object, object, None]]) -> AsyncGenerator[ScheduledWrite]:
    executor: Final = AwaitableAsyncIOExecutor()
    scheduler: Final = AsyncIOScheduler(executors={"default": executor})
    outcome: Final[asyncio.Future[None]] = asyncio.get_running_loop().create_future()

    async def run() -> None:
        try:
            await operation()
        except BaseException as error:
            outcome.set_exception(error)
            raise
        else:
            outcome.set_result(None)

    scheduler.start()
    scheduler.add_job(run, "date")  # pyright: ignore[reportUnknownMemberType]  # APScheduler's job arguments lack types
    try:
        yield ScheduledWrite(scheduler, executor, outcome)
    finally:
        await stop_in_flight_scheduler_jobs(
            scheduler, executor, finish_timeout_seconds=0, cancel_timeout_seconds=15
        )
        if outcome.done() and not outcome.cancelled():
            outcome.exception()
