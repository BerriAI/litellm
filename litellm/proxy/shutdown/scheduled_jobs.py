# pyright: reportMissingTypeStubs=false  # apscheduler ships no type information

import asyncio
from collections.abc import Collection
from typing import Final, Protocol

from apscheduler.executors.asyncio import AsyncIOExecutor

from litellm._logging import verbose_proxy_logger

JOB_CANCEL_TIMEOUT_SECONDS: Final = 5.0


class StoppableScheduler(Protocol):
    """The slice of ``AsyncIOScheduler`` shutdown uses, which ships no type information"""

    @property
    def running(self) -> bool: ...

    def pause(self) -> None: ...

    def shutdown(self, wait: bool = ...) -> None: ...


class AwaitableAsyncIOExecutor(AsyncIOExecutor):  # pyright: ignore[reportUntypedBaseClass]  # apscheduler ships no type information and is absent from the type-check env
    """``AsyncIOExecutor`` whose in-flight job tasks can be awaited after ``shutdown`` cancels them"""

    _pending_futures: Collection["asyncio.Future[object]"]

    def in_flight_jobs(self) -> tuple["asyncio.Future[object]", ...]:
        """The job tasks that are running right now, as a snapshot"""
        return tuple(future for future in self._pending_futures if not future.done())


def pause_scheduled_jobs(scheduler: StoppableScheduler) -> None:
    """Stop the scheduler from starting jobs that shutdown would only cancel; running jobs continue"""
    if scheduler.running:
        scheduler.pause()


async def cancel_in_flight_scheduler_jobs(scheduler: StoppableScheduler, executor: AwaitableAsyncIOExecutor) -> None:
    """
    Stop the scheduler and wait, bounded by JOB_CANCEL_TIMEOUT_SECONDS, for the jobs it cancels.

    Must run before the database is disconnected: a job's cancellation handler is what records
    the run's outcome, and it needs the connection the job was using.
    """
    if not scheduler.running:
        return
    in_flight: Final = executor.in_flight_jobs()
    scheduler.shutdown(wait=False)
    if not in_flight:
        return
    verbose_proxy_logger.info("Cancelling %d in-flight scheduled job(s) for shutdown", len(in_flight))
    _done, pending = await asyncio.wait(in_flight, timeout=JOB_CANCEL_TIMEOUT_SECONDS)
    if pending:
        verbose_proxy_logger.warning(
            "%d scheduled job(s) did not finish within %ss of cancellation; giving up on them",
            len(pending),
            JOB_CANCEL_TIMEOUT_SECONDS,
        )
