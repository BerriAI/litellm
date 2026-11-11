# pyright: reportMissingTypeStubs=false  # apscheduler ships no type information

import asyncio
from collections.abc import Collection
from typing import Final, Protocol

from apscheduler.executors.asyncio import AsyncIOExecutor

from litellm._logging import verbose_proxy_logger
from litellm.constants import (
    SCHEDULED_JOB_SHUTDOWN_CANCEL_TIMEOUT_SECONDS,
    SCHEDULED_JOB_SHUTDOWN_FINISH_TIMEOUT_SECONDS,
)


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


async def stop_in_flight_scheduler_jobs(
    scheduler: StoppableScheduler,
    executor: AwaitableAsyncIOExecutor,
    *,
    finish_timeout_seconds: float = SCHEDULED_JOB_SHUTDOWN_FINISH_TIMEOUT_SECONDS,
    cancel_timeout_seconds: float = SCHEDULED_JOB_SHUTDOWN_CANCEL_TIMEOUT_SECONDS,
) -> None:
    """
    Let in-flight jobs finish for up to finish_timeout_seconds, then stop the scheduler and wait, bounded by
    cancel_timeout_seconds, for the jobs it cancels.

    Must run before the database is disconnected: a write job that finishes needs its connection,
    and a job's cancellation handler is what records the run's outcome.
    """
    if not scheduler.running:
        return
    in_flight: Final = executor.in_flight_jobs()
    if in_flight:
        verbose_proxy_logger.info(
            "Waiting up to %ss for %d in-flight scheduled job(s) to finish",
            finish_timeout_seconds,
            len(in_flight),
        )
    still_running: Final = (
        (await asyncio.wait(in_flight, timeout=finish_timeout_seconds))[1] if in_flight else frozenset()
    )
    scheduler.shutdown(wait=False)
    if not still_running:
        return
    verbose_proxy_logger.info("Cancelling %d in-flight scheduled job(s) for shutdown", len(still_running))
    _done, pending = await asyncio.wait(still_running, timeout=cancel_timeout_seconds)
    if pending:
        verbose_proxy_logger.warning(
            "%d scheduled job(s) did not finish within %ss of cancellation; giving up on them",
            len(pending),
            cancel_timeout_seconds,
        )
