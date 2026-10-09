from collections.abc import Iterator, Mapping
from types import MappingProxyType
from typing import Final, Protocol

import pytest
from apscheduler.schedulers.asyncio import (  # pyright: ignore[reportMissingTypeStubs]  # APScheduler has no stubs
    AsyncIOScheduler,
)

from litellm.proxy.data_manager.spend_log_cleanup_job import (
    SPEND_LOG_CLEANUP_JOB_ID,
    SpendLogCleanupJob,
    SpendLogCleanupScheduler,
    schedule_spend_log_cleanup,
)
from litellm.proxy.utils import PrismaClient


class _TestScheduler(SpendLogCleanupScheduler, Protocol):
    def start(self, paused: bool = False) -> None: ...

    def shutdown(self, wait: bool = True) -> None: ...


@pytest.fixture
def scheduler() -> Iterator[_TestScheduler]:
    scheduler: Final[_TestScheduler] = AsyncIOScheduler()
    yield scheduler


def _settings(**values: object) -> Mapping[str, object]:
    return MappingProxyType({"maximum_spend_logs_retention_period": "30d", **values})


def _prisma_client() -> PrismaClient:
    return object.__new__(PrismaClient)


@pytest.mark.asyncio
async def test_stopped_scheduler_keeps_the_configured_interval_without_jitter(scheduler: _TestScheduler) -> None:
    schedule_spend_log_cleanup(scheduler, _settings(maximum_spend_logs_retention_interval="1m"), _prisma_client())
    scheduler.start(paused=True)

    job: Final[SpendLogCleanupJob | None] = scheduler.get_job(SPEND_LOG_CLEANUP_JOB_ID)
    assert job is not None
    assert str(job.trigger) == "interval[0:01:00]"


@pytest.mark.asyncio
async def test_running_scheduler_applies_the_configured_stagger_offset(scheduler: _TestScheduler) -> None:
    scheduler.start(paused=True)
    settings: Final = _settings(
        maximum_spend_logs_retention_interval="1m",
        scheduled_job_stagger={"offsets": {SPEND_LOG_CLEANUP_JOB_ID: 17}},
    )

    schedule_spend_log_cleanup(scheduler, settings, _prisma_client())

    job: Final[SpendLogCleanupJob | None] = scheduler.get_job(SPEND_LOG_CLEANUP_JOB_ID)
    assert job is not None
    assert str(job.trigger) == "interval[0:01:00][+17s]"


@pytest.mark.asyncio
async def test_running_scheduler_respects_disabled_stagger(scheduler: _TestScheduler) -> None:
    scheduler.start(paused=True)
    settings: Final = _settings(
        maximum_spend_logs_retention_interval="1m",
        scheduled_job_stagger={"enabled": False},
    )

    schedule_spend_log_cleanup(scheduler, settings, _prisma_client())

    job: Final[SpendLogCleanupJob | None] = scheduler.get_job(SPEND_LOG_CLEANUP_JOB_ID)
    assert job is not None
    assert str(job.trigger) == "interval[0:01:00]"


@pytest.mark.asyncio
async def test_cron_is_not_staggered_when_scheduler_is_running(scheduler: _TestScheduler) -> None:
    scheduler.start(paused=True)
    settings: Final = _settings(
        maximum_spend_logs_cleanup_cron="*/2 * * * *",
        scheduled_job_stagger={"offsets": {SPEND_LOG_CLEANUP_JOB_ID: 17}},
    )

    schedule_spend_log_cleanup(scheduler, settings, _prisma_client())

    job: Final[SpendLogCleanupJob | None] = scheduler.get_job(SPEND_LOG_CLEANUP_JOB_ID)
    assert job is not None
    assert str(job.trigger).startswith("cron[")
    assert "[+" not in str(job.trigger)


@pytest.mark.parametrize(
    "settings",
    [
        _settings(maximum_spend_logs_cleanup_cron="not a cron"),
        _settings(maximum_spend_logs_retention_interval=60),
    ],
)
@pytest.mark.asyncio
async def test_invalid_schedule_does_not_register_a_job(
    scheduler: _TestScheduler, settings: Mapping[str, object]
) -> None:
    scheduler.start(paused=True)

    schedule_spend_log_cleanup(scheduler, settings, _prisma_client())

    assert scheduler.get_job(SPEND_LOG_CLEANUP_JOB_ID) is None


@pytest.mark.asyncio
async def test_removing_all_retention_settings_removes_the_existing_job(scheduler: _TestScheduler) -> None:
    scheduler.start(paused=True)
    prisma_client: Final = _prisma_client()
    schedule_spend_log_cleanup(
        scheduler,
        _settings(maximum_spend_logs_retention_interval="1m"),
        prisma_client,
    )
    assert scheduler.get_job(SPEND_LOG_CLEANUP_JOB_ID) is not None

    schedule_spend_log_cleanup(scheduler, MappingProxyType({}), prisma_client)

    assert scheduler.get_job(SPEND_LOG_CLEANUP_JOB_ID) is None
