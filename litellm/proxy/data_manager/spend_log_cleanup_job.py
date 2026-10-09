from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Final, Protocol

from apscheduler.triggers.cron import (  # pyright: ignore[reportMissingTypeStubs]  # APScheduler has no stubs
    CronTrigger,  # pyright: ignore[reportUnknownVariableType]  # APScheduler has no stubs
)
from apscheduler.triggers.interval import (  # pyright: ignore[reportMissingTypeStubs]  # APScheduler has no stubs
    IntervalTrigger,  # pyright: ignore[reportUnknownVariableType]  # APScheduler has no stubs
)

from litellm._logging import verbose_proxy_logger
from litellm.constants import APSCHEDULER_MISFIRE_GRACE_TIME
from litellm.litellm_core_utils.duration_parser import duration_in_seconds
from litellm.proxy.common_utils.scheduled_job_stagger import Trigger, parse_stagger_settings, stagger_trigger
from litellm.proxy.db.db_transaction_queue.spend_log_cleanup import SpendLogCleanup
from litellm.proxy.utils import PrismaClient

SPEND_LOG_CLEANUP_JOB_ID: Final = "spend_log_cleanup_job"


class SpendLogCleanupJob(Protocol):
    @property
    def trigger(self) -> Trigger: ...


class SpendLogCleanupScheduler(Protocol):
    @property
    def running(self) -> bool: ...

    def get_job(self, job_id: str) -> SpendLogCleanupJob | None: ...

    def remove_job(self, job_id: str) -> None: ...

    def add_job(
        self,
        func: Callable[[PrismaClient], Awaitable[None]],
        trigger: Trigger,
        *,
        args: Sequence[object],
        id: str,
        replace_existing: bool,
        misfire_grace_time: int,
    ) -> object: ...


RETENTION_SETTING_KEYS: Final = (
    "maximum_spend_logs_retention_period",
    "maximum_autorouter_session_retention_period",
    "maximum_health_check_retention_period",
    "maximum_daily_tag_spend_retention_period",
)

CLEANUP_SCHEDULE_KEYS: Final = (
    *RETENTION_SETTING_KEYS,
    "maximum_spend_logs_cleanup_cron",
    "maximum_spend_logs_retention_interval",
)


def cleanup_schedule_of(settings: Mapping[str, object]) -> tuple[object, ...]:
    return tuple(settings.get(key) for key in CLEANUP_SCHEDULE_KEYS)


def wants_spend_log_cleanup(settings: Mapping[str, object]) -> bool:
    return any(settings.get(key) is not None for key in RETENTION_SETTING_KEYS)


def spend_log_cleanup_trigger(settings: Mapping[str, object], *, apply_stagger: bool) -> Trigger | None:
    cleanup_cron: Final[object] = settings.get("maximum_spend_logs_cleanup_cron")
    if cleanup_cron:
        try:
            cron_trigger: Final[Trigger] = CronTrigger.from_crontab(  # pyright: ignore[reportUnknownMemberType]  # APScheduler has no stubs
                cleanup_cron
            )
            return cron_trigger
        except (ValueError, TypeError, AttributeError):
            verbose_proxy_logger.error("Invalid maximum_spend_logs_cleanup_cron value: %s", cleanup_cron)
            return None

    retention_interval: Final[object] = settings.get("maximum_spend_logs_retention_interval", "1d")
    if not isinstance(retention_interval, str):
        verbose_proxy_logger.error("Invalid maximum_spend_logs_retention_interval value: %r", retention_interval)
        return None

    try:
        interval_seconds: Final = duration_in_seconds(retention_interval)
        interval_trigger: Final[Trigger] = IntervalTrigger(  # pyright: ignore[reportUnknownMemberType]  # APScheduler has no stubs
            seconds=interval_seconds
        )
    except (ValueError, OverflowError):
        verbose_proxy_logger.error("Invalid maximum_spend_logs_retention_interval value: %r", retention_interval)
        return None

    if not apply_stagger:
        return interval_trigger

    stagger_settings: Final = parse_stagger_settings(settings)
    if not stagger_settings.enabled:
        return interval_trigger

    return stagger_trigger(
        job_id=SPEND_LOG_CLEANUP_JOB_ID,
        trigger=interval_trigger,
        period_seconds=interval_seconds,
        settings=stagger_settings,
    )  # pyright: ignore[reportReturnType]  # APScheduler registers the offset trigger as a BaseTrigger


def schedule_spend_log_cleanup(
    scheduler: SpendLogCleanupScheduler,
    settings: Mapping[str, object],
    prisma_client: PrismaClient | None,
) -> None:
    if not wants_spend_log_cleanup(settings):
        if scheduler.get_job(SPEND_LOG_CLEANUP_JOB_ID) is not None:
            scheduler.remove_job(SPEND_LOG_CLEANUP_JOB_ID)
            verbose_proxy_logger.info("Removed existing spend log cleanup job")
        return

    trigger: Final[Trigger | None] = spend_log_cleanup_trigger(settings, apply_stagger=scheduler.running)
    if trigger is None:
        return

    scheduler.add_job(
        SpendLogCleanup().cleanup_old_spend_logs,
        trigger,
        args=[prisma_client],
        id=SPEND_LOG_CLEANUP_JOB_ID,
        replace_existing=True,
        misfire_grace_time=APSCHEDULER_MISFIRE_GRACE_TIME,
    )

    if scheduler.running:
        verbose_proxy_logger.info("Spend log cleanup rescheduled with trigger: %s", trigger)
    elif settings.get("maximum_spend_logs_cleanup_cron"):
        verbose_proxy_logger.info(
            "Spend log cleanup scheduled with cron: %s",
            settings.get("maximum_spend_logs_cleanup_cron"),
        )
