import asyncio
import os
import socket
import uuid
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import TYPE_CHECKING, Final, Literal, Protocol, TypeAlias

from apscheduler.triggers import cron  # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]  # untyped
from pydantic import AwareDatetime, BaseModel, ConfigDict, TypeAdapter

from litellm._logging import verbose_proxy_logger
from litellm.constants import (
    SCHEDULED_JOB_RUN_HISTORY_LIMIT,
    SCHEDULED_JOB_TICK_TIMEOUT_SECONDS,
)
from litellm.litellm_core_utils.duration_parser import duration_in_seconds
from litellm.proxy.db.db_span import db_span
from litellm.proxy.db.db_transaction_queue.spend_logs_partition_manager import bounded_tx, trusted_sql

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient

_TABLE: Final = "LiteLLM_ScheduledJobRun"
_FIRE_TIME: Final = TypeAdapter[datetime](AwareDatetime)


@dataclass(frozen=True, slots=True)
class IntervalSchedule:
    seconds: float

    @property
    def spec(self) -> str:
        return f"interval:{self.seconds:g}s"


class FireTimes(Protocol):
    def get_next_fire_time(self, previous_fire_time: datetime | None, now: datetime) -> datetime | None: ...


class JobScheduler(Protocol):
    """The slice of APScheduler's AsyncIOScheduler that registers a poll, which ships no type information."""

    def get_job(self, job_id: str) -> object | None: ...

    def remove_job(self, job_id: str) -> None: ...

    def add_job(
        self,
        func: Callable[["PrismaClient | None"], Awaitable[None]],
        trigger: str,
        *,
        args: Sequence[object],
        id: str,
        replace_existing: bool,
        max_instances: int,
        misfire_grace_time: int,
        seconds: float,
    ) -> object: ...


@dataclass(frozen=True, slots=True)
class CronSchedule:
    expression: str
    trigger: FireTimes

    @property
    def spec(self) -> str:
        return f"cron:{self.expression}"


@dataclass(frozen=True, slots=True)
class InvalidSchedule:
    reason: str


Schedule: TypeAlias = IntervalSchedule | CronSchedule


def interval_schedule(setting_name: str, value: object) -> Schedule | InvalidSchedule:
    if not isinstance(value, str):
        return InvalidSchedule(f"{setting_name} must be a duration string like '1d', got {value!r}")
    try:
        seconds: Final = duration_in_seconds(value)
        timedelta(seconds=seconds)
    except (ValueError, OverflowError):
        return InvalidSchedule(f"{setting_name} is not a valid duration: {value!r}")
    if seconds <= 0:
        return InvalidSchedule(f"{setting_name} must be positive, got {value!r}")
    return IntervalSchedule(seconds=seconds)


def _cron_trigger(expression: str) -> FireTimes:
    return cron.CronTrigger.from_crontab(expression, timezone=timezone.utc)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # untyped


def cron_schedule(setting_name: str, value: object) -> Schedule | InvalidSchedule:
    if not isinstance(value, str):
        return InvalidSchedule(f"{setting_name} must be a crontab string, got {value!r}")
    try:
        trigger: Final = _cron_trigger(value)
    except (ValueError, TypeError):
        return InvalidSchedule(f"{setting_name} is not a valid crontab expression: {value!r}")
    if trigger.get_next_fire_time(None, datetime.now(timezone.utc)) is None:
        return InvalidSchedule(f"{setting_name} never fires: {value!r}")
    return CronSchedule(expression=value, trigger=trigger)


JobResult: TypeAlias = Literal["SUCCEEDED", "FAILED"]


@dataclass(frozen=True, slots=True)
class ScheduledJob:
    """
    A job that runs on exactly one pod per occurrence.

    Jobs sharing a lock_group never run at the same time, on any pod. schedule and lease are read on every tick,
    so a settings change applies without re-registering the job.
    """

    name: str
    lock_group: str
    schedule: Callable[[], Schedule | InvalidSchedule]
    lease: Callable[[], timedelta]
    run: Callable[["PrismaClient"], Awaitable[JobResult]]


class PendingRun(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    schedule: str
    scheduled_for: AwareDatetime


class JobState(BaseModel):
    model_config = ConfigDict(frozen=True)

    now: AwareDatetime
    group_running: bool
    job_running: bool
    last_finished_at: AwareDatetime | None
    last_scheduled_for: AwareDatetime | None


@dataclass(frozen=True, slots=True)
class NewRun:
    id: str
    schedule: str
    scheduled_for: datetime


@dataclass(frozen=True, slots=True)
class TickPlan:
    cancel: tuple[str, ...]
    insert: NewRun | None
    claim: str | None


def next_run_at(schedule: Schedule, state: JobState) -> datetime:
    """
    When the next occurrence is due.

    An interval counts from the last finished run, so a restart never resets it and a missed run happens once, right
    away. A cron fires at its next time after both now and the last occurrence, so an outage collapses into the one
    occurrence that was already pending.
    """
    match schedule:
        case IntervalSchedule(seconds=seconds):
            if state.last_finished_at is None:
                return state.now
            return state.last_finished_at + timedelta(seconds=seconds)
        case CronSchedule(trigger=trigger):
            anchor: Final = max(state.now, state.last_scheduled_for or state.now)
            return _FIRE_TIME.validate_python(trigger.get_next_fire_time(None, anchor + timedelta(microseconds=1)))


def plan_tick(
    schedule: Schedule | InvalidSchedule, state: JobState, pending: tuple[PendingRun, ...], new_id: str
) -> TickPlan:
    """
    Decide one tick from the job's rows. Pure, so every pod reaches the same decision from the same rows.

    A pending row planned from a different schedule is cancelled and replaced, so a settings change applies at once.
    With an invalid schedule nothing is replanned, and a pending row from the last valid one still runs.
    """
    if isinstance(schedule, InvalidSchedule):
        return TickPlan(cancel=(), insert=None, claim=_claimable(pending[:1], state))
    current: Final = tuple(run for run in pending if run.schedule == schedule.spec)
    keep: Final = current[:1]
    cancel: Final = tuple(run.id for run in pending if run not in keep)
    if keep or state.job_running:
        return TickPlan(cancel=cancel, insert=None, claim=_claimable(keep, state))
    new_run: Final = NewRun(id=new_id, schedule=schedule.spec, scheduled_for=next_run_at(schedule, state))
    claim: Final = new_run.id if new_run.scheduled_for <= state.now and not state.group_running else None
    return TickPlan(cancel=cancel, insert=new_run, claim=claim)


def _claimable(pending: tuple[PendingRun, ...], state: JobState) -> str | None:
    if not pending or state.group_running or pending[0].scheduled_for > state.now:
        return None
    return pending[0].id


def default_worker_id() -> str:
    return f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"


class _Claimed(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    job_name: str


_STATE: Final = TypeAdapter(list[JobState])
_PENDING: Final = TypeAdapter(list[PendingRun])
_CLAIMED: Final = TypeAdapter(list[_Claimed])

_REAP_SQL: Final = f"""
    UPDATE "{_TABLE}"
    SET "status" = 'FAILED', "finished_at" = now()
    WHERE "lock_group" = $1 AND "status" = 'RUNNING' AND "lease_expires_at" < now()
    RETURNING "id", "job_name"
"""
_STATE_SQL: Final = f"""
    SELECT now() AS "now",
        EXISTS (SELECT 1 FROM "{_TABLE}" WHERE "lock_group" = $2 AND "status" = 'RUNNING') AS "group_running",
        EXISTS (SELECT 1 FROM "{_TABLE}" WHERE "job_name" = $1 AND "status" = 'RUNNING') AS "job_running",
        (SELECT max("finished_at") FROM "{_TABLE}"
            WHERE "job_name" = $1 AND "status" IN ('SUCCEEDED', 'FAILED')) AS "last_finished_at",
        (SELECT max("scheduled_for") FROM "{_TABLE}"
            WHERE "job_name" = $1 AND "status" NOT IN ('CANCELLED', 'PENDING')) AS "last_scheduled_for"
"""
_PENDING_SQL: Final = f"""
    SELECT "id", "schedule", "scheduled_for" FROM "{_TABLE}"
    WHERE "job_name" = $1 AND "status" = 'PENDING'
    ORDER BY "scheduled_for", "created_at"
"""
_CANCEL_SQL: Final = f"""
    UPDATE "{_TABLE}" SET "status" = 'CANCELLED', "finished_at" = now()
    WHERE "id" = $1 AND "status" = 'PENDING'
"""
_INSERT_SQL: Final = f"""
    INSERT INTO "{_TABLE}" ("id", "job_name", "lock_group", "schedule", "scheduled_for")
    VALUES ($1, $2, $3, $4, $5::timestamptz)
"""
_CLAIM_SQL: Final = f"""
    UPDATE "{_TABLE}"
    SET "status" = 'RUNNING', "claimed_by" = $2, "claimed_at" = now(),
        "lease_expires_at" = now() + $3::double precision * interval '1 second'
    WHERE "id" = $1 AND "status" = 'PENDING'
    RETURNING "id", "job_name"
"""
_FINISH_SQL: Final = f"""
    UPDATE "{_TABLE}" SET "status" = $3::"ScheduledJobRunStatus", "finished_at" = now()
    WHERE "id" = $1 AND "claimed_by" = $2 AND "status" = 'RUNNING'
"""
_PRUNE_SQL: Final = f"""
    DELETE FROM "{_TABLE}" WHERE "id" IN (
        SELECT "id" FROM "{_TABLE}" WHERE "job_name" = $1 AND "finished_at" IS NOT NULL
        ORDER BY "finished_at" DESC OFFSET $2
    )
"""


class ScheduledJobRunner:
    """
    Polls one ScheduledJob's rows in LiteLLM_ScheduledJobRun. Every pod runs the same tick.

    Each tick runs in one transaction under a transaction-scoped advisory lock on the job's lock_group, so only one
    pod plans and claims for a group at a time and a pod that dies mid-tick releases it. The claimed run executes
    outside any transaction under a lease. A pod that dies mid-run leaves its row RUNNING until the lease expires,
    when the next tick on any pod marks it FAILED and plans the next occurrence.
    """

    def __init__(
        self,
        job: ScheduledJob,
        *,
        worker_id: str | None = None,
        new_id: Callable[[], str] = lambda: str(uuid.uuid4()),
    ) -> None:
        self.job = job
        self.worker_id = worker_id or default_worker_id()
        self._new_id = new_id
        self._running = False

    async def tick(self, prisma_client: "PrismaClient | None") -> None:
        """
        Plan, claim and run whatever is due. Awaits the run, so graceful shutdown sees it as an in-flight job.

        Register it with max_instances=2: the extra instance lets the polls that land during a long run return at
        once instead of APScheduler warning that it skipped them.
        """
        if prisma_client is None or self._running:
            return
        self._running = True
        try:
            claimed: Final = await self._safe_plan_and_claim(prisma_client)
            if claimed is not None:
                await self._execute(prisma_client, claimed)
        finally:
            self._running = False

    async def _safe_plan_and_claim(self, prisma_client: "PrismaClient") -> str | None:
        try:
            return await self._plan_and_claim(prisma_client)
        except Exception as e:  # noqa: BLE001 - a failed tick is retried by the next one
            verbose_proxy_logger.warning("Scheduled job %s tick failed: %s: %s", self.job.name, type(e).__name__, e)
            return None

    async def _plan_and_claim(self, prisma_client: "PrismaClient") -> str | None:
        schedule: Final = self.job.schedule()
        if isinstance(schedule, InvalidSchedule):
            _log_invalid_schedule(self.job.name, schedule.reason)
        timeout_ms: Final = int(SCHEDULED_JOB_TICK_TIMEOUT_SECONDS * 1000)
        async with db_span("scheduled_job_tick", _TABLE), bounded_tx(prisma_client, timeout_ms) as tx:
            await tx.execute_raw(trusted_sql(f"SET LOCAL statement_timeout = {timeout_ms}"))
            await tx.execute_raw(trusted_sql(f"SET LOCAL lock_timeout = {timeout_ms}"))
            locked: Final = _LOCKED.validate_python(
                await tx.query_raw(trusted_sql(_GROUP_LOCK_SQL), f"litellm_scheduled_job:{self.job.lock_group}")
            )
            if not locked or not locked[0].locked:
                return None
            for reaped in _CLAIMED.validate_python(await tx.query_raw(trusted_sql(_REAP_SQL), self.job.lock_group)):
                verbose_proxy_logger.warning(
                    "Scheduled job %s run %s lost its lease; marked FAILED", reaped.job_name, reaped.id
                )
            state: Final = _STATE.validate_python(
                await tx.query_raw(trusted_sql(_STATE_SQL), self.job.name, self.job.lock_group)
            )[0]
            pending: Final = tuple(
                _PENDING.validate_python(await tx.query_raw(trusted_sql(_PENDING_SQL), self.job.name))
            )
            plan: Final = plan_tick(schedule, state, pending, self._new_id())
            for run_id in plan.cancel:
                await tx.execute_raw(trusted_sql(_CANCEL_SQL), run_id)
            if plan.insert is not None:
                await tx.execute_raw(
                    trusted_sql(_INSERT_SQL),
                    plan.insert.id,
                    self.job.name,
                    self.job.lock_group,
                    plan.insert.schedule,
                    plan.insert.scheduled_for.isoformat(),
                )
            if plan.claim is None:
                return None
            claimed: Final = _CLAIMED.validate_python(
                await tx.query_raw(
                    trusted_sql(_CLAIM_SQL), plan.claim, self.worker_id, self.job.lease().total_seconds()
                )
            )
        return claimed[0].id if claimed else None

    async def _execute(self, prisma_client: "PrismaClient", run_id: str) -> None:
        verbose_proxy_logger.info("Scheduled job %s run %s claimed by %s", self.job.name, run_id, self.worker_id)
        try:
            result: Final = await self._run_job(prisma_client)
        except asyncio.CancelledError:
            await asyncio.shield(self._finish(prisma_client, run_id, "FAILED"))
            raise
        await self._finish(prisma_client, run_id, result)
        next_run_id: Final = await self._safe_plan_and_claim(prisma_client)
        if next_run_id is not None:
            await self._execute(prisma_client, next_run_id)

    async def _run_job(self, prisma_client: "PrismaClient") -> JobResult:
        try:
            return await self.job.run(prisma_client)
        except Exception:  # noqa: BLE001 - the failure is recorded on the run row instead of escaping the task
            verbose_proxy_logger.exception("Scheduled job %s raised", self.job.name)
            return "FAILED"

    async def _finish(self, prisma_client: "PrismaClient", run_id: str, result: JobResult) -> None:
        timeout_ms: Final = int(SCHEDULED_JOB_TICK_TIMEOUT_SECONDS * 1000)
        try:
            async with db_span("scheduled_job_finish", _TABLE), bounded_tx(prisma_client, timeout_ms) as tx:
                await tx.execute_raw(trusted_sql(f"SET LOCAL statement_timeout = {timeout_ms}"))
                updated: Final = await tx.execute_raw(trusted_sql(_FINISH_SQL), run_id, self.worker_id, result)
                await tx.execute_raw(trusted_sql(_PRUNE_SQL), self.job.name, SCHEDULED_JOB_RUN_HISTORY_LIMIT)
        except Exception as e:  # noqa: BLE001 - the lease expiry marks the row when this write cannot
            verbose_proxy_logger.warning(
                "Scheduled job %s run %s could not record %s: %s: %s",
                self.job.name,
                run_id,
                result,
                type(e).__name__,
                e,
            )
            return
        if updated == 0:
            verbose_proxy_logger.warning(
                "Scheduled job %s run %s finished after its lease expired; its %s result was not recorded",
                self.job.name,
                run_id,
                result,
            )
            return
        verbose_proxy_logger.info("Scheduled job %s run %s %s", self.job.name, run_id, result)


class _Locked(BaseModel):
    model_config = ConfigDict(frozen=True)

    locked: bool


_LOCKED: Final = TypeAdapter(list[_Locked])
_GROUP_LOCK_SQL: Final = 'SELECT pg_try_advisory_xact_lock(hashtext($1)) AS "locked"'


@lru_cache(maxsize=64)
def _log_invalid_schedule(job_name: str, reason: str) -> None:
    verbose_proxy_logger.error(
        "Scheduled job %s has an invalid schedule and will not be replanned: %s", job_name, reason
    )


def register_poll(
    scheduler: JobScheduler,
    prisma_client: "PrismaClient | None",
    job_id: str,
    job: ScheduledJob,
    *,
    wanted: bool,
    poll_seconds: float,
    misfire_grace_time: int,
) -> None:
    """
    Make every pod poll job, or stop polling it. A poll that already exists is kept: its runner reads the schedule
    on every tick, so a schedule edit needs no re-registration.
    """
    registered: Final = scheduler.get_job(job_id) is not None
    if registered and not wanted:
        scheduler.remove_job(job_id)
        verbose_proxy_logger.info("Stopped polling scheduled job %s", job.name)
        return
    if registered or not wanted:
        return
    scheduler.add_job(
        ScheduledJobRunner(job).tick,
        "interval",
        args=(prisma_client,),
        id=job_id,
        replace_existing=True,
        max_instances=2,
        misfire_grace_time=misfire_grace_time,
        seconds=poll_seconds,
    )
    verbose_proxy_logger.info("Polling scheduled job %s every %ss", job.name, poll_seconds)
