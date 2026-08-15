"""
Persistence for the admin-configured periodic model cost map reload schedule stored in
``LiteLLM_Config``.

Field ownership is split by writer so concurrent writers never overwrite each other:
the schedule endpoints own the ``param_value`` JSON (``interval_hours``), while the
reload job and the manual reload endpoints own the dedicated ``last_run_at`` /
``reload_revision`` columns. ``last_run_at`` lives in the row rather than process memory
so the Admin UI still reports the last execution after a restart and across pods.
``reload_revision`` is a monotonic counter a manual reload increments; each pod records
the revision it last applied and reloads whenever the row's differs, so a request reaches
every pod exactly once without any pod clearing it and without comparing clocks. A booting
pod starts at revision 0 rather than adopting the published one, because it cannot know
whether that request predates the prices it fetched at import. Interval reloads stay
per-pod, driven by when that pod's own copy of the data was loaded.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import (
    TYPE_CHECKING,
    Protocol,
    TypedDict,
    cast,  # noqa: TID251  # prisma table access is untyped (PrismaWrapper.__getattr__)
)

from pydantic import BaseModel, ConfigDict, ValidationError

from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps
from litellm.proxy.utils import PrismaClient, evict_config_param
from litellm.repositories.config_repository import ConfigRepository

if TYPE_CHECKING:
    from prisma.models import LiteLLM_Config

MODEL_COST_MAP_RELOAD_PARAM_NAME = "model_cost_map_reload_config"


class _RevisionIncrement(TypedDict):
    increment: int


class _ConfigRowWrite(TypedDict, total=False):
    param_name: str
    param_value: str
    last_run_at: datetime
    reload_revision: int | _RevisionIncrement


class _ConfigUpsertData(TypedDict):
    create: _ConfigRowWrite
    update: _ConfigRowWrite


class _ConfigTable(Protocol):
    async def find_unique(self, where: Mapping[str, str]) -> "LiteLLM_Config | None": ...

    async def upsert(self, where: Mapping[str, str], data: _ConfigUpsertData) -> "LiteLLM_Config": ...

    async def update_many(self, data: _ConfigRowWrite, where: Mapping[str, str]) -> int: ...


def _config_table(prisma_client: PrismaClient) -> _ConfigTable:
    return cast(_ConfigTable, ConfigRepository(prisma_client).table)  # cast-ok: prisma table is untyped (Any)


@dataclass(frozen=True, slots=True)
class ReloadSchedule:
    interval_hours: int | None = None
    reload_revision: int = 0
    last_run_at: datetime | None = None


class ReloadScheduleStatus(TypedDict):
    scheduled: bool
    interval_hours: int | None
    last_run: str | None
    next_run: str | None


class _IntervalConfig(BaseModel):
    model_config = ConfigDict(strict=True)

    interval_hours: int | None = None


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_interval_hours(param_value: object) -> int | None:
    """``param_value`` is written as serialized JSON, and a raw row read can hand it back
    either decoded or still as a string depending on the driver, so accept both rather than
    reading a string as no schedule at all. Mirrors ``ConfigRepository.get_param``"""
    try:
        if isinstance(param_value, str):
            return _IntervalConfig.model_validate_json(param_value).interval_hours
        return _IntervalConfig.model_validate(param_value).interval_hours
    except ValidationError:
        return None


def _as_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


def parse_reload_schedule(row: "LiteLLM_Config") -> ReloadSchedule:
    return ReloadSchedule(
        interval_hours=_parse_interval_hours(row.param_value),
        reload_revision=int(row.reload_revision or 0),
        last_run_at=_as_utc(row.last_run_at),
    )


def next_run_at(schedule: ReloadSchedule) -> datetime | None:
    if schedule.interval_hours is None or schedule.last_run_at is None:
        return None
    return schedule.last_run_at + timedelta(hours=schedule.interval_hours)


def reload_schedule_status(schedule: ReloadSchedule | None) -> ReloadScheduleStatus:
    if schedule is None:
        return {"scheduled": False, "interval_hours": None, "last_run": None, "next_run": None}
    next_run = next_run_at(schedule)
    return {
        "scheduled": schedule.interval_hours is not None,
        "interval_hours": schedule.interval_hours,
        "last_run": schedule.last_run_at.isoformat() if schedule.last_run_at is not None else None,
        "next_run": next_run.isoformat() if next_run is not None else None,
    }


def pod_reload_is_due(
    *,
    schedule: ReloadSchedule,
    pod_applied_revision: int,
    pod_data_loaded_at: datetime,
    current_time: datetime,
    description: str,
) -> bool:
    """
    Whether this pod should reload now. A revision it has not applied means a manual reload
    it has not served. A pod starts at revision 0, so it serves any request published before
    it booted; that costs one redundant fetch per boot and is what keeps a request from being
    marked applied against data fetched before it. Interval reloads compare against this pod's
    own data, and a schedule that has never run anywhere fires immediately rather than one
    interval later
    """
    if schedule.reload_revision != pod_applied_revision:
        verbose_proxy_logger.info("%s reload triggered by manual reload request", description)
        return True
    if schedule.interval_hours is None:
        return False
    if schedule.last_run_at is None:
        verbose_proxy_logger.info("%s reload triggered - schedule has never run", description)
        return True
    hours_since_data_loaded = (current_time - pod_data_loaded_at).total_seconds() / 3600
    if hours_since_data_loaded < schedule.interval_hours:
        return False
    verbose_proxy_logger.info(
        "%s reload triggered by interval. Hours since data loaded: %.2f, Interval: %s",
        description,
        hours_since_data_loaded,
        schedule.interval_hours,
    )
    return True


async def read_reload_schedule(prisma_client: PrismaClient, param_name: str) -> ReloadSchedule | None:
    row = await _config_table(prisma_client).find_unique(where={"param_name": param_name})
    if row is None:
        return None
    return parse_reload_schedule(row)


async def write_reload_interval(prisma_client: PrismaClient, param_name: str, interval_hours: int) -> None:
    """Admin-owned write: replaces ``param_value`` without touching the job-owned columns"""
    param_value = safe_dumps({"interval_hours": interval_hours})
    await _config_table(prisma_client).upsert(
        where={"param_name": param_name},
        data={
            "create": {"param_name": param_name, "param_value": param_value},
            "update": {"param_value": param_value},
        },
    )
    await evict_config_param(param_name)


async def clear_reload_interval(prisma_client: PrismaClient, param_name: str) -> None:
    """Admin-owned write: drops the schedule but keeps the row, because the revision counter
    identifies a request rather than ordering one and so can never reuse a number. Deleting
    the row restarts it, and a reissued revision matches what pods already applied, so their
    next manual reload is silently skipped. The interval is nulled inside the JSON rather
    than by nulling the column, which prisma rejects for a ``Json?`` field"""
    await _config_table(prisma_client).update_many(
        data={"param_value": safe_dumps({"interval_hours": None})},
        where={"param_name": param_name},
    )
    await evict_config_param(param_name)


async def record_reload_run(prisma_client: PrismaClient, param_name: str, ran_at: datetime) -> None:
    """Job-owned write after this pod reloaded: stamps the shared last run only if the row
    still exists, so a schedule deleted mid-poll is not resurrected"""
    await _config_table(prisma_client).update_many(
        data={"last_run_at": ran_at},
        where={"param_name": param_name},
    )
    await evict_config_param(param_name)


async def record_manual_reload(prisma_client: PrismaClient, param_name: str, ran_at: datetime) -> int:
    """
    After a manual in-pod reload: stamp the shared last run and bump the revision every other
    pod compares against. The increment is atomic, so concurrent requests each publish a
    distinct revision instead of overwriting one another. Returns the published revision so
    the serving pod can adopt it rather than reloading again on its next poll
    """
    row = await _config_table(prisma_client).upsert(
        where={"param_name": param_name},
        data={
            "create": {"param_name": param_name, "last_run_at": ran_at, "reload_revision": 1},
            "update": {"last_run_at": ran_at, "reload_revision": {"increment": 1}},
        },
    )
    await evict_config_param(param_name)
    return int(row.reload_revision)
