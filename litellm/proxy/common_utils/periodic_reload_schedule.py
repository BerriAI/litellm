"""
Persistence for the admin-configured periodic reload schedules (model cost map,
Anthropic beta headers) stored in ``LiteLLM_Config``.

Field ownership is split by writer so concurrent writers never overwrite each other:
the schedule endpoints own the ``param_value`` JSON (``interval_hours``), while the
reload job and the manual reload endpoints own the dedicated ``last_run_at`` /
``reload_requested_at`` columns. ``last_run_at`` lives in the row rather than process
memory so the Admin UI still reports the last execution after a restart and across pods.
``reload_requested_at`` fans a manual reload out to every pod without any pod having to
write the flag back down, so no pod can starve another of the request. Each pod tracks
only when its own in-memory copy was loaded (seeded at boot) and reloads whenever that
copy is older than the request or the configured interval.
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
from litellm.proxy.utils import PrismaClient, invalidate_config_param
from litellm.repositories.config_repository import ConfigRepository

if TYPE_CHECKING:
    from prisma.models import LiteLLM_Config

MODEL_COST_MAP_RELOAD_PARAM_NAME = "model_cost_map_reload_config"
ANTHROPIC_BETA_HEADERS_RELOAD_PARAM_NAME = "anthropic_beta_headers_reload_config"


class _ConfigRowWrite(TypedDict, total=False):
    param_name: str
    param_value: str
    last_run_at: datetime
    reload_requested_at: datetime


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
    reload_requested_at: datetime | None = None
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
    try:
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
        reload_requested_at=_as_utc(row.reload_requested_at),
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
    pod_data_loaded_at: datetime,
    current_time: datetime,
    description: str,
) -> bool:
    """
    Whether this pod should reload now: when its in-memory data (loaded at boot or at its
    last reload) is older than a manual request or the configured interval. A schedule that
    has never run anywhere fires immediately so the first run is not one interval away
    """
    if schedule.reload_requested_at is not None and schedule.reload_requested_at > pod_data_loaded_at:
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
    await invalidate_config_param(param_name)


async def record_reload_run(prisma_client: PrismaClient, param_name: str, ran_at: datetime) -> None:
    """Job-owned write after this pod reloaded: stamps the shared last run only if the row
    still exists, so a schedule deleted mid-poll is not resurrected"""
    await _config_table(prisma_client).update_many(
        data={"last_run_at": ran_at},
        where={"param_name": param_name},
    )
    await invalidate_config_param(param_name)


async def record_manual_reload(prisma_client: PrismaClient, param_name: str, ran_at: datetime) -> None:
    """After a manual in-pod reload: stamps the shared last run and the request every other
    pod compares against on its next poll"""
    await _config_table(prisma_client).upsert(
        where={"param_name": param_name},
        data={
            "create": {"param_name": param_name, "last_run_at": ran_at, "reload_requested_at": ran_at},
            "update": {"last_run_at": ran_at, "reload_requested_at": ran_at},
        },
    )
    await invalidate_config_param(param_name)
