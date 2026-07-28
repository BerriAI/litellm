"""
Persistence for the admin-configured periodic reload schedules (model cost map,
Anthropic beta headers) stored in the ``LiteLLM_Config`` table.

``last_run`` is kept in the row rather than in process memory so the Admin UI still
reports the schedule and its last execution after a restart and across pods. Each pod
tracks its own last reload separately to decide when to refresh its in-memory copy of
the data.
"""

from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from typing import TypedDict

from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps
from litellm.proxy.utils import PrismaClient, invalidate_config_param
from litellm.repositories.config_repository import ConfigRepository

MODEL_COST_MAP_RELOAD_PARAM_NAME = "model_cost_map_reload_config"
ANTHROPIC_BETA_HEADERS_RELOAD_PARAM_NAME = "anthropic_beta_headers_reload_config"


@dataclass(frozen=True, slots=True)
class ReloadSchedule:
    interval_hours: int | None = None
    force_reload: bool = False
    last_run: datetime | None = None


class ReloadScheduleStatus(TypedDict):
    scheduled: bool
    interval_hours: int | None
    last_run: str | None
    next_run: str | None


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_last_run(raw: object) -> datetime | None:
    if not isinstance(raw, str):
        return None
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        verbose_proxy_logger.warning("Ignoring unparseable reload last_run: %s", raw)
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


def parse_reload_schedule(param_value: object) -> ReloadSchedule:
    if not isinstance(param_value, dict):
        return ReloadSchedule()
    interval_hours = param_value.get("interval_hours")
    return ReloadSchedule(
        interval_hours=interval_hours if isinstance(interval_hours, int) else None,
        force_reload=param_value.get("force_reload") is True,
        last_run=_parse_last_run(param_value.get("last_run")),
    )


def serialize_reload_schedule(schedule: ReloadSchedule) -> str:
    return safe_dumps(
        {
            "interval_hours": schedule.interval_hours,
            "force_reload": schedule.force_reload,
            "last_run": schedule.last_run.isoformat() if schedule.last_run is not None else None,
        }
    )


def next_run_at(schedule: ReloadSchedule) -> datetime | None:
    if schedule.interval_hours is None or schedule.last_run is None:
        return None
    return schedule.last_run + timedelta(hours=schedule.interval_hours)


def reload_schedule_status(schedule: ReloadSchedule | None) -> ReloadScheduleStatus:
    last_run = schedule.last_run if schedule is not None else None
    next_run = next_run_at(schedule) if schedule is not None else None
    return {
        "scheduled": schedule is not None and schedule.interval_hours is not None,
        "interval_hours": schedule.interval_hours if schedule is not None else None,
        "last_run": last_run.isoformat() if last_run is not None else None,
        "next_run": next_run.isoformat() if next_run is not None else None,
    }


def pod_reload_is_due(
    *,
    schedule: ReloadSchedule,
    pod_last_reload: datetime | None,
    current_time: datetime,
    description: str,
) -> bool:
    """
    Whether this pod should reload now, based on its own last reload rather than the
    persisted one, so that every pod refreshes its in-memory data on the configured interval
    """
    if schedule.force_reload:
        verbose_proxy_logger.info("%s reload triggered by force reload flag", description)
        return True
    if schedule.interval_hours is None:
        return False
    if pod_last_reload is None:
        verbose_proxy_logger.info("%s reload triggered - no previous reload time recorded", description)
        return True
    hours_since_last_reload = (current_time - pod_last_reload).total_seconds() / 3600
    if hours_since_last_reload < schedule.interval_hours:
        return False
    verbose_proxy_logger.info(
        "%s reload triggered by interval. Hours since last reload: %.2f, Interval: %s",
        description,
        hours_since_last_reload,
        schedule.interval_hours,
    )
    return True


async def read_reload_schedule(prisma_client: PrismaClient, param_name: str) -> ReloadSchedule | None:
    row = await ConfigRepository(prisma_client).table.find_unique(where={"param_name": param_name})
    if row is None or row.param_value is None:
        return None
    return parse_reload_schedule(row.param_value)


async def write_reload_schedule(prisma_client: PrismaClient, param_name: str, schedule: ReloadSchedule) -> None:
    param_value = serialize_reload_schedule(schedule)
    await ConfigRepository(prisma_client).table.upsert(
        where={"param_name": param_name},
        data={
            "create": {"param_name": param_name, "param_value": param_value},
            "update": {"param_value": param_value},
        },
    )
    await invalidate_config_param(param_name)


async def record_reload_run(prisma_client: PrismaClient, param_name: str, ran_at: datetime) -> None:
    """
    Persist a completed reload: stamp ``last_run`` and clear ``force_reload`` while keeping
    the configured interval
    """
    existing = await read_reload_schedule(prisma_client, param_name) or ReloadSchedule()
    await write_reload_schedule(
        prisma_client,
        param_name,
        replace(existing, force_reload=False, last_run=ran_at),
    )
