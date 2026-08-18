"""
Track guardrail and policy usage for the dashboard: upsert daily metrics and
insert into SpendLogGuardrailIndex when spend logs are written.
"""

import asyncio
import json
from collections import defaultdict
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from datetime import datetime, timezone
from functools import partial
from itertools import groupby
from operator import itemgetter
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final, NamedTuple, TypeVar

from litellm._logging import verbose_proxy_logger
from litellm.proxy.utils import PrismaClient
from litellm.repositories.table_repositories import (
    DailyGuardrailMetricsRepository,
    DailyGuardrailUsageUnitsRepository,
    SpendLogGuardrailIndexRepository,
)

if TYPE_CHECKING:
    from prisma import types as prisma_types


_UPSERT_RETRY_TIMES: Final = 3

_RowKey = TypeVar("_RowKey")
_RowValue = TypeVar("_RowValue")


class _UsageUnitKey(NamedTuple):
    guardrail_id: str
    date: str
    team_id: str
    api_key: str
    usage_unit: str


class _MetricsKey(NamedTuple):
    guardrail_id: str
    date: str


async def _attempt_upsert(
    upsert_row: Callable[[_RowKey, _RowValue], Awaitable[None]], key: _RowKey, value: _RowValue
) -> Exception | None:
    try:
        await upsert_row(key, value)
    except Exception as error:
        return error
    return None


async def _upsert_rows_with_retry(
    rows: Mapping[_RowKey, _RowValue],
    upsert_row: Callable[[_RowKey, _RowValue], Awaitable[None]],
    label: str,
    sleep: Callable[[float], Awaitable[None]],
    retries_left: int = _UPSERT_RETRY_TIMES,
) -> None:
    outcomes: Final = {key: await _attempt_upsert(upsert_row, key, value) for key, value in rows.items()}
    failed: Final = MappingProxyType({key: rows[key] for key, error in outcomes.items() if error is not None})
    if not failed:
        return
    if retries_left == 0:
        for key in failed:
            verbose_proxy_logger.warning(
                "Guardrail usage tracking: %s upsert failed for %s after %d retries (non-fatal): %s",
                label,
                key,
                _UPSERT_RETRY_TIMES,
                outcomes[key],
            )
        return
    await sleep(2 ** (_UPSERT_RETRY_TIMES - retries_left))
    await _upsert_rows_with_retry(failed, upsert_row, label, sleep, retries_left - 1)


def _guardrail_status_to_action(status: str | None) -> str:
    """Map StandardLogging guardrail_status to blocked/passed/flagged."""
    if not status:
        return "passed"
    s: Final = (status or "").lower()
    if "intervened" in s or "block" in s:
        return "blocked"
    if "fail" in s or "error" in s:
        return "flagged"
    return "passed"


def _parse_guardrail_info_from_payload(payload: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
    """Extract guardrail_information from spend log payload metadata."""
    meta = payload.get("metadata")
    if not meta:
        return []
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except (json.JSONDecodeError, TypeError):
            return []
    if not isinstance(meta, dict):
        return []
    info: Final = meta.get("guardrail_information") or meta.get("standard_logging_guardrail_information")
    if not isinstance(info, list):
        return []
    return info


def _date_str(dt: datetime) -> str:
    """YYYY-MM-DD in UTC."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d")


def _parse_payload_start_time(payload: Mapping[str, Any]) -> datetime | None:
    start_time: Final = payload.get("startTime")
    if isinstance(start_time, datetime):
        return start_time
    if not isinstance(start_time, str):
        return None
    try:
        return datetime.fromisoformat(start_time.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def _iter_usage_unit_increments(logs_to_process: Sequence[Mapping[str, Any]]) -> Iterator[tuple[_UsageUnitKey, int]]:
    for payload in logs_to_process:
        start_time = _parse_payload_start_time(payload)
        if not payload.get("request_id") or start_time is None:
            continue
        date_key = _date_str(start_time)
        team_id = str(payload.get("team_id") or "")
        api_key = str(payload.get("api_key") or "")
        for entry in _parse_guardrail_info_from_payload(payload):
            guardrail_id = str(entry.get("guardrail_id") or entry.get("guardrail_name") or "")
            usage = entry.get("guardrail_usage")
            if not guardrail_id or not isinstance(usage, dict):
                continue
            for unit_name, units in usage.items():
                if isinstance(units, int) and not isinstance(units, bool) and units > 0:
                    yield _UsageUnitKey(guardrail_id, date_key, team_id, api_key, str(unit_name)), units


def _sum_usage_unit_increments(logs_to_process: Sequence[Mapping[str, Any]]) -> Mapping[_UsageUnitKey, int]:
    ordered: Final = sorted(_iter_usage_unit_increments(logs_to_process), key=itemgetter(0))
    return MappingProxyType(
        {key: sum(units for _, units in group) for key, group in groupby(ordered, key=itemgetter(0))}
    )


async def _upsert_usage_unit_row(prisma_client: PrismaClient, key: _UsageUnitKey, units: int) -> None:
    row: Final[prisma_types.LiteLLM_DailyGuardrailUsageUnitsCreateInput] = {
        "guardrail_id": key.guardrail_id,
        "date": key.date,
        "team_id": key.team_id,
        "api_key": key.api_key,
        "usage_unit": key.usage_unit,
        "units": units,
    }
    where: Final[prisma_types.LiteLLM_DailyGuardrailUsageUnitsWhereUniqueInput] = {
        "guardrail_id_date_team_id_api_key_usage_unit": {
            "guardrail_id": key.guardrail_id,
            "date": key.date,
            "team_id": key.team_id,
            "api_key": key.api_key,
            "usage_unit": key.usage_unit,
        }
    }
    data: Final[prisma_types.LiteLLM_DailyGuardrailUsageUnitsUpsertInput] = {
        "create": row,
        "update": {"units": {"increment": units}},
    }
    await DailyGuardrailUsageUnitsRepository(prisma_client).table.upsert(where=where, data=data)


async def _upsert_metrics_row(prisma_client: PrismaClient, key: _MetricsKey, agg: Mapping[str, int]) -> None:
    n: Final = int(agg["requests_evaluated"])
    await DailyGuardrailMetricsRepository(prisma_client).table.upsert(
        where={"guardrail_id_date": {"guardrail_id": key.guardrail_id, "date": key.date}},
        data={
            "create": {
                "guardrail_id": key.guardrail_id,
                "date": key.date,
                "requests_evaluated": n,
                "passed_count": int(agg["passed_count"]),
                "blocked_count": int(agg["blocked_count"]),
                "flagged_count": int(agg["flagged_count"]),
            },
            "update": {
                "requests_evaluated": {"increment": n},
                "passed_count": {"increment": int(agg["passed_count"])},
                "blocked_count": {"increment": int(agg["blocked_count"])},
                "flagged_count": {"increment": int(agg["flagged_count"])},
            },
        },
    )


async def process_spend_logs_guardrail_usage(
    prisma_client: PrismaClient,
    logs_to_process: list[dict[str, Any]],
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> None:
    """
    After spend logs are written: update DailyGuardrailMetrics and insert
    SpendLogGuardrailIndex rows from guardrail_information in each payload.
    """
    if not logs_to_process:
        return
    # Aggregate daily metrics by (guardrail_id, date). Latency/score metrics dropped.
    daily_guardrail: Final[dict[_MetricsKey, dict[str, Any]]] = defaultdict(
        lambda: {
            "requests_evaluated": 0,
            "passed_count": 0,
            "blocked_count": 0,
            "flagged_count": 0,
        }
    )
    index_rows: Final[list[dict[str, Any]]] = []

    for payload in logs_to_process:
        request_id = payload.get("request_id")
        start_time = _parse_payload_start_time(payload)
        if not request_id or start_time is None:
            continue
        date_key = _date_str(start_time)

        for entry in _parse_guardrail_info_from_payload(payload):
            guardrail_id = entry.get("guardrail_id") or entry.get("guardrail_name") or ""
            if not guardrail_id:
                continue
            key = _MetricsKey(guardrail_id, date_key)
            daily_guardrail[key]["requests_evaluated"] += 1
            action = _guardrail_status_to_action(entry.get("guardrail_status"))
            if action == "passed":
                daily_guardrail[key]["passed_count"] += 1
            elif action == "blocked":
                daily_guardrail[key]["blocked_count"] += 1
            else:
                daily_guardrail[key]["flagged_count"] += 1
            policy_id = entry.get("policy_id")
            index_rows.append(
                {
                    "request_id": request_id,
                    "guardrail_id": guardrail_id,
                    "policy_id": policy_id,
                    "start_time": start_time,
                }
            )

    usage_unit_totals: Final = _sum_usage_unit_increments(logs_to_process)

    if not daily_guardrail and not index_rows and not usage_unit_totals:
        return

    try:
        # Insert index rows (skip duplicates by request_id + guardrail_id)
        if index_rows:
            try:
                await SpendLogGuardrailIndexRepository(prisma_client).table.create_many(
                    data=index_rows,
                    skip_duplicates=True,
                )
            except Exception as e:
                verbose_proxy_logger.debug("Guardrail usage tracking: index create_many skipped: %s", e)

        # Upsert daily guardrail metrics (counts only; latency/score dropped)
        metrics_rows: Final = MappingProxyType(
            {key: agg for key, agg in daily_guardrail.items() if int(agg["requests_evaluated"]) > 0}
        )
        await _upsert_rows_with_retry(metrics_rows, partial(_upsert_metrics_row, prisma_client), "daily metrics", sleep)
        await _upsert_rows_with_retry(
            usage_unit_totals, partial(_upsert_usage_unit_row, prisma_client), "usage unit", sleep
        )
    except Exception as e:
        verbose_proxy_logger.warning("Guardrail usage tracking failed (non-fatal): %s", e)
