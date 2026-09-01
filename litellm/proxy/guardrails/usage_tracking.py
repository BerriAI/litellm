"""
Track guardrail and policy usage for the dashboard: upsert daily metrics and
insert into SpendLogGuardrailIndex when spend logs are written.
"""

import asyncio
import json
from collections import defaultdict
from collections.abc import Awaitable, Callable, Iterable, Iterator, Mapping, Sequence
from datetime import datetime, timezone
from functools import partial
from itertools import groupby
from operator import itemgetter
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final, NamedTuple, TypeVar

from typing_extensions import ReadOnly, TypedDict

from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.llm_cost_calc.guardrail_cost import billed_guardrail_cost_by_unit
from litellm.proxy._types import DB_RETRY_SAFE_ERROR_TYPES
from litellm.proxy.utils import PrismaClient
from litellm.repositories.table_repositories import (
    DailyGuardrailMetricsRepository,
    DailyGuardrailUsageUnitsRepository,
    SpendLogGuardrailIndexRepository,
)

if TYPE_CHECKING:
    from prisma import types as prisma_types


_UPSERT_RETRY_TIMES: Final = 3
_MAX_PENDING_ROWS: Final = 10_000

_RowKey = TypeVar("_RowKey")
_RowValue = TypeVar("_RowValue")


class _UsageUnitKey(NamedTuple):
    guardrail_id: str
    date: str
    team_id: str
    api_key: str
    usage_unit: str


class _UsageUnitIncrement(NamedTuple):
    units: int
    cost: float | None


class _MetricsKey(NamedTuple):
    guardrail_id: str
    date: str


class _UsageUnitCompoundKey(TypedDict):
    guardrail_id: ReadOnly[str]
    date: ReadOnly[str]
    team_id: ReadOnly[str]
    api_key: ReadOnly[str]
    usage_unit: ReadOnly[str]


class _UsageUnitWhereUnique(TypedDict):
    guardrail_id_date_team_id_api_key_usage_unit: ReadOnly[_UsageUnitCompoundKey]


class PendingRollups:
    """Rollup rows whose connection-error retries exhausted, held for the next flush."""

    def __init__(self) -> None:
        self.lock: Final = asyncio.Lock()
        self.metrics: Mapping[_MetricsKey, Mapping[str, int]] = MappingProxyType({})
        self.units: Mapping[_UsageUnitKey, _UsageUnitIncrement] = MappingProxyType({})


_PENDING_ROLLUPS: Final = PendingRollups()

_NO_COUNTERS: Final[Mapping[str, int]] = MappingProxyType({})
_NO_INCREMENT: Final = _UsageUnitIncrement(units=0, cost=0.0)


def _merged_keys(base: Mapping[_RowKey, object], extra: Mapping[_RowKey, object]) -> tuple[_RowKey, ...]:
    return (*base, *(key for key in extra if key not in base))


def _summed_increments(increments: Iterable[_UsageUnitIncrement]) -> _UsageUnitIncrement:
    """Units add; cost adds too unless any increment was unpriced, which makes the sum unknown."""
    materialized: Final = tuple(increments)
    costs: Final = tuple(i.cost for i in materialized)
    return _UsageUnitIncrement(
        units=sum(i.units for i in materialized),
        cost=None if any(c is None for c in costs) else sum(c for c in costs if c is not None),
    )


def _merged_unit_rows(
    base: Mapping[_UsageUnitKey, _UsageUnitIncrement], extra: Mapping[_UsageUnitKey, _UsageUnitIncrement]
) -> Mapping[_UsageUnitKey, _UsageUnitIncrement]:
    return MappingProxyType(
        {
            key: _summed_increments((base.get(key, _NO_INCREMENT), extra.get(key, _NO_INCREMENT)))
            for key in _merged_keys(base, extra)
        }
    )


def _merged_metric_rows(
    base: Mapping[_MetricsKey, Mapping[str, int]], extra: Mapping[_MetricsKey, Mapping[str, int]]
) -> Mapping[_MetricsKey, Mapping[str, int]]:
    def merged_counters(key: _MetricsKey) -> Mapping[str, int]:
        base_counters: Final = base.get(key, _NO_COUNTERS)
        extra_counters: Final = extra.get(key, _NO_COUNTERS)
        return MappingProxyType(
            {
                counter: int(base_counters.get(counter, 0)) + int(extra_counters.get(counter, 0))
                for counter in _merged_keys(base_counters, extra_counters)
            }
        )

    return MappingProxyType({key: merged_counters(key) for key in _merged_keys(base, extra)})


def _capped(rows: Mapping[_RowKey, _RowValue], label: str) -> Mapping[_RowKey, _RowValue]:
    if len(rows) <= _MAX_PENDING_ROWS:
        return rows
    verbose_proxy_logger.warning(
        "Guardrail usage tracking: pending %s requeue exceeds %d rows; dropping the %d oldest (non-fatal)",
        label,
        _MAX_PENDING_ROWS,
        len(rows) - _MAX_PENDING_ROWS,
    )
    return MappingProxyType(dict(tuple(rows.items())[len(rows) - _MAX_PENDING_ROWS :]))


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
) -> Mapping[_RowKey, _RowValue]:
    """Returns the rows still failing with connection errors once retries exhaust, for requeueing."""
    outcomes: Final = {key: await _attempt_upsert(upsert_row, key, value) for key, value in rows.items()}
    for key, error in outcomes.items():
        if error is not None and not isinstance(error, DB_RETRY_SAFE_ERROR_TYPES):
            verbose_proxy_logger.warning(
                "Guardrail usage tracking: %s upsert failed for %s and is not safe to retry (non-fatal): %s",
                label,
                key,
                error,
            )
    retryable: Final = MappingProxyType(
        {key: rows[key] for key, error in outcomes.items() if isinstance(error, DB_RETRY_SAFE_ERROR_TYPES)}
    )
    if not retryable:
        return MappingProxyType({})
    if retries_left == 0:
        for key in retryable:
            verbose_proxy_logger.warning(
                "Guardrail usage tracking: %s upsert failed for %s after %d retries; requeued for the next flush "
                "(non-fatal): %s",
                label,
                key,
                _UPSERT_RETRY_TIMES,
                outcomes[key],
            )
        return retryable
    await sleep(2 ** (_UPSERT_RETRY_TIMES - retries_left))
    return await _upsert_rows_with_retry(retryable, upsert_row, label, sleep, retries_left - 1)


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


def _iter_usage_unit_increments(
    logs_to_process: Sequence[Mapping[str, Any]],
) -> Iterator[tuple[_UsageUnitKey, _UsageUnitIncrement]]:
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
            cost_by_unit = billed_guardrail_cost_by_unit(entry)
            for unit_name, units in usage.items():
                if isinstance(units, int) and not isinstance(units, bool) and units > 0:
                    key = _UsageUnitKey(guardrail_id, date_key, team_id, api_key, str(unit_name))
                    cost = cost_by_unit.get(str(unit_name)) if cost_by_unit is not None else None
                    yield key, _UsageUnitIncrement(units=units, cost=cost)


def _sum_usage_unit_increments(
    logs_to_process: Sequence[Mapping[str, Any]],
) -> Mapping[_UsageUnitKey, _UsageUnitIncrement]:
    ordered: Final = sorted(_iter_usage_unit_increments(logs_to_process), key=itemgetter(0))
    return MappingProxyType(
        {
            key: _summed_increments(increment for _, increment in group)
            for key, group in groupby(ordered, key=itemgetter(0))
        }
    )


async def _upsert_usage_unit_row(
    prisma_client: PrismaClient, key: _UsageUnitKey, increment: _UsageUnitIncrement
) -> None:
    row: Final[prisma_types.LiteLLM_DailyGuardrailUsageUnitsCreateInput] = {
        "guardrail_id": key.guardrail_id,
        "date": key.date,
        "team_id": key.team_id,
        "api_key": key.api_key,
        "usage_unit": key.usage_unit,
        "units": increment.units,
        "cost": increment.cost,
    }
    where: Final[_UsageUnitWhereUnique] = {
        "guardrail_id_date_team_id_api_key_usage_unit": {
            "guardrail_id": key.guardrail_id,
            "date": key.date,
            "team_id": key.team_id,
            "api_key": key.api_key,
            "usage_unit": key.usage_unit,
        }
    }
    # NULL + x stays NULL in SQL, so an unknown cost stays unknown; writing NULL outright makes it so
    data: Final[prisma_types.LiteLLM_DailyGuardrailUsageUnitsUpsertInput] = {
        "create": row,
        "update": {
            "units": {"increment": increment.units},
            "cost": {"increment": increment.cost} if increment.cost is not None else None,
        },
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
    pending: PendingRollups = _PENDING_ROLLUPS,
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

    async with pending.lock:
        pending_metrics: Final = pending.metrics
        pending_units: Final = pending.units
        pending.metrics = MappingProxyType({})
        pending.units = MappingProxyType({})

    # Upsert daily guardrail metrics (counts only; latency/score dropped)
    evaluated_metrics: Final = MappingProxyType(
        {key: agg for key, agg in daily_guardrail.items() if int(agg["requests_evaluated"]) > 0}
    )
    metrics_rows: Final = _merged_metric_rows(pending_metrics, evaluated_metrics)
    unit_rows: Final = _merged_unit_rows(pending_units, _sum_usage_unit_increments(logs_to_process))

    if not metrics_rows and not index_rows and not unit_rows:
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

        failed_metrics: Final = await _upsert_rows_with_retry(
            metrics_rows, partial(_upsert_metrics_row, prisma_client), "daily metrics", sleep
        )
        failed_units: Final = await _upsert_rows_with_retry(
            unit_rows, partial(_upsert_usage_unit_row, prisma_client), "usage unit", sleep
        )
        if failed_metrics or failed_units:
            async with pending.lock:
                pending.metrics = _capped(_merged_metric_rows(pending.metrics, failed_metrics), "daily metrics")
                pending.units = _capped(_merged_unit_rows(pending.units, failed_units), "usage unit")
    except Exception as e:
        verbose_proxy_logger.warning("Guardrail usage tracking failed (non-fatal): %s", e)
