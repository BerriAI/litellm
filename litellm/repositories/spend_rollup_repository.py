import asyncio
from collections.abc import Mapping, Sequence
from typing import Final

from typing_extensions import TypedDict

from litellm.rust_bridge.clickhouse import ClickHouseSpendStorage, RollupDimension, SpendRollupRequest
from litellm.types.repositories.daily_activity import (
    AggregatedRows,
    EntityRollupRow,
    GroupingSetsRow,
    KeyMetadataReads,
    KeyMetadataRow,
    SpendLogsWindow,
)
from litellm.types.repositories.spend_rollup import SpendRollupScope

_GRAND_TOTAL: Final = 127
_BUCKET_ONLY: Final = 63
_GROUPING_LEVELS: Final[tuple[tuple[int, tuple[RollupDimension, ...]], ...]] = (
    (_BUCKET_ONLY, ()),
    (31, (RollupDimension.API_KEY,)),
    (47, (RollupDimension.MODEL,)),
    (15, (RollupDimension.MODEL, RollupDimension.API_KEY)),
    (55, (RollupDimension.MODEL_GROUP,)),
    (23, (RollupDimension.MODEL_GROUP, RollupDimension.API_KEY)),
    (59, (RollupDimension.PROVIDER,)),
    (27, (RollupDimension.PROVIDER, RollupDimension.API_KEY)),
)

_RollupRow = Mapping[str, object]


def _text(row: _RollupRow, dimension: RollupDimension) -> str | None:
    value = row.get(dimension.value)
    return value if isinstance(value, str) and value else None


def _int(row: _RollupRow, key: str) -> int:
    value = row.get(key)
    return value if isinstance(value, int) else 0


def _float(row: _RollupRow, key: str) -> float:
    value = row.get(key)
    return float(value) if isinstance(value, (int, float)) else 0.0


def _bucket(row: _RollupRow) -> str | None:
    value = row.get("bucket")
    return value if isinstance(value, str) else None


class _Metrics(TypedDict):
    date: str | None
    api_key: str | None
    spend: float
    prompt_tokens: int
    completion_tokens: int
    cache_read_input_tokens: int
    cache_creation_input_tokens: int
    compression_saved_tokens: None
    compression_savings_spend: None
    prompt_caching_savings_spend: None
    gateway_injected_caching_savings_spend: None
    autorouter_savings_spend: None
    api_requests: int
    successful_requests: int
    failed_requests: int
    total_response_time_ms: int
    timed_requests: int


def _metrics(row: _RollupRow) -> _Metrics:
    return {
        "date": _bucket(row),
        "api_key": _text(row, RollupDimension.API_KEY),
        "spend": _float(row, "spend"),
        "prompt_tokens": _int(row, "prompt_tokens"),
        "completion_tokens": _int(row, "completion_tokens"),
        "cache_read_input_tokens": _int(row, "cache_read_input_tokens"),
        "cache_creation_input_tokens": _int(row, "cache_creation_input_tokens"),
        "compression_saved_tokens": None,
        "compression_savings_spend": None,
        "prompt_caching_savings_spend": None,
        "gateway_injected_caching_savings_spend": None,
        "autorouter_savings_spend": None,
        "api_requests": _int(row, "api_requests"),
        "successful_requests": _int(row, "successful_requests"),
        "failed_requests": _int(row, "failed_requests"),
        "total_response_time_ms": _int(row, "total_response_time_ms"),
        "timed_requests": _int(row, "timed_requests"),
    }


def grouping_row(row: _RollupRow, level: int) -> GroupingSetsRow:
    return GroupingSetsRow(
        **_metrics(row),
        model=_text(row, RollupDimension.MODEL),
        model_group=_text(row, RollupDimension.MODEL_GROUP),
        custom_llm_provider=_text(row, RollupDimension.PROVIDER),
        mcp_namespaced_tool_name=None,
        endpoint=None,
        group_level=level,
        distinct_api_keys=_int(row, "distinct_api_keys") if level == _GRAND_TOTAL else None,
    )


def entity_row(row: _RollupRow, dimension: RollupDimension, *, api_key_rolled: bool) -> EntityRollupRow:
    return EntityRollupRow(
        **_metrics(row),
        entity_id=_text(row, dimension),
        api_key_rolled=int(api_key_rolled),
        distinct_api_keys=_int(row, "distinct_api_keys") if api_key_rolled else None,
    )


def top_api_keys(rows: Sequence[GroupingSetsRow], limit: int) -> frozenset[str]:
    spend_by_key: dict[str, float] = {}
    for row in rows:
        if row.group_level == 31 and row.api_key:
            spend_by_key[row.api_key] = spend_by_key.get(row.api_key, 0.0) + (row.spend or 0.0)
    return frozenset(sorted(spend_by_key, key=lambda key: (-spend_by_key[key], key))[:limit])


class ClickHouseSpendRollupRepository:
    """Serves the Usage aggregate shape from bounded ClickHouse rollups; one query per grouping level."""

    def __init__(self, storage: ClickHouseSpendStorage, *, key_reads: KeyMetadataReads) -> None:
        self._storage = storage
        self._key_reads = key_reads

    def _request(
        self, scope: SpendRollupScope, group_by: tuple[RollupDimension, ...], *, bucketed: bool
    ) -> SpendRollupRequest:
        filters: dict[RollupDimension, tuple[str, ...]] = {}
        if scope.entity_ids:
            filters[scope.dimension] = scope.entity_ids
        if scope.api_keys:
            filters[RollupDimension.API_KEY] = scope.api_keys
        if scope.model:
            filters[RollupDimension.MODEL] = (scope.model,)
        return SpendRollupRequest(
            start=scope.window.start,
            end=scope.window.end,
            bucket=scope.window.bucket if bucketed else None,
            group_by=group_by,
            filters=filters,
        )

    async def aggregated(
        self, scope: SpendRollupScope, *, include_entity_breakdown: bool, api_key_limit: int
    ) -> AggregatedRows:
        requests: Final = [self._request(scope, (), bucketed=False)] + [
            self._request(scope, dimensions, bucketed=True) for _, dimensions in _GROUPING_LEVELS
        ]
        entity_requests: Final = (
            [
                self._request(scope, (scope.dimension,), bucketed=True),
                self._request(scope, (scope.dimension, RollupDimension.API_KEY), bucketed=True),
            ]
            if include_entity_breakdown
            else []
        )
        results: Final = await asyncio.gather(
            *(self._storage.fetch_rollup(request) for request in [*requests, *entity_requests])
        )
        grouping_rows: list[GroupingSetsRow] = [grouping_row(row, _GRAND_TOTAL) for row in results[0]]
        for (level, _), rows in zip(_GROUPING_LEVELS, results[1 : 1 + len(_GROUPING_LEVELS)], strict=True):
            grouping_rows.extend(grouping_row(row, level) for row in rows)
        keep: Final = top_api_keys(grouping_rows, api_key_limit)
        grouping_rows = [row for row in grouping_rows if not row.api_key or row.api_key in keep]
        entity_rows: tuple[EntityRollupRow, ...] | None = None
        if include_entity_breakdown:
            rolled, by_key = results[-2], results[-1]
            entity_rows = tuple(entity_row(row, scope.dimension, api_key_rolled=True) for row in rolled) + tuple(
                entity_row(row, scope.dimension, api_key_rolled=False)
                for row in by_key
                if _text(row, RollupDimension.API_KEY) in keep
            )
        distinct_api_keys: Final = next(
            (row.distinct_api_keys for row in grouping_rows if row.distinct_api_keys is not None), 0
        )
        return AggregatedRows(
            grouping_rows=tuple(grouping_rows), entity_rows=entity_rows, distinct_api_keys=distinct_api_keys
        )

    async def key_metadata(
        self, api_keys: frozenset[str], window: SpendLogsWindow | None
    ) -> Mapping[str, KeyMetadataRow]:
        return await self._key_reads.key_metadata(api_keys, window)
