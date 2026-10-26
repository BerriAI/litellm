from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Final

import pytest

from litellm.repositories.spend_rollup_repository import ClickHouseSpendRollupRepository
from litellm.rust_bridge.clickhouse import RollupDimension, SpendRollupRequest
from litellm.types.repositories.daily_activity import KeyMetadataRow, SpendLogsWindow
from litellm.types.repositories.spend_rollup import RollupWindow, SpendRollupScope

_START: Final = datetime(2026, 10, 24, 21, 0, tzinfo=timezone.utc)
_END: Final = datetime(2026, 10, 24, 23, 0, tzinfo=timezone.utc)


def _row(bucket: str | None = "2026-10-24T21:00:00", **columns: object) -> dict[str, object]:
    return {
        "bucket": bucket,
        "spend": 1.0,
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
        "api_requests": 2,
        "successful_requests": 2,
        "failed_requests": 0,
        "total_response_time_ms": 300,
        "timed_requests": 2,
        "distinct_api_keys": 3,
        **columns,
    }


class _FakeStorage:
    def __init__(self) -> None:
        self.requests: list[SpendRollupRequest] = []

    async def fetch_rollup(self, request: SpendRollupRequest) -> tuple[Mapping[str, object], ...]:
        self.requests.append(request)
        if request.bucket is None:
            return (_row(None, spend=9.0),)
        if request.group_by == (RollupDimension.API_KEY,):
            return (_row(api_key="key-top", spend=5.0), _row(api_key="key-low", spend=0.5))
        if request.group_by == (RollupDimension.MODEL, RollupDimension.API_KEY):
            return (_row(model="gpt-4o", api_key="key-top"), _row(model="gpt-4o", api_key="key-low"))
        if request.group_by == (RollupDimension.TAG, RollupDimension.API_KEY):
            return (_row(tag="prod", api_key="key-low"),)
        if request.group_by == (RollupDimension.TAG,):
            return (_row(tag="prod", spend=4.0), _row(tag="", spend=1.0))
        return (
            _row(
                **{
                    dimension.value: "key-top" if dimension is RollupDimension.API_KEY else f"{dimension.value}-1"
                    for dimension in request.group_by
                }
            ),
        )


class _KeyReads:
    async def key_metadata(
        self, api_keys: frozenset[str], window: SpendLogsWindow | None
    ) -> Mapping[str, KeyMetadataRow]:
        return {}


def _scope(**overrides: object) -> SpendRollupScope:
    values: dict[str, object] = {
        "dimension": RollupDimension.TAG,
        "entity_ids": ("prod",),
        "api_keys": None,
        "model": None,
        "window": RollupWindow(start=_START, end=_END, bucket="1h"),
    }
    values.update(overrides)
    return SpendRollupScope(**values)  # pyright: ignore[reportArgumentType]


@pytest.mark.asyncio
async def test_aggregated_maps_each_grouping_level_and_trims_keys_to_the_limit() -> None:
    storage: Final = _FakeStorage()
    repository: Final = ClickHouseSpendRollupRepository(storage, key_reads=_KeyReads())  # pyright: ignore[reportArgumentType]

    rows: Final = await repository.aggregated(_scope(), include_entity_breakdown=True, api_key_limit=1)

    levels: Final = sorted({row.group_level for row in rows.grouping_rows})
    assert levels == [15, 23, 27, 31, 47, 55, 59, 63, 127]
    grand_total: Final = rows.grouping_rows[0]
    assert (grand_total.group_level, grand_total.date, grand_total.spend, grand_total.distinct_api_keys) == (
        127,
        None,
        9.0,
        3,
    )
    assert rows.distinct_api_keys == 3
    keyed: Final = [row.api_key for row in rows.grouping_rows if row.api_key]
    assert keyed and set(keyed) == {"key-top"}
    model_row: Final = next(row for row in rows.grouping_rows if row.group_level == 47)
    assert (model_row.model, model_row.date, model_row.compression_savings_spend) == (
        "model-1",
        "2026-10-24T21:00:00",
        None,
    )
    assert rows.entity_rows is not None
    assert [(row.entity_id, row.api_key_rolled, row.distinct_api_keys, row.spend) for row in rows.entity_rows] == [
        ("prod", 1, 3, 4.0),
        (None, 1, 3, 1.0),
    ]


@pytest.mark.asyncio
async def test_aggregated_pushes_scope_filters_into_every_request() -> None:
    storage: Final = _FakeStorage()
    repository: Final = ClickHouseSpendRollupRepository(storage, key_reads=_KeyReads())  # pyright: ignore[reportArgumentType]

    await repository.aggregated(
        _scope(api_keys=("hashed-a", "hashed-b"), model="gpt-4o"), include_entity_breakdown=False, api_key_limit=10
    )

    assert len(storage.requests) == 9
    assert {request.bucket for request in storage.requests} == {None, "1h"}
    assert all(
        request.filters
        == {
            RollupDimension.TAG: ("prod",),
            RollupDimension.API_KEY: ("hashed-a", "hashed-b"),
            RollupDimension.MODEL: ("gpt-4o",),
        }
        for request in storage.requests
    )
    assert all((request.start, request.end) == (_START, _END) for request in storage.requests)
