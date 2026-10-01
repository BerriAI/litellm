from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Final

import pytest
from pydantic import ValidationError

from litellm import constants
from litellm.repositories.daily_activity_repository import DailyActivityRepository
from litellm.repositories.daily_activity_sql import (
    ExportCursor,
    build_cache_leakage_keys_sql,
    build_entity_rollup_sql,
    build_export_sql,
    build_key_page_sql,
    build_key_search_sql,
    build_model_top_keys_sql,
)
from litellm.types.repositories.daily_activity import (
    DailyActivityScope,
    DailyActivityTable,
    ExportType,
    KeyMetadataRow,
    KeyPage,
    KeySpendRow,
    SpendLogsWindow,
)


@dataclass(frozen=True, slots=True)
class _FakeVerificationToken:
    token: str
    key_alias: str | None
    team_id: str | None
    user_id: str | None
    metadata: object | None


@dataclass(frozen=True, slots=True)
class _FakeDeletedVerificationToken(_FakeVerificationToken):
    deleted_at: datetime


def _scope(
    *,
    table: DailyActivityTable = DailyActivityTable.USER,
    entity_ids: tuple[str, ...] | None = ("user-1",),
    api_keys: tuple[str, ...] | None = None,
    exclude_entity_ids: tuple[str, ...] = (),
    model: str | None = None,
) -> DailyActivityScope:
    entity_field: Final = {
        DailyActivityTable.USER: "user_id",
        DailyActivityTable.TEAM: "team_id",
        DailyActivityTable.TAG: "tag",
        DailyActivityTable.ORGANIZATION: "organization_id",
        DailyActivityTable.CUSTOMER: "end_user_id",
        DailyActivityTable.AGENT: "agent_id",
    }[table]
    return DailyActivityScope(
        table=table,
        entity_id_field=entity_field,
        entity_ids=entity_ids,
        exclude_entity_ids=exclude_entity_ids,
        api_keys=api_keys,
        start_date="2026-01-01",
        end_date="2026-01-31",
        model=model,
        timezone_offset_minutes=None,
    )


def _key_spend_row(api_key: str) -> dict[str, object]:
    return {
        "api_key": api_key,
        "spend": 1.0,
        "prompt_tokens": 10,
        "completion_tokens": 2,
        "total_tokens": 12,
        "api_requests": 1,
        "successful_requests": 1,
        "failed_requests": 0,
        "cache_read_input_tokens": 3,
        "cache_creation_input_tokens": 1,
    }


def _export_row(api_key: str | None) -> dict[str, object]:
    return {
        "date": "2026-01-01",
        "entity_id": "user-1",
        "entity_alias": None,
        "api_key": api_key,
        "key_alias": None,
        "user_id": None,
        "user_email": None,
        "model": None,
        "spend": 1.0,
        "flat_cost": 0.0,
        "prompt_tokens": 10,
        "completion_tokens": 2,
        "api_requests": 1,
        "successful_requests": 1,
        "failed_requests": 0,
        "cache_read_input_tokens": 3,
        "cache_creation_input_tokens": 1,
    }


class _FakeTable:
    def __init__(self, rows: Sequence[object] = ()) -> None:
        self.rows: Final = tuple(rows)
        self.find_many_calls: list[Mapping[str, object]] = []
        self.count_calls: list[Mapping[str, object]] = []
        self.pagination_calls: list[tuple[int | None, int | None, tuple[Mapping[str, str], ...] | None]] = []

    async def find_many(
        self,
        *,
        where: Mapping[str, object],
        skip: int | None = None,
        take: int | None = None,
        order: tuple[Mapping[str, str], ...] | None = None,
    ) -> tuple[object, ...]:
        self.find_many_calls.append(where)
        self.pagination_calls.append((skip, take, order))
        if "token" not in where:
            return self.rows
        token_filter: Final = where["token"]
        if not isinstance(token_filter, Mapping):
            return ()
        token_values: Final = token_filter.get("in")
        if not isinstance(token_values, list):
            return ()
        return tuple(row for row in self.rows if isinstance(row, _FakeVerificationToken) and row.token in token_values)

    async def count(self, *, where: Mapping[str, object]) -> int:
        self.count_calls.append(where)
        return len(self.rows)


class _FailingTable(_FakeTable):
    def __init__(self, failure: str) -> None:
        super().__init__()
        self.failure: Final = failure

    async def find_many(
        self,
        *,
        where: Mapping[str, object],
        skip: int | None = None,
        take: int | None = None,
        order: tuple[Mapping[str, str], ...] | None = None,
    ) -> tuple[object, ...]:
        raise RuntimeError(f"{self.failure}: {where!r} {skip!r} {take!r} {order!r}")


class _FakeDatabase:
    def __init__(self, responses: Sequence[Sequence[Mapping[str, object]] | None] = ()) -> None:
        self.responses = tuple(responses)
        self.query_calls: list[tuple[str, tuple[object, ...]]] = []
        self.litellm_verificationtoken = _FakeTable()
        self.litellm_deletedverificationtoken = _FakeTable()
        self.litellm_dailyuserspend = _FakeTable()
        self.litellm_dailyteamspend = _FakeTable()
        self.litellm_dailytagspend = _FakeTable()
        self.litellm_dailyorganizationspend = _FakeTable()
        self.litellm_dailyenduserspend = _FakeTable()
        self.litellm_dailyagentspend = _FakeTable()

    async def query_raw(self, query: str, *params: object) -> Sequence[Mapping[str, object]] | None:
        self.query_calls.append((query, params))
        response_index: Final = len(self.query_calls) - 1
        if response_index >= len(self.responses):
            return ()
        return self.responses[response_index]


class _FakePrismaClient:
    def __init__(self, database: _FakeDatabase) -> None:
        self.db: Final = database


class _ProxyReads:
    def __init__(self) -> None:
        self.recovery_calls: list[tuple[Mapping[str, KeyMetadataRow], frozenset[str], SpendLogsWindow | None]] = []

    async def recover_key_metadata(
        self,
        resolved: Mapping[str, KeyMetadataRow],
        api_keys: frozenset[str],
        window: SpendLogsWindow | None,
    ) -> Mapping[str, KeyMetadataRow]:
        self.recovery_calls.append((resolved, api_keys, window))
        return resolved


def _repository(
    database: _FakeDatabase, proxy_reads: _ProxyReads | None = None
) -> tuple[DailyActivityRepository, _ProxyReads]:
    reads: Final = proxy_reads if proxy_reads is not None else _ProxyReads()
    return DailyActivityRepository(_FakePrismaClient(database), proxy_reads=reads), reads


@pytest.mark.asyncio
async def test_key_methods_send_builder_queries_with_caller_limits() -> None:
    database = _FakeDatabase(((_key_spend_row("key-a"),), (_key_spend_row("key-b"),), (_key_spend_row("key-c"),)))
    repository, _ = _repository(database)
    scope = _scope()

    assert await repository.search_keys(scope, search="key", limit=2) == ("key-a",)
    model_keys: Final = await repository.model_top_keys(scope, model_group="model-a", by_model_group=True, limit=2)
    leakage_keys: Final = await repository.cache_leakage_keys(scope, limit=2)

    assert tuple(row.api_key for row in model_keys) == ("key-b",)
    assert tuple(row.api_key for row in leakage_keys) == ("key-c",)
    assert model_keys[0].spend == 1.0
    assert leakage_keys[0].prompt_tokens - leakage_keys[0].cache_read_input_tokens == 7
    assert database.query_calls == [
        (
            build_key_search_sql(scope, search="key", limit=2).sql,
            build_key_search_sql(scope, search="key", limit=2).params,
        ),
        (
            build_model_top_keys_sql(scope, model_group="model-a", by_model_group=True, limit=2).sql,
            build_model_top_keys_sql(scope, model_group="model-a", by_model_group=True, limit=2).params,
        ),
        (
            build_cache_leakage_keys_sql(scope, limit=2).sql,
            build_cache_leakage_keys_sql(scope, limit=2).params,
        ),
    ]


@pytest.mark.asyncio
async def test_key_page_maps_rows_and_keeps_total_for_an_empty_page() -> None:
    database = _FakeDatabase(
        (
            ({"total_api_keys": 2, **_key_spend_row("key-a")},),
            ({"total_api_keys": 2, "api_key": None},),
        )
    )
    repository, _ = _repository(database)
    scope = _scope()

    first_page: Final = await repository.key_page(scope, offset=0, limit=1)
    empty_page: Final = await repository.key_page(scope, offset=2, limit=1)

    assert first_page == KeyPage(
        rows=(
            KeySpendRow(
                api_key="key-a",
                spend=1.0,
                prompt_tokens=10,
                completion_tokens=2,
                total_tokens=12,
                api_requests=1,
                successful_requests=1,
                failed_requests=0,
                cache_read_input_tokens=3,
                cache_creation_input_tokens=1,
            ),
        ),
        total_api_keys=2,
    )
    assert empty_page == KeyPage(rows=(), total_api_keys=2)
    assert database.query_calls == [
        (
            build_key_page_sql(scope, offset=0, limit=1).sql,
            build_key_page_sql(scope, offset=0, limit=1).params,
        ),
        (
            build_key_page_sql(scope, offset=2, limit=1).sql,
            build_key_page_sql(scope, offset=2, limit=1).params,
        ),
    ]


@pytest.mark.asyncio
async def test_key_methods_reject_limits_outside_bounds() -> None:
    database = _FakeDatabase()
    repository, _ = _repository(database)

    with pytest.raises(ValueError, match="limit"):
        await repository.search_keys(_scope(), search="key", limit=0)
    with pytest.raises(ValueError, match="limit"):
        await repository.model_top_keys(_scope(), model_group="model-a", by_model_group=False, limit=0)
    with pytest.raises(ValueError, match="limit"):
        await repository.cache_leakage_keys(_scope(), limit=0)
    with pytest.raises(ValueError, match="limit"):
        await repository.search_keys(_scope(), search="key", limit=constants.USAGE_KEY_SEARCH_MAX + 1)
    with pytest.raises(ValueError, match="limit"):
        await repository.model_top_keys(
            _scope(), model_group="model-a", by_model_group=False, limit=constants.USAGE_MODEL_TOP_KEYS_MAX + 1
        )
    with pytest.raises(ValueError, match="limit"):
        await repository.cache_leakage_keys(_scope(), limit=constants.USAGE_CACHE_LEAKAGE_KEYS_MAX + 1)
    assert database.query_calls == []


@pytest.mark.asyncio
async def test_key_spend_validation_rejects_malformed_rows() -> None:
    repository, _ = _repository(_FakeDatabase((({"api_key": "missing-metrics"},),)))

    with pytest.raises(ValidationError):
        await repository.search_keys(_scope(), search="key", limit=1)


@pytest.mark.asyncio
async def test_key_metadata_prefers_active_rows_and_recovers_all_requested_keys() -> None:
    database = _FakeDatabase()
    active: Final = _FakeVerificationToken(
        token="active",
        key_alias="current",
        team_id="team-active",
        user_id="user-active",
        metadata={"tags": ["production", "internal"]},
    )
    deleted_active_duplicate: Final = _FakeDeletedVerificationToken(
        token="active",
        key_alias="stale",
        team_id="team-stale",
        user_id="user-stale",
        metadata={"tags": []},
        deleted_at=datetime(2026, 1, 3, tzinfo=timezone.utc),
    )
    deleted_older: Final = _FakeDeletedVerificationToken(
        token="deleted",
        key_alias="older",
        team_id=None,
        user_id=None,
        metadata={"tags": "invalid"},
        deleted_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
    )
    deleted_newer: Final = _FakeDeletedVerificationToken(
        token="deleted",
        key_alias="newer",
        team_id=None,
        user_id=None,
        metadata={"tags": ["archived"]},
        deleted_at=datetime(2026, 1, 4, tzinfo=timezone.utc),
    )
    malformed_non_list: Final = _FakeVerificationToken(
        token="malformed-non-list",
        key_alias=None,
        team_id=None,
        user_id=None,
        metadata={"tags": "invalid"},
    )
    malformed_list: Final = _FakeVerificationToken(
        token="malformed-list",
        key_alias=None,
        team_id=None,
        user_id=None,
        metadata={"tags": [1]},
    )
    database.litellm_verificationtoken = _FakeTable((active, malformed_non_list, malformed_list))
    database.litellm_deletedverificationtoken = _FakeTable((deleted_active_duplicate, deleted_older, deleted_newer))
    proxy_reads: Final = _ProxyReads()
    repository, _ = _repository(database, proxy_reads)
    window: Final = (datetime(2026, 1, 1), datetime(2026, 2, 1))
    requested: Final = frozenset(("active", "deleted", "malformed-non-list", "malformed-list", "unresolved"))

    result = await repository.key_metadata(requested, window)

    assert result["active"] == KeyMetadataRow(
        api_key="active",
        key_alias="current",
        team_id="team-active",
        user_id="user-active",
        user_email=None,
        key_exists=True,
        tags=("production", "internal"),
    )
    assert result["deleted"].key_alias == "newer"
    assert result["deleted"].key_exists is False
    assert result["deleted"].tags == ("archived",)
    assert result["malformed-non-list"].tags == ()
    assert result["malformed-list"].tags == ()
    assert len(database.litellm_deletedverificationtoken.find_many_calls) == 1
    assert set(database.litellm_deletedverificationtoken.find_many_calls[0]["token"]["in"]) == {
        "deleted",
        "unresolved",
    }
    assert proxy_reads.recovery_calls == [
        (
            result,
            requested,
            window,
        )
    ]


@pytest.mark.asyncio
async def test_key_metadata_continues_with_active_rows_when_deleted_lookup_fails() -> None:
    database = _FakeDatabase()
    active: Final = _FakeVerificationToken(
        token="active",
        key_alias="current",
        team_id=None,
        user_id=None,
        metadata={"tags": []},
    )
    database.litellm_verificationtoken = _FakeTable((active,))
    database.litellm_deletedverificationtoken = _FailingTable("deleted token query failed")
    repository, proxy_reads = _repository(database)

    result = await repository.key_metadata(frozenset(("active", "deleted")), None)

    assert result["active"].key_alias == "current"
    assert tuple(proxy_reads.recovery_calls[0][0]) == ("active",)
    assert proxy_reads.recovery_calls[0][1] == frozenset(("active", "deleted"))


@pytest.mark.asyncio
async def test_key_metadata_empty_set_does_not_query_tables() -> None:
    database = _FakeDatabase()
    repository, proxy_reads = _repository(database)

    assert await repository.key_metadata(frozenset(), None) == {}
    assert database.litellm_verificationtoken.find_many_calls == []
    assert proxy_reads.recovery_calls == []


@pytest.mark.asyncio
async def test_key_metadata_propagates_active_token_lookup_failures() -> None:
    database = _FakeDatabase()
    database.litellm_verificationtoken = _FailingTable("active token query failed")
    repository, _ = _repository(database)

    with pytest.raises(RuntimeError, match="active token query failed"):
        await repository.key_metadata(frozenset(("active",)), None)

    assert database.litellm_deletedverificationtoken.find_many_calls == []


@pytest.mark.asyncio
async def test_aggregated_normalizes_a_null_raw_query_result() -> None:
    database = _FakeDatabase((None,))
    repository, _ = _repository(database)

    result = await repository.aggregated(
        _scope(), include_entity_breakdown=False, api_key_limit=constants.USAGE_TOP_API_KEYS_DEFAULT
    )

    assert result.grouping_rows == ()
    assert result.entity_rows is None
    assert result.distinct_api_keys == 0
    assert len(database.query_calls) == 1


@pytest.mark.asyncio
async def test_aggregated_passes_api_key_limit_to_entity_rollup_query() -> None:
    database = _FakeDatabase(((), ()))
    repository, _ = _repository(database)
    scope = _scope(table=DailyActivityTable.TEAM)

    result = await repository.aggregated(scope, include_entity_breakdown=True, api_key_limit=3)

    assert result.entity_rows == ()
    assert database.query_calls[1] == (
        build_entity_rollup_sql(scope, api_key_limit=3).sql,
        build_entity_rollup_sql(scope, api_key_limit=3).params,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("table", "entity_field"),
    [
        (DailyActivityTable.USER, "user_id"),
        (DailyActivityTable.TEAM, "team_id"),
        (DailyActivityTable.TAG, "tag"),
        (DailyActivityTable.ORGANIZATION, "organization_id"),
        (DailyActivityTable.CUSTOMER, "end_user_id"),
        (DailyActivityTable.AGENT, "agent_id"),
    ],
)
async def test_daily_rows_selects_the_table_and_applies_filters_and_pagination(
    table: DailyActivityTable, entity_field: str
) -> None:
    database = _FakeDatabase()
    repository, _ = _repository(database)
    scope = _scope(
        table=table,
        entity_ids=("entity-1",),
        exclude_entity_ids=("excluded-1",),
        api_keys=("key-1",),
        model="model-1",
    )

    result = await repository.daily_rows(scope, page=3, page_size=2)

    expected_where: Final = {
        "date": {"gte": "2026-01-01", "lte": "2026-01-31"},
        entity_field: {"in": ["entity-1"], "not": {"in": ["excluded-1"]}},
        "model": "model-1",
        "api_key": {"in": ["key-1"]},
    }
    tables: Final = {
        DailyActivityTable.USER: database.litellm_dailyuserspend,
        DailyActivityTable.TEAM: database.litellm_dailyteamspend,
        DailyActivityTable.TAG: database.litellm_dailytagspend,
        DailyActivityTable.ORGANIZATION: database.litellm_dailyorganizationspend,
        DailyActivityTable.CUSTOMER: database.litellm_dailyenduserspend,
        DailyActivityTable.AGENT: database.litellm_dailyagentspend,
    }
    selected_table: Final = tables[table]

    assert result.total_count == 0
    assert result.rows == ()
    assert selected_table.count_calls == [expected_where]
    assert selected_table.find_many_calls == [expected_where]
    assert selected_table.pagination_calls == [(4, 2, ({"date": "desc"}, {"id": "asc"}))]
    assert sum(len(daily_table.find_many_calls) for daily_table in tables.values()) == 1


@pytest.mark.asyncio
async def test_export_is_lazy_and_uses_the_last_row_as_the_next_cursor(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(constants, "USAGE_EXPORT_BATCH_SIZE", 2)
    database = _FakeDatabase(
        (
            (_export_row("key-1"), _export_row("key-2")),
            (_export_row("key-3"), _export_row("key-4")),
            (_export_row("key-5"),),
        )
    )
    repository, _ = _repository(database)
    rows = repository.export_rows(_scope(), export_type=ExportType.DAILY_WITH_KEYS)

    assert database.query_calls == []
    assert (await rows.__anext__()).api_key == "key-1"
    assert len(database.query_calls) == 1
    results = [row async for row in rows]

    assert [row.api_key for row in results] == ["key-2", "key-3", "key-4", "key-5"]
    assert len(database.query_calls) == 3
    assert database.query_calls[1][1][-4:] == ("2026-01-01", "user-1", "key-2", 2)
    assert database.query_calls[2][1][-4:] == ("2026-01-01", "user-1", "key-4", 2)
    assert (
        build_export_sql(
            _scope(),
            export_type=ExportType.DAILY_WITH_KEYS,
            after=ExportCursor("2026-01-01", "user-1", "key-2"),
            batch_size=2,
        ).params
        == database.query_calls[1][1]
    )
