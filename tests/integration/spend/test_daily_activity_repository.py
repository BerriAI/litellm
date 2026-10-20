import os
import uuid
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from math import isclose
from pathlib import Path
from types import MappingProxyType
from typing import Final, cast
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import psycopg
import pytest
from integration.spend._daily_activity_fixtures import (
    seed_daily_activity_fixture,
    seed_daily_tag_activity_fixture,
    seed_daily_tag_float_tie_fixture,
    seed_daily_team_exclusion_fixture,
    seed_daily_team_unassigned_fixture,
    seed_daily_user_page_fixture,
)
from prisma import Prisma
from psycopg import sql
from pydantic import TypeAdapter

from litellm import constants
from litellm.proxy.management_endpoints.common_daily_activity import get_daily_activity_aggregated
from litellm.repositories.chunked_in import find_many_in
from litellm.repositories.daily_activity_repository import DailyActivityDatabase, DailyActivityRepository
from litellm.types.repositories.daily_activity import (
    DailyActivityProxyReads,
    DailyActivityScope,
    DailyActivityTable,
    ExportType,
    KeyMetadataRow,
    SpendLogsWindow,
)


@dataclass(frozen=True, slots=True)
class _TagRollupMetrics:
    tag: str | None
    date: str
    spend: float
    api_requests: int
    prompt_tokens: int


@dataclass(frozen=True, slots=True)
class _TagApiKeyCount:
    tag: str | None
    distinct_api_keys: int


@dataclass(frozen=True, slots=True)
class _TagKeyMembershipCount:
    api_key: str
    tag_count: int


@dataclass(frozen=True, slots=True)
class _TagFloatSpend:
    api_key: str
    spend: float


@dataclass(frozen=True, slots=True)
class _TagRankedKey:
    api_key: str


@dataclass(frozen=True, slots=True)
class _TagDistinctKeyCount:
    total_api_keys: int


_TAG_ROLLUP_METRICS_ADAPTER: Final = TypeAdapter(tuple[_TagRollupMetrics, ...])
_TAG_API_KEY_COUNT_ADAPTER: Final = TypeAdapter(tuple[_TagApiKeyCount, ...])
_TAG_KEY_MEMBERSHIP_COUNT_ADAPTER: Final = TypeAdapter(tuple[_TagKeyMembershipCount, ...])
_TAG_FLOAT_SPEND_ADAPTER: Final = TypeAdapter(tuple[_TagFloatSpend, ...])
_TAG_RANKED_KEY_ADAPTER: Final = TypeAdapter(tuple[_TagRankedKey, ...])
_TAG_DISTINCT_KEY_COUNT_ADAPTER: Final = TypeAdapter(tuple[_TagDistinctKeyCount, ...])


def _scoped_url(url: str, schema: str) -> str:
    parsed: Final = urlsplit(url)
    return urlunsplit(parsed._replace(query=urlencode({**dict(parse_qsl(parsed.query)), "schema": schema})))


@asynccontextmanager
async def _daily_activity_database(
    *,
    include_tag_activity: bool = False,
    include_tag_float_tie_activity: bool = False,
    include_team_unassigned_activity: bool = False,
    include_team_exclusion_activity: bool = False,
    include_user_page_activity: bool = False,
) -> AsyncIterator[Prisma]:
    schema: Final = f"integration_{uuid.uuid4().hex}"
    url: Final = os.environ["DATABASE_URL"]
    with psycopg.connect(url, autocommit=True) as setup:
        setup.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
        try:
            with psycopg.connect(url) as connection:
                seed_daily_activity_fixture(
                    connection,
                    schema=schema,
                    ptu_sentinel_api_key=constants.PTU_SENTINEL_API_KEY,
                )
                if include_tag_activity:
                    seed_daily_tag_activity_fixture(connection, schema=schema)
                if include_tag_float_tie_activity:
                    seed_daily_tag_float_tie_fixture(connection, schema=schema)
                if include_team_unassigned_activity:
                    seed_daily_team_unassigned_fixture(
                        connection, schema=schema, ptu_sentinel_api_key=constants.PTU_SENTINEL_API_KEY
                    )
                if include_team_exclusion_activity:
                    seed_daily_team_exclusion_fixture(connection, schema=schema)
                if include_user_page_activity:
                    seed_daily_user_page_fixture(connection, schema=schema)
            database: Final = Prisma(datasource={"url": _scoped_url(url, schema)})
            await database.connect()
            try:
                yield database
            finally:
                await database.disconnect()
        finally:
            setup.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


@dataclass(frozen=True, slots=True)
class _PrismaDatabase:
    db: Prisma


@dataclass(frozen=True, slots=True)
class _ProxyReads(DailyActivityProxyReads):
    database: Prisma

    async def recover_key_metadata(
        self, resolved: Mapping[str, KeyMetadataRow], api_keys: frozenset[str], window: SpendLogsWindow | None
    ) -> Mapping[str, KeyMetadataRow]:
        user_ids: Final = frozenset(row.user_id for row in resolved.values() if row.user_id)
        user_rows: Final = await find_many_in(self.database.litellm_usertable, "user_id", user_ids) if user_ids else ()
        user_emails: Final = MappingProxyType({row.user_id: row.user_email for row in user_rows if row.user_email})
        return MappingProxyType(
            {
                key: KeyMetadataRow(
                    api_key=row.api_key,
                    key_alias=row.key_alias,
                    team_id=row.team_id,
                    user_id=row.user_id,
                    user_email=row.user_email or user_emails.get(row.user_id),
                    key_exists=row.key_exists,
                    tags=row.tags,
                )
                for key, row in resolved.items()
            }
        )


def _repository(database: Prisma) -> DailyActivityRepository:
    client: Final = cast(DailyActivityDatabase, _PrismaDatabase(database))
    return DailyActivityRepository(client, proxy_reads=_ProxyReads(database))


def _scope(
    table: DailyActivityTable,
    entity_id_field: str,
    entity_id: str,
    api_keys: tuple[str, ...] | None = None,
) -> DailyActivityScope:
    return DailyActivityScope(
        table=table,
        entity_id_field=entity_id_field,
        entity_ids=(entity_id,),
        exclude_entity_ids=(),
        api_keys=api_keys,
        start_date="2026-06-01",
        end_date="2026-06-01",
        model=None,
        timezone_offset_minutes=None,
    )


@pytest.mark.asyncio
async def test_repository_queries_and_exports_seeded_daily_activity(monkeypatch: pytest.MonkeyPatch) -> None:
    async with _daily_activity_database() as database:
        repository: Final = _repository(database)
        team_scope: Final = _scope(DailyActivityTable.TEAM, "team_id", "team-1")
        monkeypatch.setattr(constants, "USAGE_EXPORT_BATCH_SIZE", 2)
        aggregate: Final = await repository.aggregated(team_scope, include_entity_breakdown=True, api_key_limit=3)
        totals: Final = tuple(row for row in aggregate.grouping_rows if row.group_level == 127)
        assert len(totals) == 1
        assert totals[0].spend == 1273.0
        assert totals[0].ptu_flat_cost == 42.0
        assert aggregate.distinct_api_keys == 5
        grouped_keys: Final = frozenset(
            row.api_key for row in aggregate.grouping_rows if row.group_level == 31 and row.api_key
        )
        assert grouped_keys == frozenset(("key-a", "key-b", "key-c"))
        entity_totals: Final = tuple(row for row in aggregate.entity_rows or () if row.api_key_rolled)
        assert len(entity_totals) == 1
        assert entity_totals[0].spend == 1273.0
        assert entity_totals[0].ptu_flat_cost == 42.0

        targeted_model_keys: Final = await repository.model_top_keys(
            team_scope, model_group="model-target", by_model_group=False, limit=3
        )
        assert tuple(row.api_key for row in targeted_model_keys) == ("key-target",)
        popular_model_keys: Final = await repository.model_top_keys(
            team_scope, model_group="model-popular", by_model_group=False, limit=3
        )
        assert tuple(row.api_key for row in popular_model_keys) == ("key-a", "key-b", "key-c")
        assert await repository.search_keys(team_scope, search="target", limit=10) == ("key-target",)
        assert await repository.search_keys(team_scope, search="deleted-target", limit=20) == ("key-target",)
        leakage_keys: Final = await repository.cache_leakage_keys(team_scope, limit=2)
        assert tuple(row.api_key for row in leakage_keys) == ("key-cache", "key-c")

        exports: Final = tuple(
            [row async for row in repository.export_rows(team_scope, export_type=ExportType.DAILY_WITH_KEYS)]
        )
        assert tuple(row.api_key for row in exports) == (
            "key-a",
            "key-b",
            "key-c",
            "key-cache",
            "key-target",
        )
        assert sum(row.spend for row in exports) == 273.0
        assert sum(row.flat_cost for row in exports) == 0.0
        deleted_key_export: Final = next(row for row in exports if row.api_key == "key-target")
        assert (deleted_key_export.key_alias, deleted_key_export.user_id, deleted_key_export.user_email) == (
            "deleted-target",
            "user-1",
            "user@example.com",
        )
        user_exports: Final = tuple(
            [row async for row in repository.export_rows(team_scope, export_type=ExportType.DAILY_WITH_USERS)]
        )
        assert len(user_exports) == 1
        assert (user_exports[0].user_id, user_exports[0].user_email, user_exports[0].spend) == (
            "user-1",
            "user@example.com",
            273.0,
        )
        daily_export: Final = tuple(
            [row async for row in repository.export_rows(team_scope, export_type=ExportType.DAILY)]
        )
        assert len(daily_export) == 1
        assert daily_export[0].spend == 1273.0
        assert daily_export[0].flat_cost == 42.0

        metadata: Final = await repository.key_metadata(frozenset(("key-a", "key-target")), None)
        assert metadata["key-a"].key_exists is True
        assert metadata["key-a"].key_alias == "alias-a"
        assert metadata["key-a"].tags == ("blue", "gold")
        assert metadata["key-a"].user_email == "user@example.com"
        assert metadata["key-target"].key_exists is False
        assert metadata["key-target"].key_alias == "deleted-target"
        assert metadata["key-target"].tags == ("archived",)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("search", "api_keys", "expected_keys"),
    (
        ("needle-alias", None, ("needle-key",)),
        ("needle-user", None, ("needle-key",)),
        ("needle@example.com", None, ("needle-key",)),
        ("needle-alias", ("key-a",), ()),
    ),
)
async def test_search_keys_matches_token_metadata_outside_top_n_and_respects_scope(
    search: str,
    api_keys: tuple[str, ...] | None,
    expected_keys: tuple[str, ...],
) -> None:
    async with _daily_activity_database() as database:
        await database.execute_raw(
            """
            INSERT INTO "LiteLLM_DailyTeamSpend" (
                id, team_id, date, api_key, model, model_group, custom_llm_provider,
                mcp_namespaced_tool_name, endpoint, prompt_tokens, completion_tokens,
                cache_read_input_tokens, cache_creation_input_tokens, spend, api_requests,
                successful_requests, failed_requests, ptu_flat_cost, updated_at
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19::timestamp
            )
            """,
            "needle-row",
            "team-1",
            "2026-06-01",
            "needle-key",
            "model-needle",
            "",
            "provider-a",
            None,
            "/v1/chat/completions",
            1,
            1,
            0,
            0,
            0.5,
            1,
            1,
            0,
            0.0,
            "2026-06-01 12:00:00",
        )
        await database.execute_raw(
            """
            INSERT INTO "LiteLLM_VerificationToken" (token, key_alias, team_id, user_id, metadata, models)
            VALUES ($1, $2, $3, $4, $5::jsonb, $6::text[])
            """,
            "needle-key",
            "needle-alias",
            "team-1",
            "needle-user",
            '{"tags": []}',
            [],
        )
        await database.execute_raw(
            """
            INSERT INTO "LiteLLM_UserTable" (user_id, user_email, models)
            VALUES ($1, $2, $3::text[])
            """,
            "needle-user",
            "needle@example.com",
            [],
        )

        repository: Final = _repository(database)
        team_scope: Final = _scope(DailyActivityTable.TEAM, "team_id", "team-1", api_keys=api_keys)
        aggregate: Final = await repository.aggregated(team_scope, include_entity_breakdown=False, api_key_limit=1)
        top_keys: Final = frozenset(
            row.api_key for row in aggregate.grouping_rows if row.group_level == 31 and row.api_key
        )
        assert "needle-key" not in top_keys
        assert await repository.search_keys(team_scope, search=search, limit=10) == expected_keys


@pytest.mark.asyncio
async def test_aggregated_returns_totals_with_a_one_key_limit() -> None:
    async with _daily_activity_database() as database:
        aggregate: Final = await _repository(database).aggregated(
            _scope(DailyActivityTable.TEAM, "team_id", "team-1"),
            include_entity_breakdown=False,
            api_key_limit=1,
        )

        totals: Final = tuple(row for row in aggregate.grouping_rows if row.group_level == 127)
        per_key_rows: Final = tuple(row for row in aggregate.grouping_rows if row.api_key is not None)
        per_key_names: Final = frozenset(row.api_key for row in per_key_rows)
        assert len(totals) == 1
        assert totals[0].spend == 1273.0
        assert len(per_key_rows) == 6
        assert len(per_key_names) == 1


@pytest.mark.asyncio
async def test_tag_entity_rollups_bound_keys_and_preserve_full_scope_totals() -> None:
    async with _daily_activity_database(include_tag_activity=True) as database:
        scope: Final = DailyActivityScope(
            table=DailyActivityTable.TAG,
            entity_id_field="tag",
            entity_ids=None,
            exclude_entity_ids=(),
            api_keys=None,
            start_date="2026-06-01",
            end_date="2026-06-02",
            model=None,
            timezone_offset_minutes=None,
        )
        repository: Final = _repository(database)
        aggregate: Final = await repository.aggregated(scope, include_entity_breakdown=True, api_key_limit=3)
        independent_metrics: Final = _TAG_ROLLUP_METRICS_ADAPTER.validate_python(
            await database.query_raw(
                """
                SELECT tag, date, SUM(spend)::float AS spend,
                       SUM(api_requests)::bigint AS api_requests,
                       SUM(prompt_tokens)::bigint AS prompt_tokens
                FROM "LiteLLM_DailyTagSpend"
                WHERE date >= $1 AND date <= $2
                GROUP BY tag, date
                """,
                "2026-06-01",
                "2026-06-02",
            )
        )
        independent_key_counts: Final = _TAG_API_KEY_COUNT_ADAPTER.validate_python(
            await database.query_raw(
                """
                SELECT tag, COUNT(DISTINCT api_key)::bigint AS distinct_api_keys
                FROM "LiteLLM_DailyTagSpend"
                WHERE date >= $1 AND date <= $2 AND api_key <> $3
                GROUP BY tag
                """,
                "2026-06-01",
                "2026-06-02",
                constants.PTU_SENTINEL_API_KEY,
            )
        )
        independent_key_memberships: Final = _TAG_KEY_MEMBERSHIP_COUNT_ADAPTER.validate_python(
            await database.query_raw(
                """
                SELECT api_key, COUNT(DISTINCT tag)::bigint AS tag_count
                FROM "LiteLLM_DailyTagSpend"
                WHERE date >= $1 AND date <= $2
                GROUP BY api_key
                """,
                "2026-06-01",
                "2026-06-02",
            )
        )
        expected_metrics: Final = MappingProxyType({(row.date, row.tag): row for row in independent_metrics})
        expected_key_counts: Final = MappingProxyType(
            {row.tag: row.distinct_api_keys for row in independent_key_counts}
        )
        assert {row.date for row in independent_metrics} == {"2026-06-01", "2026-06-02"}
        assert len(expected_key_counts) == 4
        assert len(independent_key_memberships) == 8
        assert all(row.tag_count == 2 for row in independent_key_memberships)
        assert max(expected_key_counts.values()) > 3

        entity_rows: Final = aggregate.entity_rows or ()
        rolled_rows: Final = tuple(row for row in entity_rows if row.api_key_rolled)
        keyed_rows: Final = tuple(row for row in entity_rows if not row.api_key_rolled and row.api_key)
        top_level_keys: Final = frozenset(
            row.api_key for row in aggregate.grouping_rows if row.group_level == 31 and row.api_key
        )
        entity_day_keys: Final = MappingProxyType(
            {
                key: frozenset(row.api_key for row in keyed_rows if (row.date, row.entity_id) == key and row.api_key)
                for key in frozenset((row.date, row.entity_id) for row in keyed_rows)
            }
        )
        assert entity_day_keys
        assert max(len(keys) for keys in entity_day_keys.values()) <= 3
        assert frozenset(row.api_key for row in keyed_rows) <= top_level_keys

        rolled_by_entity_day: Final = MappingProxyType(
            {(row.date, row.entity_id): row for row in rolled_rows if row.date is not None}
        )
        assert set(rolled_by_entity_day) == set(expected_metrics)
        for key, row in rolled_by_entity_day.items():
            assert row.spend is not None
            assert isclose(row.spend, expected_metrics[key].spend, rel_tol=1e-9, abs_tol=1e-9)
            assert row.api_requests == expected_metrics[key].api_requests
            assert row.prompt_tokens == expected_metrics[key].prompt_tokens
            assert row.distinct_api_keys == expected_key_counts[row.entity_id]

        response: Final = await get_daily_activity_aggregated(
            repository,
            scope,
            include_entity_breakdown=True,
            api_key_limit=3,
        )
        assert response.metadata.entity_total_api_keys == {
            tag: count for tag, count in expected_key_counts.items() if tag is not None
        }
        assert all(
            all(len(entity.api_key_breakdown) <= 3 for entity in day.breakdown.entities.values())
            for day in response.results
        )


@pytest.mark.asyncio
async def test_key_pages_match_full_tag_ranking_and_aggregate_top_keys() -> None:
    async with _daily_activity_database(include_tag_activity=True) as database:
        scope: Final = DailyActivityScope(
            table=DailyActivityTable.TAG,
            entity_id_field="tag",
            entity_ids=None,
            exclude_entity_ids=(),
            api_keys=None,
            start_date="2026-06-01",
            end_date="2026-06-02",
            model=None,
            timezone_offset_minutes=None,
        )
        repository: Final = _repository(database)
        first_page: Final = await repository.key_page(scope, offset=0, limit=3)
        remaining_pages: Final = tuple(
            [
                await repository.key_page(scope, offset=offset, limit=3)
                for offset in range(3, first_page.total_api_keys, 3)
            ]
        )
        pages: Final = (first_page, *remaining_pages)
        actual_keys: Final = tuple(row.api_key for page in pages for row in page.rows)
        expected_rows: Final = _TAG_RANKED_KEY_ADAPTER.validate_python(
            await database.query_raw(
                """
                SELECT api_key
                FROM "LiteLLM_DailyTagSpend"
                WHERE date >= $1 AND date <= $2 AND api_key <> $3
                GROUP BY api_key
                ORDER BY SUM(spend::numeric) DESC, api_key
                """,
                "2026-06-01",
                "2026-06-02",
                constants.PTU_SENTINEL_API_KEY,
            )
        )
        expected_keys: Final = tuple(row.api_key for row in expected_rows)
        independent_count: Final = _TAG_DISTINCT_KEY_COUNT_ADAPTER.validate_python(
            await database.query_raw(
                """
                SELECT COUNT(DISTINCT api_key)::bigint AS total_api_keys
                FROM "LiteLLM_DailyTagSpend"
                WHERE date >= $1 AND date <= $2 AND api_key <> $3
                """,
                "2026-06-01",
                "2026-06-02",
                constants.PTU_SENTINEL_API_KEY,
            )
        )[0].total_api_keys
        aggregate: Final = await repository.aggregated(scope, include_entity_breakdown=False, api_key_limit=3)
        aggregate_top_keys: Final = frozenset(
            row.api_key for row in aggregate.grouping_rows if row.group_level == 31 and row.api_key is not None
        )
        empty_page: Final = await repository.key_page(scope, offset=independent_count + 3, limit=3)

        assert actual_keys == expected_keys
        assert len(actual_keys) == len(frozenset(actual_keys))
        assert all(page.total_api_keys == independent_count for page in pages)
        assert first_page.total_api_keys == independent_count
        assert frozenset(row.api_key for row in first_page.rows) == aggregate_top_keys
        assert empty_page.rows == ()
        assert empty_page.total_api_keys == independent_count


@pytest.mark.asyncio
async def test_top_api_key_rank_is_order_independent_for_float_ties() -> None:
    async with _daily_activity_database(include_tag_float_tie_activity=True) as database:
        float_totals: Final = _TAG_FLOAT_SPEND_ADAPTER.validate_python(
            await database.query_raw(
                """
                SELECT api_key, SUM(spend)::float AS spend
                FROM "LiteLLM_DailyTagSpend"
                WHERE tag = $1 AND date = $2
                GROUP BY api_key
                """,
                "tag-float-tie",
                "2026-06-01",
            )
        )
        float_spends: Final = MappingProxyType({row.api_key: row.spend for row in float_totals})
        assert float_spends["key-z"] > float_spends["key-a"]

        scope: Final = DailyActivityScope(
            table=DailyActivityTable.TAG,
            entity_id_field="tag",
            entity_ids=None,
            exclude_entity_ids=(),
            api_keys=None,
            start_date="2026-06-01",
            end_date="2026-06-01",
            model=None,
            timezone_offset_minutes=None,
        )
        aggregate: Final = await _repository(database).aggregated(scope, include_entity_breakdown=True, api_key_limit=1)
        key_page: Final = await _repository(database).key_page(scope, offset=0, limit=1)
        top_level_keys: Final = frozenset(
            row.api_key for row in aggregate.grouping_rows if row.group_level == 31 and row.api_key
        )
        entity_keyed_keys: Final = frozenset(
            row.api_key for row in aggregate.entity_rows or () if not row.api_key_rolled and row.api_key is not None
        )
        expected_keys: Final = frozenset(("key-a",))
        assert (top_level_keys, entity_keyed_keys) == (
            expected_keys,
            expected_keys,
        ), f"plain float SUM totals: {float_spends}"
        assert frozenset(row.api_key for row in key_page.rows) == expected_keys


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("table", "entity_field", "entity_id", "fixture_name"),
    (
        (DailyActivityTable.USER, "user_id", "user-1", "daily_activity_user.json"),
        (DailyActivityTable.TEAM, "team_id", "team-1", "daily_activity_team.json"),
    ),
)
async def test_aggregated_response_matches_base_golden(
    table: DailyActivityTable, entity_field: str, entity_id: str, fixture_name: str
) -> None:
    async with _daily_activity_database() as database:
        result: Final = await get_daily_activity_aggregated(
            _repository(database),
            _scope(table, entity_field, entity_id),
            entity_metadata_field=MappingProxyType({"team-1": {"team_alias": "Usage Team"}}),
            include_entity_breakdown=True,
        )
        golden_path: Final = Path(__file__).with_name("fixtures") / fixture_name
        assert result.model_dump_json() + "\n" == golden_path.read_text()


@pytest.mark.asyncio
async def test_team_entity_rollups_merge_null_and_empty_entity_ids() -> None:
    async with _daily_activity_database(include_team_unassigned_activity=True) as database:
        scope: Final = DailyActivityScope(
            table=DailyActivityTable.TEAM,
            entity_id_field="team_id",
            entity_ids=None,
            exclude_entity_ids=(),
            api_keys=None,
            start_date="2026-06-03",
            end_date="2026-06-03",
            model=None,
            timezone_offset_minutes=None,
        )
        aggregate: Final = await _repository(database).aggregated(scope, include_entity_breakdown=True, api_key_limit=3)

        totals: Final = tuple(row for row in aggregate.grouping_rows if row.group_level == 127)
        assert len(totals) == 1
        assert totals[0].spend == 23.0
        assert aggregate.distinct_api_keys == 2

        rolled_rows: Final = tuple(row for row in aggregate.entity_rows or () if row.api_key_rolled)
        assert len(rolled_rows) == 1
        assert rolled_rows[0].entity_id == ""
        assert rolled_rows[0].spend == 23.0
        assert rolled_rows[0].ptu_flat_cost == 13.0
        assert rolled_rows[0].distinct_api_keys == 2

        keyed_rows: Final = tuple(row for row in aggregate.entity_rows or () if not row.api_key_rolled)
        assert {row.entity_id for row in keyed_rows} == {""}
        assert {row.api_key for row in keyed_rows} == {"key-unassigned-null", "key-unassigned-empty"}


@pytest.mark.asyncio
async def test_team_exclusion_keeps_null_and_empty_entity_rows() -> None:
    async with _daily_activity_database(include_team_exclusion_activity=True) as database:
        repository: Final = _repository(database)
        scope: Final = DailyActivityScope(
            table=DailyActivityTable.TEAM,
            entity_id_field="team_id",
            entity_ids=None,
            exclude_entity_ids=("litellm-dashboard",),
            api_keys=None,
            start_date="2026-06-04",
            end_date="2026-06-04",
            model=None,
            timezone_offset_minutes=None,
        )
        aggregate: Final = await repository.aggregated(scope, include_entity_breakdown=True, api_key_limit=10)

        totals: Final = tuple(row for row in aggregate.grouping_rows if row.group_level == 127)
        assert len(totals) == 1
        assert totals[0].spend == 23.0
        assert aggregate.distinct_api_keys == 3

        keyed_rows: Final = tuple(row for row in aggregate.entity_rows or () if not row.api_key_rolled)
        assert {row.api_key for row in keyed_rows} == {"key-excluded-null", "key-excluded-empty", "key-excluded-normal"}
        assert {row.entity_id for row in keyed_rows} == {"", "team-normal"}

        page: Final = await repository.key_page(scope, offset=0, limit=10)
        assert page.total_api_keys == 3
        assert {row.api_key for row in page.rows} == {"key-excluded-null", "key-excluded-empty", "key-excluded-normal"}

        daily: Final = await repository.daily_rows(scope, page=1, page_size=10)
        assert daily.total_count == 3
        assert {row.api_key for row in daily.rows} == {"key-excluded-null", "key-excluded-empty", "key-excluded-normal"}


@pytest.mark.asyncio
async def test_user_page_ranks_users_and_folds_null_ids() -> None:
    async with _daily_activity_database(include_user_page_activity=True) as database:
        scope: Final = DailyActivityScope(
            table=DailyActivityTable.USER,
            entity_id_field="user_id",
            entity_ids=None,
            exclude_entity_ids=(),
            api_keys=None,
            start_date="2026-06-01",
            end_date="2026-06-02",
            model=None,
            timezone_offset_minutes=None,
        )
        repository: Final = _repository(database)
        full_page: Final = await repository.user_page(scope, offset=0, limit=50)
        empty_page: Final = await repository.user_page(scope, offset=full_page.total_users + 1, limit=50)
        scoped_page: Final = await repository.user_page(
            replace(scope, entity_ids=("user-alpha",)), offset=0, limit=50
        )

        assert tuple((row.user_id, row.spend) for row in full_page.rows) == (
            (None, 45.0),
            ("user-zeta", 30.0),
            ("user-alpha", 20.0),
            ("user-beta", 20.0),
            ("user-other", 10.0),
        )
        assert full_page.total_users == 5
        assert empty_page.rows == ()
        assert empty_page.total_users == 5
        assert tuple(row.user_id for row in scoped_page.rows) == ("user-alpha",)
        assert scoped_page.total_users == 1
