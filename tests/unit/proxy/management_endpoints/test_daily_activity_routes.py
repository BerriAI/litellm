import csv
import io
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from dataclasses import dataclass, fields
from itertools import chain
from types import SimpleNamespace
from typing import Final
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from litellm import constants
from litellm.proxy._types import LiteLLM_TeamTable, LiteLLM_UserTable, LitellmUserRoles, Member, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.management_endpoints.daily_activity_routes import (
    _csv_cell,
    get_daily_activity_prisma_client,
    get_daily_activity_repository,
    router,
)
from litellm.types.proxy.management_endpoints.common_daily_activity import KeySpendMetrics, SpendMetrics
from litellm.types.repositories.daily_activity import (
    AggregatedRows,
    DailyActivityScope,
    DailyActivityTable,
    EntityRollupRow,
    ExportRow,
    ExportType,
    GroupingSetsRow,
    KeyMetadataRow,
    KeyPage,
    KeySpendRow,
)


@dataclass(frozen=True, slots=True)
class _Activity:
    table: DailyActivityTable
    entity_id: str
    date: str
    api_key: str
    model: str
    model_group: str
    spend: float
    flat_cost: float
    prompt_tokens: int
    completion_tokens: int
    cache_read_input_tokens: int
    cache_creation_input_tokens: int
    compression_saved_tokens: int
    compression_savings_spend: float
    prompt_caching_savings_spend: float
    gateway_injected_caching_savings_spend: float
    autorouter_savings_spend: float
    api_requests: int
    successful_requests: int
    failed_requests: int
    total_response_time_ms: int
    timed_requests: int


_ENTITY_CASES: Final[tuple[tuple[str, str, str], ...]] = (
    ("/user", "user_id", "user-a"),
    ("/team", "team_ids", "team-a"),
    ("/tag", "tags", "blue"),
    ("/organization", "organization_ids", "org-a"),
    ("/customer", "end_user_ids", "customer-a"),
    ("/agent", "agent_ids", "agent-a"),
)
_DATE_PARAMS: Final = {"start_date": "2025-01-01", "end_date": "2025-01-02"}


def _activity_for_entity(
    table: DailyActivityTable,
    entity_id: str,
    key_rows: tuple[tuple[str, str, str, float, int], ...],
) -> tuple[_Activity, ...]:
    return tuple(
        _Activity(
            table=table,
            entity_id=entity_id,
            date=date,
            api_key=api_key,
            model=model,
            model_group="rare-group" if model == "rare-model" else "popular-group",
            spend=spend,
            flat_cost=0.05,
            prompt_tokens=10,
            completion_tokens=5,
            cache_read_input_tokens=cache_read,
            cache_creation_input_tokens=1,
            compression_saved_tokens=2,
            compression_savings_spend=0.1,
            prompt_caching_savings_spend=0.2,
            gateway_injected_caching_savings_spend=0.3,
            autorouter_savings_spend=0.4,
            api_requests=1,
            successful_requests=1,
            failed_requests=0,
            total_response_time_ms=100,
            timed_requests=1,
        )
        for api_key, date, model, spend, cache_read in key_rows
    )


def _seeded_activity() -> tuple[_Activity, ...]:
    entity_ids: Final = {
        DailyActivityTable.USER: "user-a",
        DailyActivityTable.TEAM: "team-a",
        DailyActivityTable.TAG: "blue",
        DailyActivityTable.ORGANIZATION: "org-a",
        DailyActivityTable.CUSTOMER: "customer-a",
        DailyActivityTable.AGENT: "agent-a",
    }
    other_entity_ids: Final = {
        DailyActivityTable.USER: "user-b",
        DailyActivityTable.TEAM: "team-b",
        DailyActivityTable.TAG: "other-blue",
        DailyActivityTable.ORGANIZATION: "other-org",
        DailyActivityTable.CUSTOMER: "customer-b",
        DailyActivityTable.AGENT: "agent-b",
    }
    key_rows: Final = (
        ("key-alpha", "2025-01-01", "popular", 1.0, 0),
        ("key-alpha", "2025-01-02", "popular", 2.0, 0),
        ("key-beta", "2025-01-01", "popular", 4.0, 0),
        ("key-gamma", "2025-01-01", "popular", 5.0, 0),
        ("key-cache", "2025-01-01", "popular", 2.0, 20),
        ("key-target", "2025-01-01", "rare-model", 0.5, 0),
    )
    return tuple(
        chain.from_iterable(_activity_for_entity(table, entity_id, key_rows) for table, entity_id in entity_ids.items())
    ) + tuple(
        _Activity(
            table=table,
            entity_id=other_entity_ids[table],
            date="2025-01-01",
            api_key=f"key-other-{table.value}",
            model="popular",
            model_group="popular-group",
            spend=100.0,
            flat_cost=0.0,
            prompt_tokens=10,
            completion_tokens=5,
            cache_read_input_tokens=5 if table is DailyActivityTable.USER else 0,
            cache_creation_input_tokens=0,
            compression_saved_tokens=0,
            compression_savings_spend=0.0,
            prompt_caching_savings_spend=0.0,
            gateway_injected_caching_savings_spend=0.0,
            autorouter_savings_spend=0.0,
            api_requests=1,
            successful_requests=1,
            failed_requests=0,
            total_response_time_ms=100,
            timed_requests=1,
        )
        for table in entity_ids
    )


def _metrics(rows: Sequence[_Activity]) -> Mapping[str, int | float]:
    return {
        "spend": sum(row.spend for row in rows),
        "ptu_flat_cost": sum(row.flat_cost for row in rows),
        "prompt_tokens": sum(row.prompt_tokens for row in rows),
        "completion_tokens": sum(row.completion_tokens for row in rows),
        "cache_read_input_tokens": sum(row.cache_read_input_tokens for row in rows),
        "cache_creation_input_tokens": sum(row.cache_creation_input_tokens for row in rows),
        "compression_saved_tokens": sum(row.compression_saved_tokens for row in rows),
        "compression_savings_spend": sum(row.compression_savings_spend for row in rows),
        "prompt_caching_savings_spend": sum(row.prompt_caching_savings_spend for row in rows),
        "gateway_injected_caching_savings_spend": sum(row.gateway_injected_caching_savings_spend for row in rows),
        "autorouter_savings_spend": sum(row.autorouter_savings_spend for row in rows),
        "api_requests": sum(row.api_requests for row in rows),
        "successful_requests": sum(row.successful_requests for row in rows),
        "failed_requests": sum(row.failed_requests for row in rows),
        "total_response_time_ms": sum(row.total_response_time_ms for row in rows),
        "timed_requests": sum(row.timed_requests for row in rows),
    }


def _grouping_row(
    rows: Sequence[_Activity],
    *,
    date: str | None,
    api_key: str | None,
    group_level: int,
    distinct_api_keys: int | None,
) -> GroupingSetsRow:
    metric_values: Final = _metrics(rows)
    return GroupingSetsRow(
        date=date,
        api_key=api_key,
        **metric_values,
        model=None,
        model_group=None,
        custom_llm_provider=None,
        mcp_namespaced_tool_name=None,
        endpoint=None,
        group_level=group_level,
        distinct_api_keys=distinct_api_keys,
    )


def _grouping_rows_for_day(
    date: str, rows: Sequence[_Activity], top_keys: tuple[str, ...], distinct_api_keys: int
) -> tuple[GroupingSetsRow, ...]:
    date_rows: Final = tuple(row for row in rows if row.date == date)
    return (
        _grouping_row(date_rows, date=date, api_key=None, group_level=63, distinct_api_keys=distinct_api_keys),
    ) + tuple(
        _grouping_row(
            tuple(row for row in date_rows if row.api_key == api_key),
            date=date,
            api_key=api_key,
            group_level=31,
            distinct_api_keys=None,
        )
        for api_key in top_keys
        if any(row.api_key == api_key for row in date_rows)
    )


class _FakeRepository:
    def __init__(self, rows: tuple[_Activity, ...]) -> None:
        self._rows: Final = rows
        self.aggregated = AsyncMock(side_effect=self._aggregated)
        self.key_page = AsyncMock(side_effect=self._key_page)
        self.key_page_call: tuple[DailyActivityScope, int, int] | None = None
        self.search_keys = AsyncMock(side_effect=self._search_keys)
        self.model_top_keys = AsyncMock(side_effect=self._model_top_keys)
        self.cache_leakage_keys = AsyncMock(side_effect=self._cache_leakage_keys)
        self.key_metadata = AsyncMock(side_effect=self._key_metadata)
        self.export_rows_error: Exception | None = None

    def _matching_rows(self, scope: DailyActivityScope) -> tuple[_Activity, ...]:
        return tuple(
            row
            for row in self._rows
            if row.table == scope.table
            and scope.start_date <= row.date <= scope.end_date
            and (scope.entity_ids is None or row.entity_id in scope.entity_ids)
            and row.entity_id not in scope.exclude_entity_ids
            and (scope.api_keys is None or row.api_key in scope.api_keys)
            and (scope.model is None or row.model == scope.model)
        )

    async def _aggregated(
        self,
        scope: DailyActivityScope,
        *,
        include_entity_breakdown: bool = False,
        api_key_limit: int = constants.USAGE_TOP_API_KEYS_DEFAULT,
    ) -> AggregatedRows:
        rows: Final = self._matching_rows(scope)
        key_spend: Final = tuple(
            sorted(
                ((key, sum(row.spend for row in rows if row.api_key == key)) for key in {row.api_key for row in rows}),
                key=lambda item: (-item[1], item[0]),
            )
        )
        distinct_keys: Final = len(key_spend)
        top_keys: Final = tuple(key for key, _ in key_spend[:api_key_limit])
        dates: Final = tuple(sorted({row.date for row in rows}))
        grouping_rows: Final = (
            _grouping_row(rows, date=None, api_key=None, group_level=127, distinct_api_keys=distinct_keys),
        ) + tuple(
            grouping_row
            for date in dates
            for grouping_row in _grouping_rows_for_day(date, rows, top_keys, distinct_keys)
        )
        entity_rows: Final = (
            tuple(entity_row for date in dates for entity_row in _entity_rows_for_day(date, rows))
            if include_entity_breakdown
            else ()
        )
        return AggregatedRows(
            grouping_rows=grouping_rows,
            entity_rows=entity_rows if include_entity_breakdown else None,
            distinct_api_keys=distinct_keys,
        )

    async def _search_keys(self, scope: DailyActivityScope, *, search: str, limit: int) -> tuple[str, ...]:
        rows: Final = self._matching_rows(scope)
        return tuple(key for key in dict.fromkeys(row.api_key for row in rows) if search.casefold() in key.casefold())[
            :limit
        ]

    async def _key_page(self, scope: DailyActivityScope, *, offset: int, limit: int) -> KeyPage:
        self.key_page_call = (scope, offset, limit)
        rows: Final = self._matching_rows(scope)
        api_keys: Final = _ranked_keys(rows)
        return KeyPage(
            rows=tuple(_key_spend_row(api_key, rows) for api_key in api_keys[offset : offset + limit]),
            total_api_keys=len(api_keys),
        )

    async def _model_top_keys(
        self, scope: DailyActivityScope, *, model_group: str, by_model_group: bool, limit: int
    ) -> tuple[KeySpendRow, ...]:
        rows: Final = tuple(
            row
            for row in self._matching_rows(scope)
            if (row.model_group if by_model_group else row.model) == model_group
        )
        return tuple(_key_spend_row(key, rows) for key in _ranked_keys(rows)[:limit])

    async def _cache_leakage_keys(self, scope: DailyActivityScope, *, limit: int) -> tuple[KeySpendRow, ...]:
        rows: Final = tuple(row for row in self._matching_rows(scope) if row.cache_read_input_tokens > 0)
        return tuple(_key_spend_row(key, rows) for key in _ranked_keys(rows)[:limit])

    async def _key_metadata(
        self, api_keys: frozenset[str], window: tuple[object, object] | None
    ) -> Mapping[str, KeyMetadataRow]:
        return {
            key: KeyMetadataRow(
                api_key=key,
                key_alias=f"alias-{key}",
                team_id="team-a",
                user_id="user-a",
                user_email="user@example.test",
                key_exists=True,
                tags=(),
            )
            for key in api_keys
        }

    async def export_rows(self, scope: DailyActivityScope, *, export_type: ExportType) -> AsyncIterator[ExportRow]:
        if self.export_rows_error is not None:
            raise self.export_rows_error
        for row in self._matching_rows(scope):
            yield ExportRow(
                date=row.date,
                entity_id=row.entity_id,
                entity_alias="=entity",
                api_key=row.api_key,
                key_alias="+key",
                user_id="user-a",
                user_email="user@example.test",
                model=row.model,
                spend=row.spend,
                flat_cost=row.flat_cost,
                prompt_tokens=row.prompt_tokens,
                completion_tokens=row.completion_tokens,
                api_requests=row.api_requests,
                successful_requests=row.successful_requests,
                failed_requests=row.failed_requests,
                cache_read_input_tokens=row.cache_read_input_tokens,
                cache_creation_input_tokens=row.cache_creation_input_tokens,
            )


def _ranked_keys(rows: Sequence[_Activity]) -> tuple[str, ...]:
    return tuple(
        key
        for key, _ in sorted(
            ((key, sum(row.spend for row in rows if row.api_key == key)) for key in {row.api_key for row in rows}),
            key=lambda item: (-item[1], item[0]),
        )
    )


def _key_spend_row(api_key: str, rows: Sequence[_Activity]) -> KeySpendRow:
    matching: Final = tuple(row for row in rows if row.api_key == api_key)
    return KeySpendRow(
        api_key=api_key,
        spend=sum(row.spend for row in matching),
        prompt_tokens=sum(row.prompt_tokens for row in matching),
        completion_tokens=sum(row.completion_tokens for row in matching),
        total_tokens=sum(row.prompt_tokens + row.completion_tokens for row in matching),
        api_requests=sum(row.api_requests for row in matching),
        successful_requests=sum(row.successful_requests for row in matching),
        failed_requests=sum(row.failed_requests for row in matching),
        cache_read_input_tokens=sum(row.cache_read_input_tokens for row in matching),
        cache_creation_input_tokens=sum(row.cache_creation_input_tokens for row in matching),
    )


def _entity_rows_for_day(
    date: str, rows: Sequence[_Activity], distinct_api_keys: int | None = None
) -> tuple[EntityRollupRow, ...]:
    date_rows: Final = tuple(row for row in rows if row.date == date)
    entities: Final = tuple(dict.fromkeys(row.entity_id for row in date_rows))
    return tuple(
        entity_row
        for entity_id in entities
        for entity_row in _entity_rows_for_entity(date, entity_id, date_rows, distinct_api_keys)
    )


def _entity_rows_for_entity(
    date: str, entity_id: str, rows: Sequence[_Activity], distinct_api_keys: int | None
) -> tuple[EntityRollupRow, ...]:
    entity_rows: Final = tuple(row for row in rows if row.entity_id == entity_id)
    return (
        _entity_rollup(
            entity_rows,
            date=date,
            entity_id=entity_id,
            api_key=None,
            api_key_rolled=1,
            distinct_api_keys=distinct_api_keys,
        ),
    ) + tuple(
        _entity_rollup(
            tuple(row for row in entity_rows if row.api_key == api_key),
            date=date,
            entity_id=entity_id,
            api_key=api_key,
            api_key_rolled=0,
            distinct_api_keys=None,
        )
        for api_key in dict.fromkeys(row.api_key for row in entity_rows)
    )


def _entity_rollup(
    rows: Sequence[_Activity],
    *,
    date: str,
    entity_id: str,
    api_key: str | None,
    api_key_rolled: int,
    distinct_api_keys: int | None,
) -> EntityRollupRow:
    return EntityRollupRow(
        date=date,
        api_key=api_key,
        **_metrics(rows),
        entity_id=entity_id,
        api_key_rolled=api_key_rolled,
        distinct_api_keys=distinct_api_keys,
    )


class _PrismaTable:
    def __init__(self, rows: tuple[object, ...] = ()) -> None:
        self._rows: Final = rows

    async def find_many(self, *, where: Mapping[str, object] | None = None, **kwargs: object) -> tuple[object, ...]:
        if where is None:
            return self._rows
        return tuple(row for row in self._rows if _matches(row, where))

    async def find_unique(self, *, where: Mapping[str, object], **kwargs: object) -> object | None:
        return next((row for row in self._rows if _matches(row, where)), None)


def _matches(row: object, where: Mapping[str, object]) -> bool:
    return all(
        getattr(row, field_name, None) in value["in"]
        if isinstance(value, Mapping) and "in" in value
        else getattr(row, field_name, None) == value
        for field_name, value in where.items()
    )


def _prisma_client() -> object:
    user: Final = LiteLLM_UserTable(
        user_id="user-a",
        user_email="user@example.test",
        user_role=LitellmUserRoles.INTERNAL_USER.value,
        teams=["team-a"],
    )
    team: Final = LiteLLM_TeamTable(
        team_id="team-a",
        team_alias="Team A",
        members_with_roles=[Member(user_id="user-a", role="user")],
    )
    other_user: Final = LiteLLM_UserTable(
        user_id="user-b",
        user_email="other-user@example.test",
        user_role=LitellmUserRoles.INTERNAL_USER.value,
        teams=["team-b"],
    )
    other_team: Final = LiteLLM_TeamTable(
        team_id="team-b",
        team_alias="Team B",
        members_with_roles=[Member(user_id="user-b", role="user")],
    )
    db: Final = SimpleNamespace(
        litellm_usertable=_PrismaTable((user, other_user)),
        litellm_teamtable=_PrismaTable((team, other_team)),
        litellm_verificationtoken=_PrismaTable((SimpleNamespace(token="key-alpha", user_id="user-a"),)),
        litellm_organizationmembership=_PrismaTable(
            (
                SimpleNamespace(user_id="user-a", organization_id="org-a", user_role="org_admin"),
                SimpleNamespace(user_id="user-b", organization_id="other-org", user_role="org_admin"),
            )
        ),
        litellm_organizationtable=_PrismaTable(
            (
                SimpleNamespace(organization_id="org-a", organization_alias="Org A"),
                SimpleNamespace(organization_id="other-org", organization_alias="Other Org"),
            )
        ),
        litellm_endusertable=_PrismaTable(
            (
                SimpleNamespace(user_id="customer-a", alias="Customer A"),
                SimpleNamespace(user_id="customer-b", alias="Customer B"),
            )
        ),
        litellm_agentstable=_PrismaTable(
            (
                SimpleNamespace(agent_id="agent-a", agent_name="Agent A", created_by="user-a"),
                SimpleNamespace(agent_id="agent-b", agent_name="Agent B", created_by="user-b"),
            )
        ),
    )
    return SimpleNamespace(db=db, writer_db=db)


@pytest.fixture
def daily_activity_client() -> Iterator[tuple[TestClient, _FakeRepository]]:
    repository: Final = _FakeRepository(_seeded_activity())
    prisma_client: Final = _prisma_client()
    app: Final = FastAPI()
    app.include_router(router)

    def resolve_auth(request: Request) -> UserAPIKeyAuth:
        role: Final = LitellmUserRoles(request.headers.get("x-user-role", LitellmUserRoles.PROXY_ADMIN.value))
        user_id: Final[str | None] = request.headers.get("x-user-id") or (
            "admin" if role != LitellmUserRoles.INTERNAL_USER else None
        )
        return UserAPIKeyAuth(
            user_id=user_id,
            user_role=role,
            api_key=request.headers.get("x-api-key"),
        )

    app.dependency_overrides[get_daily_activity_repository] = lambda: repository
    app.dependency_overrides[get_daily_activity_prisma_client] = lambda: prisma_client
    app.dependency_overrides[user_api_key_auth] = resolve_auth
    with TestClient(app) as client:
        yield client, repository


def _entity_params(query_name: str, entity_id: str) -> dict[str, str]:
    return {**_DATE_PARAMS, query_name: entity_id}


@pytest.mark.parametrize(("prefix", "query_name", "entity_id"), _ENTITY_CASES)
def test_aggregated_routes_return_scoped_results(
    daily_activity_client: tuple[TestClient, _FakeRepository],
    prefix: str,
    query_name: str,
    entity_id: str,
) -> None:
    client, _ = daily_activity_client
    response: Final = client.get(
        f"{prefix}/daily/activity/aggregated",
        params=_entity_params(query_name, entity_id),
    )
    assert response.status_code == 200, response.text
    body: Final = response.json()
    assert body["metadata"]["total_spend"] == pytest.approx(14.5), response.text
    assert body["metadata"]["total_api_keys"] == 5, response.text
    assert len(body["results"]) == 2, response.text


@pytest.mark.parametrize(("prefix", "query_name", "entity_id"), _ENTITY_CASES)
def test_admin_aggregates_all_entities_when_filter_is_omitted(
    daily_activity_client: tuple[TestClient, _FakeRepository], prefix: str, query_name: str, entity_id: str
) -> None:
    client, _ = daily_activity_client
    response: Final = client.get(
        f"{prefix}/daily/activity/aggregated",
        params=_DATE_PARAMS,
    )
    assert response.status_code == 200, response.text
    assert response.json()["metadata"]["total_spend"] == pytest.approx(114.5), response.text
    assert response.json()["metadata"]["total_api_keys"] == 6, response.text


@pytest.mark.parametrize(("prefix", "query_name", "entity_id"), _ENTITY_CASES)
def test_search_folds_each_entity_key_across_days(
    daily_activity_client: tuple[TestClient, _FakeRepository], prefix: str, query_name: str, entity_id: str
) -> None:
    client, _ = daily_activity_client
    response: Final = client.get(
        f"{prefix}/daily/activity/aggregated/search",
        params={**_entity_params(query_name, entity_id), "search": "alpha"},
    )
    assert response.status_code == 200, response.text
    assert response.json() == {
        "api_keys": [
            {
                "api_key": "key-alpha",
                "metrics": {
                    "spend": 3.0,
                    "flat_cost": 0.0,
                    "prompt_tokens": 20,
                    "completion_tokens": 10,
                    "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 2,
                    "compression_saved_tokens": 4,
                    "compression_savings_spend": 0.2,
                    "prompt_caching_savings_spend": 0.4,
                    "gateway_injected_caching_savings_spend": 0.6,
                    "autorouter_savings_spend": 0.8,
                    "total_tokens": 30,
                    "successful_requests": 2,
                    "failed_requests": 0,
                    "api_requests": 2,
                    "total_response_time_ms": 200,
                    "timed_requests": 2,
                },
                "metadata": {
                    "key_alias": "alias-key-alpha",
                    "team_id": "team-a",
                    "user_id": "user-a",
                    "user_email": "user@example.test",
                    "key_exists": True,
                },
            }
        ]
    }


def test_search_finds_keys_outside_the_top_keys_limit_and_skips_empty_aggregate(
    daily_activity_client: tuple[TestClient, _FakeRepository],
) -> None:
    client, repository = daily_activity_client
    aggregate_response: Final = client.get(
        "/user/daily/activity/aggregated",
        params={**_entity_params("user_id", "user-a"), "api_key_limit": 3},
    )
    assert aggregate_response.status_code == 200, aggregate_response.text
    assert repository.aggregated.call_args.kwargs["api_key_limit"] == 3
    top_keys: Final = frozenset(
        chain.from_iterable(result["breakdown"]["api_keys"] for result in aggregate_response.json()["results"])
    )
    assert "key-target" not in top_keys

    search_response: Final = client.get(
        "/user/daily/activity/aggregated/search",
        params={**_entity_params("user_id", "user-a"), "search": "target", "limit": 7},
    )
    assert search_response.status_code == 200, search_response.text
    assert repository.search_keys.call_args.kwargs["limit"] == 7
    assert search_response.json()["api_keys"][0]["api_key"] == "key-target"
    assert search_response.json()["api_keys"][0]["metrics"]["spend"] == pytest.approx(0.5)
    search_metrics: Final = search_response.json()["api_keys"][0]["metrics"]
    assert set(search_metrics) == set(SpendMetrics.model_fields)
    assert search_metrics["compression_savings_spend"] == pytest.approx(0.1)
    assert search_metrics["total_response_time_ms"] == 100

    repository.aggregated.reset_mock()
    empty_response: Final = client.get(
        "/user/daily/activity/aggregated/search",
        params={**_entity_params("user_id", "user-a"), "search": "absent"},
    )
    assert empty_response.status_code == 200, empty_response.text
    assert empty_response.json() == {"api_keys": []}
    repository.aggregated.assert_not_awaited()


@pytest.mark.parametrize(("prefix", "query_name", "entity_id"), _ENTITY_CASES)
def test_key_page_routes_map_ranked_rows_and_totals(
    daily_activity_client: tuple[TestClient, _FakeRepository],
    prefix: str,
    query_name: str,
    entity_id: str,
) -> None:
    client, repository = daily_activity_client
    response: Final = client.get(
        f"{prefix}/daily/activity/aggregated/keys",
        params={**_entity_params(query_name, entity_id), "offset": 1, "limit": 2},
    )

    assert response.status_code == 200, response.text
    assert response.json() == {
        "api_keys": [
            {
                "api_key": "key-beta",
                "metrics": {
                    "spend": 4.0,
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                    "api_requests": 1,
                    "successful_requests": 1,
                    "failed_requests": 0,
                    "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 1,
                },
                "metadata": {
                    "key_alias": "alias-key-beta",
                    "team_id": "team-a",
                    "user_id": "user-a",
                    "user_email": "user@example.test",
                    "key_exists": True,
                },
            },
            {
                "api_key": "key-alpha",
                "metrics": {
                    "spend": 3.0,
                    "prompt_tokens": 20,
                    "completion_tokens": 10,
                    "total_tokens": 30,
                    "api_requests": 2,
                    "successful_requests": 2,
                    "failed_requests": 0,
                    "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 2,
                },
                "metadata": {
                    "key_alias": "alias-key-alpha",
                    "team_id": "team-a",
                    "user_id": "user-a",
                    "user_email": "user@example.test",
                    "key_exists": True,
                },
            },
        ],
        "total_api_keys": 5,
        "offset": 1,
        "limit": 2,
    }
    key_page_call: Final = repository.key_page_call
    assert key_page_call is not None
    scope: Final = key_page_call[0]
    assert scope.entity_ids == (entity_id,)
    assert key_page_call[1:] == (1, 2)


@pytest.mark.parametrize("params", ({"limit": 101}, {"offset": -1}))
def test_key_page_route_rejects_invalid_bounds(
    daily_activity_client: tuple[TestClient, _FakeRepository],
    params: Mapping[str, int],
) -> None:
    client, repository = daily_activity_client
    response: Final = client.get(
        "/user/daily/activity/aggregated/keys",
        params={**_entity_params("user_id", "user-a"), **params},
    )

    assert response.status_code == 422, response.text
    repository.key_page.assert_not_awaited()


@pytest.mark.parametrize(
    ("path", "route_params", "limit_name", "invalid_limit"),
    (
        ("/user/daily/activity/aggregated", {}, "api_key_limit", 0),
        (
            "/user/daily/activity/aggregated",
            {},
            "api_key_limit",
            constants.USAGE_TOP_API_KEYS_MAX + 1,
        ),
        ("/user/daily/activity/aggregated/search", {"search": "key"}, "limit", 0),
        (
            "/user/daily/activity/aggregated/search",
            {"search": "key"},
            "limit",
            constants.USAGE_KEY_SEARCH_MAX + 1,
        ),
        (
            "/user/daily/activity/aggregated/model_top_keys",
            {"model_group": "popular-group"},
            "limit",
            0,
        ),
        (
            "/user/daily/activity/aggregated/model_top_keys",
            {"model_group": "popular-group"},
            "limit",
            constants.USAGE_MODEL_TOP_KEYS_MAX + 1,
        ),
        (
            "/user/daily/activity/aggregated/cache_leakage_keys",
            {},
            "limit",
            0,
        ),
        (
            "/user/daily/activity/aggregated/cache_leakage_keys",
            {},
            "limit",
            constants.USAGE_CACHE_LEAKAGE_KEYS_MAX + 1,
        ),
    ),
)
def test_usage_limit_routes_reject_values_outside_bounds(
    daily_activity_client: tuple[TestClient, _FakeRepository],
    path: str,
    route_params: Mapping[str, str],
    limit_name: str,
    invalid_limit: int,
) -> None:
    client, repository = daily_activity_client
    response: Final = client.get(
        path,
        params={
            **_entity_params("user_id", "user-a"),
            **route_params,
            limit_name: invalid_limit,
        },
    )

    assert response.status_code == 422, response.text
    repository.aggregated.assert_not_awaited()
    repository.search_keys.assert_not_awaited()
    repository.model_top_keys.assert_not_awaited()
    repository.cache_leakage_keys.assert_not_awaited()


@pytest.mark.parametrize(("prefix", "query_name", "entity_id"), _ENTITY_CASES)
def test_model_top_routes_rank_keys_and_include_metadata(
    daily_activity_client: tuple[TestClient, _FakeRepository], prefix: str, query_name: str, entity_id: str
) -> None:
    client, repository = daily_activity_client
    response: Final = client.get(
        f"{prefix}/daily/activity/aggregated/model_top_keys",
        params={
            **_entity_params(query_name, entity_id),
            "model_group": "rare-group",
            "limit": 3,
        },
    )
    assert response.status_code == 200, response.text
    assert repository.model_top_keys.call_args.kwargs["limit"] == 3
    assert response.json()["model"] == "rare-group"
    assert response.json()["by_model_group"] is True
    assert tuple(row["api_key"] for row in response.json()["api_keys"]) == ("key-target",)
    metrics: Final = response.json()["api_keys"][0]["metrics"]
    expected_row: Final = KeySpendRow(
        api_key="key-target",
        spend=0.5,
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        api_requests=1,
        successful_requests=1,
        failed_requests=0,
        cache_read_input_tokens=0,
        cache_creation_input_tokens=1,
    )
    assert set(metrics) == set(KeySpendMetrics.model_fields)
    assert metrics == {field: getattr(expected_row, field) for field in KeySpendMetrics.model_fields}


@pytest.mark.parametrize(("prefix", "query_name", "entity_id"), _ENTITY_CASES)
def test_export_routes_stream_csv_and_preserve_row_counts(
    daily_activity_client: tuple[TestClient, _FakeRepository],
    prefix: str,
    query_name: str,
    entity_id: str,
) -> None:
    client, _ = daily_activity_client
    response: Final = client.get(
        f"{prefix}/daily/activity/export",
        params={**_entity_params(query_name, entity_id), "export_type": ExportType.DAILY.value},
    )
    assert response.status_code == 200, response.text
    assert response.headers["cache-control"] == "no-store"
    assert "attachment;" in response.headers["content-disposition"]
    records: Final = tuple(csv.reader(io.StringIO(response.text)))
    assert tuple(records[0]) == tuple(field.name for field in fields(ExportRow))
    assert len(records) == 7
    assert records[1][2] == "'=entity"
    assert records[1][4] == "'+key"


@pytest.mark.parametrize(("prefix", "query_name", "entity_id"), _ENTITY_CASES)
def test_export_routes_stream_json_arrays(
    daily_activity_client: tuple[TestClient, _FakeRepository],
    prefix: str,
    query_name: str,
    entity_id: str,
) -> None:
    client, _ = daily_activity_client
    response: Final = client.get(
        f"{prefix}/daily/activity/export",
        params={**_entity_params(query_name, entity_id), "format": "json"},
    )
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("application/json")
    assert response.headers["cache-control"] == "no-store"
    records: Final = response.json()
    assert isinstance(records, list) and len(records) == 6, response.text
    assert records[0]["entity_alias"] == "=entity"


def test_export_first_row_error_returns_json_error_before_streaming(
    daily_activity_client: tuple[TestClient, _FakeRepository],
) -> None:
    client, repository = daily_activity_client
    repository.export_rows_error = RuntimeError("database query failed")
    response: Final = client.get(
        "/team/daily/activity/export",
        params={**_entity_params("team_ids", "team-a"), "format": "csv"},
    )
    assert response.status_code >= 400
    assert response.headers["content-type"].startswith("application/json")
    assert response.text != ",".join(field.name for field in fields(ExportRow)) + "\r\n"
    assert "database query failed" in response.text


def test_csv_export_with_no_rows_contains_only_header(
    daily_activity_client: tuple[TestClient, _FakeRepository],
) -> None:
    client, _ = daily_activity_client
    response: Final = client.get(
        "/team/daily/activity/export",
        params={**_entity_params("team_ids", "team-a"), "api_key": "missing-key"},
    )
    assert response.status_code == 200, response.text
    assert response.text == ",".join(field.name for field in fields(ExportRow)) + "\r\n"


def test_json_export_with_no_rows_is_an_empty_array(
    daily_activity_client: tuple[TestClient, _FakeRepository],
) -> None:
    client, _ = daily_activity_client
    response: Final = client.get(
        "/team/daily/activity/export",
        params={**_entity_params("team_ids", "team-a"), "api_key": "missing-key", "format": "json"},
    )
    assert response.status_code == 200, response.text
    assert response.json() == []


def test_user_routes_preserve_scope_denials_and_service_account_guard(
    daily_activity_client: tuple[TestClient, _FakeRepository],
) -> None:
    client, repository = daily_activity_client
    denied: Final = client.get(
        "/user/daily/activity/aggregated",
        params=_entity_params("user_id", "user-b"),
        headers={"x-user-role": LitellmUserRoles.INTERNAL_USER.value, "x-user-id": "user-a"},
    )
    assert denied.status_code == 403, denied.text
    repository.aggregated.assert_not_awaited()

    service_account: Final = client.get(
        "/user/daily/activity/aggregated",
        params=_DATE_PARAMS,
        headers={"x-user-role": LitellmUserRoles.INTERNAL_USER.value},
    )
    assert service_account.status_code == 403, service_account.text
    repository.aggregated.assert_not_awaited()

    own_scope: Final = client.get(
        "/user/daily/activity/aggregated",
        params=_DATE_PARAMS,
        headers={"x-user-role": LitellmUserRoles.INTERNAL_USER.value, "x-user-id": "user-a"},
    )
    assert own_scope.status_code == 200, own_scope.text
    assert own_scope.json()["metadata"]["total_spend"] == pytest.approx(14.5)
    assert own_scope.json()["metadata"]["total_api_keys"] == 5


def test_team_scope_applies_membership_and_user_key_filter(
    daily_activity_client: tuple[TestClient, _FakeRepository],
) -> None:
    client, repository = daily_activity_client
    response: Final = client.get(
        "/team/daily/activity/aggregated",
        params={**_entity_params("team_ids", "team-a"), "timezone": "480"},
        headers={"x-user-role": LitellmUserRoles.INTERNAL_USER.value, "x-user-id": "user-a"},
    )
    assert response.status_code == 200, response.text
    assert response.json()["metadata"]["total_spend"] == pytest.approx(3.0)
    assert response.json()["metadata"]["total_api_keys"] == 1
    scope: Final = repository.aggregated.call_args.args[0]
    assert scope.api_keys == ("key-alpha",)
    assert scope.entity_ids == ("team-a",)
    assert scope.timezone_offset_minutes == 480
    assert repository.aggregated.call_args.kwargs["include_entity_breakdown"] is True


def test_team_scope_does_not_allow_an_unowned_api_key(
    daily_activity_client: tuple[TestClient, _FakeRepository],
) -> None:
    client, _ = daily_activity_client
    response: Final = client.get(
        "/team/daily/activity/aggregated",
        params={**_entity_params("team_ids", "team-a"), "api_key": "key-beta"},
        headers={"x-user-role": LitellmUserRoles.INTERNAL_USER.value, "x-user-id": "user-a"},
    )
    assert response.status_code == 200, response.text
    assert response.json()["metadata"]["total_spend"] == 0
    assert response.json()["metadata"]["total_api_keys"] == 0
    assert "key-beta" not in response.text


def _assert_customer_route_denied(client: TestClient, prefix: str, suffix: str, extra_params: dict[str, str]) -> None:
    response: Final = client.get(
        f"{prefix}/daily/activity/{suffix}",
        params={**_entity_params("end_user_ids", "customer-a"), **extra_params},
        headers={"x-user-role": LitellmUserRoles.INTERNAL_USER.value, "x-user-id": "user-a"},
    )
    assert response.status_code == 403, response.text


def test_customer_service_routes_deny_non_admins(daily_activity_client: tuple[TestClient, _FakeRepository]) -> None:
    client, repository = daily_activity_client
    route_params: Final = (
        ("aggregated", {}),
        ("aggregated/search", {"search": "alpha"}),
        ("aggregated/model_top_keys", {"model_group": "rare-group"}),
        ("export", {"export_type": ExportType.DAILY.value}),
    )
    for prefix in ("/customer", "/end_user"):
        for suffix, extra_params in route_params:
            _assert_customer_route_denied(client, prefix, suffix, extra_params)
    repository.aggregated.assert_not_awaited()


def test_customer_end_user_aliases_are_hidden_from_openapi(
    daily_activity_client: tuple[TestClient, _FakeRepository],
) -> None:
    client, _ = daily_activity_client
    paths: Final = client.get("/openapi.json").json()["paths"]
    assert "/customer/daily/activity/aggregated" in paths
    assert "/end_user/daily/activity/aggregated" not in paths
    response: Final = client.get(
        "/end_user/daily/activity/aggregated",
        params=_entity_params("end_user_ids", "customer-a"),
    )
    assert response.status_code == 200, response.text


@pytest.mark.parametrize(
    ("prefix", "query_name", "entity_id"),
    _ENTITY_CASES[:4] + (_ENTITY_CASES[-1],),
)
@pytest.mark.parametrize(
    ("family", "extra_params"),
    (
        ("aggregated", {}),
        ("aggregated/search", {"search": "key"}),
        ("aggregated/model_top_keys", {"model_group": "popular-group"}),
        ("export", {"export_type": ExportType.DAILY.value}),
    ),
)
def test_non_admin_routes_return_only_permitted_entities_and_keys(
    daily_activity_client: tuple[TestClient, _FakeRepository],
    prefix: str,
    query_name: str,
    entity_id: str,
    family: str,
    extra_params: Mapping[str, str],
) -> None:
    client, _ = daily_activity_client
    headers: Final = {"x-user-role": LitellmUserRoles.INTERNAL_USER.value, "x-user-id": "user-a"}
    response: Final = client.get(
        f"{prefix}/daily/activity/{family}",
        params={**_entity_params(query_name, entity_id), **extra_params},
        headers=headers,
    )
    assert response.status_code == 200, response.text
    assert "key-other-" not in response.text, response.text
    if prefix in ("/team", "/tag"):
        assert "key-beta" not in response.text, response.text
        assert "key-alpha" in response.text, response.text


def test_empty_scope_filters_fail_closed(daily_activity_client: tuple[TestClient, _FakeRepository]) -> None:
    client, _ = daily_activity_client
    response: Final = client.get(
        "/tag/daily/activity/aggregated",
        params=_entity_params("tags", "blue"),
        headers={"x-user-role": LitellmUserRoles.INTERNAL_USER.value, "x-user-id": "user-empty"},
    )
    assert response.status_code == 200, response.text
    assert response.json()["metadata"]["total_spend"] == 0
    assert response.json()["results"] == []


@pytest.mark.parametrize(
    ("prefix", "query_name", "entity_id", "other_entity_id", "exclude_query_name"),
    (
        ("/team", "team_ids", "team-a", "team-b", "exclude_team_ids"),
        ("/organization", "organization_ids", "org-a", "other-org", "exclude_organization_ids"),
        ("/customer", "end_user_ids", "customer-a", "customer-b", "exclude_end_user_ids"),
        ("/agent", "agent_ids", "agent-a", "agent-b", "exclude_agent_ids"),
    ),
)
def test_exclusion_filters_apply_after_entity_scope(
    daily_activity_client: tuple[TestClient, _FakeRepository],
    prefix: str,
    query_name: str,
    entity_id: str,
    other_entity_id: str,
    exclude_query_name: str,
) -> None:
    client, _ = daily_activity_client
    response: Final = client.get(
        f"{prefix}/daily/activity/aggregated",
        params={
            **_DATE_PARAMS,
            query_name: f"{entity_id},{other_entity_id}",
            exclude_query_name: entity_id,
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["metadata"]["total_spend"] == pytest.approx(100)
    assert response.json()["metadata"]["total_api_keys"] == 1


def test_csv_formula_escaping_covers_all_supported_leading_characters() -> None:
    dangerous_values: Final = ("=sum", "+sum", "-sum", "@sum", "\tsum", "\rsum")
    assert tuple(_csv_cell(value) for value in dangerous_values) == tuple(f"'{value}" for value in dangerous_values)


def test_user_cache_leakage_route_returns_cache_keys_and_metadata(
    daily_activity_client: tuple[TestClient, _FakeRepository],
) -> None:
    client, repository = daily_activity_client
    response: Final = client.get(
        "/user/daily/activity/aggregated/cache_leakage_keys",
        params={**_entity_params("user_id", "user-a"), "limit": 4},
    )
    assert response.status_code == 200, response.text
    assert repository.cache_leakage_keys.call_args.kwargs["limit"] == 4
    assert tuple(row["api_key"] for row in response.json()["api_keys"]) == ("key-cache",)
    metrics: Final = response.json()["api_keys"][0]["metrics"]
    expected_row: Final = KeySpendRow(
        api_key="key-cache",
        spend=2.0,
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        api_requests=1,
        successful_requests=1,
        failed_requests=0,
        cache_read_input_tokens=20,
        cache_creation_input_tokens=1,
    )
    assert set(metrics) == set(KeySpendMetrics.model_fields)
    assert metrics == {field: getattr(expected_row, field) for field in KeySpendMetrics.model_fields}
    assert response.json()["api_keys"][0]["metadata"]["key_alias"] == "alias-key-cache"
    repository.cache_leakage_keys.assert_awaited_once()
    repository.key_metadata.assert_awaited_once()
    assert repository.key_metadata.call_args.args[0] == frozenset(("key-cache",))
    assert repository.key_metadata.call_args.args[1] is not None


def test_user_cache_leakage_route_respects_requested_user_scope(
    daily_activity_client: tuple[TestClient, _FakeRepository],
) -> None:
    client, repository = daily_activity_client
    response: Final = client.get(
        "/user/daily/activity/aggregated/cache_leakage_keys",
        params=_entity_params("user_id", "user-missing"),
    )
    assert response.status_code == 200, response.text
    assert response.json() == {"api_keys": []}
    scope: Final = repository.cache_leakage_keys.call_args.args[0]
    assert scope.entity_ids == ("user-missing",)


def test_export_json_stream_has_all_seeded_rows(daily_activity_client: tuple[TestClient, _FakeRepository]) -> None:
    client, _ = daily_activity_client
    response: Final = client.get(
        "/user/daily/activity/export",
        params={
            **_entity_params("user_id", "user-a"),
            "export_type": ExportType.DAILY.value,
            "format": "json",
        },
    )
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("application/json")
    assert len(response.json()) == 6


def test_user_internal_role_is_scoped_to_api_key(daily_activity_client: tuple[TestClient, _FakeRepository]) -> None:
    client, _ = daily_activity_client
    response: Final = client.get(
        "/tag/daily/activity/aggregated",
        params=_entity_params("tags", "blue"),
        headers={
            "x-user-role": LitellmUserRoles.INTERNAL_USER.value,
            "x-user-id": "user-a",
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["metadata"]["total_spend"] == pytest.approx(3.0)


def test_user_aggregate_keeps_current_day_query_semantics(
    daily_activity_client: tuple[TestClient, _FakeRepository],
) -> None:
    client, repository = daily_activity_client
    response: Final = client.get(
        "/user/daily/activity/aggregated",
        params={**_DATE_PARAMS, "user_id": "user-a", "timezone": 480, "include_current_utc_day": "true"},
    )
    assert response.status_code == 200, response.text
    assert response.json()["metadata"]["total_spend"] == pytest.approx(14.5)
    scope: Final = repository.aggregated.call_args.args[0]
    assert scope.entity_ids == ("user-a",)
    assert scope.timezone_offset_minutes == 480
    assert scope.include_current_utc_day is True


@pytest.mark.parametrize(
    ("start_date", "end_date", "message"),
    (
        ("2020-01-01", "2026-12-31", "at most 400 days"),
        ("0000-01-01", "9999-12-31", "valid YYYY-MM-DD"),
        ("2024-06-01", "2024-01-01", "on or after"),
        ("not-a-date", "2024-01-31", "valid YYYY-MM-DD"),
        ("2026-9-24", "2026-09-26", "valid YYYY-MM-DD"),
        ("２０２６-09-24", "2026-09-26", "valid YYYY-MM-DD"),
        ("2026-09-01", "2026-09-4", "valid YYYY-MM-DD"),
        (None, "2024-01-31", "start_date and end_date"),
    ),
)
def test_team_aggregated_route_rejects_bad_date_ranges(
    daily_activity_client: tuple[TestClient, _FakeRepository],
    start_date: str | None,
    end_date: str | None,
    message: str,
) -> None:
    client, repository = daily_activity_client
    response: Final = client.get(
        "/team/daily/activity/aggregated",
        params={"start_date": start_date, "end_date": end_date, "team_ids": "team-a"},
    )
    assert response.status_code == 400, response.text
    assert message in str(response.json()["detail"]), response.text
    repository.aggregated.assert_not_awaited()


def test_user_aggregate_rejects_missing_dates(daily_activity_client: tuple[TestClient, _FakeRepository]) -> None:
    client, repository = daily_activity_client
    missing_dates: Final = client.get("/user/daily/activity/aggregated", params={"user_id": "user-a"})
    assert missing_dates.status_code == 400, missing_dates.text
    assert missing_dates.json()["detail"] == {"error": "Please provide start_date and end_date"}
    repository.aggregated.assert_not_awaited()


@pytest.mark.parametrize(
    ("start_date", "end_date", "message"),
    (
        ("2020-01-01", "2026-12-31", "at most 400 days"),
        ("not-a-date", "2024-01-31", "valid YYYY-MM-DD"),
        ("2024-06-01", "2024-01-01", "on or after"),
    ),
)
def test_user_key_page_rejects_bad_date_ranges(
    daily_activity_client: tuple[TestClient, _FakeRepository],
    start_date: str,
    end_date: str,
    message: str,
) -> None:
    client, repository = daily_activity_client
    response: Final = client.get(
        "/user/daily/activity/aggregated/keys",
        params={"start_date": start_date, "end_date": end_date, "user_id": "user-a"},
    )
    assert response.status_code == 400, response.text
    assert message in str(response.json()["detail"]), response.text
    repository.key_page.assert_not_awaited()


_NON_CANONICAL_DATE_RANGES: Final[tuple[tuple[str, str], ...]] = (
    ("2026-9-24", "2026-09-26"),
    ("２０２６-09-24", "2026-09-26"),
    ("2026-09-01", "2026-09-4"),
)


@pytest.mark.parametrize(("start_date", "end_date"), _NON_CANONICAL_DATE_RANGES)
def test_user_aggregate_rejects_non_canonical_dates(
    daily_activity_client: tuple[TestClient, _FakeRepository], start_date: str, end_date: str
) -> None:
    client, repository = daily_activity_client
    response: Final = client.get(
        "/user/daily/activity/aggregated",
        params={"start_date": start_date, "end_date": end_date, "user_id": "user-a"},
    )
    assert response.status_code == 400, response.text
    assert response.json()["detail"] == {"error": "start_date and end_date must be valid YYYY-MM-DD dates"}
    repository.aggregated.assert_not_awaited()


def test_user_aggregate_still_accepts_ranges_wider_than_the_team_limit(
    daily_activity_client: tuple[TestClient, _FakeRepository],
) -> None:
    client, repository = daily_activity_client
    response: Final = client.get(
        "/user/daily/activity/aggregated",
        params={"start_date": "2020-01-01", "end_date": "2026-12-31", "user_id": "user-a"},
    )
    assert response.status_code == 200, response.text
    repository.aggregated.assert_awaited_once()


@pytest.mark.parametrize(("start_date", "end_date"), _NON_CANONICAL_DATE_RANGES)
@pytest.mark.parametrize(("prefix", "query_name", "entity_id"), _ENTITY_CASES)
def test_export_routes_reject_non_canonical_dates_before_querying(
    daily_activity_client: tuple[TestClient, _FakeRepository],
    prefix: str,
    query_name: str,
    entity_id: str,
    start_date: str,
    end_date: str,
) -> None:
    client, repository = daily_activity_client
    repository.export_rows_error = AssertionError("export must not query the repository")
    response: Final = client.get(
        f"{prefix}/daily/activity/export",
        params={query_name: entity_id, "start_date": start_date, "end_date": end_date, "export_type": "daily"},
    )
    assert response.status_code == 400, response.text
    assert response.json()["detail"] == {"error": "start_date and end_date must be valid YYYY-MM-DD dates"}
    assert "content-disposition" not in response.headers


def test_export_content_disposition_is_ascii_and_built_from_canonical_dates(
    daily_activity_client: tuple[TestClient, _FakeRepository],
) -> None:
    client, _ = daily_activity_client
    response: Final = client.get(
        "/team/daily/activity/export",
        params={**_entity_params("team_ids", "team-a"), "export_type": ExportType.DAILY.value},
    )
    assert response.status_code == 200, response.text
    disposition: Final = response.headers["content-disposition"]
    assert disposition == 'attachment; filename="team-usage-2025-01-01-2025-01-02-daily.csv"'
    assert disposition.isascii()


@pytest.mark.asyncio
@pytest.mark.parametrize("grouping", ("tag", "team"))
async def test_team_tag_api_resolvers_preserve_same_authorized_intersection(grouping: str) -> None:
    from litellm.proxy.management_endpoints.daily_activity_scopes import (
        ResolvedScope,
        _resolve_tag,
        _resolve_team,
        _tag_query,
        _team_query,
    )

    auth: Final = UserAPIKeyAuth(user_id="user-a", user_role=LitellmUserRoles.INTERNAL_USER)
    prisma: Final = _prisma_client()
    tag_scope: Final = await _resolve_tag(
        auth, _tag_query(tags="shared", team_ids="team-a", group_by=grouping, **_DATE_PARAMS), prisma
    )
    team_scope: Final = await _resolve_team(
        auth, _team_query(tags="shared", team_ids="team-a", group_by=grouping, **_DATE_PARAMS), prisma
    )
    assert isinstance(tag_scope, ResolvedScope)
    assert isinstance(team_scope, ResolvedScope)
    assert tag_scope == team_scope
    assert tag_scope.scope.table is DailyActivityTable.TAG
    assert tag_scope.scope.team_ids == ("team-a",)
    assert tag_scope.scope.tags == ("shared",)
    assert tag_scope.scope.api_keys == ("key-alpha",)
    assert tag_scope.scope.entity_id_field == ("tag" if grouping == "tag" else "team_id")


@pytest.mark.parametrize("prefix", ("/tag", "/team"))
@pytest.mark.parametrize(
    "suffix", ("/aggregated", "/aggregated/keys", "/aggregated/search", "/aggregated/model_top_keys", "/export")
)
def test_cross_team_reporting_is_denied_on_every_shared_route(
    daily_activity_client: tuple[TestClient, _FakeRepository], prefix: str, suffix: str
) -> None:
    client, _ = daily_activity_client
    response: Final = client.get(
        prefix + "/daily/activity" + suffix,
        params={
            **_DATE_PARAMS,
            "team_ids": "team-b",
            "tags": "shared",
            "group_by": "team",
            "search": "key",
            "model_group": "model",
            "export_type": "daily",
            "format": "json",
        },
        headers={"x-user-role": "internal_user", "x-user-id": "user-a"},
    )
    assert response.status_code == 404, response.text


@pytest.mark.asyncio
async def test_unfiltered_team_reporting_keeps_untagged_totals_source() -> None:
    from litellm.proxy.management_endpoints.daily_activity_scopes import ResolvedScope, _resolve_team, _team_query

    result: Final = await _resolve_team(
        UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN), _team_query(**_DATE_PARAMS), _prisma_client()
    )
    assert isinstance(result, ResolvedScope)
    assert result.scope.table is DailyActivityTable.TEAM
