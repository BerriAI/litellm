from datetime import datetime, timezone
from typing import Final

import pytest

from litellm import constants
from litellm.constants import PTU_SENTINEL_API_KEY
from litellm.repositories.daily_activity_sql import (
    ExportCursor,
    adjust_dates_for_timezone,
    build_aggregated_sql,
    build_cache_leakage_keys_sql,
    build_entity_rollup_sql,
    build_export_sql,
    build_key_page_sql,
    build_key_search_sql,
    build_model_top_keys_sql,
    build_where_clause,
)
from litellm.types.proxy.management_endpoints.common_daily_activity import SpendMetrics
from litellm.types.repositories.daily_activity import DailyActivityScope, DailyActivityTable, ExportType


def _scope(
    *,
    table: DailyActivityTable = DailyActivityTable.USER,
    entity_ids: tuple[str, ...] | None = ("user-1",),
    exclude_entity_ids: tuple[str, ...] = (),
    api_keys: tuple[str, ...] | None = None,
    model: str | None = None,
    timezone_offset_minutes: int | None = None,
    include_current_utc_day: bool = False,
    start_date: str = "2026-01-01",
    end_date: str = "2026-01-31",
) -> DailyActivityScope:
    entity_field = {
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
        start_date=start_date,
        end_date=end_date,
        model=model,
        timezone_offset_minutes=timezone_offset_minutes,
        include_current_utc_day=include_current_utc_day,
    )


def test_where_clause_binds_each_filter_as_a_single_array_parameter() -> None:
    scope = _scope(
        entity_ids=("user-1", "user-2"),
        exclude_entity_ids=("user-3",),
        api_keys=("key-1", "key-2"),
        model="gpt-test",
    )

    sql, params = build_where_clause(scope)

    assert sql == (
        'date >= $1 AND date <= $2 AND "user_id" = ANY($3::text[]) '
        'AND ("user_id" IS NULL OR NOT ("user_id" = ANY($4::text[]))) AND model = $5 AND api_key = ANY($6::text[])'
    )
    assert params == (
        "2026-01-01",
        "2026-01-31",
        ["user-1", "user-2"],
        ["user-3"],
        "gpt-test",
        ["key-1", "key-2"],
    )


def test_where_clause_exclusion_keeps_null_entity_rows() -> None:
    scope = _scope(table=DailyActivityTable.TEAM, entity_ids=None, exclude_entity_ids=("litellm-dashboard",))

    sql, params = build_where_clause(scope)

    assert sql == 'date >= $1 AND date <= $2 AND ("team_id" IS NULL OR NOT ("team_id" = ANY($3::text[])))'
    assert params == ("2026-01-01", "2026-01-31", ["litellm-dashboard"])


@pytest.mark.parametrize(
    ("entity_ids", "api_keys", "expected_sql", "expected_params"),
    [
        (None, None, "date >= $1 AND date <= $2", ("2026-01-01", "2026-01-31")),
        ((), None, "date >= $1 AND date <= $2 AND FALSE", ("2026-01-01", "2026-01-31")),
        (None, (), "date >= $1 AND date <= $2 AND FALSE", ("2026-01-01", "2026-01-31")),
    ],
)
def test_where_clause_distinguishes_no_filter_from_empty_membership(
    entity_ids: tuple[str, ...] | None,
    api_keys: tuple[str, ...] | None,
    expected_sql: str,
    expected_params: tuple[object, ...],
) -> None:
    scope = _scope(entity_ids=entity_ids, api_keys=api_keys)

    sql, params = build_where_clause(scope)

    assert sql == expected_sql
    assert params == expected_params


def test_key_page_sql_orders_exact_spend_and_binds_scope_before_page() -> None:
    query = build_key_page_sql(_scope(), offset=7, limit=3)

    assert query.params == (
        "2026-01-01",
        "2026-01-31",
        ["user-1"],
        PTU_SENTINEL_API_KEY,
        3,
        7,
    )
    assert "SUM(spend::numeric) AS rank_spend" in query.sql
    assert "ORDER BY rank_spend DESC, api_key" in query.sql
    assert "(SELECT COUNT(*) FROM ranked)::bigint AS total_api_keys" in query.sql


@pytest.mark.parametrize(
    ("offset", "limit", "error"),
    (
        (0, 0, "limit must be between"),
        (0, constants.USAGE_KEY_PAGE_MAX + 1, "limit must be between"),
        (-1, 1, "offset must be non-negative"),
    ),
)
def test_key_page_sql_rejects_invalid_page_bounds(offset: int, limit: int, error: str) -> None:
    with pytest.raises(ValueError, match=error):
        build_key_page_sql(_scope(), offset=offset, limit=limit)


def test_scope_rejects_an_entity_field_not_allowed_for_its_table() -> None:
    with pytest.raises(ValueError, match="Invalid entity_id_field"):
        DailyActivityScope(
            table=DailyActivityTable.USER,
            entity_id_field="team_id",
            entity_ids=None,
            exclude_entity_ids=(),
            api_keys=None,
            start_date="2026-01-01",
            end_date="2026-01-31",
            model=None,
            timezone_offset_minutes=None,
        )


def test_timezone_adjustment_only_extends_an_opted_in_live_range() -> None:
    now = datetime(2026, 8, 6, 4, 30, tzinfo=timezone.utc)

    assert adjust_dates_for_timezone("2026-07-06", "2026-08-05", 420, include_current_utc_day=True, utc_now=now) == (
        "2026-07-06",
        "2026-08-06",
    )
    assert adjust_dates_for_timezone("2026-07-01", "2026-08-04", 420, include_current_utc_day=True, utc_now=now) == (
        "2026-07-01",
        "2026-08-04",
    )


@pytest.mark.parametrize("offset_minutes", [None, 0, -330, -540, -60, 240, 300, 480])
def test_timezone_adjustment_preserves_daily_bucket_dates(offset_minutes: int | None) -> None:
    assert adjust_dates_for_timezone("2026-05-29", "2026-05-29", offset_minutes) == (
        "2026-05-29",
        "2026-05-29",
    )


@pytest.mark.parametrize("offset_minutes", [-330, 480])
def test_timezone_adjustment_preserves_single_day_additivity(offset_minutes: int) -> None:
    days: Final = ("2026-05-29", "2026-05-30", "2026-05-31", "2026-06-01", "2026-06-02")
    single_day_ranges: Final = tuple(adjust_dates_for_timezone(day, day, offset_minutes) for day in days)
    multi_day_range: Final = adjust_dates_for_timezone(days[0], days[-1], offset_minutes)

    assert tuple(start for start, _ in single_day_ranges) == days
    assert tuple(end for _, end in single_day_ranges) == days
    assert (min(start for start, _ in single_day_ranges), max(end for _, end in single_day_ranges)) == multi_day_range


def test_timezone_adjustment_live_end_handles_offset_and_opt_in_cases() -> None:
    pt_evening: Final = datetime(2026, 8, 6, 4, 30, tzinfo=timezone.utc)
    ist_evening: Final = datetime(2026, 8, 5, 17, 0, tzinfo=timezone.utc)
    utc_noon: Final = datetime(2026, 8, 5, 12, 0, tzinfo=timezone.utc)

    assert adjust_dates_for_timezone(
        "2026-07-06", "2026-08-05", 420, include_current_utc_day=True, utc_now=pt_evening
    ) == ("2026-07-06", "2026-08-06")
    assert adjust_dates_for_timezone("2026-07-06", "2026-08-05", 420, utc_now=pt_evening) == (
        "2026-07-06",
        "2026-08-05",
    )
    assert adjust_dates_for_timezone(
        "2026-07-01", "2026-08-04", 420, include_current_utc_day=True, utc_now=pt_evening
    ) == ("2026-07-01", "2026-08-04")
    assert adjust_dates_for_timezone(
        "2026-07-07", "2026-08-06", -330, include_current_utc_day=True, utc_now=ist_evening
    ) == ("2026-07-07", "2026-08-06")
    assert adjust_dates_for_timezone(
        "2026-07-06", "2026-08-05", None, include_current_utc_day=True, utc_now=pt_evening
    ) == ("2026-07-06", "2026-08-05")
    assert adjust_dates_for_timezone("2026-07-06", "2026-08-05", 0, include_current_utc_day=True, utc_now=utc_noon) == (
        "2026-07-06",
        "2026-08-05",
    )
    assert adjust_dates_for_timezone(
        "2026-07-06", "2026-08-09", 420, include_current_utc_day=True, utc_now=pt_evening
    ) == ("2026-07-06", "2026-08-09")


@pytest.mark.parametrize("offset_minutes", [None, 0, -330, 480])
def test_aggregated_query_uses_the_caller_date_bounds(offset_minutes: int | None) -> None:
    query = build_aggregated_sql(
        _scope(
            timezone_offset_minutes=offset_minutes,
            start_date="2026-05-29",
            end_date="2026-05-29",
        ),
        api_key_limit=constants.USAGE_TOP_API_KEYS_DEFAULT,
    )

    assert query.params[:2] == ("2026-05-29", "2026-05-29")
    assert "date >= $1" in query.sql
    assert "date <= $2" in query.sql


def test_aggregate_query_sums_all_savings_drivers_and_response_time() -> None:
    query = build_aggregated_sql(_scope(), api_key_limit=constants.USAGE_TOP_API_KEYS_DEFAULT)
    fields: Final = tuple(field for field in SpendMetrics.model_fields if field.endswith("_savings_spend")) + (
        "total_response_time_ms",
        "timed_requests",
    )

    assert fields
    assert all(f"SUM({field})" in query.sql for field in fields)


def test_aggregated_query_binds_sentinel_and_api_key_limit_after_scope_values() -> None:
    scope = _scope(entity_ids=None, api_keys=("key-1",))

    query = build_aggregated_sql(scope, api_key_limit=3)

    assert "api_key <> $4" in query.sql
    assert "LIMIT $5" in query.sql
    assert 'FROM "LiteLLM_DailyUserSpend"' in query.sql
    assert query.params == (
        "2026-01-01",
        "2026-01-31",
        ["key-1"],
        PTU_SENTINEL_API_KEY,
        3,
    )


@pytest.mark.parametrize("api_key_limit", [0, constants.USAGE_TOP_API_KEYS_MAX + 1])
def test_aggregated_query_rejects_api_key_limits_outside_bounds(api_key_limit: int) -> None:
    with pytest.raises(ValueError, match="api_key_limit"):
        build_aggregated_sql(_scope(), api_key_limit=api_key_limit)


def test_entity_rollup_bounds_keys_and_reuses_scope_filters() -> None:
    query = build_entity_rollup_sql(
        _scope(table=DailyActivityTable.TEAM, entity_ids=None, api_keys=("key-1", "key-2")),
        api_key_limit=3,
    )

    assert query.sql.count("COALESCE(\"team_id\", '') AS entity_id") == 3
    assert query.sql.count("GROUP BY date, COALESCE(\"team_id\", '')") == 2
    assert '"team_id" AS entity_id' not in query.sql
    assert "JOIN top_api_keys USING (api_key)" in query.sql
    assert "api_key = ANY($3::text[])" in query.sql
    assert query.sql.count("api_key = ANY($3::text[])") == 4
    assert query.sql.count("api_key <> $4") == 2
    assert query.sql.count("ORDER BY SUM(spend::numeric) DESC, api_key") == 1
    assert "k.entity_id = e.entity_id" in query.sql
    assert "LIMIT $5" in query.sql
    assert query.params == ("2026-01-01", "2026-01-31", ["key-1", "key-2"], PTU_SENTINEL_API_KEY, 3)


@pytest.mark.parametrize("api_key_limit", [0, constants.USAGE_TOP_API_KEYS_MAX + 1])
def test_entity_rollup_rejects_api_key_limits_outside_bounds(api_key_limit: int) -> None:
    with pytest.raises(ValueError, match="api_key_limit"):
        build_entity_rollup_sql(_scope(), api_key_limit=api_key_limit)


def test_search_query_escapes_pattern_metacharacters_and_binds_limit() -> None:
    query = build_key_search_sql(_scope(entity_ids=None), search=r"foo%_\bar", limit=4)

    assert "OR api_key IN (" in query.sql
    assert 'SELECT v.token FROM "LiteLLM_VerificationToken" v' in query.sql
    assert 'LEFT JOIN "LiteLLM_UserTable" u ON u.user_id = v.user_id' in query.sql
    assert 'SELECT d.token FROM "LiteLLM_DeletedVerificationToken" d' in query.sql
    assert 'LEFT JOIN "LiteLLM_UserTable" u ON u.user_id = d.user_id' in query.sql
    assert "d.key_alias ILIKE $3 ESCAPE" in query.sql
    assert "d.user_id ILIKE $3 ESCAPE" in query.sql
    assert "api_key ILIKE $3 ESCAPE" in query.sql
    assert "v.key_alias ILIKE $3 ESCAPE" in query.sql
    assert "v.user_id ILIKE $3 ESCAPE" in query.sql
    assert "u.user_email ILIKE $3 ESCAPE" in query.sql
    assert query.sql.count("ILIKE $3 ESCAPE") == 7
    assert "api_key <> $4" in query.sql
    assert "ORDER BY SUM(spend::numeric) DESC, api_key" in query.sql
    assert "LIMIT $5" in query.sql
    assert query.params == (
        "2026-01-01",
        "2026-01-31",
        r"%foo\%\_\\bar%",
        PTU_SENTINEL_API_KEY,
        4,
    )


def test_model_and_cache_key_queries_bind_filters_sentinel_and_limits() -> None:
    model_query = build_model_top_keys_sql(
        _scope(entity_ids=None), model_group="public-model", by_model_group=True, limit=5
    )
    leakage_query = build_cache_leakage_keys_sql(_scope(entity_ids=None), limit=20)

    assert "COALESCE(NULLIF(model_group, ''), model) = $3" in model_query.sql
    assert "api_key <> $4" in model_query.sql
    assert "ORDER BY SUM(spend::numeric) DESC, api_key" in model_query.sql
    assert model_query.params == ("2026-01-01", "2026-01-31", "public-model", PTU_SENTINEL_API_KEY, 5)
    assert "HAVING SUM(prompt_tokens) - SUM(cache_read_input_tokens) > 0" in leakage_query.sql
    assert "ORDER BY SUM(prompt_tokens) - SUM(cache_read_input_tokens) DESC, api_key" in leakage_query.sql
    assert leakage_query.params == ("2026-01-01", "2026-01-31", PTU_SENTINEL_API_KEY, 20)


@pytest.mark.parametrize(
    "builder",
    [
        lambda: build_key_search_sql(_scope(), search="x", limit=0),
        lambda: build_model_top_keys_sql(_scope(), model_group="x", by_model_group=False, limit=0),
        lambda: build_cache_leakage_keys_sql(_scope(), limit=0),
        lambda: build_export_sql(_scope(), export_type=ExportType.DAILY, after=None, batch_size=0),
    ],
)
def test_query_builders_reject_nonpositive_limits(builder) -> None:
    with pytest.raises(ValueError, match="limit must be at least 1"):
        builder()


@pytest.mark.parametrize(
    ("export_type", "group_key", "key_filter", "joins"),
    [
        (ExportType.DAILY, "''", "", ""),
        (ExportType.DAILY_WITH_KEYS, "scoped.api_key", "api_key <> $3", 'LEFT JOIN "LiteLLM_VerificationToken"'),
        (ExportType.DAILY_WITH_MODELS, "COALESCE(scoped.model, '')", "api_key <> $3", ""),
        (
            ExportType.DAILY_WITH_USERS,
            "COALESCE(vt.user_id, dvt.user_id, '')",
            "api_key <> $3",
            'LEFT JOIN "LiteLLM_VerificationToken"',
        ),
    ],
)
def test_export_groups_by_requested_key_and_binds_cursor_after_scope(
    export_type: ExportType, group_key: str, key_filter: str, joins: str
) -> None:
    query = build_export_sql(
        _scope(entity_ids=None),
        export_type=export_type,
        after=ExportCursor(date="2026-01-12", entity_id="user-2", group_key="group-3"),
        batch_size=2,
    )

    assert group_key in query.sql
    assert key_filter in query.sql
    assert joins in query.sql
    assert "(scoped.date, COALESCE(scoped.\"user_id\", '')," in query.sql
    order_keys: Final = (
        "scoped.date, COALESCE(scoped.\"user_id\", '')",
        *((group_key,) if export_type is not ExportType.DAILY else ()),
    )
    assert f"ORDER BY {', '.join(order_keys)}" in query.sql
    expected_limit_index: Final = "$6" if export_type is ExportType.DAILY else "$7"
    assert f"LIMIT {expected_limit_index}" in query.sql
    assert query.params == (
        "2026-01-01",
        "2026-01-31",
        *((PTU_SENTINEL_API_KEY,) if export_type is not ExportType.DAILY else ()),
        "2026-01-12",
        "user-2",
        "group-3",
        2,
    )


@pytest.mark.parametrize("export_type", [ExportType.DAILY_WITH_KEYS, ExportType.DAILY_WITH_USERS])
def test_export_uses_latest_deleted_key_metadata(export_type: ExportType) -> None:
    query = build_export_sql(_scope(entity_ids=None), export_type=export_type, after=None, batch_size=2)

    assert 'FROM "LiteLLM_DeletedVerificationToken"' in query.sql
    assert "ORDER BY deleted_at DESC" in query.sql
    assert "COALESCE(vt.user_id, dvt.user_id)" in query.sql


@pytest.mark.parametrize("export_type", tuple(ExportType))
def test_export_without_cursor_omits_cursor_predicate_and_parameters(export_type: ExportType) -> None:
    query = build_export_sql(
        _scope(entity_ids=None),
        export_type=export_type,
        after=None,
        batch_size=2,
    )

    assert "WHERE TRUE AND (scoped.date" not in query.sql
    assert query.params == (
        "2026-01-01",
        "2026-01-31",
        *((PTU_SENTINEL_API_KEY,) if export_type is not ExportType.DAILY else ()),
        2,
    )


@pytest.mark.parametrize("table", tuple(table for table in DailyActivityTable if table is not DailyActivityTable.TAG))
def test_cross_dimension_filters_reject_tables_without_tag_attribution(table: DailyActivityTable) -> None:
    from dataclasses import replace

    with pytest.raises(ValueError, match="dimension filters"):
        replace(_scope(table=table), tags=())


def test_dimension_filters_bind_after_existing_filters_with_nondefault_offset() -> None:
    from dataclasses import replace

    scope: Final = replace(
        _scope(table=DailyActivityTable.TAG, entity_ids=("shared",), api_keys=("key",), model="model"),
        team_ids=("team-a", "team-b"),
        exclude_team_ids=("team-b",),
        tags=("shared", "x'); DROP TABLE tags; --"),
        exclude_tags=("private",),
    )
    clause, params = build_where_clause(scope, start_index=4)
    assert params == (
        "2026-01-01",
        "2026-01-31",
        ["shared"],
        "model",
        ["key"],
        ["team-a", "team-b"],
        ["team-b"],
        ["shared", "x'); DROP TABLE tags; --"],
        ["private"],
    )
    assert '"team_id" = ANY($9::text[])' in clause
    assert '"tag" = ANY($11::text[])' in clause
    assert "DROP TABLE" not in clause
