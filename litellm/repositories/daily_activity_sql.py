from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from itertools import count, islice
from types import MappingProxyType
from typing import Final

from typing_extensions import assert_never

from litellm import constants
from litellm.constants import PTU_SENTINEL_API_KEY
from litellm.types.repositories.daily_activity import DailyActivityScope, DailyActivityTable, ExportType

_API_KEY_ROLLED_UP_BIT: Final = 32
_MODEL_GROUP_EXPR: Final = "COALESCE(NULLIF(model_group, ''), model)"


@dataclass(frozen=True, slots=True)
class SqlQuery:
    sql: str
    params: tuple[object, ...]


@dataclass(frozen=True, slots=True)
class ExportCursor:
    date: str
    entity_id: str
    group_key: str


PRISMA_TO_PG_TABLE: Final[Mapping[DailyActivityTable, str]] = MappingProxyType(
    {
        DailyActivityTable.USER: "LiteLLM_DailyUserSpend",
        DailyActivityTable.TEAM: "LiteLLM_DailyTeamSpend",
        DailyActivityTable.TAG: "LiteLLM_DailyTagSpend",
        DailyActivityTable.ORGANIZATION: "LiteLLM_DailyOrganizationSpend",
        DailyActivityTable.CUSTOMER: "LiteLLM_DailyEndUserSpend",
        DailyActivityTable.AGENT: "LiteLLM_DailyAgentSpend",
    }
)


def adjust_dates_for_timezone(
    start_date: str,
    end_date: str,
    timezone_offset_minutes: int | None,
    include_current_utc_day: bool = False,
    utc_now: datetime | None = None,
) -> tuple[str, str]:
    if not include_current_utc_day or timezone_offset_minutes is None:
        return start_date, end_date
    now: Final = utc_now if utc_now is not None else datetime.now(timezone.utc)
    caller_local_today: Final = (now - timedelta(minutes=timezone_offset_minutes)).date().isoformat()
    if end_date < caller_local_today:
        return start_date, end_date
    return start_date, max(end_date, now.date().isoformat())


def build_where_clause(scope: DailyActivityScope, *, start_index: int = 1) -> tuple[str, tuple[object, ...]]:
    adjusted_start, adjusted_end = adjust_dates_for_timezone(
        scope.start_date,
        scope.end_date,
        scope.timezone_offset_minutes,
        scope.include_current_utc_day,
    )
    entity_index: Final = start_index + 2
    has_entity_array: Final = scope.entity_ids is not None and bool(scope.entity_ids)
    exclusion_index: Final = entity_index + int(has_entity_array)
    model_index: Final = exclusion_index + int(bool(scope.exclude_entity_ids))
    api_keys_index: Final = model_index + int(bool(scope.model))
    conditions: Final = (
        f"date >= ${start_index}",
        f"date <= ${start_index + 1}",
        *(
            ("FALSE",)
            if scope.entity_ids == ()
            else (f'"{scope.entity_id_field}" = ANY(${entity_index}::text[])',)
            if has_entity_array
            else ()
        ),
        *(
            (
                f'("{scope.entity_id_field}" IS NULL '
                f'OR NOT ("{scope.entity_id_field}" = ANY(${exclusion_index}::text[])))',
            )
            if scope.exclude_entity_ids
            else ()
        ),
        *((f"model = ${model_index}",) if scope.model else ()),
        *(
            ("FALSE",)
            if scope.api_keys == ()
            else (f"api_key = ANY(${api_keys_index}::text[])",)
            if scope.api_keys
            else ()
        ),
    )
    params: Final = (
        adjusted_start,
        adjusted_end,
        *((list(scope.entity_ids or ()),) if has_entity_array else ()),
        *((list(scope.exclude_entity_ids),) if scope.exclude_entity_ids else ()),
        *((scope.model,) if scope.model else ()),
        *((list(scope.api_keys),) if scope.api_keys else ()),
    )
    return " AND ".join(conditions), params


def _ptu_flat_cost_select(table: DailyActivityTable, *, aggregate: bool = True) -> str:
    if table is DailyActivityTable.TEAM:
        return "SUM(ptu_flat_cost)::float AS ptu_flat_cost" if aggregate else "SUM(scoped.ptu_flat_cost)::float"
    return "0::float AS ptu_flat_cost" if aggregate else "0::float"


def _rollup_metric_select(table: DailyActivityTable) -> str:
    return f"""
            SUM(spend)::float AS spend,
            {_ptu_flat_cost_select(table)},
            SUM(prompt_tokens)::bigint AS prompt_tokens,
            SUM(completion_tokens)::bigint AS completion_tokens,
            SUM(cache_read_input_tokens)::bigint AS cache_read_input_tokens,
            SUM(cache_creation_input_tokens)::bigint AS cache_creation_input_tokens,
            SUM(compression_saved_tokens)::bigint AS compression_saved_tokens,
            SUM(compression_savings_spend)::float AS compression_savings_spend,
            SUM(prompt_caching_savings_spend)::float AS prompt_caching_savings_spend,
            SUM(gateway_injected_caching_savings_spend)::float AS gateway_injected_caching_savings_spend,
            SUM(autorouter_savings_spend)::float AS autorouter_savings_spend,
            SUM(api_requests)::bigint AS api_requests,
            SUM(successful_requests)::bigint AS successful_requests,
            SUM(failed_requests)::bigint AS failed_requests,
            SUM(total_response_time_ms)::bigint AS total_response_time_ms,
            SUM(timed_requests)::bigint AS timed_requests"""


def _validate_api_key_limit(api_key_limit: int) -> None:
    if not 1 <= api_key_limit <= constants.USAGE_TOP_API_KEYS_MAX:
        raise ValueError(f"api_key_limit must be between 1 and {constants.USAGE_TOP_API_KEYS_MAX}")


def _top_api_keys_sql(pg_table: str, where_clause: str, *, sentinel_param: int, limit_param: int) -> str:
    return f"""
            SELECT api_key, COUNT(*) OVER () AS distinct_api_keys
            FROM "{pg_table}"
            WHERE {where_clause} AND api_key <> ${sentinel_param}
            GROUP BY api_key
            ORDER BY SUM(spend::numeric) DESC, api_key
            LIMIT ${limit_param}
        """


def build_aggregated_sql(scope: DailyActivityScope, *, api_key_limit: int) -> SqlQuery:
    pg_table: Final = PRISMA_TO_PG_TABLE[scope.table]
    where_clause, where_params = build_where_clause(scope)
    _validate_api_key_limit(api_key_limit)
    sentinel_param: Final = len(where_params) + 1
    top_keys_limit_param: Final = len(where_params) + 2
    top_api_keys: Final = _top_api_keys_sql(
        pg_table, where_clause, sentinel_param=sentinel_param, limit_param=top_keys_limit_param
    )
    metric_select: Final = _rollup_metric_select(scope.table)
    sql: Final = f"""
        (SELECT
            date,
            NULL::text AS api_key,
            model,
            {_MODEL_GROUP_EXPR} AS model_group,
            custom_llm_provider,
            mcp_namespaced_tool_name,
            endpoint,
            (GROUPING(date) << 6) | {_API_KEY_ROLLED_UP_BIT}
                | GROUPING(model, {_MODEL_GROUP_EXPR},
                           custom_llm_provider, mcp_namespaced_tool_name,
                           endpoint) AS group_level,
            NULL::bigint AS distinct_api_keys,{metric_select}
        FROM "{pg_table}"
        WHERE {where_clause}
        GROUP BY GROUPING SETS (
            (date),
            (date, model),
            (date, {_MODEL_GROUP_EXPR}),
            (date, custom_llm_provider),
            (date, mcp_namespaced_tool_name),
            (date, endpoint),
            ()
        ))
        UNION ALL
        (WITH top_api_keys AS (
            {top_api_keys}
        )
        SELECT
            date,
            api_key,
            model,
            {_MODEL_GROUP_EXPR} AS model_group,
            custom_llm_provider,
            mcp_namespaced_tool_name,
            endpoint,
            GROUPING(date, api_key, model, {_MODEL_GROUP_EXPR},
                     custom_llm_provider, mcp_namespaced_tool_name,
                     endpoint) AS group_level,
            MAX(top_api_keys.distinct_api_keys) AS distinct_api_keys,{metric_select}
        FROM "{pg_table}" JOIN top_api_keys USING (api_key)
        WHERE {where_clause}
        GROUP BY GROUPING SETS (
            (date, api_key),
            (date, model, api_key),
            (date, {_MODEL_GROUP_EXPR}, api_key),
            (date, custom_llm_provider, api_key),
            (date, mcp_namespaced_tool_name, api_key),
            (date, endpoint, api_key)
        ))
    """
    return SqlQuery(
        sql=sql,
        params=(*where_params, PTU_SENTINEL_API_KEY, api_key_limit),
    )


def build_entity_rollup_sql(scope: DailyActivityScope, *, api_key_limit: int) -> SqlQuery:
    pg_table: Final = PRISMA_TO_PG_TABLE[scope.table]
    where_clause, where_params = build_where_clause(scope)
    _validate_api_key_limit(api_key_limit)
    sentinel_param: Final = len(where_params) + 1
    top_keys_limit_param: Final = len(where_params) + 2
    top_api_keys: Final = _top_api_keys_sql(
        pg_table, where_clause, sentinel_param=sentinel_param, limit_param=top_keys_limit_param
    )
    metric_select: Final = _rollup_metric_select(scope.table)
    sql: Final = f"""
        WITH top_api_keys AS (
            {top_api_keys}
        ),
        entity_api_keys AS (
            SELECT COALESCE("{scope.entity_id_field}", '') AS entity_id,
                   COUNT(DISTINCT api_key)::bigint AS distinct_api_keys
            FROM "{pg_table}"
            WHERE {where_clause} AND api_key <> ${sentinel_param}
            GROUP BY COALESCE("{scope.entity_id_field}", '')
        )
        (SELECT e.*, COALESCE(k.distinct_api_keys, 0)::bigint AS distinct_api_keys
         FROM (
             SELECT COALESCE("{scope.entity_id_field}", '') AS entity_id,
                    date,
                    NULL::text AS api_key,
                    1 AS api_key_rolled,{metric_select}
             FROM "{pg_table}"
             WHERE {where_clause}
             GROUP BY date, COALESCE("{scope.entity_id_field}", '')
         ) e
         LEFT JOIN entity_api_keys k ON k.entity_id = e.entity_id)
        UNION ALL
        (SELECT COALESCE("{scope.entity_id_field}", '') AS entity_id,
                date,
                api_key,
                0 AS api_key_rolled,{metric_select},
                NULL::bigint AS distinct_api_keys
         FROM "{pg_table}" JOIN top_api_keys USING (api_key)
         WHERE {where_clause}
         GROUP BY date, COALESCE("{scope.entity_id_field}", ''), api_key)
    """
    return SqlQuery(sql=sql, params=(*where_params, PTU_SENTINEL_API_KEY, api_key_limit))


def _key_spend_select() -> str:
    return """
            COALESCE(SUM(spend), 0)::float AS spend,
            COALESCE(SUM(prompt_tokens), 0)::bigint AS prompt_tokens,
            COALESCE(SUM(completion_tokens), 0)::bigint AS completion_tokens,
            (COALESCE(SUM(prompt_tokens), 0) + COALESCE(SUM(completion_tokens), 0))::bigint AS total_tokens,
            COALESCE(SUM(api_requests), 0)::bigint AS api_requests,
            COALESCE(SUM(successful_requests), 0)::bigint AS successful_requests,
            COALESCE(SUM(failed_requests), 0)::bigint AS failed_requests,
            COALESCE(SUM(cache_read_input_tokens), 0)::bigint AS cache_read_input_tokens,
            COALESCE(SUM(cache_creation_input_tokens), 0)::bigint AS cache_creation_input_tokens"""


def build_key_page_sql(scope: DailyActivityScope, *, offset: int, limit: int) -> SqlQuery:
    if not 1 <= limit <= constants.USAGE_KEY_PAGE_MAX:
        raise ValueError(f"limit must be between 1 and {constants.USAGE_KEY_PAGE_MAX}")
    if offset < 0:
        raise ValueError("offset must be non-negative")
    where_clause, where_params = build_where_clause(scope)
    sentinel_param: Final = len(where_params) + 1
    limit_param: Final = sentinel_param + 1
    offset_param: Final = limit_param + 1
    sql: Final = f"""
        WITH ranked AS (
            SELECT api_key,{_key_spend_select()}, SUM(spend::numeric) AS rank_spend
            FROM "{PRISMA_TO_PG_TABLE[scope.table]}"
            WHERE {where_clause} AND api_key <> ${sentinel_param}
            GROUP BY api_key
        )
        SELECT (SELECT COUNT(*) FROM ranked)::bigint AS total_api_keys, page.*
        FROM (SELECT 1) AS one
        LEFT JOIN LATERAL (
            SELECT * FROM ranked
            ORDER BY rank_spend DESC, api_key
            LIMIT ${limit_param} OFFSET ${offset_param}
        ) AS page ON TRUE
    """
    return SqlQuery(sql=sql, params=(*where_params, PTU_SENTINEL_API_KEY, limit, offset))


def _user_spend_select() -> str:
    return """
            COALESCE(SUM(spend), 0)::float AS spend,
            COALESCE(SUM(prompt_tokens), 0)::bigint AS prompt_tokens,
            COALESCE(SUM(completion_tokens), 0)::bigint AS completion_tokens,
            (COALESCE(SUM(prompt_tokens), 0) + COALESCE(SUM(completion_tokens), 0))::bigint AS total_tokens,
            COALESCE(SUM(api_requests), 0)::bigint AS api_requests,
            COALESCE(SUM(successful_requests), 0)::bigint AS successful_requests,
            COALESCE(SUM(failed_requests), 0)::bigint AS failed_requests"""


def build_user_page_sql(scope: DailyActivityScope, *, offset: int, limit: int) -> SqlQuery:
    if not 1 <= limit <= constants.USAGE_USER_PAGE_MAX:
        raise ValueError(f"limit must be between 1 and {constants.USAGE_USER_PAGE_MAX}")
    if offset < 0:
        raise ValueError("offset must be non-negative")
    where_clause, where_params = build_where_clause(scope)
    limit_param: Final = len(where_params) + 1
    offset_param: Final = limit_param + 1
    group_expr: Final = f"NULLIF(\"{scope.entity_id_field}\", '')"
    sql: Final = f"""
        WITH ranked AS (
            SELECT {group_expr} AS user_id,{_user_spend_select()}, SUM(spend::numeric) AS rank_spend
            FROM "{PRISMA_TO_PG_TABLE[scope.table]}"
            WHERE {where_clause}
            GROUP BY {group_expr}
        )
        SELECT (SELECT COUNT(*) FROM ranked)::bigint AS total_users, page.*
        FROM (SELECT 1) AS one
        LEFT JOIN LATERAL (
            SELECT * FROM ranked
            ORDER BY rank_spend DESC, user_id ASC NULLS LAST
            LIMIT ${limit_param} OFFSET ${offset_param}
        ) AS page ON TRUE
    """
    return SqlQuery(sql=sql, params=(*where_params, limit, offset))


def _bounded_limit(limit: int, *, minimum: int = 1) -> None:
    if limit < minimum:
        raise ValueError(f"limit must be at least {minimum}")


def build_key_search_sql(scope: DailyActivityScope, *, search: str, limit: int) -> SqlQuery:
    _bounded_limit(limit)
    where_clause, where_params = build_where_clause(scope)
    search_param: Final = len(where_params) + 1
    sentinel_param: Final = search_param + 1
    limit_param: Final = sentinel_param + 1
    escaped: Final = search.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    sql: Final = f"""
        SELECT api_key,{_key_spend_select()}
        FROM "{PRISMA_TO_PG_TABLE[scope.table]}"
        WHERE {where_clause}
          AND api_key <> ${sentinel_param}
          AND (
            api_key ILIKE ${search_param} ESCAPE '\\'
            OR api_key IN (
                SELECT v.token FROM "LiteLLM_VerificationToken" v
                LEFT JOIN "LiteLLM_UserTable" u ON u.user_id = v.user_id
                WHERE v.key_alias ILIKE ${search_param} ESCAPE '\\'
                   OR v.user_id ILIKE ${search_param} ESCAPE '\\'
                   OR u.user_email ILIKE ${search_param} ESCAPE '\\'
                UNION
                SELECT d.token FROM "LiteLLM_DeletedVerificationToken" d
                LEFT JOIN "LiteLLM_UserTable" u ON u.user_id = d.user_id
                WHERE d.key_alias ILIKE ${search_param} ESCAPE '\\'
                   OR d.user_id ILIKE ${search_param} ESCAPE '\\'
                   OR u.user_email ILIKE ${search_param} ESCAPE '\\'
            )
          )
        GROUP BY api_key
        ORDER BY SUM(spend::numeric) DESC, api_key
        LIMIT ${limit_param}
    """
    return SqlQuery(sql=sql, params=(*where_params, f"%{escaped}%", PTU_SENTINEL_API_KEY, limit))


def build_model_top_keys_sql(
    scope: DailyActivityScope, *, model_group: str, by_model_group: bool, limit: int
) -> SqlQuery:
    _bounded_limit(limit)
    where_clause, where_params = build_where_clause(scope)
    model_param: Final = len(where_params) + 1
    sentinel_param: Final = model_param + 1
    limit_param: Final = sentinel_param + 1
    model_clause: Final = (
        f"COALESCE(NULLIF(model_group, ''), model) = ${model_param}" if by_model_group else f"model = ${model_param}"
    )
    sql: Final = f"""
        SELECT api_key,{_key_spend_select()}
        FROM "{PRISMA_TO_PG_TABLE[scope.table]}"
        WHERE {where_clause} AND {model_clause} AND api_key <> ${sentinel_param}
        GROUP BY api_key
        ORDER BY SUM(spend::numeric) DESC, api_key
        LIMIT ${limit_param}
    """
    return SqlQuery(sql=sql, params=(*where_params, model_group, PTU_SENTINEL_API_KEY, limit))


def build_cache_leakage_keys_sql(scope: DailyActivityScope, *, limit: int) -> SqlQuery:
    _bounded_limit(limit)
    where_clause, where_params = build_where_clause(scope)
    sentinel_param: Final = len(where_params) + 1
    limit_param: Final = sentinel_param + 1
    sql: Final = f"""
        SELECT api_key,{_key_spend_select()}
        FROM "{PRISMA_TO_PG_TABLE[scope.table]}"
        WHERE {where_clause} AND api_key <> ${sentinel_param}
        GROUP BY api_key
        HAVING SUM(prompt_tokens) - SUM(cache_read_input_tokens) > 0
        ORDER BY SUM(prompt_tokens) - SUM(cache_read_input_tokens) DESC, api_key
        LIMIT ${limit_param}
    """
    return SqlQuery(sql=sql, params=(*where_params, PTU_SENTINEL_API_KEY, limit))


def build_export_sql(
    scope: DailyActivityScope, *, export_type: ExportType, after: ExportCursor | None, batch_size: int
) -> SqlQuery:
    _bounded_limit(batch_size)
    where_clause, where_params = build_where_clause(scope)
    group_key, output_key, user_fields, type_joins = _export_grouping(export_type)
    grouping_keys: Final = (
        f"scoped.date, COALESCE(scoped.\"{scope.entity_id_field}\", '')",
        *((group_key,) if export_type is not ExportType.DAILY else ()),
    )
    entity_joins: Final = (
        ('LEFT JOIN "LiteLLM_TeamTable" tt ON tt.team_id = scoped.team_id',)
        if scope.table is DailyActivityTable.TEAM
        else ('LEFT JOIN "LiteLLM_OrganizationTable" ot ON ot.organization_id = scoped.organization_id',)
        if scope.table is DailyActivityTable.ORGANIZATION
        else ()
    )
    joins: Final = (*type_joins, *entity_joins)
    alias_expression: Final = (
        "MAX(tt.team_alias)"
        if scope.table is DailyActivityTable.TEAM
        else "MAX(ot.organization_alias)"
        if scope.table is DailyActivityTable.ORGANIZATION
        else "NULL::text"
    )
    parameter_indexes: Final = count(len(where_params) + 1)
    sentinel_param: Final = next(parameter_indexes) if export_type is not ExportType.DAILY else None
    cursor_indexes: Final = tuple(islice(parameter_indexes, 3)) if after is not None else ()
    limit_param: Final = next(parameter_indexes)
    cursor_clause, cursor_params = _export_cursor_clause(
        scope, after=after, cursor_indexes=cursor_indexes, group_key=group_key
    )
    sentinel_clause: Final = f" AND api_key <> ${sentinel_param}" if sentinel_param is not None else ""
    table: Final = PRISMA_TO_PG_TABLE[scope.table]
    flat_cost: Final = _ptu_flat_cost_select(scope.table, aggregate=False)
    sql: Final = f"""
        WITH scoped AS (
            SELECT * FROM "{table}"
            WHERE {where_clause}{sentinel_clause}
        )
        SELECT
            scoped.date,
            COALESCE(scoped."{scope.entity_id_field}", '') AS entity_id,
            {alias_expression} AS entity_alias,
            {output_key} AS api_key,
            {user_fields},
            {"NULLIF(COALESCE(scoped.model, ''), '')" if export_type is ExportType.DAILY_WITH_MODELS else "NULL::text"} AS model,
            COALESCE(SUM(scoped.spend), 0)::float AS spend,
            {flat_cost} AS flat_cost,
            COALESCE(SUM(scoped.prompt_tokens), 0)::bigint AS prompt_tokens,
            COALESCE(SUM(scoped.completion_tokens), 0)::bigint AS completion_tokens,
            COALESCE(SUM(scoped.api_requests), 0)::bigint AS api_requests,
            COALESCE(SUM(scoped.successful_requests), 0)::bigint AS successful_requests,
            COALESCE(SUM(scoped.failed_requests), 0)::bigint AS failed_requests,
            COALESCE(SUM(scoped.cache_read_input_tokens), 0)::bigint AS cache_read_input_tokens,
            COALESCE(SUM(scoped.cache_creation_input_tokens), 0)::bigint AS cache_creation_input_tokens
        FROM scoped
        {" ".join(joins)}
        WHERE TRUE{cursor_clause}
        GROUP BY {", ".join(grouping_keys)}
        ORDER BY {", ".join(grouping_keys)}
        LIMIT ${limit_param}
    """
    return SqlQuery(
        sql=sql,
        params=(
            *where_params,
            *((PTU_SENTINEL_API_KEY,) if export_type is not ExportType.DAILY else ()),
            *cursor_params,
            batch_size,
        ),
    )


def _export_grouping(export_type: ExportType) -> tuple[str, str, str, tuple[str, ...]]:
    if export_type is ExportType.DAILY:
        return (
            "''",
            "NULL::text",
            "NULL::text AS key_alias, NULL::text AS user_id, NULL::text AS user_email",
            (),
        )
    if export_type is ExportType.DAILY_WITH_KEYS:
        return (
            "scoped.api_key",
            "NULLIF(scoped.api_key, '')",
            "MAX(COALESCE(vt.key_alias, dvt.key_alias)) AS key_alias, "
            "MAX(COALESCE(vt.user_id, dvt.user_id)) AS user_id, MAX(u.user_email) AS user_email",
            (
                'LEFT JOIN "LiteLLM_VerificationToken" vt ON vt.token = scoped.api_key',
                """LEFT JOIN LATERAL (
                    SELECT key_alias, user_id
                    FROM "LiteLLM_DeletedVerificationToken"
                    WHERE token = scoped.api_key
                    ORDER BY deleted_at DESC
                    LIMIT 1
                ) dvt ON vt.token IS NULL""",
                'LEFT JOIN "LiteLLM_UserTable" u ON u.user_id = COALESCE(vt.user_id, dvt.user_id)',
            ),
        )
    if export_type is ExportType.DAILY_WITH_MODELS:
        return (
            "COALESCE(scoped.model, '')",
            "NULL::text",
            "NULL::text AS key_alias, NULL::text AS user_id, NULL::text AS user_email",
            (),
        )
    if export_type is ExportType.DAILY_WITH_USERS:
        return (
            "COALESCE(vt.user_id, dvt.user_id, '')",
            "NULL::text",
            "NULL::text AS key_alias, MAX(COALESCE(vt.user_id, dvt.user_id)) AS user_id, "
            "MAX(u.user_email) AS user_email",
            (
                'LEFT JOIN "LiteLLM_VerificationToken" vt ON vt.token = scoped.api_key',
                """LEFT JOIN LATERAL (
                    SELECT key_alias, user_id
                    FROM "LiteLLM_DeletedVerificationToken"
                    WHERE token = scoped.api_key
                    ORDER BY deleted_at DESC
                    LIMIT 1
                ) dvt ON vt.token IS NULL""",
                'LEFT JOIN "LiteLLM_UserTable" u ON u.user_id = COALESCE(vt.user_id, dvt.user_id)',
            ),
        )
    assert_never(export_type)
    raise AssertionError("unreachable")


def _export_cursor_clause(
    scope: DailyActivityScope,
    *,
    after: ExportCursor | None,
    cursor_indexes: tuple[int, ...],
    group_key: str,
) -> tuple[str, tuple[object, ...]]:
    if after is None:
        return "", ()
    first_cursor_index: Final = cursor_indexes[0]
    clause: Final = (
        f""" AND (scoped.date, COALESCE(scoped."{scope.entity_id_field}", ''), {group_key}) """
        f"> (${first_cursor_index}, ${cursor_indexes[1]}, ${cursor_indexes[2]})"
    )
    return clause, (after.date, after.entity_id, after.group_key)
