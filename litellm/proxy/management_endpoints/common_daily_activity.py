import asyncio
from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from fastapi import HTTPException, status

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import CommonProxyErrors
from litellm.proxy.utils import PrismaClient
from litellm.types.proxy.management_endpoints.common_daily_activity import (
    BreakdownMetrics,
    DailySpendData,
    DailySpendMetadata,
    KeyMetadata,
    KeyMetricWithMetadata,
    MetricWithMetadata,
    SpendAnalyticsPaginatedResponse,
    SpendMetrics,
)

# Mapping from Prisma accessor names to actual PostgreSQL table names.
_PRISMA_TO_PG_TABLE: Dict[str, str] = {
    "litellm_dailyuserspend": "LiteLLM_DailyUserSpend",
    "litellm_dailyteamspend": "LiteLLM_DailyTeamSpend",
    "litellm_dailyorganizationspend": "LiteLLM_DailyOrganizationSpend",
    "litellm_dailyenduserspend": "LiteLLM_DailyEndUserSpend",
    "litellm_dailyagentspend": "LiteLLM_DailyAgentSpend",
    "litellm_dailytagspend": "LiteLLM_DailyTagSpend",
}


def update_metrics(existing_metrics: SpendMetrics, record: Any) -> SpendMetrics:
    """Update metrics with new record data."""
    existing_metrics.spend += record.spend
    existing_metrics.prompt_tokens += record.prompt_tokens
    existing_metrics.completion_tokens += record.completion_tokens
    existing_metrics.total_tokens += record.prompt_tokens + record.completion_tokens
    existing_metrics.cache_read_input_tokens += record.cache_read_input_tokens
    existing_metrics.cache_creation_input_tokens += record.cache_creation_input_tokens
    existing_metrics.api_requests += record.api_requests
    existing_metrics.successful_requests += record.successful_requests
    existing_metrics.failed_requests += record.failed_requests
    return existing_metrics


def _is_user_agent_tag(tag: Optional[str]) -> bool:
    """Determine whether a tag should be treated as a User-Agent tag."""
    if not tag:
        return False
    normalized_tag = tag.strip().lower()
    return normalized_tag.startswith("user-agent:") or normalized_tag.startswith("user agent:")


def compute_tag_metadata_totals(records: List[Any]) -> SpendMetrics:
    """
    Deduplicate spend metrics for tags using request_id, ignoring User-Agent prefixed tags.

    Each unique request_id contributes at most one record (the tag with max spend) to metadata.
    """
    deduped_records: Dict[str, Any] = {}
    for record in records:
        request_id = getattr(record, "request_id", None)
        if not request_id:
            continue

        tag_value = getattr(record, "tag", None)
        if _is_user_agent_tag(tag_value):
            continue

        current_best = deduped_records.get(request_id)
        if current_best is None or record.spend > current_best.spend:
            deduped_records[request_id] = record

    metadata_metrics = SpendMetrics()
    for record in deduped_records.values():
        update_metrics(metadata_metrics, record)
    return metadata_metrics


def update_breakdown_metrics(
    breakdown: BreakdownMetrics,
    record: Any,
    model_metadata: Dict[str, Dict[str, Any]],
    provider_metadata: Dict[str, Dict[str, Any]],
    api_key_metadata: Dict[str, Dict[str, Any]],
    entity_id_field: Optional[str] = None,
    entity_metadata_field: Optional[Dict[str, dict]] = None,
) -> BreakdownMetrics:
    """Updates breakdown metrics for a single record using the existing update_metrics function"""

    # Update model breakdown
    if record.model and record.model not in breakdown.models:
        breakdown.models[record.model] = MetricWithMetadata(
            metrics=SpendMetrics(),
            metadata=model_metadata.get(
                record.model, {}
            ),  # Add any model-specific metadata here
        )
    if record.model:
        breakdown.models[record.model].metrics = update_metrics(
            breakdown.models[record.model].metrics, record
        )

        # Update API key breakdown for this model
        if record.api_key not in breakdown.models[record.model].api_key_breakdown:
            breakdown.models[record.model].api_key_breakdown[record.api_key] = (
                KeyMetricWithMetadata(
                    metrics=SpendMetrics(),
                    metadata=KeyMetadata(
                        key_alias=api_key_metadata.get(record.api_key, {}).get(
                            "key_alias", None
                        ),
                        team_id=api_key_metadata.get(record.api_key, {}).get(
                            "team_id", None
                        ),
                    ),
                )
            )
        breakdown.models[record.model].api_key_breakdown[record.api_key].metrics = (
            update_metrics(
                breakdown.models[record.model]
                .api_key_breakdown[record.api_key]
                .metrics,
                record,
            )
        )

    # Update model group breakdown
    if record.model_group and record.model_group not in breakdown.model_groups:
        breakdown.model_groups[record.model_group] = MetricWithMetadata(
            metrics=SpendMetrics(),
            metadata=model_metadata.get(record.model_group, {}),
        )
    if record.model_group:
        breakdown.model_groups[record.model_group].metrics = update_metrics(
            breakdown.model_groups[record.model_group].metrics, record
        )

        # Update API key breakdown for this model
        if (
            record.api_key
            not in breakdown.model_groups[record.model_group].api_key_breakdown
        ):
            breakdown.model_groups[record.model_group].api_key_breakdown[
                record.api_key
            ] = KeyMetricWithMetadata(
                metrics=SpendMetrics(),
                metadata=KeyMetadata(
                    key_alias=api_key_metadata.get(record.api_key, {}).get(
                        "key_alias", None
                    ),
                    team_id=api_key_metadata.get(record.api_key, {}).get(
                        "team_id", None
                    ),
                ),
            )
        breakdown.model_groups[record.model_group].api_key_breakdown[
            record.api_key
        ].metrics = update_metrics(
            breakdown.model_groups[record.model_group]
            .api_key_breakdown[record.api_key]
            .metrics,
            record,
        )

    if record.mcp_namespaced_tool_name:
        if record.mcp_namespaced_tool_name not in breakdown.mcp_servers:
            breakdown.mcp_servers[record.mcp_namespaced_tool_name] = MetricWithMetadata(
                metrics=SpendMetrics(),
                metadata={},
            )
        breakdown.mcp_servers[record.mcp_namespaced_tool_name].metrics = update_metrics(
            breakdown.mcp_servers[record.mcp_namespaced_tool_name].metrics, record
        )

        # Update API key breakdown for this MCP server
        if (
            record.api_key
            not in breakdown.mcp_servers[
                record.mcp_namespaced_tool_name
            ].api_key_breakdown
        ):
            breakdown.mcp_servers[record.mcp_namespaced_tool_name].api_key_breakdown[
                record.api_key
            ] = KeyMetricWithMetadata(
                metrics=SpendMetrics(),
                metadata=KeyMetadata(
                    key_alias=api_key_metadata.get(record.api_key, {}).get(
                        "key_alias", None
                    ),
                    team_id=api_key_metadata.get(record.api_key, {}).get(
                        "team_id", None
                    ),
                ),
            )

        breakdown.mcp_servers[record.mcp_namespaced_tool_name].api_key_breakdown[
            record.api_key
        ].metrics = update_metrics(
            breakdown.mcp_servers[record.mcp_namespaced_tool_name]
            .api_key_breakdown[record.api_key]
            .metrics,
            record,
        )

    # Update provider breakdown
    provider = record.custom_llm_provider or "unknown"
    if provider not in breakdown.providers:
        breakdown.providers[provider] = MetricWithMetadata(
            metrics=SpendMetrics(),
            metadata=provider_metadata.get(
                provider, {}
            ),  # Add any provider-specific metadata here
        )
    breakdown.providers[provider].metrics = update_metrics(
        breakdown.providers[provider].metrics, record
    )

    # Update API key breakdown for this provider
    if record.api_key not in breakdown.providers[provider].api_key_breakdown:
        breakdown.providers[provider].api_key_breakdown[record.api_key] = (
            KeyMetricWithMetadata(
                metrics=SpendMetrics(),
                metadata=KeyMetadata(
                    key_alias=api_key_metadata.get(record.api_key, {}).get(
                        "key_alias", None
                    ),
                    team_id=api_key_metadata.get(record.api_key, {}).get(
                        "team_id", None
                    ),
                ),
            )
        )
    breakdown.providers[provider].api_key_breakdown[record.api_key].metrics = (
        update_metrics(
            breakdown.providers[provider].api_key_breakdown[record.api_key].metrics,
            record,
        )
    )

    # Update endpoint breakdown
    if record.endpoint:
        if record.endpoint not in breakdown.endpoints:
            breakdown.endpoints[record.endpoint] = MetricWithMetadata(
                metrics=SpendMetrics(),
                metadata={},
            )
        breakdown.endpoints[record.endpoint].metrics = update_metrics(
            breakdown.endpoints[record.endpoint].metrics, record
        )

        # Update API key breakdown for this endpoint
        if record.api_key not in breakdown.endpoints[record.endpoint].api_key_breakdown:
            breakdown.endpoints[record.endpoint].api_key_breakdown[record.api_key] = (
                KeyMetricWithMetadata(
                    metrics=SpendMetrics(),
                    metadata=KeyMetadata(
                        key_alias=api_key_metadata.get(record.api_key, {}).get(
                            "key_alias", None
                        ),
                        team_id=api_key_metadata.get(record.api_key, {}).get(
                            "team_id", None
                        ),
                    ),
                )
            )
        breakdown.endpoints[record.endpoint].api_key_breakdown[record.api_key].metrics = (
            update_metrics(
                breakdown.endpoints[record.endpoint]
                .api_key_breakdown[record.api_key]
                .metrics,
                record,
            )
        )

    # Update api key breakdown
    if record.api_key not in breakdown.api_keys:
        breakdown.api_keys[record.api_key] = KeyMetricWithMetadata(
            metrics=SpendMetrics(),
            metadata=KeyMetadata(
                key_alias=api_key_metadata.get(record.api_key, {}).get(
                    "key_alias", None
                ),
                team_id=api_key_metadata.get(record.api_key, {}).get("team_id", None),
            ),  # Add any api_key-specific metadata here
        )
    breakdown.api_keys[record.api_key].metrics = update_metrics(
        breakdown.api_keys[record.api_key].metrics, record
    )

    # Update entity-specific metrics if entity_id_field is provided
    if entity_id_field:
        entity_value = getattr(record, entity_id_field, None)
        entity_value = (
            entity_value if entity_value else "Unassigned"
        )  # allow for null entity_id_field
        if entity_value not in breakdown.entities:
            breakdown.entities[entity_value] = MetricWithMetadata(
                metrics=SpendMetrics(),
                metadata=(
                    entity_metadata_field.get(entity_value, {})
                    if entity_metadata_field
                    else {}
                ),
            )
        breakdown.entities[entity_value].metrics = update_metrics(
            breakdown.entities[entity_value].metrics, record
        )

        # Update API key breakdown for this entity
        if record.api_key not in breakdown.entities[entity_value].api_key_breakdown:
            breakdown.entities[entity_value].api_key_breakdown[record.api_key] = (
                KeyMetricWithMetadata(
                    metrics=SpendMetrics(),
                    metadata=KeyMetadata(
                        key_alias=api_key_metadata.get(record.api_key, {}).get(
                            "key_alias", None
                        ),
                        team_id=api_key_metadata.get(record.api_key, {}).get(
                            "team_id", None
                        ),
                    ),
                )
            )
        breakdown.entities[entity_value].api_key_breakdown[record.api_key].metrics = (
            update_metrics(
                breakdown.entities[entity_value]
                .api_key_breakdown[record.api_key]
                .metrics,
                record,
            )
        )

    return breakdown


async def get_api_key_metadata(
    prisma_client: PrismaClient,
    api_keys: Set[str],
) -> Dict[str, Dict[str, Any]]:
    """Get api key metadata, falling back to deleted keys table for keys not found in active table.

    This ensures that key_alias and team_id are preserved in historical activity logs
    even after a key is deleted or regenerated.
    """
    key_records = await prisma_client.db.litellm_verificationtoken.find_many(
        where={"token": {"in": list(api_keys)}}
    )
    result = {
        k.token: {"key_alias": k.key_alias, "team_id": k.team_id}
        for k in key_records
    }

    # For any keys not found in the active table, check the deleted keys table
    missing_keys = api_keys - set(result.keys())
    if missing_keys:
        try:
            deleted_key_records = (
                await prisma_client.db.litellm_deletedverificationtoken.find_many(
                    where={"token": {"in": list(missing_keys)}},
                    order={"deleted_at": "desc"},
                )
            )
            # Use the most recent deleted record for each token (ordered by deleted_at desc)
            for k in deleted_key_records:
                if k.token not in result:
                    result[k.token] = {
                        "key_alias": k.key_alias,
                        "team_id": k.team_id,
                    }
        except Exception as e:
            verbose_proxy_logger.warning(
                "Failed to fetch deleted key metadata for %d missing keys: %s",
                len(missing_keys),
                e,
            )

    return result


def _adjust_dates_for_timezone(
    start_date: str,
    end_date: str,
    timezone_offset_minutes: Optional[int],
) -> Tuple[str, str]:
    """
    Adjust date range to account for timezone differences.

    The database stores dates in UTC. When a user in a different timezone
    selects a local date range, we need to expand the UTC query range to
    capture all records that fall within their local date range.

    Args:
        start_date: Start date in YYYY-MM-DD format (user's local date)
        end_date: End date in YYYY-MM-DD format (user's local date)
        timezone_offset_minutes: Minutes behind UTC (positive = west of UTC)
            This matches JavaScript's Date.getTimezoneOffset() convention.
            For example: PST = +480 (8 hours * 60 = 480 minutes behind UTC)

    Returns:
        Tuple of (adjusted_start_date, adjusted_end_date) in YYYY-MM-DD format
    """
    if timezone_offset_minutes is None or timezone_offset_minutes == 0:
        return start_date, end_date

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    if timezone_offset_minutes > 0:
        # West of UTC (Americas): local evening extends into next UTC day
        # e.g., Feb 4 23:59 PST = Feb 5 07:59 UTC
        end = end + timedelta(days=1)
    else:
        # East of UTC (Asia/Europe): local morning starts in previous UTC day
        # e.g., Feb 4 00:00 IST = Feb 3 18:30 UTC
        start = start - timedelta(days=1)

    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def _build_where_conditions(
    *,
    entity_id_field: str,
    entity_id: Optional[Union[str, List[str]]],
    start_date: str,
    end_date: str,
    model: Optional[str],
    api_key: Optional[Union[str, List[str]]],
    exclude_entity_ids: Optional[List[str]] = None,
    timezone_offset_minutes: Optional[int] = None,
) -> Dict[str, Any]:
    """Build prisma where clause for daily activity queries."""
    # Adjust dates for timezone if provided
    adjusted_start, adjusted_end = _adjust_dates_for_timezone(
        start_date, end_date, timezone_offset_minutes
    )

    where_conditions: Dict[str, Any] = {
        "date": {
            "gte": adjusted_start,
            "lte": adjusted_end,
        }
    }

    if model:
        where_conditions["model"] = model
    if api_key:
        if isinstance(api_key, list):
            where_conditions["api_key"] = {"in": api_key}
        else:
            where_conditions["api_key"] = api_key

    if entity_id is not None:
        if isinstance(entity_id, list):
            where_conditions[entity_id_field] = {"in": entity_id}
        else:
            where_conditions[entity_id_field] = {"equals": entity_id}

    if exclude_entity_ids:
        current = where_conditions.get(entity_id_field, {})
        if isinstance(current, str):
            current = {"equals": current}
        current["not"] = {"in": exclude_entity_ids}
        where_conditions[entity_id_field] = current

    return where_conditions


# ---------------------------------------------------------------------------
# Helpers for the optimized multi-query aggregation
# ---------------------------------------------------------------------------

# SQL SUM expressions reused across all aggregation queries.
_METRIC_SUM_EXPR = (
    "SUM(spend)::float AS spend, "
    "SUM(prompt_tokens)::bigint AS prompt_tokens, "
    "SUM(completion_tokens)::bigint AS completion_tokens, "
    "SUM(cache_read_input_tokens)::bigint AS cache_read_input_tokens, "
    "SUM(cache_creation_input_tokens)::bigint AS cache_creation_input_tokens, "
    "SUM(api_requests)::bigint AS api_requests, "
    "SUM(successful_requests)::bigint AS successful_requests, "
    "SUM(failed_requests)::bigint AS failed_requests"
)


def _build_base_where_clause_sql(
    *,
    entity_id_field: str,
    entity_id: Optional[Union[str, List[str]]],
    start_date: str,
    end_date: str,
    model: Optional[str],
    api_key: Optional[Union[str, List[str]]],
    exclude_entity_ids: Optional[List[str]],
) -> Tuple[str, List[Any]]:
    """Build a parameterised SQL WHERE clause shared by all aggregation queries.

    Returns (where_clause_str, params_list).  The same clause and params are
    reused across every GROUP BY query so that only the SELECT / GROUP BY
    portion differs.
    """
    conditions: List[str] = []
    params: List[Any] = []
    p = 1  # 1-based for PostgreSQL $N placeholders

    # Date range (always present)
    conditions.append(f"date >= ${p}")
    params.append(start_date)
    p += 1
    conditions.append(f"date <= ${p}")
    params.append(end_date)
    p += 1

    # Entity filter
    if entity_id is not None:
        if isinstance(entity_id, list):
            placeholders = ", ".join(f"${p + i}" for i in range(len(entity_id)))
            conditions.append(f'"{entity_id_field}" IN ({placeholders})')
            params.extend(entity_id)
            p += len(entity_id)
        else:
            conditions.append(f'"{entity_id_field}" = ${p}')
            params.append(entity_id)
            p += 1

    # Exclude specific entities
    if exclude_entity_ids:
        placeholders = ", ".join(
            f"${p + i}" for i in range(len(exclude_entity_ids))
        )
        conditions.append(f'"{entity_id_field}" NOT IN ({placeholders})')
        params.extend(exclude_entity_ids)
        p += len(exclude_entity_ids)

    # Optional model filter
    if model:
        conditions.append(f"model = ${p}")
        params.append(model)
        p += 1

    # Optional api_key filter
    if api_key:
        if isinstance(api_key, list):
            placeholders = ", ".join(f"${p + i}" for i in range(len(api_key)))
            conditions.append(f"api_key IN ({placeholders})")
            params.extend(api_key)
            p += len(api_key)
        else:
            conditions.append(f"api_key = ${p}")
            params.append(api_key)
            p += 1

    return " AND ".join(conditions), params


def _row_to_spend_metrics(row: dict) -> SpendMetrics:
    """Convert a SQL result dict to a SpendMetrics instance."""
    pt = row.get("prompt_tokens") or 0
    ct = row.get("completion_tokens") or 0
    return SpendMetrics(
        spend=row.get("spend") or 0.0,
        prompt_tokens=pt,
        completion_tokens=ct,
        total_tokens=pt + ct,
        cache_read_input_tokens=row.get("cache_read_input_tokens") or 0,
        cache_creation_input_tokens=row.get("cache_creation_input_tokens") or 0,
        api_requests=row.get("api_requests") or 0,
        successful_requests=row.get("successful_requests") or 0,
        failed_requests=row.get("failed_requests") or 0,
    )


def _add_metrics(target: SpendMetrics, source: SpendMetrics) -> None:
    """Accumulate *source* metrics into *target* in-place."""
    target.spend += source.spend
    target.prompt_tokens += source.prompt_tokens
    target.completion_tokens += source.completion_tokens
    target.total_tokens += source.total_tokens
    target.cache_read_input_tokens += source.cache_read_input_tokens
    target.cache_creation_input_tokens += source.cache_creation_input_tokens
    target.api_requests += source.api_requests
    target.successful_requests += source.successful_requests
    target.failed_requests += source.failed_requests


def _build_aggregated_sql_query(
    *,
    table_name: str,
    entity_id_field: str,
    entity_id: Optional[Union[str, List[str]]],
    start_date: str,
    end_date: str,
    model: Optional[str],
    api_key: Optional[Union[str, List[str]]],
    exclude_entity_ids: Optional[List[str]] = None,
    timezone_offset_minutes: Optional[int] = None,
    include_entity_id: bool = False,
) -> Tuple[str, List[Any]]:
    """Build a parameterized SQL GROUP BY query for aggregated daily activity.

    Groups by (date, api_key, model, model_group, custom_llm_provider,
    mcp_namespaced_tool_name, endpoint) with SUMs on all metric columns.

    When include_entity_id is False (default), the entity_id column is omitted
    from GROUP BY to collapse rows across entities.

    When include_entity_id is True, the entity_id column is included in both
    SELECT and GROUP BY, preserving per-entity breakdown in the results.

    Returns:
        Tuple of (sql_query, params_list) ready for prisma_client.db.query_raw().
    """
    pg_table = _PRISMA_TO_PG_TABLE.get(table_name)
    if pg_table is None:
        raise ValueError(f"Unknown table name: {table_name}")

    adjusted_start, adjusted_end = _adjust_dates_for_timezone(
        start_date, end_date, timezone_offset_minutes
    )

    sql_conditions: List[str] = []
    sql_params: List[Any] = []
    p = 1  # parameter index (1-based for PostgreSQL $N placeholders)

    # Date range (always present)
    sql_conditions.append(f"date >= ${p}")
    sql_params.append(adjusted_start)
    p += 1

    sql_conditions.append(f"date <= ${p}")
    sql_params.append(adjusted_end)
    p += 1

    # Optional entity filter
    if entity_id is not None:
        if isinstance(entity_id, list):
            placeholders = ", ".join(f"${p + i}" for i in range(len(entity_id)))
            sql_conditions.append(f'"{entity_id_field}" IN ({placeholders})')
            sql_params.extend(entity_id)
            p += len(entity_id)
        else:
            sql_conditions.append(f'"{entity_id_field}" = ${p}')
            sql_params.append(entity_id)
            p += 1

    # Exclude specific entities
    if exclude_entity_ids:
        placeholders = ", ".join(
            f"${p + i}" for i in range(len(exclude_entity_ids))
        )
        sql_conditions.append(f'"{entity_id_field}" NOT IN ({placeholders})')
        sql_params.extend(exclude_entity_ids)
        p += len(exclude_entity_ids)

    # Optional model filter
    if model:
        sql_conditions.append(f"model = ${p}")
        sql_params.append(model)
        p += 1

    # Optional api_key filter
    if api_key:
        if isinstance(api_key, list):
            placeholders = ", ".join(f"${p + i}" for i in range(len(api_key)))
            sql_conditions.append(f"api_key IN ({placeholders})")
            sql_params.extend(api_key)
            p += len(api_key)
        else:
            sql_conditions.append(f"api_key = ${p}")
            sql_params.append(api_key)
            p += 1

    where_clause = " AND ".join(sql_conditions)

    entity_select = f'"{entity_id_field}",' if include_entity_id else ""
    entity_group_by = f'"{entity_id_field}",' if include_entity_id else ""

    sql_query = f"""
        SELECT
            {entity_select}
            date,
            api_key,
            model,
            model_group,
            custom_llm_provider,
            mcp_namespaced_tool_name,
            endpoint,
            SUM(spend)::float AS spend,
            SUM(prompt_tokens)::bigint AS prompt_tokens,
            SUM(completion_tokens)::bigint AS completion_tokens,
            SUM(cache_read_input_tokens)::bigint AS cache_read_input_tokens,
            SUM(cache_creation_input_tokens)::bigint AS cache_creation_input_tokens,
            SUM(api_requests)::bigint AS api_requests,
            SUM(successful_requests)::bigint AS successful_requests,
            SUM(failed_requests)::bigint AS failed_requests
        FROM "{pg_table}"
        WHERE {where_clause}
        GROUP BY {entity_group_by} date, api_key, model, model_group, custom_llm_provider,
                 mcp_namespaced_tool_name, endpoint
        ORDER BY date DESC
    """

    return sql_query, sql_params


async def _aggregate_spend_records(
    *,
    prisma_client: PrismaClient,
    records: List[Any],
    entity_id_field: Optional[str],
    entity_metadata_field: Optional[Dict[str, dict]],
    api_key_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Aggregate rows into DailySpendData list and total metrics.

    If *api_key_metadata* is provided it will be used directly, avoiding a
    redundant DB round-trip when the caller already fetched it.
    """
    if api_key_metadata is None:
        api_keys: Set[str] = set()
        for record in records:
            if record.api_key:
                api_keys.add(record.api_key)

        api_key_metadata = {}
        if api_keys:
            api_key_metadata = await get_api_key_metadata(prisma_client, api_keys)

    model_metadata: Dict[str, Dict[str, Any]] = {}
    provider_metadata: Dict[str, Dict[str, Any]] = {}

    results: List[DailySpendData] = []
    total_metrics = SpendMetrics()
    grouped_data: Dict[str, Dict[str, Any]] = {}

    for record in records:
        date_str = record.date
        if date_str not in grouped_data:
            grouped_data[date_str] = {
                "metrics": SpendMetrics(),
                "breakdown": BreakdownMetrics(),
            }

        grouped_data[date_str]["metrics"] = update_metrics(
            grouped_data[date_str]["metrics"], record
        )

        grouped_data[date_str]["breakdown"] = update_breakdown_metrics(
            grouped_data[date_str]["breakdown"],
            record,
            model_metadata,
            provider_metadata,
            api_key_metadata,
            entity_id_field=entity_id_field,
            entity_metadata_field=entity_metadata_field,
        )

        total_metrics = update_metrics(total_metrics, record)

    for date_str, data in grouped_data.items():
        results.append(
            DailySpendData(
                date=datetime.strptime(date_str, "%Y-%m-%d").date(),
                metrics=data["metrics"],
                breakdown=data["breakdown"],
            )
        )

    results.sort(key=lambda x: x.date, reverse=True)

    return {"results": results, "totals": total_metrics}


async def get_daily_activity(
    prisma_client: Optional[PrismaClient],
    table_name: str,
    entity_id_field: str,
    entity_id: Optional[Union[str, List[str]]],
    entity_metadata_field: Optional[Dict[str, dict]],
    start_date: Optional[str],
    end_date: Optional[str],
    model: Optional[str],
    api_key: Optional[Union[str, List[str]]],
    page: int,
    page_size: int,
    exclude_entity_ids: Optional[List[str]] = None,
    metadata_metrics_func: Optional[Callable[[List[Any]], SpendMetrics]] = None,
    timezone_offset_minutes: Optional[int] = None,
) -> SpendAnalyticsPaginatedResponse:
    """Common function to get daily activity for any entity type."""

    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )

    if start_date is None or end_date is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "Please provide start_date and end_date"},
        )

    try:
        where_conditions = _build_where_conditions(
            entity_id_field=entity_id_field,
            entity_id=entity_id,
            start_date=start_date,
            end_date=end_date,
            model=model,
            api_key=api_key,
            exclude_entity_ids=exclude_entity_ids,
            timezone_offset_minutes=timezone_offset_minutes,
        )

        # Get total count for pagination
        total_count = await getattr(prisma_client.db, table_name).count(
            where=where_conditions
        )

        # Fetch paginated results
        daily_spend_data = await getattr(prisma_client.db, table_name).find_many(
            where=where_conditions,
            order=[
                {"date": "desc"},
            ],
            skip=(page - 1) * page_size,
            take=page_size,
        )

        aggregated = await _aggregate_spend_records(
            prisma_client=prisma_client,
            records=daily_spend_data,
            entity_id_field=entity_id_field,
            entity_metadata_field=entity_metadata_field,
        )

        metadata_metrics = aggregated["totals"]
        if metadata_metrics_func:
            metadata_metrics = metadata_metrics_func(daily_spend_data)

        return SpendAnalyticsPaginatedResponse(
            results=aggregated["results"],
            metadata=DailySpendMetadata(
                total_spend=metadata_metrics.spend,
                total_prompt_tokens=metadata_metrics.prompt_tokens,
                total_completion_tokens=metadata_metrics.completion_tokens,
                total_tokens=metadata_metrics.total_tokens,
                total_api_requests=metadata_metrics.api_requests,
                total_successful_requests=metadata_metrics.successful_requests,
                total_failed_requests=metadata_metrics.failed_requests,
                total_cache_read_input_tokens=metadata_metrics.cache_read_input_tokens,
                total_cache_creation_input_tokens=metadata_metrics.cache_creation_input_tokens,
                page=page,
                total_pages=-(-total_count // page_size),  # Ceiling division
                has_more=(page * page_size) < total_count,
            ),
        )

    except Exception as e:
        verbose_proxy_logger.exception(f"Error fetching daily activity: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": f"Failed to fetch analytics: {str(e)}"},
        )


async def _get_aggregated_with_entity_breakdown(
    *,
    prisma_client: PrismaClient,
    table_name: str,
    entity_id_field: str,
    entity_id: Optional[Union[str, List[str]]],
    entity_metadata_field: Optional[Dict[str, dict]],
    start_date: str,
    end_date: str,
    model: Optional[str],
    api_key: Optional[Union[str, List[str]]],
    exclude_entity_ids: Optional[List[str]],
) -> SpendAnalyticsPaginatedResponse:
    """Multi-query aggregation for entity-breakdown views (e.g. team usage).

    Runs three concurrent queries instead of one whose GROUP BY matches
    the table's unique key (~150k rows unchanged):

      Q0 — detail query WITHOUT entity_id  (~5-10k rows)
      Q1 — per-entity per-date             (~entities × days rows)
      Q2 — entity × API key                (~entities × keys rows)

    Q0 feeds model / key / provider / endpoint breakdowns via the
    existing ``_aggregate_spend_records`` helper.  Q1 and Q2 provide
    per-entity data that is merged into the result.
    """
    pg_table = _PRISMA_TO_PG_TABLE.get(table_name)
    if pg_table is None:
        raise ValueError(f"Unknown table name: {table_name}")

    where_clause, params = _build_base_where_clause_sql(
        entity_id_field=entity_id_field,
        entity_id=entity_id,
        start_date=start_date,
        end_date=end_date,
        model=model,
        api_key=api_key,
        exclude_entity_ids=exclude_entity_ids,
    )

    m = _METRIC_SUM_EXPR
    base = f'FROM "{pg_table}" WHERE {where_clause}'

    # Q0 — detail without entity_id (collapses entity dimension)
    q_detail = (
        f"SELECT date, api_key, model, model_group, custom_llm_provider, "
        f"mcp_namespaced_tool_name, endpoint, {m} {base} "
        f"GROUP BY date, api_key, model, model_group, custom_llm_provider, "
        f"mcp_namespaced_tool_name, endpoint ORDER BY date DESC"
    )
    # Q1 — per-entity per-date (for chart tooltip + daily entity table)
    q_entity_daily = (
        f'SELECT "{entity_id_field}", date, {m} {base} '
        f'GROUP BY "{entity_id_field}", date ORDER BY date DESC'
    )
    # Q2 — entity × API key (for entity → key correlation)
    q_entity_key = (
        f'SELECT "{entity_id_field}", api_key, {m} {base} '
        f'GROUP BY "{entity_id_field}", api_key'
    )

    raw_detail, raw_entity_daily, raw_entity_key = await asyncio.gather(
        prisma_client.db.query_raw(q_detail, *params),
        prisma_client.db.query_raw(q_entity_daily, *params),
        prisma_client.db.query_raw(q_entity_key, *params),
    )

    # Collect all api_keys from Q0 + Q2 and fetch metadata once
    all_api_keys: Set[str] = set()
    for row in (raw_detail or []):
        ak = row.get("api_key")
        if ak:
            all_api_keys.add(ak)
    for row in (raw_entity_key or []):
        ak = row.get("api_key")
        if ak:
            all_api_keys.add(ak)

    api_key_metadata: Dict[str, Dict[str, Any]] = {}
    if all_api_keys:
        api_key_metadata = await get_api_key_metadata(
            prisma_client, all_api_keys
        )

    # Process detail rows for model / key / provider / endpoint breakdowns
    detail_records = [SimpleNamespace(**row) for row in (raw_detail or [])]
    aggregated = await _aggregate_spend_records(
        prisma_client=prisma_client,
        records=detail_records,
        entity_id_field=None,
        entity_metadata_field=None,
        api_key_metadata=api_key_metadata,
    )

    # Build entity → key mapping (cross-date totals)
    entity_key_map: Dict[str, Dict[str, SpendMetrics]] = {}
    for row in (raw_entity_key or []):
        eid = row.get(entity_id_field) or "Unassigned"
        ak = row.get("api_key")
        if ak:
            entity_key_map.setdefault(eid, {})[ak] = (
                _row_to_spend_metrics(row)
            )

    # Group entity-daily rows by date
    date_entity_rows: Dict[str, List[dict]] = {}
    for row in (raw_entity_daily or []):
        date_entity_rows.setdefault(row["date"], []).append(row)

    # Index detail-aggregated results by date for merging
    date_to_detail: Dict[str, DailySpendData] = {
        str(r.date): r for r in aggregated["results"]
    }

    # Merge entity data into per-date results
    total_metrics = SpendMetrics()
    results: List[DailySpendData] = []
    sorted_dates = sorted(date_entity_rows.keys(), reverse=True)
    # Track which entities already got their api_key_breakdown attached
    # so we only attach it once (on the most recent date each entity appears).
    entities_with_key_breakdown: set = set()

    for date_str in sorted_dates:
        day_metrics = SpendMetrics()
        day_entities: Dict[str, MetricWithMetadata] = {}

        for row in date_entity_rows[date_str]:
            eid = row.get(entity_id_field) or "Unassigned"
            entity_metrics = _row_to_spend_metrics(row)
            _add_metrics(day_metrics, entity_metrics)

            entity_meta = (
                entity_metadata_field.get(eid, {})
                if entity_metadata_field
                else {}
            )
            entity_entry = MetricWithMetadata(
                metrics=entity_metrics, metadata=entity_meta
            )

            # Attach per-key breakdown on the first (most recent) date
            # each entity appears. The UI sums across dates for the
            # entity → key table, so we only need it once per entity.
            if eid not in entities_with_key_breakdown:
                for ak, ak_metrics in entity_key_map.get(eid, {}).items():
                    meta = api_key_metadata.get(ak, {})
                    entity_entry.api_key_breakdown[ak] = (
                        KeyMetricWithMetadata(
                            metrics=ak_metrics,
                            metadata=KeyMetadata(
                                key_alias=meta.get("key_alias"),
                                team_id=meta.get("team_id"),
                            ),
                        )
                    )
                entities_with_key_breakdown.add(eid)

            day_entities[eid] = entity_entry

        _add_metrics(total_metrics, day_metrics)

        # Merge model / key / provider breakdowns from the detail query
        detail_day = date_to_detail.get(date_str)
        if detail_day:
            day_breakdown = detail_day.breakdown
            day_breakdown.entities = day_entities
        else:
            day_breakdown = BreakdownMetrics(entities=day_entities)

        results.append(
            DailySpendData(
                date=datetime.strptime(date_str, "%Y-%m-%d").date(),
                metrics=day_metrics,
                breakdown=day_breakdown,
            )
        )

    return SpendAnalyticsPaginatedResponse(
        results=results,
        metadata=DailySpendMetadata(
            total_spend=total_metrics.spend,
            total_prompt_tokens=total_metrics.prompt_tokens,
            total_completion_tokens=total_metrics.completion_tokens,
            total_tokens=total_metrics.total_tokens,
            total_api_requests=total_metrics.api_requests,
            total_successful_requests=total_metrics.successful_requests,
            total_failed_requests=total_metrics.failed_requests,
            total_cache_read_input_tokens=total_metrics.cache_read_input_tokens,
            total_cache_creation_input_tokens=total_metrics.cache_creation_input_tokens,
            page=1,
            total_pages=1,
            has_more=False,
        ),
    )


async def get_daily_activity_aggregated(
    prisma_client: Optional[PrismaClient],
    table_name: str,
    entity_id_field: str,
    entity_id: Optional[Union[str, List[str]]],
    entity_metadata_field: Optional[Dict[str, dict]],
    start_date: Optional[str],
    end_date: Optional[str],
    model: Optional[str],
    api_key: Optional[Union[str, List[str]]],
    exclude_entity_ids: Optional[List[str]] = None,
    timezone_offset_minutes: Optional[int] = None,
    include_entity_breakdown: bool = False,
) -> SpendAnalyticsPaginatedResponse:
    """Aggregated variant that returns the full result set (no pagination).

    When *include_entity_breakdown* is **False** the entity column is dropped
    from the GROUP BY, collapsing rows across entities.  This already reduces
    ~150k raw rows to ~5-10k, so we process them with the standard
    ``_aggregate_spend_records`` helper.

    When *include_entity_breakdown* is **True** (e.g. team usage) the entity
    column would be added back to the GROUP BY, matching the table's unique
    key and returning all ~150k rows — far too slow.  Instead we run three
    concurrent queries with coarser GROUP BYs:

      1. A *detail* query **without** the entity column (~5-10k rows) to
         produce model / key / provider / endpoint breakdowns.
      2. A per-entity-per-date query (~teams × days rows) for the daily
         chart and entity table.
      3. A per-entity-per-key query for the entity → key correlation.

    Both paths produce the same ``SpendAnalyticsPaginatedResponse`` shape.
    """
    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )

    if start_date is None or end_date is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "Please provide start_date and end_date"},
        )

    try:
        adjusted_start, adjusted_end = _adjust_dates_for_timezone(
            start_date, end_date, timezone_offset_minutes
        )

        if not include_entity_breakdown:
            # ---- Original single-query path (already fast) ----
            sql_query, sql_params = _build_aggregated_sql_query(
                table_name=table_name,
                entity_id_field=entity_id_field,
                entity_id=entity_id,
                start_date=adjusted_start,
                end_date=adjusted_end,
                model=model,
                api_key=api_key,
                exclude_entity_ids=exclude_entity_ids,
                include_entity_id=False,
            )

            rows = await prisma_client.db.query_raw(sql_query, *sql_params)
            records = [SimpleNamespace(**row) for row in (rows or [])]

            aggregated = await _aggregate_spend_records(
                prisma_client=prisma_client,
                records=records,
                entity_id_field=None,
                entity_metadata_field=None,
            )

            return SpendAnalyticsPaginatedResponse(
                results=aggregated["results"],
                metadata=DailySpendMetadata(
                    total_spend=aggregated["totals"].spend,
                    total_prompt_tokens=aggregated["totals"].prompt_tokens,
                    total_completion_tokens=aggregated["totals"].completion_tokens,
                    total_tokens=aggregated["totals"].total_tokens,
                    total_api_requests=aggregated["totals"].api_requests,
                    total_successful_requests=aggregated["totals"].successful_requests,
                    total_failed_requests=aggregated["totals"].failed_requests,
                    total_cache_read_input_tokens=aggregated["totals"].cache_read_input_tokens,
                    total_cache_creation_input_tokens=aggregated["totals"].cache_creation_input_tokens,
                    page=1,
                    total_pages=1,
                    has_more=False,
                ),
            )

        # ---- Optimised multi-query path (entity breakdown) ----
        return await _get_aggregated_with_entity_breakdown(
            prisma_client=prisma_client,
            table_name=table_name,
            entity_id_field=entity_id_field,
            entity_id=entity_id,
            entity_metadata_field=entity_metadata_field,
            start_date=adjusted_start,
            end_date=adjusted_end,
            model=model,
            api_key=api_key,
            exclude_entity_ids=exclude_entity_ids,
        )

    except Exception as e:
        verbose_proxy_logger.exception(
            f"Error fetching aggregated daily activity: {str(e)}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": f"Failed to fetch analytics: {str(e)}"},
        )
