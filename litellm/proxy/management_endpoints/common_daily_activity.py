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
    return normalized_tag.startswith("user-agent:") or normalized_tag.startswith(
        "user agent:"
    )


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
        breakdown.endpoints[record.endpoint].api_key_breakdown[
            record.api_key
        ].metrics = update_metrics(
            breakdown.endpoints[record.endpoint]
            .api_key_breakdown[record.api_key]
            .metrics,
            record,
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
        k.token: {"key_alias": k.key_alias, "team_id": k.team_id} for k in key_records
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


def _build_aggregated_sql_query(
    *,
    table_name: str,
    entity_id_field: str,
    entity_id: Optional[Union[str, List[str]]],
    start_date: str,
    end_date: str,
    model: Optional[str],
    api_key: Optional[str],
    exclude_entity_ids: Optional[List[str]] = None,
    timezone_offset_minutes: Optional[int] = None,
) -> Tuple[str, List[Any]]:
    """Build a parameterized SQL GROUP BY query for aggregated daily activity.

    Groups by (date, api_key, model, model_group, custom_llm_provider,
    mcp_namespaced_tool_name, endpoint) with SUMs on all metric columns.
    The entity_id column is intentionally omitted from GROUP BY to collapse
    rows across entities — this is where the biggest row reduction comes from.

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
        placeholders = ", ".join(f"${p + i}" for i in range(len(exclude_entity_ids)))
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
        sql_conditions.append(f"api_key = ${p}")
        sql_params.append(api_key)
        p += 1

    where_clause = " AND ".join(sql_conditions)

    # Postgres computes every rollup level the response needs — per-date
    # totals, per-(date, model), per-(date, model, api_key), per-provider,
    # etc. — in a single pass via GROUPING SETS. The GROUPING() bitmask
    # encodes which level a row belongs to so Python can dispatch rows
    # straight into their buckets without re-summing. The leaf grouping
    # is omitted on purpose: nothing in the response shape needs it once
    # all the rollups are present.
    sql_query = f"""
        SELECT
            date,
            api_key,
            model,
            model_group,
            custom_llm_provider,
            mcp_namespaced_tool_name,
            endpoint,
            GROUPING(date, api_key, model, model_group,
                     custom_llm_provider, mcp_namespaced_tool_name,
                     endpoint) AS group_level,
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
        GROUP BY GROUPING SETS (
            (date),
            (date, api_key),
            (date, model),
            (date, model, api_key),
            (date, model_group),
            (date, model_group, api_key),
            (date, custom_llm_provider),
            (date, custom_llm_provider, api_key),
            (date, mcp_namespaced_tool_name),
            (date, mcp_namespaced_tool_name, api_key),
            (date, endpoint),
            (date, endpoint, api_key),
            ()
        )
    """

    return sql_query, sql_params


def _aggregate_spend_records_sync(
    *,
    records: List[Any],
    api_key_metadata: Dict[str, Dict[str, Any]],
    entity_id_field: Optional[str],
    entity_metadata_field: Optional[Dict[str, dict]],
) -> Dict[str, Any]:
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


async def _aggregate_spend_records(
    *,
    prisma_client: PrismaClient,
    records: List[Any],
    entity_id_field: Optional[str],
    entity_metadata_field: Optional[Dict[str, dict]],
) -> Dict[str, Any]:
    """Aggregate rows into DailySpendData list and total metrics.

    The per-row loop is offloaded to a worker thread via asyncio.to_thread so
    a large result set doesn't peg the event loop.
    """
    api_keys: Set[str] = {record.api_key for record in records if record.api_key}

    api_key_metadata: Dict[str, Dict[str, Any]] = {}
    if api_keys:
        api_key_metadata = await get_api_key_metadata(prisma_client, api_keys)

    return await asyncio.to_thread(
        _aggregate_spend_records_sync,
        records=records,
        api_key_metadata=api_key_metadata,
        entity_id_field=entity_id_field,
        entity_metadata_field=entity_metadata_field,
    )


# GROUPING() bitmask values for each grouping set emitted by
# _build_aggregated_sql_query. Per Postgres semantics, the rightmost argument
# is the least-significant bit. Argument order:
#   date, api_key, model, model_group, custom_llm_provider,
#   mcp_namespaced_tool_name, endpoint
# A bit is 1 when the corresponding column is rolled up (i.e. NOT in the
# current grouping set's key), 0 when the column is part of the key.
_GROUP_GRAND_TOTAL = 127  # 0b1111111 — all rolled up
_GROUP_DATE = 63  # 0b0111111 — only date kept
_GROUP_DATE_API_KEY = 31  # 0b0011111
_GROUP_DATE_MODEL = 47  # 0b0101111
_GROUP_DATE_MODEL_API_KEY = 15  # 0b0001111
_GROUP_DATE_MODEL_GROUP = 55  # 0b0110111
_GROUP_DATE_MODEL_GROUP_API_KEY = 23  # 0b0010111
_GROUP_DATE_PROVIDER = 59  # 0b0111011
_GROUP_DATE_PROVIDER_API_KEY = 27  # 0b0011011
_GROUP_DATE_MCP = 61  # 0b0111101
_GROUP_DATE_MCP_API_KEY = 29  # 0b0011101
_GROUP_DATE_ENDPOINT = 62  # 0b0111110
_GROUP_DATE_ENDPOINT_API_KEY = 30  # 0b0011110


def _record_to_spend_metrics(record: Any) -> SpendMetrics:
    """Build a SpendMetrics directly from one already-aggregated rollup row."""
    return SpendMetrics(
        spend=record.spend,
        prompt_tokens=record.prompt_tokens,
        completion_tokens=record.completion_tokens,
        total_tokens=record.prompt_tokens + record.completion_tokens,
        cache_read_input_tokens=record.cache_read_input_tokens,
        cache_creation_input_tokens=record.cache_creation_input_tokens,
        api_requests=record.api_requests,
        successful_requests=record.successful_requests,
        failed_requests=record.failed_requests,
    )


def _key_metadata(
    api_key_metadata: Dict[str, Dict[str, Any]], api_key: str
) -> KeyMetadata:
    meta = api_key_metadata.get(api_key, {})
    return KeyMetadata(key_alias=meta.get("key_alias"), team_id=meta.get("team_id"))


def _aggregate_grouping_sets_records_sync(  # noqa: PLR0915
    *,
    records: List[Any],
    api_key_metadata: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Build the response from rollup rows produced by the GROUPING SETS query.

    Each row carries a `group_level` bitmask (from Postgres GROUPING()) that
    identifies which rollup level it belongs to. We dispatch the row's
    pre-aggregated metrics straight into the matching bucket — no per-row
    summing in Python and no nested update_metrics calls.
    """
    total_metrics = SpendMetrics()
    grouped_data: Dict[str, Dict[str, Any]] = {}

    def ensure_date(date_str: str) -> Dict[str, Any]:
        bucket = grouped_data.get(date_str)
        if bucket is None:
            bucket = {"metrics": SpendMetrics(), "breakdown": BreakdownMetrics()}
            grouped_data[date_str] = bucket
        return bucket

    def assign_metric_with_metadata(
        target: Dict[str, MetricWithMetadata], key: str, metrics: SpendMetrics
    ) -> None:
        existing = target.get(key)
        if existing is None:
            target[key] = MetricWithMetadata(metrics=metrics, metadata={})
        else:
            existing.metrics = metrics

    def assign_api_key_breakdown(
        target: Dict[str, MetricWithMetadata],
        parent_key: str,
        api_key: str,
        metrics: SpendMetrics,
    ) -> None:
        parent = target.get(parent_key)
        if parent is None:
            parent = MetricWithMetadata(metrics=SpendMetrics(), metadata={})
            target[parent_key] = parent
        parent.api_key_breakdown[api_key] = KeyMetricWithMetadata(
            metrics=metrics, metadata=_key_metadata(api_key_metadata, api_key)
        )

    for record in records:
        level = record.group_level
        metrics = _record_to_spend_metrics(record)

        if level == _GROUP_GRAND_TOTAL:
            total_metrics = metrics
            continue

        if level == _GROUP_DATE:
            ensure_date(record.date)["metrics"] = metrics
            continue

        breakdown = ensure_date(record.date)["breakdown"]

        if level == _GROUP_DATE_API_KEY:
            if record.api_key:
                breakdown.api_keys[record.api_key] = KeyMetricWithMetadata(
                    metrics=metrics,
                    metadata=_key_metadata(api_key_metadata, record.api_key),
                )
        elif level == _GROUP_DATE_MODEL:
            if record.model:
                assign_metric_with_metadata(breakdown.models, record.model, metrics)
        elif level == _GROUP_DATE_MODEL_API_KEY:
            if record.model and record.api_key:
                assign_api_key_breakdown(
                    breakdown.models, record.model, record.api_key, metrics
                )
        elif level == _GROUP_DATE_MODEL_GROUP:
            if record.model_group:
                assign_metric_with_metadata(
                    breakdown.model_groups, record.model_group, metrics
                )
        elif level == _GROUP_DATE_MODEL_GROUP_API_KEY:
            if record.model_group and record.api_key:
                assign_api_key_breakdown(
                    breakdown.model_groups,
                    record.model_group,
                    record.api_key,
                    metrics,
                )
        elif level == _GROUP_DATE_PROVIDER:
            provider = record.custom_llm_provider or "unknown"
            assign_metric_with_metadata(breakdown.providers, provider, metrics)
        elif level == _GROUP_DATE_PROVIDER_API_KEY:
            if record.api_key:
                provider = record.custom_llm_provider or "unknown"
                assign_api_key_breakdown(
                    breakdown.providers, provider, record.api_key, metrics
                )
        elif level == _GROUP_DATE_MCP:
            if record.mcp_namespaced_tool_name:
                assign_metric_with_metadata(
                    breakdown.mcp_servers, record.mcp_namespaced_tool_name, metrics
                )
        elif level == _GROUP_DATE_MCP_API_KEY:
            if record.mcp_namespaced_tool_name and record.api_key:
                assign_api_key_breakdown(
                    breakdown.mcp_servers,
                    record.mcp_namespaced_tool_name,
                    record.api_key,
                    metrics,
                )
        elif level == _GROUP_DATE_ENDPOINT:
            if record.endpoint:
                assign_metric_with_metadata(
                    breakdown.endpoints, record.endpoint, metrics
                )
        elif level == _GROUP_DATE_ENDPOINT_API_KEY:
            if record.endpoint and record.api_key:
                assign_api_key_breakdown(
                    breakdown.endpoints, record.endpoint, record.api_key, metrics
                )

    results = [
        DailySpendData(
            date=datetime.strptime(date_str, "%Y-%m-%d").date(),
            metrics=data["metrics"],
            breakdown=data["breakdown"],
        )
        for date_str, data in grouped_data.items()
    ]
    results.sort(key=lambda x: x.date, reverse=True)

    return {"results": results, "totals": total_metrics}


async def _aggregate_grouping_sets_records(
    *,
    prisma_client: PrismaClient,
    records: List[Any],
) -> Dict[str, Any]:
    """Async wrapper: fetch api_key_metadata, then dispatch on a worker thread."""
    api_keys: Set[str] = {r.api_key for r in records if r.api_key}

    api_key_metadata: Dict[str, Dict[str, Any]] = {}
    if api_keys:
        api_key_metadata = await get_api_key_metadata(prisma_client, api_keys)

    return await asyncio.to_thread(
        _aggregate_grouping_sets_records_sync,
        records=records,
        api_key_metadata=api_key_metadata,
    )


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


async def get_daily_activity_aggregated(
    prisma_client: Optional[PrismaClient],
    table_name: str,
    entity_id_field: str,
    entity_id: Optional[Union[str, List[str]]],
    entity_metadata_field: Optional[Dict[str, dict]],
    start_date: Optional[str],
    end_date: Optional[str],
    model: Optional[str],
    api_key: Optional[str],
    exclude_entity_ids: Optional[List[str]] = None,
    timezone_offset_minutes: Optional[int] = None,
) -> SpendAnalyticsPaginatedResponse:
    """Aggregated variant that returns the full result set (no pagination).

    Uses SQL GROUP BY to aggregate rows in the database rather than fetching
    all individual rows into Python. This collapses rows across entities
    (users/teams/orgs), reducing ~150k rows to ~2-3k grouped rows.

    Matches the response model of the paginated endpoint so the UI does not need to transform.
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
        sql_query, sql_params = _build_aggregated_sql_query(
            table_name=table_name,
            entity_id_field=entity_id_field,
            entity_id=entity_id,
            start_date=start_date,
            end_date=end_date,
            model=model,
            api_key=api_key,
            exclude_entity_ids=exclude_entity_ids,
            timezone_offset_minutes=timezone_offset_minutes,
        )

        # Execute GROUPING SETS query — returns one row per rollup level.
        rows = await prisma_client.db.query_raw(sql_query, *sql_params)
        if rows is None:
            rows = []

        records = [SimpleNamespace(**row) for row in rows]

        # The grouping-sets dispatcher places each row directly in its bucket
        # using the row's GROUPING() bitmask. No Python-side summing needed.
        aggregated = await _aggregate_grouping_sets_records(
            prisma_client=prisma_client,
            records=records,
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
                total_cache_read_input_tokens=aggregated[
                    "totals"
                ].cache_read_input_tokens,
                total_cache_creation_input_tokens=aggregated[
                    "totals"
                ].cache_creation_input_tokens,
                page=1,
                total_pages=1,
                has_more=False,
            ),
        )

    except Exception as e:
        verbose_proxy_logger.exception(
            f"Error fetching aggregated daily activity: {str(e)}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": f"Failed to fetch analytics: {str(e)}"},
        )
