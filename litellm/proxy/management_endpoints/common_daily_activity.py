from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Union

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
    """Update api key metadata for a single record."""
    key_records = await prisma_client.db.litellm_verificationtoken.find_many(
        where={"token": {"in": list(api_keys)}}
    )
    return {
        k.token: {"key_alias": k.key_alias, "team_id": k.team_id} for k in key_records
    }


def _build_where_conditions(
    *,
    entity_id_field: str,
    entity_id: Optional[Union[str, List[str]]],
    start_date: str,
    end_date: str,
    model: Optional[str],
    api_key: Optional[str],
    exclude_entity_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Build prisma where clause for daily activity queries."""
    where_conditions: Dict[str, Any] = {
        "date": {
            "gte": start_date,
            "lte": end_date,
        }
    }

    if model:
        where_conditions["model"] = model
    if api_key:
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


async def _aggregate_spend_records(
    *,
    prisma_client: PrismaClient,
    records: List[Any],
    entity_id_field: Optional[str],
    entity_metadata_field: Optional[Dict[str, dict]],
) -> Dict[str, Any]:
    """Aggregate rows into DailySpendData list and total metrics."""
    api_keys: Set[str] = set()
    for record in records:
        if record.api_key:
            api_keys.add(record.api_key)

    api_key_metadata: Dict[str, Dict[str, Any]] = {}
    model_metadata: Dict[str, Dict[str, Any]] = {}
    provider_metadata: Dict[str, Dict[str, Any]] = {}
    if api_keys:
        api_key_metadata = await get_api_key_metadata(prisma_client, api_keys)

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
    api_key: Optional[str],
    page: int,
    page_size: int,
    exclude_entity_ids: Optional[List[str]] = None,
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
) -> SpendAnalyticsPaginatedResponse:
    """Aggregated variant that returns the full result set (no pagination).

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
        where_conditions = _build_where_conditions(
            entity_id_field=entity_id_field,
            entity_id=entity_id,
            start_date=start_date,
            end_date=end_date,
            model=model,
            api_key=api_key,
            exclude_entity_ids=exclude_entity_ids,
        )

        # Fetch all matching results (no pagination)
        daily_spend_data = await getattr(prisma_client.db, table_name).find_many(
            where=where_conditions,
            order=[
                {"date": "desc"},
            ],
        )

        aggregated = await _aggregate_spend_records(
            prisma_client=prisma_client,
            records=daily_spend_data,
            entity_id_field=entity_id_field,
            entity_metadata_field=entity_metadata_field,
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

    except Exception as e:
        verbose_proxy_logger.exception(
            f"Error fetching aggregated daily activity: {str(e)}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": f"Failed to fetch analytics: {str(e)}"},
        )
