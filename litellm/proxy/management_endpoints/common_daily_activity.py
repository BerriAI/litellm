from datetime import date, datetime
from typing import Any, Dict, List, Optional, Union

from fastapi import HTTPException, status
from pydantic import BaseModel, Field

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import CommonProxyErrors
from litellm.proxy.utils import PrismaClient


class SpendMetrics(BaseModel):
    spend: float = Field(default=0.0)
    prompt_tokens: int = Field(default=0)
    completion_tokens: int = Field(default=0)
    cache_read_input_tokens: int = Field(default=0)
    cache_creation_input_tokens: int = Field(default=0)
    total_tokens: int = Field(default=0)
    successful_requests: int = Field(default=0)
    failed_requests: int = Field(default=0)
    api_requests: int = Field(default=0)


class BreakdownMetrics(BaseModel):
    models: Dict[str, SpendMetrics] = Field(default_factory=dict)
    api_keys: Dict[str, SpendMetrics] = Field(default_factory=dict)
    providers: Dict[str, SpendMetrics] = Field(default_factory=dict)
    entities: Dict[str, SpendMetrics] = Field(default_factory=dict)


class DailySpendData(BaseModel):
    date: date
    metrics: SpendMetrics
    breakdown: BreakdownMetrics = Field(default_factory=BreakdownMetrics)


class DailySpendMetadata(BaseModel):
    total_spend: float = Field(default=0.0)
    total_prompt_tokens: int = Field(default=0)
    total_completion_tokens: int = Field(default=0)
    total_tokens: int = Field(default=0)
    total_api_requests: int = Field(default=0)
    total_successful_requests: int = Field(default=0)
    total_failed_requests: int = Field(default=0)
    total_cache_read_input_tokens: int = Field(default=0)
    total_cache_creation_input_tokens: int = Field(default=0)
    page: int = Field(default=1)
    total_pages: int = Field(default=1)
    has_more: bool = Field(default=False)


class SpendAnalyticsPaginatedResponse(BaseModel):
    results: List[DailySpendData]
    metadata: DailySpendMetadata = Field(default_factory=DailySpendMetadata)


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
) -> BreakdownMetrics:
    """Update breakdown metrics with new record data."""
    # Update model metrics
    model_key = record.model
    if model_key not in breakdown.models:
        breakdown.models[model_key] = SpendMetrics()
    breakdown.models[model_key] = update_metrics(breakdown.models[model_key], record)

    # Update API key metrics
    api_key = record.api_key
    if api_key not in breakdown.api_keys:
        breakdown.api_keys[api_key] = SpendMetrics()
    breakdown.api_keys[api_key] = update_metrics(breakdown.api_keys[api_key], record)

    # Update provider metrics
    provider = record.custom_llm_provider or "default"
    if provider not in breakdown.providers:
        breakdown.providers[provider] = SpendMetrics()
    breakdown.providers[provider] = update_metrics(
        breakdown.providers[provider], record
    )

    # Update entity-specific metrics if entity_id_field is provided
    if entity_id_field:
        entity_value = getattr(record, entity_id_field, None)
        if entity_value:
            if entity_value not in breakdown.entities:
                breakdown.entities[entity_value] = SpendMetrics()
            breakdown.entities[entity_value] = update_metrics(
                breakdown.entities[entity_value], record
            )

    return breakdown


async def get_daily_activity(
    prisma_client: Optional[PrismaClient],
    table_name: str,
    entity_id_field: str,
    entity_id: Optional[Union[str, List[str]]],
    start_date: Optional[str],
    end_date: Optional[str],
    model: Optional[str],
    api_key: Optional[str],
    page: int,
    page_size: int,
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
        # Build filter conditions
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
        if entity_id:
            if isinstance(entity_id, list):
                where_conditions[entity_id_field] = {"in": entity_id}
            else:
                where_conditions[entity_id_field] = entity_id

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

        # Get all unique API keys from the spend data
        api_keys = set()
        for record in daily_spend_data:
            if record.api_key:
                api_keys.add(record.api_key)

        # Fetch key aliases in bulk
        api_key_metadata: Dict[str, Dict[str, Any]] = {}
        model_metadata: Dict[str, Dict[str, Any]] = {}
        provider_metadata: Dict[str, Dict[str, Any]] = {}
        if api_keys:
            key_records = await prisma_client.db.litellm_verificationtoken.find_many(
                where={"token": {"in": list(api_keys)}}
            )
            api_key_metadata.update(
                {k.token: {"key_alias": k.key_alias} for k in key_records}
            )

        # Process results
        results = []
        total_metrics = SpendMetrics()
        grouped_data: Dict[str, Dict[str, Any]] = {}

        for record in daily_spend_data:
            date_str = record.date
            if date_str not in grouped_data:
                grouped_data[date_str] = {
                    "metrics": SpendMetrics(),
                    "breakdown": BreakdownMetrics(),
                }

            # Update metrics
            grouped_data[date_str]["metrics"] = update_metrics(
                grouped_data[date_str]["metrics"], record
            )
            # Update breakdowns
            grouped_data[date_str]["breakdown"] = update_breakdown_metrics(
                grouped_data[date_str]["breakdown"],
                record,
                model_metadata,
                provider_metadata,
                api_key_metadata,
                entity_id_field=entity_id_field,
            )

            # Update total metrics
            total_metrics.spend += record.spend
            total_metrics.prompt_tokens += record.prompt_tokens
            total_metrics.completion_tokens += record.completion_tokens
            total_metrics.total_tokens += (
                record.prompt_tokens + record.completion_tokens
            )
            total_metrics.cache_read_input_tokens += record.cache_read_input_tokens
            total_metrics.cache_creation_input_tokens += (
                record.cache_creation_input_tokens
            )
            total_metrics.api_requests += record.api_requests
            total_metrics.successful_requests += record.successful_requests
            total_metrics.failed_requests += record.failed_requests

        # Convert grouped data to response format
        for date_str, data in grouped_data.items():
            results.append(
                DailySpendData(
                    date=datetime.strptime(date_str, "%Y-%m-%d").date(),
                    metrics=data["metrics"],
                    breakdown=data["breakdown"],
                )
            )

        # Sort results by date
        results.sort(key=lambda x: x.date, reverse=True)

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
