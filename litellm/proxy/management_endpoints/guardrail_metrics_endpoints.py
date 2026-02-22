import json
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from litellm.proxy._types import CommonProxyErrors
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.types.proxy.management_endpoints.guardrail_metrics import (
    GuardrailDailyMetrics,
    GuardrailDetailMetrics,
    GuardrailLogEntry,
    GuardrailLogsResponse,
    GuardrailMetricsResponse,
    GuardrailSummary,
)

router = APIRouter()


@router.get(
    "/guardrail/metrics",
    tags=["guardrails"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=GuardrailMetricsResponse,
)
async def get_guardrail_metrics(
    start_date: str = Query(..., description="Start date YYYY-MM-DD"),
    end_date: str = Query(..., description="End date YYYY-MM-DD"),
    guardrail_name: Optional[str] = Query(None),
    provider: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
):
    """
    Get aggregated guardrail metrics for dashboard table view.

    Returns list of guardrails with total requests, fail rate, avg latency.
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )

    # Build query filters
    where_conditions: Dict[str, Any] = {
        "date": {
            "gte": start_date,
            "lte": end_date,
        }
    }

    if guardrail_name:
        where_conditions["guardrail_name"] = guardrail_name

    if provider:
        where_conditions["guardrail_provider"] = provider

    # Query daily metrics
    daily_metrics = await prisma_client.db.litellm_dailyguardrailmetrics.find_many(
        where=where_conditions,
        order=[{"date": "desc"}],
    )

    # Aggregate by guardrail_name across all dates
    guardrail_aggregates = {}
    for record in daily_metrics:
        name = record.guardrail_name
        if name not in guardrail_aggregates:
            guardrail_aggregates[name] = {
                "provider": record.guardrail_provider or "unknown",
                "total_requests": 0,
                "intervened_count": 0,
                "total_latency_ms": 0.0,
            }

        agg = guardrail_aggregates[name]
        agg["total_requests"] += int(record.total_requests)
        agg["intervened_count"] += int(record.intervened_count)
        agg["total_latency_ms"] += float(record.total_latency_ms)

    # Calculate fail rate and avg latency
    results = []
    for name, agg in guardrail_aggregates.items():
        fail_rate = (
            (agg["intervened_count"] / agg["total_requests"] * 100)
            if agg["total_requests"] > 0
            else 0.0
        )
        avg_latency = (
            agg["total_latency_ms"] / agg["total_requests"]
            if agg["total_requests"] > 0
            else 0.0
        )

        results.append(
            GuardrailSummary(
                guardrail_name=name,
                provider=agg["provider"],
                total_requests=agg["total_requests"],
                fail_rate=round(fail_rate, 2),
                avg_latency_ms=round(avg_latency, 2),
            )
        )

    # Sort by fail rate descending
    results.sort(key=lambda x: x.fail_rate, reverse=True)

    # Pagination
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    paginated_results = results[start_idx:end_idx]

    total_count = len(results)

    return GuardrailMetricsResponse(
        results=paginated_results,
        metadata={
            "page": page,
            "total_pages": (total_count + page_size - 1) // page_size,
            "has_more": end_idx < total_count,
            "total_count": total_count,
        },
    )


@router.get(
    "/guardrail/{guardrail_name}/metrics",
    tags=["guardrails"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=GuardrailDetailMetrics,
)
async def get_guardrail_detail_metrics(
    guardrail_name: str,
    start_date: str = Query(..., description="Start date YYYY-MM-DD"),
    end_date: str = Query(..., description="End date YYYY-MM-DD"),
):
    """
    Get detailed metrics for a specific guardrail (for overview tab).

    Returns aggregated metrics plus daily time-series data.
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )

    # Query daily metrics for this guardrail
    daily_records = await prisma_client.db.litellm_dailyguardrailmetrics.find_many(
        where={
            "guardrail_name": guardrail_name,
            "date": {
                "gte": start_date,
                "lte": end_date,
            },
        },
        order=[{"date": "asc"}],
    )

    if not daily_records:
        return GuardrailDetailMetrics(
            requests_evaluated=0,
            fail_rate=0.0,
            avg_latency_ms=0.0,
            blocked_count=0,
            daily_metrics=[],
        )

    # Calculate totals
    total_requests = sum(int(r.total_requests) for r in daily_records)
    total_intervened = sum(int(r.intervened_count) for r in daily_records)
    total_latency_ms = sum(float(r.total_latency_ms) for r in daily_records)

    fail_rate = (total_intervened / total_requests * 100) if total_requests > 0 else 0.0
    avg_latency_ms = total_latency_ms / total_requests if total_requests > 0 else 0.0

    # Build daily time-series
    daily_metrics = []
    for record in daily_records:
        requests = int(record.total_requests)
        intervened = int(record.intervened_count)
        latency = float(record.total_latency_ms)

        daily_fail_rate = (intervened / requests * 100) if requests > 0 else 0.0
        daily_avg_latency = latency / requests if requests > 0 else 0.0

        daily_metrics.append(
            GuardrailDailyMetrics(
                date=record.date,
                total_requests=requests,
                intervened_count=intervened,
                success_count=int(record.success_count),
                fail_rate=round(daily_fail_rate, 2),
                avg_latency_ms=round(daily_avg_latency, 2),
            )
        )

    return GuardrailDetailMetrics(
        requests_evaluated=total_requests,
        fail_rate=round(fail_rate, 2),
        avg_latency_ms=round(avg_latency_ms, 2),
        blocked_count=total_intervened,
        daily_metrics=daily_metrics,
    )


@router.get(
    "/guardrail/{guardrail_name}/logs",
    tags=["guardrails"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=GuardrailLogsResponse,
)
async def get_guardrail_logs(
    guardrail_name: str,
    start_date: str = Query(..., description="Start date YYYY-MM-DD"),
    end_date: str = Query(..., description="End date YYYY-MM-DD"),
    status_filter: Optional[str] = Query(None, description="'blocked' or 'passed'"),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=100),
):
    """
    Get individual request logs for a guardrail (for logs tab).

    Queries LiteLLM_SpendLogs and filters by guardrail_information in metadata.
    """
    from datetime import datetime

    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )

    # Convert dates to datetime for filtering
    start_datetime = datetime.fromisoformat(start_date).isoformat()
    end_datetime = datetime.fromisoformat(end_date + "T23:59:59").isoformat()

    # Query spend logs with metadata containing guardrail_information
    # Note: This is a simplified approach - may need optimization for large datasets
    spend_logs = await prisma_client.db.litellm_spendlogs.find_many(
        where={
            "startTime": {
                "gte": start_datetime,
                "lte": end_datetime,
            },
        },
        order=[{"startTime": "desc"}],
        skip=(page - 1) * page_size,
        take=page_size * 3,  # Over-fetch to account for filtering
    )

    # Parse and filter logs
    filtered_logs = []
    for log in spend_logs:
        try:
            metadata = json.loads(log.metadata) if isinstance(log.metadata, str) else log.metadata
            guardrail_info = metadata.get("guardrail_information", [])

            # Find matching guardrail in list
            for g in guardrail_info:
                if g.get("guardrail_name") == guardrail_name:
                    status = g.get("guardrail_status", "")

                    # Map status to blocked/passed
                    if status == "guardrail_intervened":
                        log_status = "blocked"
                    elif status == "success":
                        log_status = "passed"
                    else:
                        continue  # Skip other statuses

                    # Apply status filter
                    if status_filter and log_status != status_filter:
                        continue

                    # Extract request content
                    request_content = None
                    messages = metadata.get("messages", [])
                    if messages and isinstance(messages, list) and len(messages) > 0:
                        last_msg = messages[-1]
                        if isinstance(last_msg, dict):
                            request_content = last_msg.get("content", "")

                    filtered_logs.append(
                        GuardrailLogEntry(
                            request_id=log.request_id,
                            timestamp=log.startTime.isoformat() if log.startTime else "",
                            model=log.model or "unknown",
                            status=log_status,
                            guardrail_response=g.get("guardrail_response"),
                            request_content=request_content,
                            latency_ms=round((g.get("duration") or 0) * 1000, 2),
                        )
                    )

                    if len(filtered_logs) >= page_size:
                        break

            if len(filtered_logs) >= page_size:
                break

        except Exception as e:
            continue

    return GuardrailLogsResponse(
        logs=filtered_logs[:page_size],
        total_count=len(filtered_logs),  # Approximate
        page=page,
        page_size=page_size,
    )
