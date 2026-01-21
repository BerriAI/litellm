"""
Admin User Usage Endpoints

Provides endpoints for admins to view usage metrics across all users.
Follows the same pattern as /user/daily/activity but for admin-level user analytics.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

import fastapi
from fastapi import APIRouter, Depends, HTTPException, Query, status

from litellm.proxy._types import CommonProxyErrors, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.management_endpoints.common_daily_activity import (
    SpendAnalyticsPaginatedResponse,
)
from litellm.proxy.utils import PrismaClient

router = APIRouter()


def _user_has_admin_view(user_api_key_dict: UserAPIKeyAuth) -> bool:
    """Check if user has admin permissions."""
    from litellm.proxy.proxy_server import premium_user
    from litellm.utils.utils import get_utc_datetime

    from litellm.proxy.management_endpoints.internal_user_endpoints import (
        _user_has_admin_view as internal_admin_check,
    )

    return internal_admin_check(user_api_key_dict)


async def get_admin_users_usage(
    prisma_client: Optional[PrismaClient],
    start_date: str,
    end_date: str,
    tag_filters: Optional[List[str]] = None,
    min_spend: Optional[float] = None,
    max_spend: Optional[float] = None,
    sort_by: str = "spend",
    sort_order: str = "desc",
    page: int = 1,
    page_size: int = 50,
    top_n: int = 10,
) -> Dict[str, Any]:
    """
    Get admin-level user usage analytics with pagination.

    Returns:
    - summary: Aggregate stats across all users
    - top_users: Top N users for bar chart visualization
    - users: Paginated list of users for table
    - pagination: Pagination metadata
    """
    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )

    # Build WHERE conditions for queries
    where_conditions = []
    params_summary = [start_date, end_date]
    params_top = [start_date, end_date]
    params_paginated = [start_date, end_date]

    # Base WHERE clause
    base_where = "dus.date >= $1 AND dus.date <= $2 AND vt.user_id IS NOT NULL"

    # Add tag filters
    tag_filter_clause = ""
    if tag_filters and len(tag_filters) > 0:
        tag_placeholders = ", ".join([f"${i+3}" for i in range(len(tag_filters))])
        tag_filter_clause = f" AND dts.tag IN ({tag_placeholders})"
        params_summary.extend(tag_filters)
        params_top.extend(tag_filters)
        params_paginated.extend(tag_filters)

    # Query 1: Get summary statistics
    summary_sql = f"""
    WITH user_totals AS (
        SELECT
            vt.user_id,
            SUM(dus.spend) as user_spend,
            SUM(dus.api_requests) as user_requests,
            SUM(dus.successful_requests) as user_successful_requests,
            SUM(dus.failed_requests) as user_failed_requests,
            SUM(dus.prompt_tokens + dus.completion_tokens) as user_tokens
        FROM "LiteLLM_DailyUserSpend" dus
        INNER JOIN "LiteLLM_VerificationToken" vt ON dus.api_key = vt.token
        {f'INNER JOIN "LiteLLM_DailyTagSpend" dts ON dus.api_key = dts.api_key AND dus.date = dts.date' if tag_filters else ''}
        WHERE {base_where}{tag_filter_clause if tag_filters else ''}
        GROUP BY vt.user_id
    )
    SELECT
        COUNT(DISTINCT user_id) as total_users,
        COALESCE(SUM(user_spend), 0) as total_spend,
        COALESCE(SUM(user_requests), 0) as total_requests,
        COALESCE(SUM(user_successful_requests), 0) as total_successful_requests,
        COALESCE(SUM(user_failed_requests), 0) as total_failed_requests,
        COALESCE(SUM(user_tokens), 0) as total_tokens,
        COALESCE(AVG(user_spend), 0) as avg_spend_per_user,
        COUNT(CASE WHEN user_spend > 200 THEN 1 END) as power_users_count,
        COUNT(CASE WHEN user_spend < 10 THEN 1 END) as low_users_count
    FROM user_totals
    """

    if min_spend is not None:
        summary_sql += f" WHERE user_spend >= ${len(params_summary) + 1}"
        params_summary.append(min_spend)

    if max_spend is not None:
        summary_sql += (
            f" {'AND' if min_spend is not None else 'WHERE'} user_spend <= ${len(params_summary) + 1}"
        )
        params_summary.append(max_spend)

    summary_result = await prisma_client.db.query_raw(summary_sql, *params_summary)

    summary = {
        "total_users": summary_result[0].get("total_users", 0) if summary_result else 0,
        "total_spend": float(summary_result[0].get("total_spend", 0.0)) if summary_result else 0.0,
        "total_requests": int(summary_result[0].get("total_requests", 0)) if summary_result else 0,
        "total_successful_requests": int(summary_result[0].get("total_successful_requests", 0))
        if summary_result
        else 0,
        "total_failed_requests": int(summary_result[0].get("total_failed_requests", 0))
        if summary_result
        else 0,
        "total_tokens": int(summary_result[0].get("total_tokens", 0)) if summary_result else 0,
        "avg_spend_per_user": float(summary_result[0].get("avg_spend_per_user", 0.0))
        if summary_result
        else 0.0,
        "power_users_count": summary_result[0].get("power_users_count", 0) if summary_result else 0,
        "low_users_count": summary_result[0].get("low_users_count", 0) if summary_result else 0,
    }

    # Query 2: Get top N users for bar chart
    sort_field_map = {
        "spend": "total_spend",
        "requests": "total_requests",
        "tokens": "total_tokens",
    }
    db_sort_field = sort_field_map.get(sort_by, "total_spend")

    spend_filter_clause = ""
    if min_spend is not None:
        spend_filter_clause += f" HAVING SUM(dus.spend) >= ${len(params_top) + 1}"
        params_top.append(min_spend)
        if max_spend is not None:
            spend_filter_clause += f" AND SUM(dus.spend) <= ${len(params_top) + 1}"
            params_top.append(max_spend)
    elif max_spend is not None:
        spend_filter_clause += f" HAVING SUM(dus.spend) <= ${len(params_top) + 1}"
        params_top.append(max_spend)

    top_users_sql = f"""
    SELECT
        vt.user_id,
        ut.user_email,
        SUM(dus.spend) as total_spend,
        SUM(dus.api_requests) as total_requests,
        SUM(dus.successful_requests) as total_successful_requests,
        SUM(dus.failed_requests) as total_failed_requests,
        SUM(dus.prompt_tokens) as total_prompt_tokens,
        SUM(dus.completion_tokens) as total_completion_tokens,
        SUM(dus.prompt_tokens + dus.completion_tokens) as total_tokens,
        COUNT(DISTINCT dus.date) as days_active,
        MIN(dus.date) as first_request_date,
        MAX(dus.date) as last_request_date,
        ARRAY_AGG(DISTINCT dts.tag) FILTER (WHERE dts.tag IS NOT NULL) as tags,
        ARRAY_AGG(DISTINCT dus.model) FILTER (WHERE dus.model IS NOT NULL) as models_used
    FROM "LiteLLM_DailyUserSpend" dus
    INNER JOIN "LiteLLM_VerificationToken" vt ON dus.api_key = vt.token
    LEFT JOIN "LiteLLM_UserTable" ut ON vt.user_id = ut.user_id
    LEFT JOIN "LiteLLM_DailyTagSpend" dts ON dus.api_key = dts.api_key AND dus.date = dts.date
    WHERE {base_where}{tag_filter_clause if tag_filters else ''}
    GROUP BY vt.user_id, ut.user_email
    {spend_filter_clause}
    ORDER BY {db_sort_field} {sort_order.upper()}
    LIMIT {top_n}
    """

    top_users_result = await prisma_client.db.query_raw(top_users_sql, *params_top)

    top_users = [
        {
            "user_id": row["user_id"],
            "user_email": row["user_email"] or row["user_id"],
            "spend": float(row["total_spend"]),
            "requests": int(row["total_requests"]),
            "successful_requests": int(row["total_successful_requests"]),
            "failed_requests": int(row["total_failed_requests"]),
            "prompt_tokens": int(row["total_prompt_tokens"]),
            "completion_tokens": int(row["total_completion_tokens"]),
            "tokens": int(row["total_tokens"]),
            "days_active": int(row["days_active"]),
            "first_request_date": str(row["first_request_date"]),
            "last_request_date": str(row["last_request_date"]),
            "tags": row["tags"] or [],
            "models_used": row["models_used"] or [],
        }
        for row in top_users_result
    ]

    # Query 3: Get paginated users for table
    offset = (page - 1) * page_size

    # First get total count for pagination
    count_sql = f"""
    SELECT COUNT(DISTINCT vt.user_id) as total_count
    FROM "LiteLLM_DailyUserSpend" dus
    INNER JOIN "LiteLLM_VerificationToken" vt ON dus.api_key = vt.token
    {f'INNER JOIN "LiteLLM_DailyTagSpend" dts ON dus.api_key = dts.api_key AND dus.date = dts.date' if tag_filters else ''}
    WHERE {base_where}{tag_filter_clause if tag_filters else ''}
    """

    count_result = await prisma_client.db.query_raw(count_sql, *params_paginated[: len(params_paginated) - (2 if min_spend or max_spend else 0)])
    total_count = count_result[0]["total_count"] if count_result else 0

    # Get paginated users
    paginated_sql = f"""
    SELECT
        vt.user_id,
        ut.user_email,
        SUM(dus.spend) as total_spend,
        SUM(dus.api_requests) as total_requests,
        SUM(dus.successful_requests) as total_successful_requests,
        SUM(dus.failed_requests) as total_failed_requests,
        SUM(dus.prompt_tokens) as total_prompt_tokens,
        SUM(dus.completion_tokens) as total_completion_tokens,
        SUM(dus.prompt_tokens + dus.completion_tokens) as total_tokens,
        COUNT(DISTINCT dus.date) as days_active,
        MIN(dus.date) as first_request_date,
        MAX(dus.date) as last_request_date,
        ARRAY_AGG(DISTINCT dts.tag) FILTER (WHERE dts.tag IS NOT NULL) as tags,
        ARRAY_AGG(DISTINCT dus.model) FILTER (WHERE dus.model IS NOT NULL) as models_used
    FROM "LiteLLM_DailyUserSpend" dus
    INNER JOIN "LiteLLM_VerificationToken" vt ON dus.api_key = vt.token
    LEFT JOIN "LiteLLM_UserTable" ut ON vt.user_id = ut.user_id
    LEFT JOIN "LiteLLM_DailyTagSpend" dts ON dus.api_key = dts.api_key AND dus.date = dts.date
    WHERE {base_where}{tag_filter_clause if tag_filters else ''}
    GROUP BY vt.user_id, ut.user_email
    {spend_filter_clause}
    ORDER BY {db_sort_field} {sort_order.upper()}
    LIMIT {page_size} OFFSET {offset}
    """

    paginated_result = await prisma_client.db.query_raw(paginated_sql, *params_paginated)

    users = [
        {
            "user_id": row["user_id"],
            "user_email": row["user_email"] or row["user_id"],
            "spend": float(row["total_spend"]),
            "requests": int(row["total_requests"]),
            "successful_requests": int(row["total_successful_requests"]),
            "failed_requests": int(row["total_failed_requests"]),
            "prompt_tokens": int(row["total_prompt_tokens"]),
            "completion_tokens": int(row["total_completion_tokens"]),
            "tokens": int(row["total_tokens"]),
            "days_active": int(row["days_active"]),
            "first_request_date": str(row["first_request_date"]),
            "last_request_date": str(row["last_request_date"]),
            "tags": row["tags"] or [],
            "models_used": row["models_used"] or [],
        }
        for row in paginated_result
    ]

    total_pages = (total_count + page_size - 1) // page_size if total_count > 0 else 0

    return {
        "summary": summary,
        "top_users": top_users,
        "users": users,
        "pagination": {
            "page": page,
            "page_size": page_size,
            "total_count": total_count,
            "total_pages": total_pages,
        },
    }


@router.get(
    "/admin/users/daily/activity",
    tags=["Budget & Spend Tracking", "Admin User Analytics"],
    dependencies=[Depends(user_api_key_auth)],
)
async def get_admin_users_daily_activity(
    start_date: str = fastapi.Query(
        ...,
        description="Start date in YYYY-MM-DD format",
    ),
    end_date: str = fastapi.Query(
        ...,
        description="End date in YYYY-MM-DD format",
    ),
    tag_filters: Optional[List[str]] = fastapi.Query(
        default=None,
        description="Filter by specific tags (e.g., User-Agent:claude-code)",
    ),
    min_spend: Optional[float] = fastapi.Query(
        default=None,
        description="Minimum spend threshold",
    ),
    max_spend: Optional[float] = fastapi.Query(
        default=None,
        description="Maximum spend threshold",
    ),
    sort_by: str = fastapi.Query(
        default="spend",
        description="Sort by field: spend, requests, or tokens",
    ),
    sort_order: str = fastapi.Query(
        default="desc",
        description="Sort order: asc or desc",
    ),
    page: int = fastapi.Query(
        default=1, description="Page number for pagination", ge=1
    ),
    page_size: int = fastapi.Query(
        default=50, description="Items per page", ge=1, le=100
    ),
    top_n: int = fastapi.Query(
        default=10, description="Number of top users to return", ge=1, le=50
    ),
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Get admin-level user usage analytics with pagination.

    Returns usage metrics for all users with:
    - Summary: Aggregate statistics across all users
    - Top Users: Top N users for visualization (bar chart)
    - Users: Paginated list of users for table display
    - Pagination: Metadata for pagination

    Requires admin permissions.
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )

    # Check admin permissions
    if not _user_has_admin_view(user_api_key_dict):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"error": "Admin permissions required"},
        )

    # Validate date format
    try:
        datetime.strptime(start_date, "%Y-%m-%d")
        datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "Invalid date format. Use YYYY-MM-DD"},
        )

    # Validate sort parameters
    if sort_by not in ["spend", "requests", "tokens"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "sort_by must be one of: spend, requests, tokens"},
        )

    if sort_order not in ["asc", "desc"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "sort_order must be one of: asc, desc"},
        )

    try:
        result = await get_admin_users_usage(
            prisma_client=prisma_client,
            start_date=start_date,
            end_date=end_date,
            tag_filters=tag_filters,
            min_spend=min_spend,
            max_spend=max_spend,
            sort_by=sort_by,
            sort_order=sort_order,
            page=page,
            page_size=page_size,
            top_n=top_n,
        )
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": f"Failed to fetch admin user usage: {str(e)}"},
        )
