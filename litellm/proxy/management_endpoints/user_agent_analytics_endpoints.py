"""
User Agent Analytics Endpoints

This module provides optimized endpoints for tracking user agent activity metrics including:
- Daily Active Users (DAU) by tags for configurable number of days
- Weekly Active Users (WAU) by tags for configurable number of weeks  
- Monthly Active Users (MAU) by tags for configurable number of months
- Summary analytics by tags

These endpoints use optimized single SQL queries with joins to efficiently calculate
user metrics from tag activity data and return time series for dashboard visualization.
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from litellm.proxy._types import CommonProxyErrors, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

# Constants for analytics periods
MAX_DAYS = 7  # Number of days to show in DAU analytics
MAX_WEEKS = 7  # Number of weeks to show in WAU analytics
MAX_MONTHS = 7  # Number of months to show in MAU analytics
MAX_TAGS = 250  # Maximum number of distinct tags to return

router = APIRouter()


class TagActiveUsersResponse(BaseModel):
    """Response for tag active users metrics"""
    tag: str
    active_users: int
    date: str  # The specific date or period identifier
    period_start: Optional[str] = None  # For WAU/MAU, this will be the start of the period
    period_end: Optional[str] = None  # For WAU/MAU, this will be the end of the period


class ActiveUsersAnalyticsResponse(BaseModel):
    """Response for active users analytics"""
    results: List[TagActiveUsersResponse]


class TagSummaryMetrics(BaseModel):
    """Summary metrics for a tag"""
    tag: str
    unique_users: int
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_tokens: int
    total_spend: float


class TagSummaryResponse(BaseModel):
    """Response for tag summary analytics"""
    results: List[TagSummaryMetrics]


class DistinctTagResponse(BaseModel):
    """Response for distinct user agent tags"""
    tag: str


class DistinctTagsResponse(BaseModel):
    """Response for all distinct user agent tags"""
    results: List[DistinctTagResponse]



class PerUserMetrics(BaseModel):
    """Metrics for individual user"""
    user_id: str
    user_email: Optional[str] = None
    user_agent: Optional[str] = None
    successful_requests: int = 0
    failed_requests: int = 0
    total_requests: int = 0
    total_tokens: int = 0
    spend: float = 0.0


class PerUserAnalyticsResponse(BaseModel):
    """Response for per-user analytics"""
    results: List[PerUserMetrics]
    total_count: int
    page: int
    page_size: int
    total_pages: int


@router.get(
    "/tag/distinct",
    response_model=DistinctTagsResponse,
    tags=["tag management", "user agent analytics"],
    dependencies=[Depends(user_api_key_auth)],
)
async def get_distinct_user_agent_tags(
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Get all distinct user agent tags up to a maximum of {MAX_TAGS} tags.
    
    This endpoint returns all unique user agent tags found in the database,
    sorted by frequency of usage.
    
    Returns:
        DistinctTagsResponse: List of distinct user agent tags
    """
    from litellm.proxy.proxy_server import prisma_client
    
    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )
    
    try:
        sql_query = f"""
        SELECT 
            dts.tag,
            COUNT(*) as usage_count
        FROM "LiteLLM_DailyTagSpend" dts
        WHERE dts.tag LIKE 'User-Agent:%' OR dts.tag NOT LIKE '%:%'
        GROUP BY dts.tag
        ORDER BY usage_count DESC
        LIMIT {MAX_TAGS}
        """
        
        db_response = await prisma_client.db.query_raw(sql_query)
        
        results = [
            DistinctTagResponse(tag=row["tag"])
            for row in db_response
        ]
        
        return DistinctTagsResponse(results=results)
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch distinct user agent tags: {str(e)}",
        )


@router.get(
    "/tag/dau",
    response_model=ActiveUsersAnalyticsResponse,
    tags=["tag management", "user agent analytics"],
    dependencies=[Depends(user_api_key_auth)],
)
async def get_daily_active_users(
    start_date: Optional[str] = Query(
        default=None,
        description="Start date in YYYY-MM-DD format (defaults to 7 days ago)",
    ),
    end_date: Optional[str] = Query(
        default=None,
        description="End date in YYYY-MM-DD format (defaults to today)",
    ),
    tag_filter: Optional[str] = Query(
        default=None,
        description="Filter by specific tag (optional)",
    ),
    tag_filters: Optional[List[str]] = Query(
        default=None,
        description="Filter by multiple specific tags (optional, takes precedence over tag_filter)",
    ),
    custom_llm_provider: Optional[str] = Query(
        default=None,
        description="Filter by custom LLM provider (e.g., 'hosted_vllm') (optional)",
    ),
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Get Daily Active Users (DAU) by tags for a customizable date range.

    This endpoint calculates unique users per tag for each day in the selected range
    using a single optimized SQL query, perfect for dashboard time series visualization.

    Args:
        start_date: Start date for the analytics period (YYYY-MM-DD, defaults to 7 days ago)
        end_date: End date for the analytics period (YYYY-MM-DD, defaults to today)
        tag_filter: Optional filter to specific tag (legacy)
        tag_filters: Optional filter to multiple specific tags (takes precedence over tag_filter)

    Returns:
        ActiveUsersAnalyticsResponse: DAU data by tag for each day in the date range
    """
    from litellm.proxy.proxy_server import prisma_client
    
    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )
    
    try:
        # Calculate date range
        from datetime import timezone

        if end_date:
            # User provided specific end date - interpret as inclusive calendar day
            # We add 1 day and use the resulting date as the (exclusive) upper bound in the SQL query
            end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(
                hour=0, minute=0, second=0, microsecond=0
            ) + timedelta(days=1)
        else:
            # Default: use today + 1 day for inclusive query
            end_dt = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)

        if start_date:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(
                hour=0, minute=0, second=0, microsecond=0
            )
        else:
            # Default to 7 days ago
            start_dt = end_dt - timedelta(days=7)

        end_date_str = end_dt.strftime("%Y-%m-%d")
        start_date_str = start_dt.strftime("%Y-%m-%d")

        # Build SQL query with optional tag filter(s) and custom_llm_provider filter
        where_clause = "WHERE dts.date >= $1 AND dts.date < $2 AND vt.user_id IS NOT NULL"
        params = [start_date_str, end_date_str]

        # Add custom_llm_provider filter if provided
        if custom_llm_provider:
            where_clause += f" AND dts.custom_llm_provider = ${len(params) + 1}"
            params.append(custom_llm_provider)

        # Handle multiple tag filters (takes precedence over single tag filter)
        if tag_filters and len(tag_filters) > 0:
            tag_conditions = []
            for tag in tag_filters:
                param_index = len(params) + 1
                tag_conditions.append(f"dts.tag = ${param_index}")
                params.append(tag)
            where_clause += f" AND ({' OR '.join(tag_conditions)})"
        elif tag_filter:
            where_clause += " AND dts.tag ILIKE $3"
            params.append(f"%{tag_filter}%")
       

        sql_query = f"""
        SELECT
            dts.tag,
            dts.date,
            COUNT(DISTINCT vt.user_id) as active_users
        FROM "LiteLLM_DailyTagSpend" dts
        INNER JOIN "LiteLLM_VerificationToken" vt ON dts.api_key = vt.token
        {where_clause}
        GROUP BY dts.tag, dts.date
        ORDER BY dts.date DESC, active_users DESC
        """
        
        db_response = await prisma_client.db.query_raw(sql_query, *params)
        
        results = [
            TagActiveUsersResponse(
                tag=row["tag"],
                active_users=row["active_users"],
                date=row["date"]
            )
            for row in db_response
        ]
        
        return ActiveUsersAnalyticsResponse(results=results)

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid date format. Use YYYY-MM-DD: {str(e)}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch DAU analytics: {str(e)}",
        )


@router.get(
    "/tag/wau",
    response_model=ActiveUsersAnalyticsResponse,
    tags=["tag management", "user agent analytics"],
    dependencies=[Depends(user_api_key_auth)],
)
async def get_weekly_active_users(
    tag_filter: Optional[str] = Query(
        default=None,
        description="Filter by specific tag (optional)",
    ),
    tag_filters: Optional[List[str]] = Query(
        default=None,
        description="Filter by multiple specific tags (optional, takes precedence over tag_filter)",
    ),
    custom_llm_provider: Optional[str] = Query(
        default=None,
        description="Filter by custom LLM provider (e.g., 'hosted_vllm') (optional)",
    ),
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Get Weekly Active Users (WAU) by tags for the last {MAX_WEEKS} weeks ending on UTC today + 1 day.
    
    Shows week-by-week breakdown:
    - Week 1 (Jan 1): Earliest week (7 weeks ago)
    - Week 2 (Jan 8): Next week (6 weeks ago)
    - Week 3 (Jan 15): Next week (5 weeks ago)
    - ... and so on for {MAX_WEEKS} weeks total
    - Week 7: Most recent week ending on UTC today + 1 day
    
    Args:
        tag_filter: Optional filter to specific tag (legacy)
        tag_filters: Optional filter to multiple specific tags (takes precedence over tag_filter)
        
    Returns:
        ActiveUsersAnalyticsResponse: WAU data by tag for each of the last {MAX_WEEKS} weeks with descriptive week labels (e.g., "Week 1 (Jan 1)")
    """
    from litellm.proxy.proxy_server import prisma_client
    
    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )
    
    try:
        # Calculate end_date as UTC today + 1 day
        from datetime import timezone
        end_dt = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        end_date = end_dt.strftime("%Y-%m-%d")
        
        # Calculate date range for all weeks (49 days total)
        # Start from 48 days before end_date to cover exactly MAX_WEEKS complete weeks
        start_dt = end_dt - timedelta(days=(MAX_WEEKS * 7 - 1))  # MAX_WEEKS weeks * 7 days - 1
        start_date = start_dt.strftime("%Y-%m-%d")
        
        # Build SQL query with optional tag filter(s) and custom_llm_provider filter
        where_clause = "WHERE dts.date >= $1 AND dts.date <= $2 AND vt.user_id IS NOT NULL"
        params = [start_date, end_date]

        # Add custom_llm_provider filter if provided
        if custom_llm_provider:
            where_clause += f" AND dts.custom_llm_provider = ${len(params) + 1}"
            params.append(custom_llm_provider)

        # Handle multiple tag filters (takes precedence over single tag filter)
        if tag_filters and len(tag_filters) > 0:
            tag_conditions = []
            for tag in tag_filters:
                param_index = len(params) + 1
                tag_conditions.append(f"dts.tag = ${param_index}")
                params.append(tag)
            where_clause += f" AND ({' OR '.join(tag_conditions)})"
        elif tag_filter:
            where_clause += " AND dts.tag ILIKE $3"
            params.append(f"%{tag_filter}%")

        # Use window function to group by weeks with clear week numbering
        sql_query = f"""
        WITH weekly_data AS (
            SELECT 
                dts.tag,
                dts.date,
                vt.user_id,
                -- Calculate week number (0 = Week 1 most recent, 1 = Week 2, etc.)
                FLOOR((DATE '{end_date}' - dts.date::date) / 7) as week_offset
            FROM "LiteLLM_DailyTagSpend" dts
            INNER JOIN "LiteLLM_VerificationToken" vt ON dts.api_key = vt.token
            {where_clause}
        )
        SELECT 
            tag,
            COUNT(DISTINCT user_id) as active_users,
            -- Week identifier with month and day (Week 1 (earliest), Week 2, etc.)
            'Week ' || ({MAX_WEEKS} - week_offset)::text || ' (' || 
            TO_CHAR(DATE '{end_date}' - (week_offset * 7 || ' days')::interval - '6 days'::interval, 'Mon DD') || ')' as date,
            -- Calculate week start and end dates for each week
            (DATE '{end_date}' - (week_offset * 7 || ' days')::interval - '6 days'::interval)::text as period_start,
            (DATE '{end_date}' - (week_offset * 7 || ' days')::interval)::text as period_end,
            week_offset
        FROM weekly_data
        WHERE week_offset < {MAX_WEEKS}
        GROUP BY tag, week_offset
        ORDER BY week_offset DESC, active_users DESC
        """
        
        db_response = await prisma_client.db.query_raw(sql_query, *params)
        
        results = [
            TagActiveUsersResponse(
                tag=row["tag"],
                active_users=row["active_users"],
                date=row["date"],  # This will be "Week 1 (Jan 15)", "Week 2 (Jan 8)", etc.
                period_start=row["period_start"],
                period_end=row["period_end"]
            )
            for row in db_response
        ]
        
        return ActiveUsersAnalyticsResponse(results=results)
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch WAU analytics: {str(e)}",
        )


@router.get(
    "/tag/mau",
    response_model=ActiveUsersAnalyticsResponse,
    tags=["tag management", "user agent analytics"],
    dependencies=[Depends(user_api_key_auth)],
)
async def get_monthly_active_users(
    months: int = Query(default=7, ge=1, le=12, description="Number of months to show (1-12)"),
    tag_filter: Optional[str] = Query(
        default=None,
        description="Filter by specific tag (optional)",
    ),
    tag_filters: Optional[List[str]] = Query(
        default=None,
        description="Filter by multiple specific tags (optional, takes precedence over tag_filter)",
    ),
    custom_llm_provider: Optional[str] = Query(
        default=None,
        description="Filter by custom LLM provider (e.g., 'hosted_vllm') (optional)",
    ),
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Get Monthly Active Users (MAU) by tags for the last N months ending on UTC today + 1 day.

    Shows month-by-month breakdown with proper month names (e.g., "December 2025").

    Args:
        months: Number of months to show (1-12, default: 7)
        tag_filter: Optional filter to specific tag (legacy)
        tag_filters: Optional filter to multiple specific tags (takes precedence over tag_filter)

    Returns:
        ActiveUsersAnalyticsResponse: MAU data by tag for each of the last N months
    """
    from litellm.proxy.proxy_server import prisma_client
    
    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )
    
    try:
        # Calculate end_date as UTC today + 1 day
        from datetime import timezone
        end_dt = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        end_date = end_dt.strftime("%Y-%m-%d")

        # Calculate date range for N months
        # Start from (months * 30 - 1) days before end_date
        start_dt = end_dt - timedelta(days=(months * 30 - 1))
        start_date = start_dt.strftime("%Y-%m-%d")

        # Build SQL query with optional tag filter(s) and custom_llm_provider filter
        where_clause = "WHERE dts.date >= $1 AND dts.date <= $2 AND vt.user_id IS NOT NULL"
        params = [start_date, end_date]

        # Add custom_llm_provider filter if provided
        if custom_llm_provider:
            where_clause += f" AND dts.custom_llm_provider = ${len(params) + 1}"
            params.append(custom_llm_provider)

        # Handle multiple tag filters (takes precedence over single tag filter)
        if tag_filters and len(tag_filters) > 0:
            tag_conditions = []
            for tag in tag_filters:
                param_index = len(params) + 1
                tag_conditions.append(f"dts.tag = ${param_index}")
                params.append(tag)
            where_clause += f" AND ({' OR '.join(tag_conditions)})"
        elif tag_filter:
            where_clause += " AND dts.tag ILIKE $3"
            params.append(f"%{tag_filter}%")

        # Use window function to group by months with proper month name labels
        sql_query = f"""
        WITH monthly_data AS (
            SELECT
                dts.tag,
                dts.date,
                vt.user_id,
                -- Calculate month number (0 = most recent month, 1 = month before, etc.)
                FLOOR((DATE '{end_date}' - dts.date::date) / 30) as month_offset
            FROM "LiteLLM_DailyTagSpend" dts
            INNER JOIN "LiteLLM_VerificationToken" vt ON dts.api_key = vt.token
            {where_clause}
        )
        SELECT
            tag,
            COUNT(DISTINCT user_id) as active_users,
            -- Month label with proper month name and year (e.g., "December 2025")
            TO_CHAR(DATE '{end_date}' - (month_offset * 30 || ' days')::interval - '29 days'::interval, 'Mon YYYY') as date,
            -- Calculate month start and end dates
            (DATE '{end_date}' - (month_offset * 30 || ' days')::interval - '29 days'::interval)::text as period_start,
            (DATE '{end_date}' - (month_offset * 30 || ' days')::interval)::text as period_end,
            month_offset
        FROM monthly_data
        WHERE month_offset < {months}
        GROUP BY tag, month_offset
        ORDER BY month_offset DESC, active_users DESC
        """
        
        db_response = await prisma_client.db.query_raw(sql_query, *params)
        
        results = [
            TagActiveUsersResponse(
                tag=row["tag"],
                active_users=row["active_users"],
                date=row["date"].strip(),  # Remove extra whitespace from Month format
                period_start=row["period_start"],
                period_end=row["period_end"]
            )
            for row in db_response
        ]
        
        return ActiveUsersAnalyticsResponse(results=results)
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch MAU analytics: {str(e)}",
        )


@router.get(
    "/tag/summary",
    response_model=TagSummaryResponse,
    tags=["tag management", "user agent analytics"],
    dependencies=[Depends(user_api_key_auth)],
)
async def get_tag_summary(
    start_date: str = Query(
        description="Start date in YYYY-MM-DD format"
    ),
    end_date: str = Query(
        description="End date in YYYY-MM-DD format"
    ),
    tag_filter: Optional[str] = Query(
        default=None,
        description="Filter by specific tag (optional)",
    ),
    tag_filters: Optional[List[str]] = Query(
        default=None,
        description="Filter by multiple specific tags (optional, takes precedence over tag_filter)",
    ),
    custom_llm_provider: Optional[str] = Query(
        default=None,
        description="Filter by custom LLM provider (e.g., 'hosted_vllm') (optional)",
    ),
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Get summary analytics for tags including unique users, requests, tokens, and spend.
    
    Args:
        start_date: Start date for the analytics period (YYYY-MM-DD)
        end_date: End date for the analytics period (YYYY-MM-DD)
        tag_filter: Optional filter to specific tag (legacy)
        tag_filters: Optional filter to multiple specific tags (takes precedence over tag_filter)
        
    Returns:
        TagSummaryResponse: Summary analytics data by tag
    """
    from litellm.proxy.proxy_server import prisma_client
    
    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )
    
    try:
        # Validate date format
        datetime.strptime(start_date, "%Y-%m-%d")
        datetime.strptime(end_date, "%Y-%m-%d")
        
        # Build SQL query with optional tag filter(s) and custom_llm_provider filter
        where_clause = "WHERE dts.date >= $1 AND dts.date <= $2"
        params = [start_date, end_date]

        # Add custom_llm_provider filter if provided
        if custom_llm_provider:
            where_clause += f" AND dts.custom_llm_provider = ${len(params) + 1}"
            params.append(custom_llm_provider)
       

        # Handle multiple tag filters (takes precedence over single tag filter)
        if tag_filters and len(tag_filters) > 0:
            tag_conditions = []
            for tag in tag_filters:
                param_index = len(params) + 1
                tag_conditions.append(f"dts.tag = ${param_index}")
                params.append(tag)
            where_clause += f" AND ({' OR '.join(tag_conditions)})"
        elif tag_filter:
            where_clause += " AND dts.tag ILIKE $3"
            params.append(f"%{tag_filter}%")

        sql_query = f"""
        SELECT 
            dts.tag,
            COUNT(DISTINCT vt.user_id) as unique_users,
            SUM(dts.api_requests) as total_requests,
            SUM(dts.successful_requests) as successful_requests,
            SUM(dts.failed_requests) as failed_requests,
            SUM(dts.prompt_tokens + dts.completion_tokens) as total_tokens,
            SUM(dts.spend) as total_spend
        FROM "LiteLLM_DailyTagSpend" dts
        LEFT JOIN "LiteLLM_VerificationToken" vt ON dts.api_key = vt.token
        {where_clause}
        GROUP BY dts.tag
        ORDER BY total_requests DESC
        """
        
        db_response = await prisma_client.db.query_raw(sql_query, *params)
        
        results = [
            TagSummaryMetrics(
                tag=row["tag"],
                unique_users=row["unique_users"] or 0,
                total_requests=int(row["total_requests"] or 0),
                successful_requests=int(row["successful_requests"] or 0),
                failed_requests=int(row["failed_requests"] or 0),
                total_tokens=int(row["total_tokens"] or 0),
                total_spend=float(row["total_spend"] or 0.0)
            )
            for row in db_response
        ]
        
        return TagSummaryResponse(results=results)
        
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid date format. Use YYYY-MM-DD: {str(e)}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch tag summary analytics: {str(e)}",
        )


@router.get(
    "/tag/user-agent/per-user-analytics",
    response_model=PerUserAnalyticsResponse,
    tags=["tag management", "user agent analytics"],
    dependencies=[Depends(user_api_key_auth)],
)
async def get_per_user_analytics(
    tag_filter: Optional[str] = Query(
        default=None,
        description="Filter by specific tag (optional)",
    ),
    tag_filters: Optional[List[str]] = Query(
        default=None,
        description="Filter by multiple specific tags (optional, takes precedence over tag_filter)",
    ),
    page: int = Query(default=1, description="Page number for pagination", ge=1),
    page_size: int = Query(
        default=50, description="Items per page", ge=1, le=1000
    ),
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Get per-user analytics including successful requests, tokens, and spend by individual users.
    
    This endpoint provides usage metrics broken down by individual users based on their
    tag activity during the last 30 days ending on UTC today + 1 day.
    
    Args:
        tag_filter: Optional filter to specific tag (legacy)
        tag_filters: Optional filter to multiple specific tags (takes precedence over tag_filter)
        page: Page number for pagination
        page_size: Number of items per page
        
    Returns:
        PerUserAnalyticsResponse: Analytics data broken down by individual users for the last 30 days
    """
    from litellm.proxy.proxy_server import prisma_client
    
    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )
    
    try:
        # Calculate end_date as UTC today + 1 day
        from datetime import timezone
        end_dt = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        end_date = end_dt.strftime("%Y-%m-%d")
        
        # Calculate date range (last 30 days)
        start_dt = end_dt - timedelta(days=30)
        start_date = start_dt.strftime("%Y-%m-%d")
        
        # Build where clause with date range
        where_clause: Dict[str, Any] = {
            "date": {"gte": start_date, "lte": end_date}
        }
        
        # Add tag filtering if provided
        if tag_filters and len(tag_filters) > 0:
            where_clause["tag"] = {"in": tag_filters}
        elif tag_filter:
            where_clause["tag"] = {"contains": tag_filter}
        
        # Get all tag records in the date range with optional tag filtering
        tag_records = await prisma_client.db.litellm_dailytagspend.find_many(
            where=where_clause
        )
        
        # Get unique api_keys
        api_keys = set(record.api_key for record in tag_records if record.api_key)
        
        if not api_keys:
            return PerUserAnalyticsResponse(
                results=[],
                total_count=0,
                page=page,
                page_size=page_size,
                total_pages=0,
            )
        
        # Lookup user_id for each api_key
        api_key_records = await prisma_client.db.litellm_verificationtoken.find_many(
            where={"token": {"in": list(api_keys)}}
        )
        
        # Create mapping from api_key to user_id
        api_key_to_user_id = {
            record.token: record.user_id 
            for record in api_key_records 
            if record.user_id
        }
        
        # Get user emails for the user_ids
        user_ids = list(set(api_key_to_user_id.values()))
        user_records = await prisma_client.db.litellm_usertable.find_many(
            where={"user_id": {"in": user_ids}}
        )
        
        # Create mapping from user_id to user_email
        user_id_to_email = {
            record.user_id: record.user_email 
            for record in user_records
        }
        
        # Aggregate metrics by user
        user_metrics: Dict[str, PerUserMetrics] = {}

        for record in tag_records:
            if record.api_key in api_key_to_user_id:
                user_id = api_key_to_user_id[record.api_key]
                tag = record.tag  # Use the full tag as user_agent
                
                if user_id not in user_metrics:
                    user_metrics[user_id] = PerUserMetrics(
                        user_id=user_id,
                        user_email=user_id_to_email.get(user_id),
                        user_agent=tag
                    )
                else:
                    # If tag is different, keep the first one or prioritize certain ones
                    if tag and not user_metrics[user_id].user_agent:
                        user_metrics[user_id].user_agent = tag
                
                # Aggregate metrics
                user_metrics[user_id].successful_requests += record.successful_requests or 0
                user_metrics[user_id].failed_requests += record.failed_requests or 0
                user_metrics[user_id].total_requests += record.api_requests or 0
                # Calculate total_tokens from prompt_tokens + completion_tokens
                prompt_tokens = record.prompt_tokens or 0
                completion_tokens = record.completion_tokens or 0
                user_metrics[user_id].total_tokens += int(prompt_tokens + completion_tokens)
                user_metrics[user_id].spend += record.spend or 0.0
        
        # Convert to list and sort by successful requests (descending)
        results = sorted(
            list(user_metrics.values()),
            key=lambda x: x.successful_requests,
            reverse=True
        )
        
        # Apply pagination
        total_count = len(results)
        total_pages = (total_count + page_size - 1) // page_size
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated_results = results[start_idx:end_idx]
        
        return PerUserAnalyticsResponse(
            results=paginated_results,
            total_count=total_count,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch per-user analytics: {str(e)}",
        )


class LeaderboardUser(BaseModel):
    """User entry in the leaderboard"""
    user_id: str
    user_email: Optional[str] = None
    request_count: int


class LeaderboardResponse(BaseModel):
    """Response for user leaderboard - returns all users sorted by request count"""
    results: List[LeaderboardUser]
    total_count: int


@router.get(
    "/user/analytics/leaderboard",
    response_model=LeaderboardResponse,
    tags=["user agent analytics"],
    dependencies=[Depends(user_api_key_auth)],
)
async def get_user_leaderboard(
    start_date: Optional[str] = Query(
        default=None,
        description="Start date in YYYY-MM-DD format (defaults to 7 days ago)",
    ),
    end_date: Optional[str] = Query(
        default=None,
        description="End date in YYYY-MM-DD format (defaults to today)",
    ),
    custom_llm_provider: Optional[str] = Query(
        default=None,
        description="Filter by custom LLM provider (e.g., 'hosted_vllm') (optional)",
    ),
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Get all active users by request count with customizable date range.

    Returns ALL users sorted by their total request count.
    Frontend handles pagination and email search.
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail="Database not connected",
        )

    try:
        # Calculate date range
        if end_date:
            # User provided specific end date - interpret as inclusive calendar day
            # We add 1 day and use the resulting date as the (exclusive) upper bound in the SQL query
            end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(
                hour=0, minute=0, second=0, microsecond=0
            ) + timedelta(days=1)
        else:
            # Default: use today + 1 day for inclusive query
            end_dt = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)

        if start_date:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(
                hour=0, minute=0, second=0, microsecond=0
            )
        else:
            # Default to 7 days ago
            start_dt = end_dt - timedelta(days=7)

        end_date_str = end_dt.strftime("%Y-%m-%d")
        start_date_str = start_dt.strftime("%Y-%m-%d")

        # Build SQL query with proper pagination using OFFSET/LIMIT for better performance
        where_clause = "WHERE dts.date >= $1 AND dts.date < $2"
        params = [start_date_str, end_date_str]

        # Add custom_llm_provider filter if provided
        if custom_llm_provider:
            where_clause += f" AND dts.custom_llm_provider = ${len(params) + 1}"
            params.append(custom_llm_provider)

        # First, get all matching api_keys and aggregate counts in a single query
        sql_query = f"""
        SELECT
            dts.api_key,
            SUM(dts.api_requests) as request_count
        FROM "LiteLLM_DailyTagSpend" dts
        {where_clause}
        AND dts.api_key IS NOT NULL
        GROUP BY dts.api_key
        """

        db_response = await prisma_client.db.query_raw(sql_query, *params)

        if not db_response:
            return LeaderboardResponse(
                results=[],
                total_count=0,
            )

        # Aggregate request count by api_key
        api_key_counts: Dict[str, int] = {}
        for row in db_response:
            api_key_counts[row["api_key"]] = row["request_count"]

        # Get unique api_keys
        api_keys = list(api_key_counts.keys())

        # Lookup user_id for each api_key
        api_key_records = await prisma_client.db.litellm_verificationtoken.find_many(
            where={"token": {"in": api_keys}}
        )

        # Create mapping from api_key to user_id
        api_key_to_user_id = {
            record.token: record.user_id
            for record in api_key_records
            if record.user_id
        }

        # Get user emails for the user_ids
        user_ids = list(set(api_key_to_user_id.values()))
        user_records = await prisma_client.db.litellm_usertable.find_many(
            where={"user_id": {"in": user_ids}}
        )

        # Create mapping from user_id to user_email
        user_id_to_email = {
            record.user_id: record.user_email
            for record in user_records
        }

        # Aggregate request counts by user_id (summing across api_keys)
        user_counts: Dict[str, int] = {}
        for api_key, count in api_key_counts.items():
            if api_key in api_key_to_user_id:
                user_id = api_key_to_user_id[api_key]
                user_counts[user_id] = user_counts.get(user_id, 0) + count

        # Build leaderboard entries
        leaderboard_entries: List[LeaderboardUser] = []
        for user_id, request_count in user_counts.items():
            leaderboard_entries.append(
                LeaderboardUser(
                    user_id=user_id,
                    user_email=user_id_to_email.get(user_id),
                    request_count=request_count,
                )
            )

        # Sort by request count (descending)
        leaderboard_entries.sort(key=lambda x: x.request_count, reverse=True)

        # Return all users - frontend handles pagination and search
        return LeaderboardResponse(
            results=leaderboard_entries,
            total_count=len(leaderboard_entries),
        )

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid date format. Use YYYY-MM-DD: {str(e)}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch leaderboard: {str(e)}",
        )


class UserActiveUsersResponse(BaseModel):
    """Response for user-count based active users analytics (not broken down by tag)"""
    results: List[Dict[str, Any]]


class UserActiveUsersItem(BaseModel):
    """Single user active users item"""
    date: str  # "YYYY-MM-DD" or "Week X (Mon DD)" or "Mon YYYY"
    active_users: int
    period_start: Optional[str] = None
    period_end: Optional[str] = None


@router.get(
    "/user/dau",
    response_model=UserActiveUsersResponse,
    tags=["user agent analytics"],
    dependencies=[Depends(user_api_key_auth)],
)
async def get_user_daily_active_users(
    start_date: Optional[str] = Query(
        default=None,
        description="Start date in YYYY-MM-DD format (defaults to 7 days ago)",
    ),
    end_date: Optional[str] = Query(
        default=None,
        description="End date in YYYY-MM-DD format (defaults to today)",
    ),
    custom_llm_provider: Optional[str] = Query(
        default=None,
        description="Filter by custom LLM provider (e.g., 'hosted_vllm') (optional)",
    ),
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Get daily unique user count (not broken down by user-agent tag).

    Returns the total count of unique users per day for the selected date range.
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )

    try:
        from datetime import timezone

        if end_date:
            end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(
                hour=0, minute=0, second=0, microsecond=0
            ) + timedelta(days=1)
        else:
            end_dt = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)

        if start_date:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(
                hour=0, minute=0, second=0, microsecond=0
            )
        else:
            start_dt = end_dt - timedelta(days=7)

        end_date_str = end_dt.strftime("%Y-%m-%d")
        start_date_str = start_dt.strftime("%Y-%m-%d")

        # Build SQL query
        where_clause = "WHERE dts.date >= $1 AND dts.date < $2 AND vt.user_id IS NOT NULL"
        params = [start_date_str, end_date_str]

        if custom_llm_provider:
            where_clause += f" AND dts.custom_llm_provider = ${len(params) + 1}"
            params.append(custom_llm_provider)

        sql_query = f"""
        SELECT
            dts.date,
            COUNT(DISTINCT vt.user_id) as active_users
        FROM "LiteLLM_DailyTagSpend" dts
        INNER JOIN "LiteLLM_VerificationToken" vt ON dts.api_key = vt.token
        {where_clause}
        GROUP BY dts.date
        ORDER BY dts.date DESC
        """

        db_response = await prisma_client.db.query_raw(sql_query, *params)

        results = [
            {
                "date": row["date"],
                "active_users": row["active_users"],
            }
            for row in db_response
        ]

        return UserActiveUsersResponse(results=results)

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid date format. Use YYYY-MM-DD: {str(e)}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch user DAU analytics: {str(e)}",
        )


@router.get(
    "/user/wau",
    response_model=UserActiveUsersResponse,
    tags=["user agent analytics"],
    dependencies=[Depends(user_api_key_auth)],
)
async def get_user_weekly_active_users(
    custom_llm_provider: Optional[str] = Query(
        default=None,
        description="Filter by custom LLM provider (e.g., 'hosted_vllm') (optional)",
    ),
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Get weekly unique user count for the last 7 weeks.

    Returns total unique users per week (not broken down by user-agent tag).
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )

    try:
        from datetime import timezone
        end_dt = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        end_date = end_dt.strftime("%Y-%m-%d")

        # Calculate start date for 7 weeks
        start_dt = end_dt - timedelta(days=(MAX_WEEKS * 7 - 1))
        start_date = start_dt.strftime("%Y-%m-%d")

        where_clause = "WHERE dts.date >= $1 AND dts.date <= $2 AND vt.user_id IS NOT NULL"
        params = [start_date, end_date]

        if custom_llm_provider:
            where_clause += f" AND dts.custom_llm_provider = ${len(params) + 1}"
            params.append(custom_llm_provider)

        sql_query = f"""
        WITH weekly_data AS (
            SELECT
                dts.date,
                vt.user_id,
                FLOOR((DATE '{end_date}' - dts.date::date) / 7) as week_offset
            FROM "LiteLLM_DailyTagSpend" dts
            INNER JOIN "LiteLLM_VerificationToken" vt ON dts.api_key = vt.token
            {where_clause}
        )
        SELECT
            COUNT(DISTINCT user_id) as active_users,
            'Week ' || ({MAX_WEEKS} - week_offset)::text || ' (' ||
            TO_CHAR(DATE '{end_date}' - (week_offset * 7 || ' days')::interval - '6 days'::interval, 'Mon DD') || ')' as date,
            (DATE '{end_date}' - (week_offset * 7 || ' days')::interval - '6 days'::interval)::text as period_start,
            (DATE '{end_date}' - (week_offset * 7 || ' days')::interval)::text as period_end,
            week_offset
        FROM weekly_data
        WHERE week_offset < {MAX_WEEKS}
        GROUP BY week_offset
        ORDER BY week_offset DESC
        """

        db_response = await prisma_client.db.query_raw(sql_query, *params)

        results = [
            {
                "date": row["date"],
                "active_users": row["active_users"],
                "period_start": row["period_start"],
                "period_end": row["period_end"],
            }
            for row in db_response
        ]

        return UserActiveUsersResponse(results=results)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch user WAU analytics: {str(e)}",
        )


@router.get(
    "/user/mau",
    response_model=UserActiveUsersResponse,
    tags=["user agent analytics"],
    dependencies=[Depends(user_api_key_auth)],
)
async def get_user_monthly_active_users(
    months: int = Query(default=7, ge=1, le=12, description="Number of months to show (1-12)"),
    custom_llm_provider: Optional[str] = Query(
        default=None,
        description="Filter by custom LLM provider (e.g., 'hosted_vllm') (optional)",
    ),
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Get monthly unique user count for the last N months.

    Returns total unique users per month (not broken down by user-agent tag).
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )

    try:
        from datetime import timezone
        end_dt = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        end_date = end_dt.strftime("%Y-%m-%d")

        start_dt = end_dt - timedelta(days=(months * 30 - 1))
        start_date = start_dt.strftime("%Y-%m-%d")

        where_clause = "WHERE dts.date >= $1 AND dts.date <= $2 AND vt.user_id IS NOT NULL"
        params = [start_date, end_date]

        if custom_llm_provider:
            where_clause += f" AND dts.custom_llm_provider = ${len(params) + 1}"
            params.append(custom_llm_provider)

        sql_query = f"""
        WITH monthly_data AS (
            SELECT
                dts.date,
                vt.user_id,
                FLOOR((DATE '{end_date}' - dts.date::date) / 30) as month_offset
            FROM "LiteLLM_DailyTagSpend" dts
            INNER JOIN "LiteLLM_VerificationToken" vt ON dts.api_key = vt.token
            {where_clause}
        )
        SELECT
            COUNT(DISTINCT user_id) as active_users,
            TO_CHAR(DATE '{end_date}' - (month_offset * 30 || ' days')::interval - '29 days'::interval, 'Mon YYYY') as date,
            (DATE '{end_date}' - (month_offset * 30 || ' days')::interval - '29 days'::interval)::text as period_start,
            (DATE '{end_date}' - (month_offset * 30 || ' days')::interval)::text as period_end,
            month_offset
        FROM monthly_data
        WHERE month_offset < {months}
        GROUP BY month_offset
        ORDER BY month_offset DESC
        """

        db_response = await prisma_client.db.query_raw(sql_query, *params)

        results = [
            {
                "date": row["date"].strip(),
                "active_users": row["active_users"],
                "period_start": row["period_start"],
                "period_end": row["period_end"],
            }
            for row in db_response
        ]

        return UserActiveUsersResponse(results=results)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch user MAU analytics: {str(e)}",
        )