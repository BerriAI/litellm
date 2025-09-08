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

from datetime import datetime, timedelta
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
    tag_filter: Optional[str] = Query(
        default=None,
        description="Filter by specific tag (optional)",
    ),
    tag_filters: Optional[List[str]] = Query(
        default=None,
        description="Filter by multiple specific tags (optional, takes precedence over tag_filter)",
    ),
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Get Daily Active Users (DAU) by tags for the last {MAX_DAYS} days ending on UTC today + 1 day.
    
    This endpoint efficiently calculates unique users per tag for each of the last {MAX_DAYS} days
    using a single optimized SQL query, perfect for dashboard time series visualization.
    
    Args:
        tag_filter: Optional filter to specific tag (legacy)
        tag_filters: Optional filter to multiple specific tags (takes precedence over tag_filter)
        
    Returns:
        ActiveUsersAnalyticsResponse: DAU data by tag for each of the last {MAX_DAYS} days
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
        
        # Calculate date range (last MAX_DAYS days)
        start_dt = end_dt - timedelta(days=MAX_DAYS)
        start_date = start_dt.strftime("%Y-%m-%d")
        
        # Build SQL query with optional tag filter(s)
        where_clause = "WHERE dts.date >= $1 AND dts.date <= $2 AND vt.user_id IS NOT NULL"
        params = [start_date, end_date]
        
        # Handle multiple tag filters (takes precedence over single tag filter)
        if tag_filters and len(tag_filters) > 0:
            tag_conditions = []
            for i, tag in enumerate(tag_filters):
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
        
        # Build SQL query with optional tag filter(s)
        where_clause = "WHERE dts.date >= $1 AND dts.date <= $2 AND vt.user_id IS NOT NULL"
        params = [start_date, end_date]
        
        # Handle multiple tag filters (takes precedence over single tag filter)
        if tag_filters and len(tag_filters) > 0:
            tag_conditions = []
            for i, tag in enumerate(tag_filters):
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
    tag_filter: Optional[str] = Query(
        default=None,
        description="Filter by specific tag (optional)",
    ),
    tag_filters: Optional[List[str]] = Query(
        default=None,
        description="Filter by multiple specific tags (optional, takes precedence over tag_filter)",
    ),
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Get Monthly Active Users (MAU) by tags for the last {MAX_MONTHS} months ending on UTC today + 1 day.
    
    Shows month-by-month breakdown:
    - Month 1 (Nov): Earliest month (7 months ago, 30-day period)
    - Month 2 (Dec): Next month (6 months ago)
    - Month 3 (Jan): Next month (5 months ago)
    - ... and so on for {MAX_MONTHS} months total
    - Month 7: Most recent month ending on UTC today + 1 day
    
    Args:
        tag_filter: Optional filter to specific tag (legacy)
        tag_filters: Optional filter to multiple specific tags (takes precedence over tag_filter)
        
    Returns:
        ActiveUsersAnalyticsResponse: MAU data by tag for each of the last {MAX_MONTHS} months with descriptive month labels (e.g., "Month 1 (Nov)")
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
        
        # Calculate date range for all months (210 days total)
        # Start from 209 days before end_date to cover exactly MAX_MONTHS complete months
        start_dt = end_dt - timedelta(days=(MAX_MONTHS * 30 - 1))  # MAX_MONTHS months * 30 days - 1
        start_date = start_dt.strftime("%Y-%m-%d")
        
        # Build SQL query with optional tag filter(s)
        where_clause = "WHERE dts.date >= $1 AND dts.date <= $2 AND vt.user_id IS NOT NULL"
        params = [start_date, end_date]
        
        # Handle multiple tag filters (takes precedence over single tag filter)
        if tag_filters and len(tag_filters) > 0:
            tag_conditions = []
            for i, tag in enumerate(tag_filters):
                param_index = len(params) + 1
                tag_conditions.append(f"dts.tag = ${param_index}")
                params.append(tag)
            where_clause += f" AND ({' OR '.join(tag_conditions)})"
        elif tag_filter:
            where_clause += " AND dts.tag ILIKE $3"
            params.append(f"%{tag_filter}%")
        
        # Use window function to group by months (30-day periods) with clear month numbering
        sql_query = f"""
        WITH monthly_data AS (
            SELECT 
                dts.tag,
                dts.date,
                vt.user_id,
                -- Calculate month number (0 = Month 1 most recent, 1 = Month 2, etc.)
                FLOOR((DATE '{end_date}' - dts.date::date) / 30) as month_offset
            FROM "LiteLLM_DailyTagSpend" dts
            INNER JOIN "LiteLLM_VerificationToken" vt ON dts.api_key = vt.token
            {where_clause}
        )
        SELECT 
            tag,
            COUNT(DISTINCT user_id) as active_users,
            -- Month identifier with month name (Month 1 (earliest), Month 2, etc.)
            'Month ' || ({MAX_MONTHS} - month_offset)::text || ' (' || 
            TO_CHAR(DATE '{end_date}' - (month_offset * 30 || ' days')::interval - '29 days'::interval, 'Mon') || ')' as date,
            -- Calculate month start and end dates for each month
            (DATE '{end_date}' - (month_offset * 30 || ' days')::interval - '29 days'::interval)::text as period_start,
            (DATE '{end_date}' - (month_offset * 30 || ' days')::interval)::text as period_end,
            month_offset
        FROM monthly_data
        WHERE month_offset < {MAX_MONTHS}
        GROUP BY tag, month_offset
        ORDER BY month_offset DESC, active_users DESC
        """
        
        db_response = await prisma_client.db.query_raw(sql_query, *params)
        
        results = [
            TagActiveUsersResponse(
                tag=row["tag"],
                active_users=row["active_users"],
                date=row["date"],  # This will be "Month 1 (Jan)", "Month 2 (Dec)", etc.
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
        
        # Build SQL query with optional tag filter(s)
        where_clause = "WHERE dts.date >= $1 AND dts.date <= $2"
        params = [start_date, end_date]
        
        # Handle multiple tag filters (takes precedence over single tag filter)
        if tag_filters and len(tag_filters) > 0:
            tag_conditions = []
            for i, tag in enumerate(tag_filters):
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