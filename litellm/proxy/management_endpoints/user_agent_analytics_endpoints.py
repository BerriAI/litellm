"""
User Agent Analytics Endpoints

This module provides optimized endpoints for tracking user agent activity metrics including:
- Daily Active Users (DAU) by tags for last 7 days
- Weekly Active Users (WAU) by tags for last 7 weeks  
- Monthly Active Users (MAU) by tags for last 7 months
- Summary analytics by tags

These endpoints use optimized single SQL queries with joins to efficiently calculate
user metrics from tag activity data and return time series for dashboard visualization.
"""

from datetime import datetime, timedelta
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from litellm.proxy._types import CommonProxyErrors, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

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


@router.get(
    "/tag/dau",
    response_model=ActiveUsersAnalyticsResponse,
    tags=["tag management", "user agent analytics"],
    dependencies=[Depends(user_api_key_auth)],
)
async def get_daily_active_users(
    end_date: str = Query(
        description="End date in YYYY-MM-DD format (will show DAU for last 7 days ending on this date)"
    ),
    tag_filter: Optional[str] = Query(
        default=None,
        description="Filter by specific tag (optional)",
    ),
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Get Daily Active Users (DAU) by tags for the last 7 days ending on the specified date.
    
    This endpoint efficiently calculates unique users per tag for each of the last 7 days
    using a single optimized SQL query, perfect for dashboard time series visualization.
    
    Args:
        end_date: End date for DAU calculation (YYYY-MM-DD) - will show 7 days ending on this date
        tag_filter: Optional filter to specific tag
        
    Returns:
        ActiveUsersAnalyticsResponse: DAU data by tag for each of the last 7 days
    """
    from litellm.proxy.proxy_server import prisma_client
    
    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )
    
    try:
        # Validate and calculate date range (last 7 days)
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        start_dt = end_dt - timedelta(days=6)
        start_date = start_dt.strftime("%Y-%m-%d")
        
        # Build SQL query with optional tag filter
        where_clause = "WHERE dts.date >= $1 AND dts.date <= $2 AND vt.user_id IS NOT NULL"
        params = [start_date, end_date]
        
        if tag_filter:
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
    end_date: str = Query(
        description="End date in YYYY-MM-DD format (will show WAU for last 7 weeks ending on this date)"
    ),
    tag_filter: Optional[str] = Query(
        default=None,
        description="Filter by specific tag (optional)",
    ),
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Get Weekly Active Users (WAU) by tags for the last 7 weeks ending on the specified date.
    
    Each week is a 7-day period ending on the same weekday as the end_date.
    
    Args:
        end_date: End date for WAU calculation (YYYY-MM-DD) - will show 7 weeks ending on this date
        tag_filter: Optional filter to specific tag
        
    Returns:
        ActiveUsersAnalyticsResponse: WAU data by tag for each of the last 7 weeks
    """
    from litellm.proxy.proxy_server import prisma_client
    
    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )
    
    try:
        # Validate date format
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
        # Calculate date range for all 7 weeks (49 days total)
        start_dt = end_dt - timedelta(days=48)  # 7 weeks * 7 days - 1
        start_date = start_dt.strftime("%Y-%m-%d")
        
        # Build SQL query with optional tag filter
        where_clause = "WHERE dts.date >= $1 AND dts.date <= $2 AND vt.user_id IS NOT NULL"
        params = [start_date, end_date]
        
        if tag_filter:
            where_clause += " AND dts.tag ILIKE $3"
            params.append(f"%{tag_filter}%")
        
        # Use window function to group by weeks
        sql_query = f"""
        WITH weekly_data AS (
            SELECT 
                dts.tag,
                dts.date,
                vt.user_id,
                -- Calculate week number (0 = most recent week)
                FLOOR((DATE '{end_date}' - dts.date::date) / 7) as week_offset
            FROM "LiteLLM_DailyTagSpend" dts
            INNER JOIN "LiteLLM_VerificationToken" vt ON dts.api_key = vt.token
            {where_clause}
        )
        SELECT 
            tag,
            COUNT(DISTINCT user_id) as active_users,
            -- Calculate week end date for each week
            (DATE '{end_date}' - (week_offset * 7 || ' days')::interval)::text as date,
            (DATE '{end_date}' - (week_offset * 7 || ' days')::interval - '6 days'::interval)::text as period_start,
            (DATE '{end_date}' - (week_offset * 7 || ' days')::interval)::text as period_end
        FROM weekly_data
        WHERE week_offset < 7
        GROUP BY tag, week_offset
        ORDER BY week_offset ASC, active_users DESC
        """
        
        db_response = await prisma_client.db.query_raw(sql_query, *params)
        
        results = [
            TagActiveUsersResponse(
                tag=row["tag"],
                active_users=row["active_users"],
                date=row["date"],
                period_start=row["period_start"],
                period_end=row["period_end"]
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
            detail=f"Failed to fetch WAU analytics: {str(e)}",
        )


@router.get(
    "/tag/mau",
    response_model=ActiveUsersAnalyticsResponse,
    tags=["tag management", "user agent analytics"],
    dependencies=[Depends(user_api_key_auth)],
)
async def get_monthly_active_users(
    end_date: str = Query(
        description="End date in YYYY-MM-DD format (will show MAU for last 7 months ending on this date)"
    ),
    tag_filter: Optional[str] = Query(
        default=None,
        description="Filter by specific tag (optional)",
    ),
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Get Monthly Active Users (MAU) by tags for the last 7 months ending on the specified date.
    
    Each month is a 30-day period ending on the same day of month as the end_date.
    
    Args:
        end_date: End date for MAU calculation (YYYY-MM-DD) - will show 7 months ending on this date
        tag_filter: Optional filter to specific tag
        
    Returns:
        ActiveUsersAnalyticsResponse: MAU data by tag for each of the last 7 months
    """
    from litellm.proxy.proxy_server import prisma_client
    
    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )
    
    try:
        # Validate date format
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
        # Calculate date range for all 7 months (210 days total)
        start_dt = end_dt - timedelta(days=209)  # 7 months * 30 days - 1
        start_date = start_dt.strftime("%Y-%m-%d")
        
        # Build SQL query with optional tag filter
        where_clause = "WHERE dts.date >= $1 AND dts.date <= $2 AND vt.user_id IS NOT NULL"
        params = [start_date, end_date]
        
        if tag_filter:
            where_clause += " AND dts.tag ILIKE $3"
            params.append(f"%{tag_filter}%")
        
        # Use window function to group by months (30-day periods)
        sql_query = f"""
        WITH monthly_data AS (
            SELECT 
                dts.tag,
                dts.date,
                vt.user_id,
                -- Calculate month number (0 = most recent month)
                FLOOR((DATE '{end_date}' - dts.date::date) / 30) as month_offset
            FROM "LiteLLM_DailyTagSpend" dts
            INNER JOIN "LiteLLM_VerificationToken" vt ON dts.api_key = vt.token
            {where_clause}
        )
        SELECT 
            tag,
            COUNT(DISTINCT user_id) as active_users,
            -- Calculate month end date for each month
            (DATE '{end_date}' - (month_offset * 30 || ' days')::interval)::text as date,
            (DATE '{end_date}' - (month_offset * 30 || ' days')::interval - '29 days'::interval)::text as period_start,
            (DATE '{end_date}' - (month_offset * 30 || ' days')::interval)::text as period_end
        FROM monthly_data
        WHERE month_offset < 7
        GROUP BY tag, month_offset
        ORDER BY month_offset ASC, active_users DESC
        """
        
        db_response = await prisma_client.db.query_raw(sql_query, *params)
        
        results = [
            TagActiveUsersResponse(
                tag=row["tag"],
                active_users=row["active_users"],
                date=row["date"],
                period_start=row["period_start"],
                period_end=row["period_end"]
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
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Get summary analytics for tags including unique users, requests, tokens, and spend.
    
    Args:
        start_date: Start date for the analytics period (YYYY-MM-DD)
        end_date: End date for the analytics period (YYYY-MM-DD)
        tag_filter: Optional filter to specific tag
        
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
        
        # Build SQL query with optional tag filter
        where_clause = "WHERE dts.date >= $1 AND dts.date <= $2"
        params = [start_date, end_date]
        
        if tag_filter:
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