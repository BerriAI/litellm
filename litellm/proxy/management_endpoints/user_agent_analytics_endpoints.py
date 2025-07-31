"""
User Agent Analytics Endpoints

This module provides endpoints for tracking user agent activity metrics including:
- Daily Active Users (DAU) by tags
- Weekly Active Users (WAU) by tags  
- Monthly Active Users (MAU) by tags
- Successful requests by tags
- Completed tokens by tags

These endpoints extend the existing tag daily activity functionality to provide 
user agent specific analytics by using the user-agent tags that are automatically
tracked by the system.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, cast

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from litellm.proxy._types import CommonProxyErrors, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.management_endpoints.common_daily_activity import get_daily_activity
from litellm.types.proxy.management_endpoints.common_daily_activity import (
    DailySpendData,
)

router = APIRouter()


class UserAgentMetrics(BaseModel):
    """Metrics for user agent activity"""
    dau: int = 0  # Daily Active Users
    wau: int = 0  # Weekly Active Users  
    mau: int = 0  # Monthly Active Users
    successful_requests: int = 0
    failed_requests: int = 0
    total_requests: int = 0
    completed_tokens: int = 0
    total_tokens: int = 0
    spend: float = 0.0


class UserAgentActivityData(BaseModel):
    """User agent activity data for a specific date"""
    date: str
    tag: str
    user_agent: Optional[str] = None
    metrics: UserAgentMetrics


class UserAgentAnalyticsResponse(BaseModel):
    """Response for user agent analytics"""
    results: List[UserAgentActivityData]
    total_count: int
    page: int
    page_size: int
    total_pages: int


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


async def _get_unique_users_for_tags(
    prisma_client,
    tags: List[str],
    start_date: str,
    end_date: str,
) -> Dict[str, Set[str]]:
    """
    Get unique users for each tag by looking up api_key -> user_id mappings
    """
    from litellm.proxy.proxy_server import prisma_client as db_client
    
    if not db_client:
        return {}
    
    # Get all records for the specified tags and date range
    tag_records = await db_client.db.litellm_dailytagspend.find_many(
        where={
            "tag": {"in": tags},
            "date": {"gte": start_date, "lte": end_date}
        }
    )
    
    # Get unique api_keys
    api_keys = set(record.api_key for record in tag_records if record.api_key)
    
    if not api_keys:
        return {}
    
    # Lookup user_id for each api_key
    api_key_records = await db_client.db.litellm_verificationtoken.find_many(
        where={"token": {"in": list(api_keys)}}
    )
    
    # Create mapping from api_key to user_id
    api_key_to_user_id = {
        record.token: record.user_id 
        for record in api_key_records 
        if record.user_id
    }
    
    # Group unique users by tag
    tag_users: Dict[str, Set[str]] = {}
    for record in tag_records:
        if record.api_key in api_key_to_user_id:
            user_id = api_key_to_user_id[record.api_key]
            tag = record.tag
            if tag not in tag_users:
                tag_users[tag] = set()
            tag_users[tag].add(user_id)
    
    return tag_users


async def _calculate_dau_wau_mau(
    prisma_client,
    tags: List[str],
    target_date: str,
) -> Dict[str, Dict[str, int]]:
    """
    Calculate DAU, WAU, MAU for given tags and date
    """
    target_dt = datetime.strptime(target_date, "%Y-%m-%d")
    
    # Calculate date ranges
    dau_start = target_date
    dau_end = target_date
    
    wau_start = (target_dt - timedelta(days=6)).strftime("%Y-%m-%d")
    wau_end = target_date
    
    mau_start = (target_dt - timedelta(days=29)).strftime("%Y-%m-%d")
    mau_end = target_date
    
    # Get unique users for each period
    dau_users = await _get_unique_users_for_tags(prisma_client, tags, dau_start, dau_end)
    wau_users = await _get_unique_users_for_tags(prisma_client, tags, wau_start, wau_end)
    mau_users = await _get_unique_users_for_tags(prisma_client, tags, mau_start, mau_end)
    
    result = {}
    for tag in tags:
        result[tag] = {
            "dau": len(dau_users.get(tag, set())),
            "wau": len(wau_users.get(tag, set())),
            "mau": len(mau_users.get(tag, set())),
        }
    
    return result


def _extract_user_agent_from_tag(tag: str) -> Optional[str]:
    """
    Extract user agent name from tag.
    Tags are in format "User-Agent: <agent_name>" or "User-Agent: <agent_name>/<version>"
    """
    if not tag.startswith("User-Agent: "):
        return None
    
    user_agent = tag[12:]  # Remove "User-Agent: " prefix
    
    # If it contains a version, extract just the name part
    if "/" in user_agent:
        return user_agent.split("/")[0]
    
    return user_agent


@router.get(
    "/tag/user-agent/analytics",
    response_model=UserAgentAnalyticsResponse,
    tags=["tag management", "user agent analytics"],
    dependencies=[Depends(user_api_key_auth)],
)
async def get_user_agent_analytics(
    start_date: Optional[str] = Query(
        default=None,
        description="Start date in YYYY-MM-DD format",
    ),
    end_date: Optional[str] = Query(
        default=None,
        description="End date in YYYY-MM-DD format",
    ),
    user_agent_filter: Optional[str] = Query(
        default=None,
        description="Filter by specific user agent (e.g., 'curl', 'litellm')",
    ),
    page: int = Query(default=1, description="Page number for pagination", ge=1),
    page_size: int = Query(
        default=50, description="Items per page", ge=1, le=1000
    ),
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Get user agent analytics including DAU, WAU, MAU, successful requests, and completed tokens by user agent tags.
    
    This endpoint analyzes the user-agent tags that are automatically tracked by the system
    and provides analytics broken down by user agent.
    
    Args:
        start_date: Start date for the analytics period (YYYY-MM-DD)
        end_date: End date for the analytics period (YYYY-MM-DD)
        user_agent_filter: Filter results to specific user agent name
        page: Page number for pagination
        page_size: Number of items per page
        
    Returns:
        UserAgentAnalyticsResponse: Analytics data broken down by user agent and date
    """
    from litellm.proxy.proxy_server import prisma_client
    
    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )
    
    if start_date is None or end_date is None:
        raise HTTPException(
            status_code=400,
            detail={"error": "Please provide start_date and end_date"},
        )
    
    try:
        # Get all user-agent tags from the database
        user_agent_tags_records = await prisma_client.db.litellm_dailytagspend.find_many(
            where={
                "tag": {"startswith": "User-Agent: "},
                "date": {"gte": start_date, "lte": end_date},
            },
            distinct=["tag"],
        )
        
        user_agent_tags = [record.tag for record in user_agent_tags_records]
        
        # Filter by user agent if specified
        if user_agent_filter:
            user_agent_tags = [
                tag for tag in user_agent_tags
                if user_agent_filter.lower() in tag.lower()
            ]
        
        if not user_agent_tags:
            return UserAgentAnalyticsResponse(
                results=[],
                total_count=0,
                page=page,
                page_size=page_size,
                total_pages=0,
            )
        
        # Get daily activity data for user-agent tags
        daily_activity_response = await get_daily_activity(
            prisma_client=prisma_client,
            table_name="litellm_dailytagspend",
            entity_id_field="tag",
            entity_id=user_agent_tags,
            entity_metadata_field=None,
            start_date=start_date,
            end_date=end_date,
            model=None,
            api_key=None,
            page=1,  # Get all data first, then paginate our results
            page_size=10000,  # Large page size to get all data
        )
        
        # Process the results to calculate DAU/WAU/MAU and organize by user agent
        results = []
        daily_data_by_tag_and_date: Dict[str, Dict[str, DailySpendData]] = {}
        
        # Organize data by tag and date
        for daily_data in daily_activity_response.results:
            date_str = daily_data.date.strftime("%Y-%m-%d")
            
            # Get tag from breakdown data
            for tag, tag_metrics in daily_data.breakdown.entities.items():
                if tag.startswith("User-Agent: "):
                    if tag not in daily_data_by_tag_and_date:
                        daily_data_by_tag_and_date[tag] = {}
                    daily_data_by_tag_and_date[tag][date_str] = daily_data
        
        # Calculate DAU/WAU/MAU for each date and tag combination
        unique_dates: set[str] = set()
        for tag_data in daily_data_by_tag_and_date.values():
            unique_dates.update(tag_data.keys())
        
        for tag in user_agent_tags:
            user_agent = _extract_user_agent_from_tag(tag)
            
            for date_str in sorted(unique_dates):
                if tag in daily_data_by_tag_and_date and date_str in daily_data_by_tag_and_date[tag]:
                    daily_data = daily_data_by_tag_and_date[tag][date_str]
                    tag_breakdown = daily_data.breakdown.entities.get(tag)
                    
                    if tag_breakdown:
                        # Calculate DAU/WAU/MAU for this specific date and tag
                        dau_wau_mau = await _calculate_dau_wau_mau(
                            prisma_client, [tag], date_str
                        )
                        
                        metrics = UserAgentMetrics(
                            dau=dau_wau_mau.get(tag, {}).get("dau", 0),
                            wau=dau_wau_mau.get(tag, {}).get("wau", 0),
                            mau=dau_wau_mau.get(tag, {}).get("mau", 0),
                            successful_requests=tag_breakdown.metrics.successful_requests,
                            failed_requests=tag_breakdown.metrics.failed_requests,
                            total_requests=tag_breakdown.metrics.api_requests,
                            completed_tokens=tag_breakdown.metrics.completion_tokens,
                            total_tokens=tag_breakdown.metrics.total_tokens,
                            spend=tag_breakdown.metrics.spend,
                        )
                        
                        results.append(
                            UserAgentActivityData(
                                date=date_str,
                                tag=tag,
                                user_agent=user_agent,
                                metrics=metrics,
                            )
                        )
        
        # Sort results by date (most recent first) and then by user agent
        results.sort(key=lambda x: (x.date, x.user_agent or ""), reverse=True)
        
        # Apply pagination
        total_count = len(results)
        total_pages = (total_count + page_size - 1) // page_size
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated_results = results[start_idx:end_idx]
        
        return UserAgentAnalyticsResponse(
            results=paginated_results,
            total_count=total_count,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch user agent analytics: {str(e)}",
        )


@router.get(
    "/tag/user-agent/summary",
    tags=["tag management", "user agent analytics"],
    dependencies=[Depends(user_api_key_auth)],
)
async def get_user_agent_summary(
    start_date: Optional[str] = Query(
        default=None,
        description="Start date in YYYY-MM-DD format",
    ),
    end_date: Optional[str] = Query(
        default=None,
        description="End date in YYYY-MM-DD format",
    ),
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Get summary statistics for user agent activity.
    
    Returns aggregated metrics across all user agents for the specified time period.
    """
    from litellm.proxy.proxy_server import prisma_client
    
    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )
    
    if start_date is None or end_date is None:
        raise HTTPException(
            status_code=400,
            detail={"error": "Please provide start_date and end_date"},
        )
    
    try:
        # Get all user-agent tags
        user_agent_tags_records = await prisma_client.db.litellm_dailytagspend.find_many(
            where={
                "tag": {"startswith": "User-Agent: "},
                "date": {"gte": start_date, "lte": end_date},
            },
            distinct=["tag"],
        )
        
        user_agent_tags = [record.tag for record in user_agent_tags_records]
        
        if not user_agent_tags:
            return {
                "total_user_agents": 0,
                "total_requests": 0,
                "total_successful_requests": 0,
                "total_failed_requests": 0,
                "total_tokens": 0,
                "total_spend": 0.0,
                "top_user_agents": [],
            }
        
        # Get aggregated data
        daily_activity_response = await get_daily_activity(
            prisma_client=prisma_client,
            table_name="litellm_dailytagspend",
            entity_id_field="tag",
            entity_id=user_agent_tags,
            entity_metadata_field=None,
            start_date=start_date,
            end_date=end_date,
            model=None,
            api_key=None,
            page=1,
            page_size=10000,
        )
        
        # Aggregate metrics by user agent
        user_agent_totals: Dict[str, UserAgentMetrics] = {}
        
        for daily_data in daily_activity_response.results:
            for tag, tag_metrics in daily_data.breakdown.entities.items():
                if tag.startswith("User-Agent: "):
                    user_agent = _extract_user_agent_from_tag(tag)
                    if user_agent is not None and user_agent not in user_agent_totals:
                        user_agent_totals[user_agent] = UserAgentMetrics()
                    
                    if user_agent is not None:
                        totals = user_agent_totals[user_agent]
                        totals.successful_requests += tag_metrics.metrics.successful_requests
                        totals.failed_requests += tag_metrics.metrics.failed_requests
                        totals.total_requests += tag_metrics.metrics.api_requests
                        totals.completed_tokens += tag_metrics.metrics.completion_tokens
                        totals.total_tokens += tag_metrics.metrics.total_tokens
                        totals.spend += tag_metrics.metrics.spend
        
        # Calculate summary statistics
        total_requests = sum(ua.total_requests for ua in user_agent_totals.values())
        total_successful_requests = sum(ua.successful_requests for ua in user_agent_totals.values())
        total_failed_requests = sum(ua.failed_requests for ua in user_agent_totals.values())
        total_tokens = sum(ua.total_tokens for ua in user_agent_totals.values())
        total_spend = sum(ua.spend for ua in user_agent_totals.values())
        
        # Get top user agents by request count
        top_user_agents = sorted(
            [
                {
                    "user_agent": ua,
                    "requests": metrics.total_requests,
                    "successful_requests": metrics.successful_requests,
                    "failed_requests": metrics.failed_requests,
                    "tokens": metrics.total_tokens,
                    "spend": metrics.spend,
                }
                for ua, metrics in user_agent_totals.items()
            ],
            key=lambda x: cast(int, x["requests"]),
            reverse=True,
        )[:10]  # Top 10
        
        return {
            "total_user_agents": len(user_agent_totals),
            "total_requests": total_requests,
            "total_successful_requests": total_successful_requests,
            "total_failed_requests": total_failed_requests,
            "total_tokens": total_tokens,
            "total_spend": total_spend,
            "top_user_agents": top_user_agents,
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch user agent summary: {str(e)}",
        )


@router.get(
    "/tag/user-agent/per-user-analytics",
    response_model=PerUserAnalyticsResponse,
    tags=["tag management", "user agent analytics"],
    dependencies=[Depends(user_api_key_auth)],
)
async def get_per_user_analytics(
    start_date: Optional[str] = Query(
        default=None,
        description="Start date in YYYY-MM-DD format",
    ),
    end_date: Optional[str] = Query(
        default=None,
        description="End date in YYYY-MM-DD format",
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
    user-agent activity during the specified time period.
    
    Args:
        start_date: Start date for the analytics period (YYYY-MM-DD)
        end_date: End date for the analytics period (YYYY-MM-DD)
        page: Page number for pagination
        page_size: Number of items per page
        
    Returns:
        PerUserAnalyticsResponse: Analytics data broken down by individual users
    """
    from litellm.proxy.proxy_server import prisma_client
    
    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )
    
    if start_date is None or end_date is None:
        raise HTTPException(
            status_code=400,
            detail={"error": "Please provide start_date and end_date"},
        )
    
    try:
        # Get all user-agent tags from the database
        user_agent_tags_records = await prisma_client.db.litellm_dailytagspend.find_many(
            where={
                "tag": {"startswith": "User-Agent: "},
                "date": {"gte": start_date, "lte": end_date},
            },
            distinct=["tag"],
        )
        
        user_agent_tags = [record.tag for record in user_agent_tags_records]
        
        if not user_agent_tags:
            return PerUserAnalyticsResponse(
                results=[],
                total_count=0,
                page=page,
                page_size=page_size,
                total_pages=0,
            )
        
        # Get all records for user-agent tags in the date range
        tag_records = await prisma_client.db.litellm_dailytagspend.find_many(
            where={
                "tag": {"in": user_agent_tags},
                "date": {"gte": start_date, "lte": end_date}
            }
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
                user_agent = _extract_user_agent_from_tag(record.tag)
                
                if user_id not in user_metrics:
                    user_metrics[user_id] = PerUserMetrics(
                        user_id=user_id,
                        user_email=user_id_to_email.get(user_id),
                        user_agent=user_agent
                    )
                else:
                    # If user agent is different, keep the first one or prioritize certain ones
                    if user_agent and not user_metrics[user_id].user_agent:
                        user_metrics[user_id].user_agent = user_agent
                
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