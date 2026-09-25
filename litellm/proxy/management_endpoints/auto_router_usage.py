from datetime import date, datetime, time, timedelta
from typing import Annotated, Final, Protocol

from fastapi import APIRouter, Depends, HTTPException, Query

from litellm.proxy._types import UserAPIKeyAuth, user_api_key_has_admin_view
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.management_endpoints.common_utils import require_caller_user_id_for_non_admin
from litellm.types.management_endpoints.auto_router_endpoints import AutoRouterUsage

router: Final = APIRouter()


class RoutingUsageDatabase(Protocol):
    async def get_auto_router_usage(
        self,
        *,
        start_time: datetime,
        end_time: datetime,
        user_id: str | None = None,
        api_key: str | None = None,
        router_name: str | None = None,
        router_type: str | None = None,
        destination_model: str | None = None,
    ) -> tuple[AutoRouterUsage, ...]: ...


def routing_usage_database() -> RoutingUsageDatabase:
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=503, detail="Database is not connected")
    return prisma_client


MAX_ROUTING_USAGE_DAYS: Final = 93


@router.get(
    "/auto_router/usage",
    tags=["auto router"],  # mutable-ok: FastAPI's decorator requires a list
    response_model=tuple[AutoRouterUsage, ...],
)
async def get_auto_router_usage(
    start_date: date,
    end_date: date,
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
    db: Annotated[RoutingUsageDatabase, Depends(routing_usage_database)],
    destination_model: Annotated[str | None, Query(min_length=1)] = None,
    router_name: Annotated[str | None, Query(min_length=1)] = None,
    router_type: Annotated[str | None, Query(min_length=1)] = None,
    user_id: Annotated[str | None, Query(min_length=1)] = None,
    api_key: Annotated[str | None, Query(min_length=1)] = None,
) -> tuple[AutoRouterUsage, ...]:
    """Requests and destination-model spend from retained logs in inclusive UTC days.

    Select one model or one router over at most 93 inclusive UTC days.
    Internal classifier and shadow-evaluation calls
    are excluded. Non-admins can only see requests attributed to their own user.
    """
    if (destination_model is None) == (router_name is None):
        raise HTTPException(status_code=400, detail="Select exactly one model or router")
    if end_date < start_date or end_date == date.max:
        raise HTTPException(status_code=400, detail="Invalid date range")
    if (end_date - start_date).days >= MAX_ROUTING_USAGE_DAYS:
        raise HTTPException(status_code=400, detail=f"Select a range of {MAX_ROUTING_USAGE_DAYS} days or fewer")
    if router_type is not None and router_name is None:
        raise HTTPException(status_code=400, detail="router_type requires router_name")
    scoped_user: Final = (
        user_id
        if user_api_key_has_admin_view(user_api_key_dict)
        else require_caller_user_id_for_non_admin(user_api_key_dict)
    )
    if user_id is not None and user_id != scoped_user:
        raise HTTPException(status_code=403, detail="Cannot view another user's routing usage")
    return await db.get_auto_router_usage(
        start_time=datetime.combine(start_date, time.min),
        end_time=datetime.combine(end_date + timedelta(days=1), time.min),
        user_id=scoped_user,
        api_key=api_key,
        router_name=router_name,
        router_type=router_type,
        destination_model=destination_model,
    )
