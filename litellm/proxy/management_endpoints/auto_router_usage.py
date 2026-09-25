from collections.abc import Mapping, Sequence
from datetime import date, datetime, time, timedelta
from typing import Annotated, Final, Protocol

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, TypeAdapter

from litellm.constants import INTERNAL_CALL_ORIGIN_METADATA_KEY
from litellm.proxy._types import UserAPIKeyAuth, user_api_key_has_admin_view
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.management_endpoints.common_utils import require_caller_user_id_for_non_admin

router: Final = APIRouter()


class AutoRouterUsage(BaseModel):
    model: str
    router_name: str | None
    router_type: str | None
    tier: str | None
    requests: int
    spend: float


class RoutingUsageDatabase(Protocol):
    async def query_raw(self, query: str, *args: object) -> Sequence[Mapping[str, object]]: ...


def routing_usage_database() -> RoutingUsageDatabase:
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=503, detail="Database is not connected")
    return prisma_client.db  # pyright: ignore[reportReturnType]  # wrapper forwards query_raw through __getattr__


ROUTING_USAGE_SQL: Final = f"""
WITH requests AS (
    SELECT model, spend,
        CASE WHEN jsonb_typeof(metadata->'routing_decision') = 'object'
            AND metadata->'routing_decision' <> '{{}}'::jsonb
            THEN COALESCE(NULLIF(metadata#>>'{{routing_decision,router_model_name}}', ''), NULLIF(model_group, ''))
        END AS router_name,
        NULLIF(metadata#>>'{{routing_decision,router_type}}', '') AS router_type,
        NULLIF(metadata#>>'{{routing_decision,tier}}', '') AS tier
    FROM "LiteLLM_SpendLogs"
    WHERE "startTime" >= $1::timestamp AND "startTime" < $2::timestamp
        AND ($3::text IS NULL OR "user" = $3)
        AND ($4::text IS NULL OR api_key = $4)
        AND NULLIF(metadata->>'{INTERNAL_CALL_ORIGIN_METADATA_KEY}', '') IS NULL
        AND ($7::text IS NULL OR model = $7)
)
SELECT model, router_name, router_type, tier, COUNT(*)::int AS requests,
    COALESCE(SUM(spend), 0)::float8 AS spend
FROM requests
WHERE ($5::text IS NULL OR router_name = $5)
    AND ($6::text IS NULL OR router_type = $6)
GROUP BY model, router_name, router_type, tier
ORDER BY spend DESC, model, router_name, tier
"""

_USAGE_ROWS: Final = TypeAdapter(tuple[AutoRouterUsage, ...])


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

    Select one model or one router. Internal classifier and shadow-evaluation calls
    are excluded. Non-admins can only see requests attributed to their own user.
    """
    if (destination_model is None) == (router_name is None):
        raise HTTPException(status_code=400, detail="Select exactly one model or router")
    if end_date < start_date or end_date == date.max:
        raise HTTPException(status_code=400, detail="Invalid date range")
    if router_type is not None and router_name is None:
        raise HTTPException(status_code=400, detail="router_type requires router_name")
    scoped_user: Final = (
        user_id
        if user_api_key_has_admin_view(user_api_key_dict)
        else require_caller_user_id_for_non_admin(user_api_key_dict)
    )
    if user_id is not None and user_id != scoped_user:
        raise HTTPException(status_code=403, detail="Cannot view another user's routing usage")
    rows: Final = await db.query_raw(
        ROUTING_USAGE_SQL,
        datetime.combine(start_date, time.min).isoformat(),
        datetime.combine(end_date + timedelta(days=1), time.min).isoformat(),
        scoped_user,
        api_key,
        router_name,
        router_type,
        destination_model,
    )
    return _USAGE_ROWS.validate_python(rows)
