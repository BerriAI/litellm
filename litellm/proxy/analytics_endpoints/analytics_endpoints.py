#### Analytics Endpoints #####
from datetime import datetime, timezone
from typing import Annotated, Final

import fastapi
from fastapi import APIRouter, Depends, HTTPException, status

from litellm.proxy._types import *
from litellm.proxy.analytics_endpoints.cache_activity import CacheActivityResponse, get_cache_activity
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

router: Final = APIRouter()


def _parse_date(value: str, param_name: str) -> datetime:
    try:
        return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": f"{param_name} must be a YYYY-MM-DD date, got {value!r}"},
        )


@router.get(
    "/global/activity/cache_hits",
    tags=["Budget & Spend Tracking"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=CacheActivityResponse,
    include_in_schema=False,
)
async def get_global_activity(
    start_date: Annotated[str, fastapi.Query(description="Time from which to start viewing spend")],
    end_date: Annotated[str, fastapi.Query(description="Time till which to view spend")],
    key_aliases: Annotated[
        list[str] | None, fastapi.Query(description="Only include spend from these key aliases")
    ] = None,
    models: Annotated[list[str] | None, fastapi.Query(description="Only include spend for these models")] = None,
) -> CacheActivityResponse:
    """
    Cache activity for the Admin UI cache dashboard, aggregated per call_type:
    cache hits vs successful LLM API requests vs failed requests, plus totals
    for the stat cards and the available key-alias/model filter options.
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Database not connected. Connect a database to your proxy - https://docs.litellm.ai/docs/simple_proxy#managing-auth---virtual-keys"
            },
        )

    return await get_cache_activity(
        prisma_client=prisma_client,
        start_date=_parse_date(start_date, "start_date"),
        end_date=_parse_date(end_date, "end_date"),
        key_aliases=key_aliases or [],
        models=models or [],
    )
