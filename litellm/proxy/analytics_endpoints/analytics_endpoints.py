#### Analytics Endpoints #####
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import fastapi
from fastapi import APIRouter, Depends, Header, HTTPException, Request, status

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import *
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

router = APIRouter()


@router.get(
    "/global/activity/cache_hits",
    tags=["Budget & Spend Tracking"],
    dependencies=[Depends(user_api_key_auth)],
    responses={
        200: {"model": List[LiteLLM_SpendLogs]},
    },
    include_in_schema=False,
)
async def get_global_activity(
    start_date: Optional[str] = fastapi.Query(
        default=None,
        description="Time from which to start viewing spend",
    ),
    end_date: Optional[str] = fastapi.Query(
        default=None,
        description="Time till which to view spend",
    ),
):
    """
    Get number of cache hits, vs misses

    {
        "daily_data": [
                const chartdata = [
                {
                    date: 'Jan 22',
                    cache_hits: 10,
                    llm_api_calls: 2000
                },
                {
                    date: 'Jan 23',
                    cache_hits: 10,
                    llm_api_calls: 12
                },
        ],
        "sum_cache_hits": 20,
        "sum_llm_api_calls": 2012
    }
    """
    from collections import defaultdict

    if start_date is None or end_date is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "Please provide start_date and end_date"},
        )

    start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
    end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")

    from litellm.proxy.proxy_server import llm_router, prisma_client

    try:
        if prisma_client is None:
            raise ValueError(
                f"Database not connected. Connect a database to your proxy - https://docs.litellm.ai/docs/simple_proxy#managing-auth---virtual-keys"
            )

        sql_query = """
            SELECT
                CASE 
                    WHEN vt."key_alias" IS NOT NULL THEN vt."key_alias"
                    ELSE 'Unnamed Key'
                END AS api_key,
                sl."call_type",
                sl."model",
                COUNT(*) AS total_rows,
                SUM(CASE WHEN sl."cache_hit" = 'True' THEN 1 ELSE 0 END) AS cache_hit_true_rows,
                SUM(CASE WHEN sl."cache_hit" = 'True' THEN sl."completion_tokens" ELSE 0 END) AS cached_completion_tokens,
                SUM(CASE WHEN sl."cache_hit" != 'True' THEN sl."completion_tokens" ELSE 0 END) AS generated_completion_tokens
            FROM "LiteLLM_SpendLogs" sl
            LEFT JOIN "LiteLLM_VerificationToken" vt ON sl."api_key" = vt."token"
            WHERE 
                sl."startTime" BETWEEN $1::date AND $2::date + interval '1 day'
            GROUP BY 
                vt."key_alias",
                sl."call_type",
                sl."model"
        """
        db_response = await prisma_client.db.query_raw(
            sql_query, start_date_obj, end_date_obj
        )

        if db_response is None:
            return []

        return db_response

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": str(e)},
        )
