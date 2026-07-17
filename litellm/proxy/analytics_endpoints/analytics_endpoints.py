#### Analytics Endpoints #####
from datetime import datetime, timezone
from typing import List, Optional

import fastapi
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, TypeAdapter

from litellm.proxy._types import *
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

router = APIRouter()


class PromptCacheActivityRow(BaseModel):
    api_key: str
    model: str | None = None
    prompt_tokens: int
    cache_read_input_tokens: int
    cache_creation_input_tokens: int
    api_requests: int


_PROMPT_CACHE_ROWS_ADAPTER = TypeAdapter(list[PromptCacheActivityRow])


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

    if start_date is None or end_date is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "Please provide start_date and end_date"},
        )

    start_date_obj = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_date_obj = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    from litellm.proxy.proxy_server import prisma_client

    try:
        if prisma_client is None:
            raise ValueError(
                "Database not connected. Connect a database to your proxy - https://docs.litellm.ai/docs/simple_proxy#managing-auth---virtual-keys"
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
                sl."startTime" >= ($1::timestamptz AT TIME ZONE 'UTC')
                AND sl."startTime" <  (($2::timestamptz + INTERVAL '1 day') AT TIME ZONE 'UTC')
            GROUP BY 
                vt."key_alias",
                sl."call_type",
                sl."model"
        """
        db_response = await prisma_client.db.query_raw(sql_query, start_date_obj, end_date_obj)

        if db_response is None:
            return []

        return db_response

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": str(e)},
        )


@router.get(
    "/global/activity/cache_hits/prompt_caching",
    tags=["Budget & Spend Tracking"],
    dependencies=[Depends(user_api_key_auth)],
    responses={
        200: {"model": list[PromptCacheActivityRow]},
    },
    include_in_schema=False,
)
async def get_prompt_cache_activity(
    start_date: str | None = fastapi.Query(
        default=None,
        description="Time from which to start viewing prompt-cache usage",
    ),
    end_date: str | None = fastapi.Query(
        default=None,
        description="Time till which to view prompt-cache usage",
    ),
) -> list[PromptCacheActivityRow]:
    if start_date is None or end_date is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "Please provide start_date and end_date"},
        )

    for label, value in (("start_date", start_date), ("end_date", end_date)):
        try:
            datetime.strptime(value, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": f"{label} must be in YYYY-MM-DD format"},
            )

    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "Database not connected. Connect a database to your proxy - https://docs.litellm.ai/docs/simple_proxy#managing-auth---virtual-keys"
            },
        )

    sql_query = """
        SELECT
            CASE
                WHEN vt."key_alias" IS NOT NULL THEN vt."key_alias"
                ELSE 'Unnamed Key'
            END AS api_key,
            ds."model",
            SUM(ds."prompt_tokens")::bigint AS prompt_tokens,
            SUM(ds."cache_read_input_tokens")::bigint AS cache_read_input_tokens,
            SUM(ds."cache_creation_input_tokens")::bigint AS cache_creation_input_tokens,
            SUM(ds."api_requests")::bigint AS api_requests
        FROM "LiteLLM_DailyUserSpend" ds
        LEFT JOIN "LiteLLM_VerificationToken" vt ON ds."api_key" = vt."token"
        WHERE
            ds."date" >= $1
            AND ds."date" <= $2
        GROUP BY
            vt."key_alias",
            ds."model"
    """
    db_response = await prisma_client.db.query_raw(sql_query, start_date, end_date)

    if db_response is None:
        return []

    return _PROMPT_CACHE_ROWS_ADAPTER.validate_python(db_response)
