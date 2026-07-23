#### Analytics Endpoints #####
from datetime import datetime, timezone
from typing import List, Optional

import fastapi
from fastapi import APIRouter, Depends, HTTPException, status

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
            WITH spend_logs_with_prompt_cache AS (
                SELECT
                    sl.*,
                    COALESCE(
                        NULLIF(sl."metadata" #>> '{usage_object,cache_read_input_tokens}', '')::numeric,
                        NULLIF(sl."metadata" #>> '{usage_object,prompt_tokens_details,cached_tokens}', '')::numeric,
                        0
                    ) AS prompt_cache_read_input_tokens,
                    COALESCE(
                        NULLIF(sl."metadata" #>> '{usage_object,cache_creation_input_tokens}', '')::numeric,
                        NULLIF(sl."metadata" #>> '{usage_object,prompt_tokens_details,cache_write_tokens}', '')::numeric,
                        NULLIF(sl."metadata" #>> '{usage_object,prompt_tokens_details,cache_creation_tokens}', '')::numeric,
                        0
                    ) AS prompt_cache_creation_input_tokens
                FROM "LiteLLM_SpendLogs" sl
                WHERE
                    sl."startTime" >= ($1::timestamptz AT TIME ZONE 'UTC')
                    AND sl."startTime" <  (($2::timestamptz + INTERVAL '1 day') AT TIME ZONE 'UTC')
            )
            SELECT
                CASE
                    WHEN vt."key_alias" IS NOT NULL THEN vt."key_alias"
                    ELSE 'Unnamed Key'
                END AS api_key,
                sl."call_type",
                sl."model",
                COUNT(*) AS total_rows,
                SUM(CASE WHEN sl."cache_hit" = 'True' THEN 1 ELSE 0 END) AS cache_hit_true_rows,
                SUM(CASE WHEN sl.prompt_cache_read_input_tokens > 0 THEN 1 ELSE 0 END) AS cache_read_input_token_rows,
                SUM(CASE WHEN sl."cache_hit" = 'True' OR sl.prompt_cache_read_input_tokens > 0 THEN 1 ELSE 0 END) AS cache_activity_rows,
                SUM(CASE WHEN sl."cache_hit" = 'True' THEN sl."completion_tokens" ELSE 0 END) AS cached_completion_tokens,
                SUM(CASE WHEN sl."cache_hit" != 'True' THEN sl."completion_tokens" ELSE 0 END) AS generated_completion_tokens,
                SUM(sl.prompt_cache_read_input_tokens)::bigint AS cache_read_input_tokens,
                SUM(sl.prompt_cache_creation_input_tokens)::bigint AS cache_creation_input_tokens
            FROM spend_logs_with_prompt_cache sl
            LEFT JOIN "LiteLLM_VerificationToken" vt ON sl."api_key" = vt."token"
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
