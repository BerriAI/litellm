import asyncio
import json
from datetime import datetime
from typing import TYPE_CHECKING, Sequence

from pydantic import BaseModel, TypeAdapter

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient

UNKNOWN_CALL_TYPE = "Unknown"


class CacheActivityGroup(BaseModel):
    call_type: str
    api_requests: int
    cache_hits: int
    failed_requests: int
    cached_completion_tokens: int
    generated_completion_tokens: int


class CacheActivityTotals(BaseModel):
    api_requests: int
    cache_hits: int
    failed_requests: int
    cached_completion_tokens: int
    cache_hit_ratio: float


class CacheActivityFilterOptions(BaseModel):
    key_aliases: list[str]
    models: list[str]


class CacheActivityResponse(BaseModel):
    groups: list[CacheActivityGroup]
    totals: CacheActivityTotals
    filter_options: CacheActivityFilterOptions


GROUPS_SQL = """
    SELECT
        CASE WHEN sl."call_type" = '' THEN 'Unknown' ELSE sl."call_type" END AS call_type,
        (COUNT(*)
            - SUM(CASE WHEN COALESCE(sl."cache_hit", '') = 'True' THEN 1 ELSE 0 END)
            - SUM(CASE WHEN sl."status" = 'failure' THEN 1 ELSE 0 END))::int AS api_requests,
        SUM(CASE WHEN COALESCE(sl."cache_hit", '') = 'True' THEN 1 ELSE 0 END)::int AS cache_hits,
        SUM(CASE WHEN sl."status" = 'failure' THEN 1 ELSE 0 END)::int AS failed_requests,
        SUM(CASE WHEN COALESCE(sl."cache_hit", '') = 'True' THEN sl."completion_tokens" ELSE 0 END)::int
            AS cached_completion_tokens,
        SUM(CASE WHEN COALESCE(sl."cache_hit", '') != 'True' THEN sl."completion_tokens" ELSE 0 END)::int
            AS generated_completion_tokens
    FROM "LiteLLM_SpendLogs" sl
    LEFT JOIN "LiteLLM_VerificationToken" vt ON sl."api_key" = vt."token"
    WHERE
        sl."startTime" >= ($1::timestamptz AT TIME ZONE 'UTC')
        AND sl."startTime" <  (($2::timestamptz + INTERVAL '1 day') AT TIME ZONE 'UTC')
        AND ($3::jsonb = '[]'::jsonb
            OR COALESCE(vt."key_alias", 'Unnamed Key') IN (SELECT jsonb_array_elements_text($3::jsonb)))
        AND ($4::jsonb = '[]'::jsonb
            OR sl."model" IN (SELECT jsonb_array_elements_text($4::jsonb)))
    GROUP BY 1
    ORDER BY (COUNT(*)) DESC
"""

KEY_ALIAS_OPTIONS_SQL = """
    SELECT DISTINCT COALESCE(vt."key_alias", 'Unnamed Key') AS key_alias
    FROM "LiteLLM_SpendLogs" sl
    LEFT JOIN "LiteLLM_VerificationToken" vt ON sl."api_key" = vt."token"
    WHERE
        sl."startTime" >= ($1::timestamptz AT TIME ZONE 'UTC')
        AND sl."startTime" <  (($2::timestamptz + INTERVAL '1 day') AT TIME ZONE 'UTC')
    ORDER BY 1
"""

MODEL_OPTIONS_SQL = """
    SELECT DISTINCT sl."model" AS model
    FROM "LiteLLM_SpendLogs" sl
    WHERE
        sl."startTime" >= ($1::timestamptz AT TIME ZONE 'UTC')
        AND sl."startTime" <  (($2::timestamptz + INTERVAL '1 day') AT TIME ZONE 'UTC')
        AND sl."model" != ''
    ORDER BY 1
"""


class _KeyAliasRow(BaseModel):
    key_alias: str


class _ModelRow(BaseModel):
    model: str


_groups_adapter = TypeAdapter(list[CacheActivityGroup])
_key_alias_rows_adapter = TypeAdapter(list[_KeyAliasRow])
_model_rows_adapter = TypeAdapter(list[_ModelRow])


def compute_totals(groups: Sequence[CacheActivityGroup]) -> CacheActivityTotals:
    api_requests = sum(group.api_requests for group in groups)
    cache_hits = sum(group.cache_hits for group in groups)
    failed_requests = sum(group.failed_requests for group in groups)
    all_requests = api_requests + cache_hits + failed_requests
    return CacheActivityTotals(
        api_requests=api_requests,
        cache_hits=cache_hits,
        failed_requests=failed_requests,
        cached_completion_tokens=sum(group.cached_completion_tokens for group in groups),
        cache_hit_ratio=(cache_hits / all_requests) * 100 if all_requests > 0 else 0.0,
    )


async def get_cache_activity(
    prisma_client: "PrismaClient",
    start_date: datetime,
    end_date: datetime,
    key_aliases: Sequence[str],
    models: Sequence[str],
) -> CacheActivityResponse:
    group_rows, key_alias_rows, model_rows = await asyncio.gather(
        prisma_client.db.query_raw(
            GROUPS_SQL, start_date, end_date, json.dumps(list(key_aliases)), json.dumps(list(models))
        ),
        prisma_client.db.query_raw(KEY_ALIAS_OPTIONS_SQL, start_date, end_date),
        prisma_client.db.query_raw(MODEL_OPTIONS_SQL, start_date, end_date),
    )
    groups = _groups_adapter.validate_python(group_rows or [])
    return CacheActivityResponse(
        groups=groups,
        totals=compute_totals(groups),
        filter_options=CacheActivityFilterOptions(
            key_aliases=[row.key_alias for row in _key_alias_rows_adapter.validate_python(key_alias_rows or [])],
            models=[row.model for row in _model_rows_adapter.validate_python(model_rows or [])],
        ),
    )
