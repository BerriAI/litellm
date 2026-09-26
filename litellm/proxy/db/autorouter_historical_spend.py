import json
import sys
from contextlib import AbstractAsyncContextManager
from datetime import timedelta
from typing import TYPE_CHECKING, Final, Protocol, cast

from litellm.constants import MAX_SPENDLOG_ROWS_TO_QUERY
from litellm.proxy.db.autorouter_daily_spend import DAILY_COSTS_COMPLETE_SQL, AutoRouterDailyCosts
from litellm.proxy.db.create_views import SupportsRawQueries

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient


class _ReadTransactions(Protocol):
    def tx(self, *, timeout: timedelta, max_wait: timedelta) -> AbstractAsyncContextManager[SupportsRawQueries]: ...


async def recover_daily_router_costs(
    prisma_client: "PrismaClient", start_date: str, end_date: str, api_key: str | None, user_id: str | None
) -> AutoRouterDailyCosts | None:
    from litellm.proxy.route_llm_request import ROUTE_ENDPOINT_MAPPING

    reader: Final = cast(_ReadTransactions, prisma_client.read_db)  # cast-ok: untyped Prisma tx delegate
    async with reader.tx(timeout=timedelta(seconds=3), max_wait=timedelta(seconds=1)) as transaction:
        await transaction.execute_raw("SET TRANSACTION READ ONLY")
        await transaction.execute_raw("SET LOCAL statement_timeout = 2000")
        rows: Final = await transaction.query_raw(
            AUTOROUTER_HISTORICAL_COSTS_SQL, start_date, end_date, api_key, user_id, json.dumps(ROUTE_ENDPOINT_MAPPING)
        )
        return AutoRouterDailyCosts.model_validate(rows[0]) if rows else None


AUTOROUTER_HISTORICAL_COSTS_SQL: Final = f"""
WITH daily AS MATERIALIZED (
    SELECT
        date, COALESCE(user_id, '') AS user_id, api_key,
        COALESCE(model, '') AS model,
        COALESCE(custom_llm_provider, '') AS custom_llm_provider,
        COALESCE(mcp_namespaced_tool_name, '') AS mcp_namespaced_tool_name,
        COALESCE(endpoint, '') AS endpoint,
        SUM(api_requests)::bigint AS api_requests,
        SUM(successful_requests)::bigint AS successful_requests,
        SUM(failed_requests)::bigint AS failed_requests,
        SUM(prompt_tokens)::bigint AS prompt_tokens,
        SUM(completion_tokens)::bigint AS completion_tokens,
        SUM(spend)::float8 AS spend,
        SUM(autorouter_requests)::bigint AS requests,
        SUM(autorouter_llm_spend)::float8 AS llm_spend,
        SUM(autorouter_classifier_cost)::float8 AS classifier_cost,
        SUM(autorouter_classifier_cost_recorded_requests)::bigint AS classifier_requests,
        SUM(autorouter_estimated_requests)::bigint AS estimated_requests,
        SUM(autorouter_estimated_actual_spend)::float8 AS estimated_actual_spend,
        SUM(autorouter_savings_spend)::float8 AS saved_spend,
        BOOL_AND({DAILY_COSTS_COMPLETE_SQL}) AS tracked_complete
    FROM "LiteLLM_DailyUserSpend"
    WHERE date >= $1::text AND date <= $2::text
      AND ($3::text IS NULL OR api_key = $3::text)
      AND ($4::text IS NULL OR user_id = $4::text)
    GROUP BY 1, 2, 3, 4, 5, 6, 7
), limited_logs AS MATERIALIZED (
    SELECT
        to_char(logs."startTime", 'YYYY-MM-DD') AS date,
        COALESCE(logs."user", '') AS user_id, logs.api_key,
        COALESCE(logs.model, '') AS model,
        COALESCE(logs.custom_llm_provider, '') AS custom_llm_provider,
        COALESCE(logs.mcp_namespaced_tool_name, '') AS mcp_namespaced_tool_name,
        COALESCE($5::jsonb ->> logs.call_type, '') AS endpoint,
        logs.spend, logs.prompt_tokens, logs.completion_tokens, logs.status,
        jsonb_build_object(
            'internal_call_origin', logs.metadata::jsonb -> 'internal_call_origin',
            'status', logs.metadata::jsonb -> 'status',
            'routing_decision', logs.metadata::jsonb -> 'routing_decision',
            'autorouter_savings', logs.metadata::jsonb -> 'autorouter_savings',
            'autorouter_savings_estimate', logs.metadata::jsonb -> 'autorouter_savings_estimate'
        ) AS metadata
    FROM "LiteLLM_SpendLogs" AS logs
    WHERE logs."startTime" >= $1::text::timestamp
      AND logs."startTime" < $2::text::timestamp + INTERVAL '1 day'
      AND ($3::text IS NULL OR logs.api_key = $3::text)
      AND ($4::text IS NULL OR logs."user" = $4::text)
      AND EXISTS (
          SELECT 1 FROM daily
          WHERE NOT daily.tracked_complete
            AND daily.date = to_char(logs."startTime", 'YYYY-MM-DD')
            AND daily.user_id = COALESCE(logs."user", '')
            AND daily.api_key = logs.api_key
            AND daily.model = COALESCE(logs.model, '')
            AND daily.custom_llm_provider = COALESCE(logs.custom_llm_provider, '')
            AND daily.mcp_namespaced_tool_name = COALESCE(logs.mcp_namespaced_tool_name, '')
            AND daily.endpoint = COALESCE($5::jsonb ->> logs.call_type, '')
      )
    LIMIT {MAX_SPENDLOG_ROWS_TO_QUERY + 1}
), log_facts AS (
    SELECT *,
        COALESCE(metadata ->> 'internal_call_origin', '') = '' AS external,
        COALESCE(metadata ->> 'status', '') <> 'failure' AS successful,
        COALESCE(metadata ->> 'internal_call_origin', '') = ''
            AND status = 'success'
            AND jsonb_typeof(metadata -> 'routing_decision') = 'object'
            AND metadata -> 'routing_decision' <> '{{}}'::jsonb AS routed,
        CASE WHEN jsonb_typeof(metadata #> '{{routing_decision,classifier_cost}}') = 'number'
            THEN CASE WHEN (metadata #>> '{{routing_decision,classifier_cost}}')::numeric
                BETWEEN -{sys.float_info.max} AND {sys.float_info.max}
                THEN (metadata #>> '{{routing_decision,classifier_cost}}')::float8 END END AS classifier,
        CASE WHEN jsonb_typeof(metadata -> 'autorouter_savings') = 'number'
            THEN CASE WHEN (metadata ->> 'autorouter_savings')::numeric
                BETWEEN -{sys.float_info.max} AND {sys.float_info.max}
                THEN (metadata ->> 'autorouter_savings')::float8 END END AS recorded_savings,
        metadata -> 'autorouter_savings_estimate' AS estimate
    FROM limited_logs
), classified_logs AS (
    SELECT *,
        recorded_savings IS NOT NULL AND (
            estimate IS NULL OR estimate = 'null'::jsonb OR (
                jsonb_typeof(estimate -> 'version') = 'number'
                AND estimate ->> 'version' IN ('1', '2', '3')
                AND estimate ->> 'status' = 'estimated'
            )
        ) AS estimated,
        jsonb_typeof(estimate -> 'version') = 'number'
            AND estimate ->> 'version' IN ('1', '2', '3')
            AND estimate ->> 'status' = 'unknown' AS unknown
    FROM log_facts
), log_totals AS (
    SELECT date, user_id, api_key, model, custom_llm_provider, mcp_namespaced_tool_name, endpoint,
        COUNT(*) FILTER (WHERE external)::bigint AS api_requests,
        COUNT(*) FILTER (WHERE external AND successful)::bigint AS successful_requests,
        COUNT(*) FILTER (WHERE external AND NOT successful)::bigint AS failed_requests,
        SUM(prompt_tokens)::bigint AS prompt_tokens,
        SUM(completion_tokens)::bigint AS completion_tokens,
        SUM(spend)::float8 AS spend,
        COUNT(*) FILTER (WHERE routed)::bigint AS requests,
        COALESCE(SUM(spend) FILTER (WHERE routed), 0)::float8 AS llm_spend,
        COALESCE(SUM(classifier) FILTER (WHERE routed), 0)::float8 AS classifier_cost,
        COUNT(*) FILTER (WHERE routed AND classifier IS NOT NULL)::bigint AS classifier_requests,
        COUNT(*) FILTER (WHERE routed AND estimated)::bigint AS estimated_requests,
        COALESCE(SUM(spend + COALESCE(classifier, 0)) FILTER (WHERE routed AND estimated), 0)::float8
            AS estimated_actual_spend,
        COALESCE(SUM(recorded_savings) FILTER (WHERE routed AND estimated), 0)::float8 AS saved_spend,
        COALESCE(BOOL_AND(COALESCE(estimated OR unknown, FALSE)) FILTER (WHERE routed), TRUE)
            AS comparison_complete
    FROM classified_logs
    GROUP BY 1, 2, 3, 4, 5, 6, 7
), reconciled AS (
    SELECT daily.*, logs.requests AS recovered_requests, logs.llm_spend AS recovered_llm_spend,
        logs.classifier_cost AS recovered_classifier_cost,
        logs.classifier_requests AS recovered_classifier_requests,
        logs.estimated_requests AS recovered_estimated_requests,
        logs.estimated_actual_spend AS recovered_estimated_actual_spend,
        COALESCE(logs.comparison_complete
            AND ABS(logs.saved_spend - daily.saved_spend)
                <= GREATEST(1e-9, ABS(daily.saved_spend) * 1e-9), FALSE) AS recovered_comparison_complete,
        COALESCE((SELECT COUNT(*) FROM limited_logs) <= {MAX_SPENDLOG_ROWS_TO_QUERY}
            AND logs.api_requests = daily.api_requests
            AND logs.successful_requests = daily.successful_requests
            AND logs.failed_requests = daily.failed_requests
            AND logs.prompt_tokens = daily.prompt_tokens
            AND logs.completion_tokens = daily.completion_tokens
            AND ABS(logs.spend - daily.spend) <= GREATEST(1e-9, ABS(daily.spend) * 1e-9)
            AND (daily.saved_spend = 0 OR logs.requests > 0), FALSE) AS recovered
    FROM daily LEFT JOIN log_totals AS logs
        USING (date, user_id, api_key, model, custom_llm_provider, mcp_namespaced_tool_name, endpoint)
)
SELECT
    COALESCE(SUM(CASE WHEN recovered THEN recovered_requests ELSE requests END), 0)::bigint AS requests,
    COALESCE(SUM(CASE WHEN recovered THEN recovered_llm_spend ELSE llm_spend END), 0)::float8 AS llm_spend,
    COALESCE(SUM(CASE WHEN recovered THEN recovered_classifier_cost ELSE classifier_cost END), 0)::float8
        AS classifier_cost,
    COALESCE(SUM(CASE WHEN recovered THEN recovered_classifier_requests ELSE classifier_requests END), 0)::bigint
        AS classifier_requests,
    COALESCE(SUM(CASE WHEN recovered THEN recovered_estimated_requests ELSE estimated_requests END), 0)::bigint
        AS estimated_requests,
    COALESCE(SUM(CASE WHEN recovered THEN recovered_estimated_actual_spend ELSE estimated_actual_spend END), 0)::float8
        AS estimated_actual_spend,
    COALESCE(SUM(saved_spend), 0)::float8 AS saved_spend,
    COALESCE(BOOL_AND(tracked_complete OR recovered), TRUE) AS complete,
    COALESCE(BOOL_AND(tracked_complete OR (recovered AND recovered_comparison_complete)), TRUE)
        AS comparison_complete
FROM reconciled
"""
