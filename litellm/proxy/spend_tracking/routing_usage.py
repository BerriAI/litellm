from typing import Final, Literal

from pydantic import BaseModel, ConfigDict


class RoutingUsageRow(BaseModel):
    model_config = ConfigDict(frozen=True)

    attribution: Literal["direct", "router", "unattributed"]
    router_name: str | None
    model: str
    provider: str
    requests: int
    failed_attempts: int
    cache_hits: int
    inference_spend: float
    classifier_spend: float
    classifier_cost_known_requests: int


class RoutingUsageResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    results: tuple[RoutingUsageRow, ...]
    start_date: str
    end_date: str
    spend_logs_disabled: bool
    configured_retention: str | None
    coverage: Literal["retained_spend_logs"] = "retained_spend_logs"


ROUTING_USAGE_SQL: Final = """
WITH scoped AS (
    SELECT request_id, litellm_call_id, model, custom_llm_provider,
           status, cache_hit, spend, metadata, "startTime"
    FROM "LiteLLM_SpendLogs"
    WHERE "startTime" >= ($1::timestamptz AT TIME ZONE 'UTC')
      AND "startTime" < ($2::timestamptz AT TIME ZONE 'UTC')
      AND ($3::boolean OR "user" = $4::text OR team_id = ANY($5::text[]))
      AND ($6::text[] IS NULL OR team_id = ANY($6::text[]))
      AND ($7::text IS NULL OR "user" = $7)
      AND ($8::text IS NULL OR api_key = $8)
      AND api_key != ALL($11::text[])
      AND NULLIF(metadata->>'internal_call_origin', '') IS NULL
      AND call_type NOT IN ('call_mcp_tool', 'list_mcp_tools', 'asend_message',
                            'acreate_batch', 'create_batch', 'aretrieve_batch', 'retrieve_batch')
), attributed AS (
    SELECT *,
        COALESCE(
            CASE WHEN metadata->'routing_origin'->>'kind' = 'router'
                 THEN NULLIF(metadata->'routing_origin'->>'router_name', '') END,
            NULLIF(metadata->'router_metadata'->>'requested_model', ''),
            NULLIF(metadata->'routing_decision'->>'router_model_name', '')
        ) AS router_name,
        CASE WHEN jsonb_typeof(metadata->'routing_decision'->'classifier_cost') = 'number'
             THEN CASE WHEN (metadata->'routing_decision'->>'classifier_cost')::numeric >= 0
                       THEN (metadata->'routing_decision'->>'classifier_cost')::numeric END END AS classifier_cost
    FROM scoped
), ranked AS (
    SELECT *, ROW_NUMBER() OVER (
        PARTITION BY COALESCE(NULLIF(metadata->'routing_decision'->>'decision_id', ''),
                             NULLIF(litellm_call_id, ''),
                             NULLIF(metadata->>'litellm_call_id', ''), request_id), router_name
        ORDER BY classifier_cost DESC NULLS LAST, "startTime", request_id
    ) AS classifier_rank
    FROM attributed
)
SELECT CASE WHEN router_name IS NOT NULL THEN 'router'
            WHEN metadata->'routing_origin'->>'kind' = 'direct' THEN 'direct'
            ELSE 'unattributed' END AS attribution,
       router_name,
       COALESCE(NULLIF(model, ''), 'Unknown model') AS model,
       COALESCE(custom_llm_provider, '') AS provider,
       COUNT(*) FILTER (WHERE status = 'success' OR status IS NULL)::integer AS requests,
       COUNT(*) FILTER (WHERE status IS NOT NULL AND status != 'success')::integer AS failed_attempts,
       COUNT(*) FILTER (WHERE (status = 'success' OR status IS NULL)
                          AND LOWER(cache_hit) = 'true')::integer AS cache_hits,
       COALESCE(SUM(spend), 0)::double precision AS inference_spend,
       COALESCE(SUM(classifier_cost) FILTER (WHERE classifier_rank = 1 AND router_name IS NOT NULL), 0)
           ::double precision AS classifier_spend,
       COUNT(*) FILTER (WHERE (status = 'success' OR status IS NULL)
                          AND router_name IS NOT NULL AND classifier_cost IS NOT NULL)
           ::integer AS classifier_cost_known_requests
FROM ranked
WHERE ($9::text IS NULL OR router_name = $9)
  AND ($10::text IS NULL OR model = $10)
GROUP BY attribution, router_name, COALESCE(NULLIF(model, ''), 'Unknown model'),
         COALESCE(custom_llm_provider, '')
ORDER BY inference_spend DESC, attribution, router_name, model, provider
"""
