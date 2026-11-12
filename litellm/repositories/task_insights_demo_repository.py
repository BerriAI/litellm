"""Writes and removes the Task Insights sample data. Every statement is scoped to the demo id prefix or to sample
taxonomy rows, so seeding and resetting never touch anyone else's records."""

from collections.abc import Mapping, Sequence
from datetime import timedelta
from typing import Final

from litellm.litellm_core_utils.safe_json_dumps import safe_dumps
from litellm.repositories.task_insights_repository import TaskInsightsDatabase

_TIMEOUT: Final = timedelta(seconds=120)

_RESET: Final = (
    'DELETE FROM "LiteLLM_TaskAssignment" WHERE request_id LIKE $1',
    'DELETE FROM "LiteLLM_TaskMapPoint" WHERE request_id LIKE $1',
    'DELETE FROM "LiteLLM_TaskObservation" WHERE request_id LIKE $1',
    'DELETE FROM "LiteLLM_SpendLogDelivery" WHERE request_id LIKE $1',
    'DELETE FROM "LiteLLM_SpendLogs" WHERE request_id LIKE $1',
    'DELETE FROM "LiteLLM_TeamTable" WHERE team_id LIKE $1',
)
_RESET_SAMPLE_TAXONOMY: Final = (
    """DELETE FROM "LiteLLM_TaskDefinition" WHERE source = 'sample'""",
    """DELETE FROM "LiteLLM_TaskInsightsState" WHERE id = 'default'
    AND discovery_provenance->>'source' = 'sample'""",
)
_FOREIGN_TAXONOMY: Final = """SELECT EXISTS (SELECT 1 FROM "LiteLLM_TaskDefinition" WHERE source <> 'sample')
    AS found"""
_INSERT_TEAMS: Final = """INSERT INTO "LiteLLM_TeamTable" (team_id, team_alias, admins, members, models, metadata)
SELECT team_id, team_alias, '{}'::text[], '{}'::text[], '{}'::text[], '{"task_insights_sample": true}'::jsonb
FROM jsonb_to_recordset($1::jsonb) AS r(team_id text, team_alias text)
ON CONFLICT (team_id) DO NOTHING"""
_INSERT_SPEND_LOGS: Final = """INSERT INTO "LiteLLM_SpendLogs" (request_id, call_type, api_key, spend, total_tokens,
    prompt_tokens, completion_tokens, "startTime", "endTime", request_duration_ms, model, model_id, model_group,
    custom_llm_provider, team_id, session_id, status, metadata, request_tags, proxy_server_request)
SELECT request_id, call_type, api_key, COALESCE(spend, 0), total_tokens, prompt_tokens, completion_tokens,
    "startTime", "endTime", request_duration_ms, model, model_id, model_group, custom_llm_provider, team_id,
    session_id, status, metadata, request_tags, proxy_server_request
FROM jsonb_to_recordset($1::jsonb) AS r(request_id text, call_type text, api_key text, spend double precision,
    total_tokens integer, prompt_tokens integer, completion_tokens integer, "startTime" timestamp(3),
    "endTime" timestamp(3), request_duration_ms integer, model text, model_id text, model_group text,
    custom_llm_provider text, team_id text, session_id text, status text, metadata jsonb, request_tags jsonb,
    proxy_server_request jsonb)
ON CONFLICT (request_id) DO NOTHING"""
_INSERT_DELIVERIES: Final = """INSERT INTO "LiteLLM_SpendLogDelivery" (delivery_id, request_id, state,
    accounting_payload, is_batch, applied_at, created_at, next_attempt_at)
SELECT delivery_id, request_id, 'applied', accounting_payload, false, $2::timestamp, created_at, created_at
FROM jsonb_to_recordset($1::jsonb) AS r(delivery_id text, request_id text, accounting_payload jsonb,
    created_at timestamp(3))
ON CONFLICT (delivery_id) DO NOTHING"""


class TaskInsightsDemoRepository:
    def __init__(self, database: TaskInsightsDatabase, prefix: str) -> None:
        self._database = database
        self._pattern = f"{prefix}%"

    async def has_foreign_taxonomy(self) -> bool:
        rows: Final = await self._database.query_raw(_FOREIGN_TAXONOMY)
        return bool(rows and rows[0].get("found"))

    async def reset(self) -> None:
        async with self._database.tx(timeout=_TIMEOUT) as transaction:
            for statement in _RESET:
                await transaction.execute_raw(statement, self._pattern)
            for statement in _RESET_SAMPLE_TAXONOMY:
                await transaction.execute_raw(statement)

    async def insert_traffic(
        self,
        teams: Sequence[Mapping[str, object]],
        spend_logs: Sequence[Mapping[str, object]],
        deliveries: Sequence[Mapping[str, object]],
        applied_at: str,
    ) -> None:
        async with self._database.tx(timeout=_TIMEOUT) as transaction:
            await transaction.execute_raw(_INSERT_TEAMS, safe_dumps(tuple(teams)))
            await transaction.execute_raw(_INSERT_SPEND_LOGS, safe_dumps(tuple(spend_logs), non_finite_as_null=True))
            await transaction.execute_raw(
                _INSERT_DELIVERIES, safe_dumps(tuple(deliveries), non_finite_as_null=True), applied_at
            )
