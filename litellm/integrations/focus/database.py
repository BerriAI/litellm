"""Database access helpers for Focus export."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

import polars as pl


class FocusLiteLLMDatabase:
    """Retrieves LiteLLM usage data for Focus export workflows."""

    def __init__(self, *, include_end_user: bool = False) -> None:
        self.include_end_user = include_end_user

    def _ensure_prisma_client(self):
        from litellm.proxy.proxy_server import prisma_client

        if prisma_client is None:
            raise RuntimeError(
                "Database not connected. Connect a database to your proxy - "
                "https://docs.litellm.ai/docs/simple_proxy#managing-auth---virtual-keys"
            )
        return prisma_client

    async def get_usage_data(
        self,
        *,
        limit: Optional[int] = None,
        start_time_utc: Optional[datetime] = None,
        end_time_utc: Optional[datetime] = None,
    ) -> pl.DataFrame:
        """Return usage data for the requested window."""
        client = self._ensure_prisma_client()

        if self.include_end_user:
            return await self._get_spend_log_usage_data(
                client=client,
                limit=limit,
                start_time_utc=start_time_utc,
                end_time_utc=end_time_utc,
            )

        where_clauses: list[str] = []
        query_params: list[Any] = []
        placeholder_index = 1
        if start_time_utc:
            where_clauses.append(f"dus.updated_at >= ${placeholder_index}::timestamptz")
            query_params.append(start_time_utc)
            placeholder_index += 1
        if end_time_utc:
            where_clauses.append(f"dus.updated_at <= ${placeholder_index}::timestamptz")
            query_params.append(end_time_utc)
            placeholder_index += 1

        where_clause = ""
        if where_clauses:
            where_clause = "WHERE " + " AND ".join(where_clauses)

        limit_clause = ""
        if limit is not None:
            try:
                limit_value = int(limit)
            except (TypeError, ValueError) as exc:  # pragma: no cover - defensive guard
                raise ValueError("limit must be an integer") from exc
            if limit_value < 0:
                raise ValueError("limit must be non-negative")
            limit_clause = f" LIMIT ${placeholder_index}"
            query_params.append(limit_value)

        query = f"""
        SELECT
            dus.id,
            dus.date,
            dus.user_id,
            dus.api_key,
            dus.model,
            dus.model_group,
            dus.custom_llm_provider,
            dus.prompt_tokens,
            dus.completion_tokens,
            dus.spend,
            dus.api_requests,
            dus.successful_requests,
            dus.failed_requests,
            dus.cache_creation_input_tokens,
            dus.cache_read_input_tokens,
            dus.created_at,
            dus.updated_at,
            vt.team_id,
            vt.key_alias as api_key_alias,
            tt.team_alias,
            ut.user_email as user_email,
            COALESCE(vt.organization_id, tt.organization_id) as organization_id,
            ot.organization_alias as organization_alias
        FROM "LiteLLM_DailyUserSpend" dus
        LEFT JOIN "LiteLLM_VerificationToken" vt ON dus.api_key = vt.token
        LEFT JOIN "LiteLLM_TeamTable" tt ON vt.team_id = tt.team_id
        LEFT JOIN "LiteLLM_UserTable" ut ON dus.user_id = ut.user_id
        LEFT JOIN "LiteLLM_OrganizationTable" ot
            ON ot.organization_id = COALESCE(vt.organization_id, tt.organization_id)
        {where_clause}
        ORDER BY dus.date DESC, dus.created_at DESC
        {limit_clause}
        """

        try:
            db_response = await client.db.query_raw(query, *query_params)
            return pl.DataFrame(db_response, infer_schema_length=None)
        except Exception as exc:
            raise RuntimeError(f"Error retrieving usage data: {exc}") from exc

    async def _get_spend_log_usage_data(
        self,
        *,
        client: Any,
        limit: Optional[int],
        start_time_utc: Optional[datetime],
        end_time_utc: Optional[datetime],
    ) -> pl.DataFrame:
        """Return exact request spend grouped by end user for Focus export."""
        where_clauses: list[str] = []
        query_params: list[Any] = []
        placeholder_index = 1
        if start_time_utc:
            where_clauses.append(f'sl."startTime" >= ${placeholder_index}::timestamptz')
            query_params.append(start_time_utc)
            placeholder_index += 1
        if end_time_utc:
            where_clauses.append(f'sl."startTime" < ${placeholder_index}::timestamptz')
            query_params.append(end_time_utc)
            placeholder_index += 1

        where_clause = ""
        if where_clauses:
            where_clause = "WHERE " + " AND ".join(where_clauses)

        limit_clause = ""
        if limit is not None:
            try:
                limit_value = int(limit)
            except (TypeError, ValueError) as exc:
                raise ValueError("limit must be an integer") from exc
            if limit_value < 0:
                raise ValueError("limit must be non-negative")
            limit_clause = f" LIMIT ${placeholder_index}"
            query_params.append(limit_value)

        query = f"""
        SELECT
            MIN(sl.request_id) as id,
            TO_CHAR(sl."startTime" AT TIME ZONE 'UTC', 'YYYY-MM-DD') as date,
            NULLIF(sl."user", '') as user_id,
            NULLIF(sl.end_user, '') as end_user,
            sl.api_key,
            NULLIF(sl.model, '') as model,
            NULLIF(sl.model_group, '') as model_group,
            NULLIF(sl.custom_llm_provider, '') as custom_llm_provider,
            SUM(sl.prompt_tokens) as prompt_tokens,
            SUM(sl.completion_tokens) as completion_tokens,
            SUM(sl.spend) as spend,
            COUNT(*) as api_requests,
            COUNT(*) FILTER (WHERE sl.status = 'success') as successful_requests,
            COUNT(*) FILTER (WHERE sl.status IS NOT NULL AND sl.status != 'success') as failed_requests,
            MIN(sl."startTime") as created_at,
            MAX(sl."endTime") as updated_at,
            COALESCE(sl.team_id, vt.team_id) as team_id,
            vt.key_alias as api_key_alias,
            tt.team_alias,
            ut.user_email as user_email,
            COALESCE(sl.organization_id, vt.organization_id, tt.organization_id) as organization_id,
            ot.organization_alias as organization_alias
        FROM "LiteLLM_SpendLogs" sl
        LEFT JOIN "LiteLLM_VerificationToken" vt ON sl.api_key = vt.token
        LEFT JOIN "LiteLLM_TeamTable" tt
            ON COALESCE(sl.team_id, vt.team_id) = tt.team_id
        LEFT JOIN "LiteLLM_UserTable" ut
            ON COALESCE(NULLIF(sl."user", ''), vt.user_id) = ut.user_id
        LEFT JOIN "LiteLLM_OrganizationTable" ot
            ON ot.organization_id = COALESCE(
                sl.organization_id, vt.organization_id, tt.organization_id
            )
        {where_clause}
        GROUP BY
            TO_CHAR(sl."startTime" AT TIME ZONE 'UTC', 'YYYY-MM-DD'),
            NULLIF(sl."user", ''),
            NULLIF(sl.end_user, ''),
            sl.api_key,
            NULLIF(sl.model, ''),
            NULLIF(sl.model_group, ''),
            NULLIF(sl.custom_llm_provider, ''),
            COALESCE(sl.team_id, vt.team_id),
            vt.key_alias,
            tt.team_alias,
            ut.user_email,
            COALESCE(sl.organization_id, vt.organization_id, tt.organization_id),
            ot.organization_alias
        ORDER BY date DESC, created_at DESC
        {limit_clause}
        """

        try:
            db_response = await client.db.query_raw(query, *query_params)
            return pl.DataFrame(db_response, infer_schema_length=None)
        except Exception as exc:
            raise RuntimeError(f"Error retrieving end-user usage data: {exc}") from exc

    async def get_table_info(self) -> Dict[str, Any]:
        """Return metadata about the spend table for diagnostics."""
        client = self._ensure_prisma_client()

        info_query = """
        SELECT column_name, data_type, is_nullable
        FROM information_schema.columns
        WHERE table_name = 'LiteLLM_DailyUserSpend'
        ORDER BY ordinal_position;
        """
        try:
            columns_response = await client.db.query_raw(info_query)
            return {"columns": columns_response, "table_name": "LiteLLM_DailyUserSpend"}
        except Exception as exc:
            raise RuntimeError(f"Error getting table info: {exc}") from exc
