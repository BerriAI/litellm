"""Database access helpers for Focus SKU export from LiteLLM_SpendLogs."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

import polars as pl


class FocusSpendLogsDatabase:
    """Retrieves per-request usage data from LiteLLM_SpendLogs for SKU export.

    Unlike FocusLiteLLMDatabase (which reads the daily-aggregated
    LiteLLM_DailyUserSpend table), this class reads the per-request
    LiteLLM_SpendLogs table so that each row maps to a single API call and
    carries a real ``request_id``.

    Cache token counts (cache_creation_input_tokens, cache_read_input_tokens)
    are not top-level columns in LiteLLM_SpendLogs; they are stored inside the
    ``metadata`` JSON column under ``additional_usage_values``.  This class
    extracts them via a PostgreSQL JSON path expression so that
    FocusSkuTransformer can explode them into individual SKU rows.

    Returned column names match those expected by FocusSkuTransformer:
        request_id, api_key, api_key_alias, model, model_group,
        custom_llm_provider, prompt_tokens, completion_tokens,
        cache_creation_input_tokens, cache_read_input_tokens,
        total_tokens, spend, date, team_id, team_alias, user_id, user_email
    """

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
        """Return per-request usage data for the requested window.

        Time filters apply to ``startTime`` (the moment the LLM call began).
        """
        client = self._ensure_prisma_client()

        where_clauses: list[str] = []
        query_params: list[Any] = []
        placeholder_index = 1

        if start_time_utc:
            where_clauses.append(
                f'sl."startTime" >= ${placeholder_index}::timestamptz'
            )
            query_params.append(start_time_utc)
            placeholder_index += 1
        if end_time_utc:
            where_clauses.append(
                f'sl."startTime" <= ${placeholder_index}::timestamptz'
            )
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
            limit_clause = f"LIMIT ${placeholder_index}"
            query_params.append(limit_value)

        # Cache tokens are stored in metadata->additional_usage_values as JSON.
        # COALESCE handles NULL metadata, missing keys, and non-numeric values
        # gracefully — falling back to 0 in all cases.
        #
        # NOTE: The ::jsonb cast and -> / ->> operators below are PostgreSQL-
        # specific. This class intentionally bypasses the Prisma ORM (which is
        # database-agnostic) to access these operators. If LiteLLM ever adds
        # support for a second database backend, this query will need a dialect-
        # specific implementation.
        query = f"""
        SELECT
            sl.request_id,
            sl.api_key,
            sl.spend,
            sl.prompt_tokens,
            sl.completion_tokens,
            sl.total_tokens,
            COALESCE(
                (sl.metadata::jsonb -> 'additional_usage_values'
                    ->> 'cache_creation_input_tokens')::bigint,
                0
            ) AS cache_creation_input_tokens,
            COALESCE(
                (sl.metadata::jsonb -> 'additional_usage_values'
                    ->> 'cache_read_input_tokens')::bigint,
                0
            ) AS cache_read_input_tokens,
            sl.model,
            sl.model_group,
            sl.custom_llm_provider,
            sl.team_id,
            sl."user" AS user_id,
            TO_CHAR(sl."startTime" AT TIME ZONE 'UTC', 'YYYY-MM-DD') AS date,
            vt.key_alias AS api_key_alias,
            tt.team_alias,
            ut.user_email
        FROM "LiteLLM_SpendLogs" sl
        LEFT JOIN "LiteLLM_VerificationToken" vt ON sl.api_key = vt.token
        LEFT JOIN "LiteLLM_TeamTable" tt ON sl.team_id = tt.team_id
        LEFT JOIN "LiteLLM_UserTable" ut ON sl."user" = ut.user_id
        {where_clause}
        ORDER BY sl."startTime" DESC
        {limit_clause}
        """

        try:
            db_response = await client.db.query_raw(query, *query_params)
            return pl.DataFrame(db_response, infer_schema_length=None)
        except Exception as exc:
            raise RuntimeError(f"Error retrieving spend logs data: {exc}") from exc
