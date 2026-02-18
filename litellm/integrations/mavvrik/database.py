"""Database extraction layer for the Mavvrik integration.

Queries LiteLLM_DailyUserSpend joined with VerificationToken, TeamTable,
and UserTable — identical to the CloudZero approach, returning a Polars DataFrame.

Additionally provides get/set helpers for Mavvrik settings (including the
marker that tracks the last successfully uploaded interval) stored in LiteLLM_Config.
"""

import json
from datetime import datetime
from typing import Any, List, Optional

import polars as pl


class LiteLLMDatabase:
    """Handle LiteLLM database queries for the Mavvrik integration."""

    def _ensure_prisma_client(self):
        from litellm.proxy.proxy_server import prisma_client

        if prisma_client is None:
            raise Exception(
                "Database not connected. Connect a database to your proxy — "
                "https://docs.litellm.ai/docs/simple_proxy#managing-auth---virtual-keys"
            )
        return prisma_client

    async def get_usage_data(
        self,
        limit: Optional[int] = None,
        start_time_utc: Optional[datetime] = None,
        end_time_utc: Optional[datetime] = None,
    ) -> pl.DataFrame:
        """Retrieve spend rows from LiteLLM_DailyUserSpend with team/key metadata.

        Filters by updated_at to support the time-windowed export pattern.
        When start_time_utc is None, no lower bound is applied (backfill mode).
        """
        client = self._ensure_prisma_client()

        query = """
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
            vt.key_alias  AS api_key_alias,
            tt.team_alias,
            ut.user_email AS user_email
        FROM "LiteLLM_DailyUserSpend" dus
        LEFT JOIN "LiteLLM_VerificationToken" vt  ON dus.api_key   = vt.token
        LEFT JOIN "LiteLLM_TeamTable"         tt  ON vt.team_id    = tt.team_id
        LEFT JOIN "LiteLLM_UserTable"         ut  ON dus.user_id   = ut.user_id
        WHERE ($1::timestamptz IS NULL OR dus.updated_at >= $1::timestamptz)
          AND ($2::timestamptz IS NULL OR dus.updated_at <= $2::timestamptz)
        ORDER BY dus.date ASC, dus.created_at ASC
        """

        params: List[Any] = [start_time_utc, end_time_utc]

        if limit is not None:
            try:
                params.append(int(limit))
            except (TypeError, ValueError):
                raise ValueError("limit must be an integer")
            query += " LIMIT $3"

        try:
            db_response = await client.db.query_raw(query, *params)
            return pl.DataFrame(db_response, infer_schema_length=None)
        except Exception as exc:
            raise Exception(f"Error retrieving Mavvrik usage data: {exc}") from exc

    # ------------------------------------------------------------------
    # Mavvrik settings + marker helpers (stored in LiteLLM_Config table)
    # ------------------------------------------------------------------

    async def get_mavvrik_settings(self) -> dict:
        """Return the stored Mavvrik settings dict (API key already decrypted by caller)."""
        client = self._ensure_prisma_client()

        row = await client.db.litellm_config.find_first(
            where={"param_name": "mavvrik_settings"}
        )
        if row is None or row.param_value is None:
            return {}

        value = row.param_value
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                return {}
        return value if isinstance(value, dict) else {}

    async def set_mavvrik_settings(self, settings: dict) -> None:
        """Upsert Mavvrik settings into LiteLLM_Config."""
        client = self._ensure_prisma_client()

        payload = json.dumps(settings)
        await client.db.litellm_config.upsert(
            where={"param_name": "mavvrik_settings"},
            data={
                "create": {"param_name": "mavvrik_settings", "param_value": payload},
                "update": {"param_value": payload},
            },
        )

    async def advance_marker(self, new_marker: str) -> None:
        """Advance the upload marker to new_marker (ISO-8601 UTC string).

        Reads current settings, updates only the marker field, and writes back.
        This is called after each successful GCS upload so the next scheduled
        run starts from where we left off (delta / incremental pattern).
        """
        settings = await self.get_mavvrik_settings()
        settings["marker"] = new_marker
        await self.set_mavvrik_settings(settings)
