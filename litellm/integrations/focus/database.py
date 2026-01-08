"""
Database connection and data extraction for FOCUS export.

Provides access to LiteLLM usage data for FOCUS format export.
"""

from datetime import datetime
from typing import Any, Dict, Optional

import polars as pl


class FOCUSDatabase:
    """Handle LiteLLM database connections and queries for FOCUS export."""

    def _ensure_prisma_client(self):
        """Ensure prisma client is available."""
        from litellm.proxy.proxy_server import prisma_client

        if prisma_client is None:
            raise Exception(
                "Database not connected. Connect a database to your proxy - "
                "https://docs.litellm.ai/docs/simple_proxy#managing-auth---virtual-keys"
            )
        return prisma_client

    async def get_usage_data(
        self,
        limit: Optional[int] = None,
        start_time_utc: Optional[datetime] = None,
        end_time_utc: Optional[datetime] = None,
    ) -> pl.DataFrame:
        """
        Retrieve usage data from LiteLLM daily user spend table.

        Args:
            limit: Optional limit on number of records
            start_time_utc: Optional start time filter
            end_time_utc: Optional end time filter

        Returns:
            Polars DataFrame with usage data
        """
        client = self._ensure_prisma_client()

        # Build WHERE clause for time filtering
        where_conditions = []
        if start_time_utc:
            where_conditions.append(
                f"dus.updated_at >= '{start_time_utc.isoformat()}'"
            )
        if end_time_utc:
            where_conditions.append(
                f"dus.updated_at <= '{end_time_utc.isoformat()}'"
            )

        where_clause = ""
        if where_conditions:
            where_clause = "WHERE " + " AND ".join(where_conditions)

        # Query to get user spend data with team information
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
            tt.team_alias
        FROM "LiteLLM_DailyUserSpend" dus
        LEFT JOIN "LiteLLM_VerificationToken" vt ON dus.api_key = vt.token
        LEFT JOIN "LiteLLM_TeamTable" tt ON vt.team_id = tt.team_id
        {where_clause}
        ORDER BY dus.date DESC, dus.created_at DESC
        """

        if limit:
            query += f" LIMIT {limit}"

        try:
            db_response = await client.db.query_raw(query)
            # Convert the response to polars DataFrame with full schema inference
            return pl.DataFrame(db_response, infer_schema_length=None)
        except Exception as e:
            raise Exception(f"Error retrieving usage data: {str(e)}")

    async def get_table_info(self) -> Dict[str, Any]:
        """Get information about the daily user spend table."""
        client = self._ensure_prisma_client()

        try:
            # Get row count from user spend table
            count_query = 'SELECT COUNT(*) as count FROM "LiteLLM_DailyUserSpend"'
            count_response = await client.db.query_raw(count_query)
            row_count = (
                count_response[0].get("count", 0)
                if count_response
                else 0
            )

            return {
                "table_name": "LiteLLM_DailyUserSpend",
                "row_count": row_count,
            }
        except Exception as e:
            raise Exception(f"Error getting table info: {str(e)}")
