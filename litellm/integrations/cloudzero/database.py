# Copyright 2025 CloudZero
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# CHANGELOG: 2025-01-19 - Refactored to use daily spend tables for proper CBF mapping (erik.peterson)
# CHANGELOG: 2025-01-19 - Migrated from pandas to polars for database operations (erik.peterson)
# CHANGELOG: 2025-01-19 - Initial database module for LiteLLM data extraction (erik.peterson)

"""Database connection and data extraction for LiteLLM."""

from datetime import datetime
from typing import Any, Optional, List

import polars as pl


class LiteLLMDatabase:
    """Handle LiteLLM PostgreSQL database connections and queries."""

    def _ensure_prisma_client(self):
        from litellm.proxy.proxy_server import prisma_client

        """Ensure prisma client is available."""
        if prisma_client is None:
            raise Exception(
                "Database not connected. Connect a database to your proxy - https://docs.litellm.ai/docs/simple_proxy#managing-auth---virtual-keys"
            )
        return prisma_client

    async def get_usage_data(
        self,
        limit: Optional[int] = None,
        start_time_utc: Optional[datetime] = None,
        end_time_utc: Optional[datetime] = None,
    ) -> pl.DataFrame:
        """Retrieve usage data from LiteLLM daily user spend table."""
        client = self._ensure_prisma_client()

        # Query to get user spend data with team information. Use parameter binding to
        # avoid SQL injection from user-supplied timestamps or limits.
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
            vt.key_alias as api_key_alias,
            tt.team_alias,
            ut.user_email as user_email
        FROM "LiteLLM_DailyUserSpend" dus
        LEFT JOIN "LiteLLM_VerificationToken" vt ON dus.api_key = vt.token
        LEFT JOIN "LiteLLM_TeamTable" tt ON vt.team_id = tt.team_id
        LEFT JOIN "LiteLLM_UserTable" ut ON dus.user_id = ut.user_id
        WHERE ($1::timestamptz IS NULL OR dus.updated_at >= $1::timestamptz)
          AND ($2::timestamptz IS NULL OR dus.updated_at <= $2::timestamptz)
        ORDER BY dus.date DESC, dus.created_at DESC
        """

        params: List[Any] = [
            start_time_utc,
            end_time_utc,
        ]

        if limit is not None:
            try:
                params.append(int(limit))
            except (TypeError, ValueError):
                raise ValueError("limit must be an integer")
            query += " LIMIT $3"

        try:
            db_response = await client.db.query_raw(query, *params)
            # Convert the response to polars DataFrame with full schema inference
            # This prevents schema mismatch errors when data types vary across rows
            return pl.DataFrame(db_response, infer_schema_length=None)
        except Exception as e:
            raise Exception(f"Error retrieving usage data: {str(e)}")
