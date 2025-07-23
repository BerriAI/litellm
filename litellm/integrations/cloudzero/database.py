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

from typing import Any, Dict, Optional

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

    async def get_usage_data(self, limit: Optional[int] = None) -> pl.DataFrame:
        """Retrieve consolidated usage data from LiteLLM daily spend tables."""
        client = self._ensure_prisma_client()
        
        # Union query to combine user, team, and tag spend data
        query = """
        WITH consolidated_spend AS (
            -- User spend data
            SELECT
                id,
                date,
                user_id as entity_id,
                'user' as entity_type,
                api_key,
                model,
                model_group,
                custom_llm_provider,
                prompt_tokens,
                completion_tokens,
                spend,
                api_requests,
                successful_requests,
                failed_requests,
                cache_creation_input_tokens,
                cache_read_input_tokens,
                created_at,
                updated_at
            FROM "LiteLLM_DailyUserSpend"

            UNION ALL

            -- Team spend data
            SELECT
                id,
                date,
                team_id as entity_id,
                'team' as entity_type,
                api_key,
                model,
                model_group,
                custom_llm_provider,
                prompt_tokens,
                completion_tokens,
                spend,
                api_requests,
                successful_requests,
                failed_requests,
                cache_creation_input_tokens,
                cache_read_input_tokens,
                created_at,
                updated_at
            FROM "LiteLLM_DailyTeamSpend"

            UNION ALL

            -- Tag spend data
            SELECT
                id,
                date,
                tag as entity_id,
                'tag' as entity_type,
                api_key,
                model,
                model_group,
                custom_llm_provider,
                prompt_tokens,
                completion_tokens,
                spend,
                api_requests,
                successful_requests,
                failed_requests,
                cache_creation_input_tokens,
                cache_read_input_tokens,
                created_at,
                updated_at
            FROM "LiteLLM_DailyTagSpend"
        )
        SELECT * FROM consolidated_spend
        ORDER BY date DESC, created_at DESC
        """

        if limit:
            query += f" LIMIT {limit}"

        try:
            db_response = await client.db.query_raw(query)
            # Convert the response to polars DataFrame
            return pl.DataFrame(db_response)
        except Exception as e:
            raise Exception(f"Error retrieving usage data: {str(e)}")

    async def get_table_info(self) -> Dict[str, Any]:
        """Get information about the consolidated daily spend tables."""
        client = self._ensure_prisma_client()
        
        try:
            # Get combined row count from both tables
            user_count = await self._get_table_row_count('LiteLLM_DailyUserSpend')
            team_count = await self._get_table_row_count('LiteLLM_DailyTeamSpend')
            tag_count = await self._get_table_row_count('LiteLLM_DailyTagSpend')

            # Get column structure from user spend table (representative)
            query = """
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_name = 'LiteLLM_DailyUserSpend'
            ORDER BY ordinal_position;
            """
            columns_response = await client.db.query_raw(query)

            return {
                'columns': columns_response,
                'row_count': user_count + team_count + tag_count,
                'table_breakdown': {
                    'user_spend': user_count,
                    'team_spend': team_count,
                    'tag_spend': tag_count
                }
            }
        except Exception as e:
            raise Exception(f"Error getting table info: {str(e)}")

    async def _get_table_row_count(self, table_name: str) -> int:
        """Get row count from specified table."""
        client = self._ensure_prisma_client()
        
        try:
            query = f'SELECT COUNT(*) as count FROM "{table_name}"'
            response = await client.db.query_raw(query)
            
            if response and len(response) > 0:
                return response[0].get('count', 0)
            return 0
        except Exception:
            return 0

    async def discover_all_tables(self) -> Dict[str, Any]:
        """Discover all tables in the LiteLLM database and their schemas."""
        client = self._ensure_prisma_client()
        
        try:
            # Get all LiteLLM tables
            litellm_tables_query = """
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name LIKE 'LiteLLM_%'
            ORDER BY table_name;
            """
            tables_response = await client.db.query_raw(litellm_tables_query)
            table_names = [row['table_name'] for row in tables_response]

            # Get detailed schema for each table
            tables_info = {}
            for table_name in table_names:
                # Get column information
                columns_query = """
                SELECT 
                    column_name,
                    data_type,
                    is_nullable,
                    column_default,
                    character_maximum_length,
                    numeric_precision,
                    numeric_scale,
                    ordinal_position
                FROM information_schema.columns 
                WHERE table_name = $1
                AND table_schema = 'public'
                ORDER BY ordinal_position;
                """
                columns_response = await client.db.query_raw(columns_query, table_name)

                # Get primary key information
                pk_query = """
                SELECT a.attname
                FROM pg_index i
                JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey)
                WHERE i.indrelid = $1::regclass AND i.indisprimary;
                """
                pk_response = await client.db.query_raw(pk_query, f'"{table_name}"')
                primary_keys = [row['attname'] for row in pk_response] if pk_response else []

                # Get foreign key information
                fk_query = """
                SELECT
                    tc.constraint_name,
                    kcu.column_name,
                    ccu.table_name AS foreign_table_name,
                    ccu.column_name AS foreign_column_name
                FROM information_schema.table_constraints AS tc
                JOIN information_schema.key_column_usage AS kcu
                    ON tc.constraint_name = kcu.constraint_name
                JOIN information_schema.constraint_column_usage AS ccu
                    ON ccu.constraint_name = tc.constraint_name
                WHERE tc.constraint_type = 'FOREIGN KEY'
                AND tc.table_name = $1;
                """
                fk_response = await client.db.query_raw(fk_query, table_name)
                foreign_keys = fk_response if fk_response else []

                # Get indexes
                indexes_query = """
                SELECT
                    i.relname AS index_name,
                    array_agg(a.attname ORDER BY a.attnum) AS column_names,
                    ix.indisunique AS is_unique
                FROM pg_class t
                JOIN pg_index ix ON t.oid = ix.indrelid
                JOIN pg_class i ON i.oid = ix.indexrelid
                JOIN pg_attribute a ON a.attrelid = t.oid AND a.attnum = ANY(ix.indkey)
                WHERE t.relname = $1
                AND t.relkind = 'r'
                GROUP BY i.relname, ix.indisunique
                ORDER BY i.relname;
                """
                indexes_response = await client.db.query_raw(indexes_query, table_name)
                indexes = indexes_response if indexes_response else []

                # Get row count
                try:
                    row_count = await self._get_table_row_count(table_name)
                except Exception:
                    row_count = 0

                tables_info[table_name] = {
                    'columns': columns_response,
                    'primary_keys': primary_keys,
                    'foreign_keys': foreign_keys,
                    'indexes': indexes,
                    'row_count': row_count
                }

            return {
                'tables': tables_info,
                'table_count': len(table_names),
                'table_names': table_names
            }
        except Exception as e:
            raise Exception(f"Error discovering tables: {str(e)}")

