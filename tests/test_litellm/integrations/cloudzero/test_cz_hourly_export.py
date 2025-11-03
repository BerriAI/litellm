import sqlite3
import pytest

from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
from litellm.integrations.cloudzero.cloudzero import CloudZeroLogger
from litellm.integrations.cloudzero.cz_stream_api import CloudZeroStreamer
from litellm.integrations.cloudzero.database import LiteLLMDatabase

class TestCloudZeroHourlyExport:
    @pytest.mark.asyncio
    async def test_hourly_export(self):
        with sqlite3.connect(":memory:") as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            _ = cursor.execute("""
                CREATE TABLE "LiteLLM_DailyUserSpend"
                (id TEXT PRIMARY KEY,
                user_id TEXT,
                date TEXT,
                api_key TEXT,
                model TEXT,
                model_group TEXT,
                custom_llm_provider TEXT,
                prompt_tokens INTEGER,
                completion_tokens INTEGER,
                spend REAL,
                api_requests INTEGER,
                successful_requests INTEGER,
                failed_requests INTEGER,
                cache_creation_input_tokens INTEGER,
                cache_read_input_tokens INTEGER,
                created_at TEXT,
                updated_at TEXT);
                """)
            # Recently created and updated record
            _ = cursor.execute("""
                INSERT INTO "LiteLLM_DailyUserSpend"
                VALUES ('09327a4f-fa99-4613-86c5-23efb03640b1',
                '069e8205-8f55-44fd-870b-0c036cab600c',
                '2025-11-01',
                'c1465c9a821f420927b3d81972323fb516745bc93a4a54ceca0ce6ddf6100c39',
                'model_1',
                'model_group_1',
                'provider_1',
                60,
                71,
                0.0005,
                1,
                1,
                0,
                0,
                0,
                '2025-11-01T12:00:00.000000+00:00',
                '2025-11-01T12:00:00.000000+00:00');
                """)
            # Record created a while ago and updated recently
            _ = cursor.execute("""
                INSERT INTO "LiteLLM_DailyUserSpend"
                VALUES ('c7bcec65-0d76-4126-93b6-50fea1cdd2bd',
                '069e8205-8f55-44fd-870b-0c036cab600c',
                '2025-11-01',
                'c1465c9a821f420927b3d81972323fb516745bc93a4a54ceca0ce6ddf6100c39',
                'model_2',
                'model_group_2',
                'provider_2',
                60,
                71,
                0.0005,
                1,
                1,
                0,
                0,
                0,
                '2025-11-01T02:00:00.000000+00:00',
                '2025-11-01T12:00:00.000000+00:00');
                """)

            _ = cursor.execute("""
                CREATE TABLE "LiteLLM_VerificationToken"
                (team_id TEXT,
                key_alias TEXT,
                token TEXT);
                """)
            _ = cursor.execute("""
                INSERT INTO "LiteLLM_VerificationToken"
                VALUES ('a3d6b0bb-098f-4260-81d6-fabae695b622',
                'key_1',
                'c1465c9a821f420927b3d81972323fb516745bc93a4a54ceca0ce6ddf6100c39');
                """)

            _ = cursor.execute("""
                CREATE TABLE "LiteLLM_TeamTable"
                (team_id TEXT, team_alias TEXT);
                """)
            _ = cursor.execute("""
                INSERT INTO "LiteLLM_TeamTable"
                VALUES ('a3d6b0bb-098f-4260-81d6-fabae695b622', 'team_1');
                """)

            with (
                patch.object(LiteLLMDatabase, "_ensure_prisma_client") as mock_prisma_client_getter,
                patch.object(CloudZeroStreamer, "send_batched") as send_batched_mock,
                patch("litellm.integrations.cloudzero.cloudzero.datetime") as mock_datetime
            ):
                fake_client = MagicMock()
                fake_db = MagicMock()

                async def query_raw_mock(query: str):
                    rows: list[sqlite3.Row] = cursor.execute(query).fetchall()
                    result = [dict(row) for row in rows]
                    return [dict(row) for row in rows]


                fake_db.query_raw = AsyncMock(side_effect=query_raw_mock)
                fake_client.db = fake_db
                mock_prisma_client_getter.return_value = fake_client
                
                mock_datetime.now.return_value = datetime(2025, 11, 1, 12, 0, 1)

                def export_verifier(cbf_data, operation):
                    assert operation == "replace_hourly"
                    assert len(cbf_data) == 2

                send_batched_mock.side_effect = export_verifier

                logger = CloudZeroLogger(api_key="test", connection_id="test")

                await logger._hourly_usage_data_export()
