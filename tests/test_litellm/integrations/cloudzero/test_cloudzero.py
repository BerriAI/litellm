import pytest
import polars as pl

from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
from litellm.integrations.cloudzero.cloudzero import CloudZeroLogger
from litellm.integrations.cloudzero.cz_stream_api import CloudZeroStreamer
from litellm.integrations.cloudzero.database import LiteLLMDatabase


class TestCloudZeroHourlyExport:
    @pytest.mark.asyncio
    async def test_hourly_export(self):
        spend_mock_data = pl.LazyFrame(
            {
                "id": ["09327a4f-fa99-4613-86c5-23efb03640b1", "c7bcec65-0d76-4126-93b6-50fea1cdd2b"],
                "user_id": ["069e8205-8f55-44fd-870b-0c036cab600c", "069e8205-8f55-44fd-870b-0c036cab600c"],
                "date": ["2025-11-01", "2025-11-01"],
                "api_key": [
                    "c1465c9a821f420927b3d81972323fb516745bc93a4a54ceca0ce6ddf6100c39",
                    "c1465c9a821f420927b3d81972323fb516745bc93a4a54ceca0ce6ddf6100c39",
                ],
                "model": ["model_1", "model_2"],
                "model_group": ["model_group_1", "model_group_2"],
                "custom_llm_provider": ["provider_1", "provider_2"],
                "prompt_tokens": [60, 60],
                "completion_tokens": [71, 71],
                "spend": [0.005, 0.005],
                "api_requests": [1, 1],
                "successful_requests": [1, 1],
                "failed_requests": [0, 0],
                "cache_creation_input_tokens": [0, 0],
                "cache_read_input_tokens": [0, 0],
                "created_at": [datetime(2025, 11, 1, 12), datetime(2025, 11, 1, 2)],
                "updated_at": [datetime(2025, 11, 1, 12), datetime(2025, 11, 1, 12)],
            }
        )

        team_mock_data = pl.LazyFrame(
            {
                "team_id": ["a3d6b0bb-098f-4260-81d6-fabae695b622"],
                "team_alias": ["team_1"],
            }
        )
        verification_mock_data = pl.LazyFrame(
            {
                "team_id": ["a3d6b0bb-098f-4260-81d6-fabae695b622"],
                "key_alias": ["key_1"],
                "token": ["c1465c9a821f420927b3d81972323fb516745bc93a4a54ceca0ce6ddf6100c39"],
            }
        )

        with (
            patch.object(LiteLLMDatabase, "_ensure_prisma_client") as mock_prisma_client_getter,
            patch.object(CloudZeroStreamer, "send_batched") as send_batched_mock,
            patch("litellm.integrations.cloudzero.cloudzero.datetime") as mock_datetime,
        ):
            fake_client = MagicMock()
            fake_db = MagicMock()

            async def query_raw_mock(query: str):
                sql_context = pl.SQLContext(
                    LiteLLM_DailyUserSpend=spend_mock_data,
                    LiteLLM_VerificationToken=verification_mock_data,
                    LiteLLM_TeamTable=team_mock_data,
                )
                result = sql_context.execute(query).collect()

                return result

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
