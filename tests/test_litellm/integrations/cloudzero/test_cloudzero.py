from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import polars as pl
import pytest

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
                "token": ["sk-test-cloudzero-token-010"],
            }
        )
        user_mock_data = pl.LazyFrame(
            {
                "user_id": ["069e8205-8f55-44fd-870b-0c036cab600c"],
                "user_email": ["user@example.com"],
            }
        )

        with (
            patch.object(LiteLLMDatabase, "_ensure_prisma_client") as mock_prisma_client_getter,
            patch.object(CloudZeroStreamer, "send_batched") as send_batched_mock,
            patch("litellm.integrations.cloudzero.cloudzero.datetime") as mock_datetime,
        ):
            fake_client = MagicMock()
            fake_db = MagicMock()

            async def query_raw_mock(query: str, *params):
                start_time_utc = params[0] if len(params) > 0 else None
                end_time_utc = params[1] if len(params) > 1 else None
                limit = params[2] if len(params) > 2 else None

                spend_df = spend_mock_data.collect()
                verification_df = verification_mock_data.collect().rename(
                    {"key_alias": "api_key_alias"}
                )
                team_df = team_mock_data.collect()
                user_df = user_mock_data.collect()

                joined = (
                    spend_df.join(
                        verification_df, left_on="api_key", right_on="token", how="left"
                    )
                    .join(
                        team_df,
                        left_on="team_id",
                        right_on="team_id",
                        how="left",
                        suffix="_team",
                    )
                    .join(
                        user_df,
                        left_on="user_id",
                        right_on="user_id",
                        how="left",
                        suffix="_user",
                    )
                )

                for duplicate_column in ("team_id_team", "user_id_user"):
                    if duplicate_column in joined.columns:
                        joined = joined.drop(duplicate_column)

                if start_time_utc is not None:
                    joined = joined.filter(pl.col("updated_at") >= start_time_utc)
                if end_time_utc is not None:
                    joined = joined.filter(pl.col("updated_at") <= end_time_utc)

                joined = joined.select(
                    [
                        "id",
                        "date",
                        "user_id",
                        "api_key",
                        "model",
                        "model_group",
                        "custom_llm_provider",
                        "prompt_tokens",
                        "completion_tokens",
                        "spend",
                        "api_requests",
                        "successful_requests",
                        "failed_requests",
                        "cache_creation_input_tokens",
                        "cache_read_input_tokens",
                        "created_at",
                        "updated_at",
                        "team_id",
                        "api_key_alias",
                        "team_alias",
                        "user_email",
                    ]
                ).sort(["date", "created_at"], descending=[True, True])

                if limit is not None:
                    joined = joined.head(int(limit))

                return joined

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
