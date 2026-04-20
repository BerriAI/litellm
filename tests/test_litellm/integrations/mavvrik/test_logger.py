"""Unit tests for MavvrikUploader — upload_usage_data, dry_run, _validate_config."""

import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import polars as pl
import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.integrations.mavvrik.uploader import MavvrikUploader


def _make_uploader(**kwargs) -> MavvrikUploader:
    defaults = dict(
        api_key="mav_key",
        api_endpoint="https://api.mavvrik.dev/acme",
        connection_id="litellm-test",
    )
    defaults.update(kwargs)
    return MavvrikUploader(**defaults)


def _make_df(rows=1) -> pl.DataFrame:
    """Return a minimal spend DataFrame."""
    return pl.DataFrame(
        {
            "date": ["2026-04-10"] * rows,
            "user_id": ["user-alice"] * rows,
            "api_key": ["sk-hash"] * rows,
            "model": ["gpt-4o"] * rows,
            "model_group": ["gpt-4o"] * rows,
            "custom_llm_provider": ["openai"] * rows,
            "prompt_tokens": [1000] * rows,
            "completion_tokens": [500] * rows,
            "spend": [0.015] * rows,
            "api_requests": [10] * rows,
            "successful_requests": [10] * rows,
            "failed_requests": [0] * rows,
            "cache_creation_input_tokens": [0] * rows,
            "cache_read_input_tokens": [0] * rows,
            "created_at": ["2026-04-10T00:00:00Z"] * rows,
            "updated_at": ["2026-04-10T00:00:00Z"] * rows,
            "team_id": ["team-1"] * rows,
            "api_key_alias": ["prod-key"] * rows,
            "team_alias": ["Engineering"] * rows,
            "user_email": ["alice@example.com"] * rows,
        }
    )


# ---------------------------------------------------------------------------
# upload_usage_data
# ---------------------------------------------------------------------------


class TestUploadUsageData:
    @pytest.mark.asyncio
    async def test_returns_record_count_on_success(self):
        uploader = _make_uploader()
        mock_db = MagicMock()
        mock_db.get_usage_data = AsyncMock(return_value=_make_df(rows=5))
        mock_client = MagicMock()
        mock_client.upload = AsyncMock()

        uploader._mavvrik_client = mock_client

        with patch(
            "litellm.integrations.mavvrik.uploader.MavvrikDatabase",
            return_value=mock_db,
        ):
            count = await uploader.upload_usage_data(date_str="2026-04-10")

        assert count == 5
        mock_client.upload.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_zero_when_no_data(self):
        uploader = _make_uploader()
        mock_db = MagicMock()
        mock_db.get_usage_data = AsyncMock(return_value=pl.DataFrame())
        mock_client = MagicMock()
        mock_client.upload = AsyncMock()

        with patch(
            "litellm.integrations.mavvrik.uploader.MavvrikDatabase",
            return_value=mock_db,
        ), patch(
            "litellm.integrations.mavvrik.uploader.MavvrikClient",
            return_value=mock_client,
        ):
            count = await uploader.upload_usage_data(date_str="2026-04-10")

        assert count == 0
        mock_client.upload.assert_not_called()

    @pytest.mark.asyncio
    async def test_raises_value_error_when_config_missing(self):
        uploader = MavvrikUploader(api_key="", api_endpoint="", connection_id="")
        with pytest.raises(ValueError, match="missing required config fields"):
            await uploader.upload_usage_data(date_str="2026-04-10")
