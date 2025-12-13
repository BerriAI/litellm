"""
Unit tests for FOCUS exporter module.
"""

import json
import os
import sys
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import polars as pl
import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.integrations.focus.focus import FOCUSExporter


class TestFOCUSExporter:
    """Test suite for FOCUSExporter class."""

    def test_init_default(self):
        """Test FOCUSExporter initialization with defaults."""
        exporter = FOCUSExporter()
        assert exporter.timezone == "UTC"
        assert exporter.include_tags is True
        assert exporter.include_token_breakdown is True

    def test_init_custom(self):
        """Test FOCUSExporter initialization with custom settings."""
        exporter = FOCUSExporter(
            timezone="America/New_York",
            include_tags=False,
            include_token_breakdown=False,
        )
        assert exporter.timezone == "America/New_York"
        assert exporter.include_tags is False
        assert exporter.include_token_breakdown is False

    def test_init_from_env(self):
        """Test FOCUSExporter initialization from environment variables."""
        with patch.dict(os.environ, {"FOCUS_EXPORT_TIMEZONE": "Europe/London"}):
            exporter = FOCUSExporter()
            assert exporter.timezone == "Europe/London"

    @pytest.mark.asyncio
    async def test_get_focus_data_empty(self):
        """Test get_focus_data with no data."""
        exporter = FOCUSExporter()

        with patch(
            "litellm.integrations.focus.focus.FOCUSDatabase"
        ) as mock_db_class:
            mock_db = MagicMock()
            mock_db.get_usage_data = AsyncMock(return_value=pl.DataFrame())
            mock_db_class.return_value = mock_db

            result = await exporter.get_focus_data()

            assert result.is_empty()

    @pytest.mark.asyncio
    async def test_get_focus_data_with_data(self):
        """Test get_focus_data with valid data."""
        exporter = FOCUSExporter()

        mock_data = pl.DataFrame(
            {
                "date": ["2025-01-19"],
                "successful_requests": [5],
                "spend": [10.0],
                "model": ["gpt-4"],
                "custom_llm_provider": ["openai"],
                "prompt_tokens": [100],
                "completion_tokens": [50],
            }
        )

        with patch(
            "litellm.integrations.focus.focus.FOCUSDatabase"
        ) as mock_db_class:
            mock_db = MagicMock()
            mock_db.get_usage_data = AsyncMock(return_value=mock_data)
            mock_db_class.return_value = mock_db

            result = await exporter.get_focus_data()

            assert len(result) == 1
            record = result.to_dicts()[0]
            assert record["BilledCost"] == 10.0

    @pytest.mark.asyncio
    async def test_export_json_empty(self):
        """Test export_json with no data."""
        exporter = FOCUSExporter()

        with patch(
            "litellm.integrations.focus.focus.FOCUSDatabase"
        ) as mock_db_class:
            mock_db = MagicMock()
            mock_db.get_usage_data = AsyncMock(return_value=pl.DataFrame())
            mock_db_class.return_value = mock_db

            result = await exporter.export_json()

            data = json.loads(result)
            assert data["records"] == []
            assert data["record_count"] == 0

    @pytest.mark.asyncio
    async def test_export_json_with_data(self):
        """Test export_json with valid data."""
        exporter = FOCUSExporter()

        mock_data = pl.DataFrame(
            {
                "date": ["2025-01-19"],
                "successful_requests": [5],
                "spend": [10.0],
                "model": ["gpt-4"],
                "custom_llm_provider": ["openai"],
                "prompt_tokens": [100],
                "completion_tokens": [50],
            }
        )

        with patch(
            "litellm.integrations.focus.focus.FOCUSDatabase"
        ) as mock_db_class:
            mock_db = MagicMock()
            mock_db.get_usage_data = AsyncMock(return_value=mock_data)
            mock_db_class.return_value = mock_db

            result = await exporter.export_json()

            data = json.loads(result)
            assert data["focus_version"] == "1.0"
            assert data["record_count"] == 1
            assert len(data["records"]) == 1
            assert data["records"][0]["BilledCost"] == 10.0

    @pytest.mark.asyncio
    async def test_export_csv_empty(self):
        """Test export_csv with no data."""
        exporter = FOCUSExporter()

        with patch(
            "litellm.integrations.focus.focus.FOCUSDatabase"
        ) as mock_db_class:
            mock_db = MagicMock()
            mock_db.get_usage_data = AsyncMock(return_value=pl.DataFrame())
            mock_db_class.return_value = mock_db

            result = await exporter.export_csv()

            assert result == ""

    @pytest.mark.asyncio
    async def test_export_csv_with_data(self):
        """Test export_csv with valid data."""
        exporter = FOCUSExporter()

        mock_data = pl.DataFrame(
            {
                "date": ["2025-01-19"],
                "successful_requests": [5],
                "spend": [10.0],
                "model": ["gpt-4"],
                "custom_llm_provider": ["openai"],
                "prompt_tokens": [100],
                "completion_tokens": [50],
            }
        )

        with patch(
            "litellm.integrations.focus.focus.FOCUSDatabase"
        ) as mock_db_class:
            mock_db = MagicMock()
            mock_db.get_usage_data = AsyncMock(return_value=mock_data)
            mock_db_class.return_value = mock_db

            result = await exporter.export_csv()

            # CSV should contain header and data row
            assert "BilledCost" in result
            assert "10.0" in result

    @pytest.mark.asyncio
    async def test_export_to_dict_empty(self):
        """Test export_to_dict with no data."""
        exporter = FOCUSExporter()

        with patch(
            "litellm.integrations.focus.focus.FOCUSDatabase"
        ) as mock_db_class:
            mock_db = MagicMock()
            mock_db.get_usage_data = AsyncMock(return_value=pl.DataFrame())
            mock_db_class.return_value = mock_db

            result = await exporter.export_to_dict()

            assert result["focus_version"] == "1.0"
            assert result["records"] == []
            assert result["summary"]["total_records"] == 0
            assert result["summary"]["total_billed_cost"] == 0.0

    @pytest.mark.asyncio
    async def test_export_to_dict_with_data(self):
        """Test export_to_dict with valid data."""
        exporter = FOCUSExporter()

        mock_data = pl.DataFrame(
            {
                "date": ["2025-01-19", "2025-01-20"],
                "successful_requests": [5, 3],
                "spend": [10.0, 5.0],
                "model": ["gpt-4", "claude-3"],
                "custom_llm_provider": ["openai", "anthropic"],
                "prompt_tokens": [100, 50],
                "completion_tokens": [50, 25],
                "team_id": ["team-1", "team-2"],
            }
        )

        with patch(
            "litellm.integrations.focus.focus.FOCUSDatabase"
        ) as mock_db_class:
            mock_db = MagicMock()
            mock_db.get_usage_data = AsyncMock(return_value=mock_data)
            mock_db_class.return_value = mock_db

            result = await exporter.export_to_dict()

            assert result["focus_version"] == "1.0"
            assert len(result["records"]) == 2
            assert result["summary"]["total_records"] == 2
            assert result["summary"]["total_billed_cost"] == 15.0
            assert result["summary"]["total_consumed_quantity"] == 225  # 150 + 75
            assert result["summary"]["unique_providers"] == 2
            assert result["summary"]["unique_sub_accounts"] == 2

    @pytest.mark.asyncio
    async def test_dry_run_export_empty(self):
        """Test dry_run_export with no data."""
        exporter = FOCUSExporter()

        with patch(
            "litellm.integrations.focus.focus.FOCUSDatabase"
        ) as mock_db_class:
            mock_db = MagicMock()
            mock_db.get_usage_data = AsyncMock(return_value=pl.DataFrame())
            mock_db_class.return_value = mock_db

            result = await exporter.dry_run_export()

            assert result["raw_data_sample"] == []
            assert result["focus_data"] == []
            assert result["summary"]["total_records"] == 0

    @pytest.mark.asyncio
    async def test_dry_run_export_with_data(self):
        """Test dry_run_export with valid data."""
        exporter = FOCUSExporter()

        mock_data = pl.DataFrame(
            {
                "date": ["2025-01-19"],
                "successful_requests": [5],
                "spend": [10.0],
                "model": ["gpt-4"],
                "custom_llm_provider": ["openai"],
                "prompt_tokens": [100],
                "completion_tokens": [50],
            }
        )

        with patch(
            "litellm.integrations.focus.focus.FOCUSDatabase"
        ) as mock_db_class:
            mock_db = MagicMock()
            mock_db.get_usage_data = AsyncMock(return_value=mock_data)
            mock_db_class.return_value = mock_db

            result = await exporter.dry_run_export()

            assert len(result["raw_data_sample"]) == 1
            assert len(result["focus_data"]) == 1
            assert result["summary"]["total_records"] == 1
            assert result["summary"]["total_billed_cost"] == 10.0
