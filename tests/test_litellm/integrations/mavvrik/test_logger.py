"""Unit tests for MavvrikExporter — export_usage_data, dry_run, _validate_config.

Scheduler method tests (initialize_mavvrik_export_job, _scheduled_export,
_resolve_first_run_start_date) are commented out below — they will be moved
to test_scheduler.py in the next task once MavvrikScheduler is created.
"""

import os
import sys
from datetime import date, datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import polars as pl
import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.integrations.mavvrik.exporter import MavvrikExporter


def _make_exporter(**kwargs) -> MavvrikExporter:
    defaults = dict(
        api_key="mav_key",
        api_endpoint="https://api.mavvrik.dev/acme",
        connection_id="litellm-test",
    )
    defaults.update(kwargs)
    return MavvrikExporter(**defaults)


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
# _resolve_first_run_start_date
# (TODO: move to test_scheduler.py once MavvrikScheduler exists)
# ---------------------------------------------------------------------------

# class TestResolveFirstRunStartDate:
#     @pytest.mark.asyncio
#     async def test_uses_lookback_start_date_when_set(self):
#         """MAVVRIK_LOOKBACK_START_DATE takes priority over MIN(date) in DB."""
#         exporter = _make_exporter()
#         mock_db = MagicMock()
#         mock_db.get_earliest_date = AsyncMock(return_value="2026-01-01")
#         yesterday = date(2026, 4, 15)
#
#         with patch.dict("os.environ", {"MAVVRIK_LOOKBACK_START_DATE": "2026-03-01"}):
#             with patch(
#                 "litellm.integrations.mavvrik.exporter.MAVVRIK_LOOKBACK_START_DATE",
#                 "2026-03-01",
#             ):
#                 result = await exporter._resolve_first_run_start_date(mock_db, yesterday)
#
#         assert result == date(2026, 3, 1)
#
#     @pytest.mark.asyncio
#     async def test_clamps_lookback_to_earliest_db_date(self):
#         """If LOOKBACK_START_DATE is before MIN(date), use MIN(date)."""
#         exporter = _make_exporter()
#         mock_db = MagicMock()
#         mock_db.get_earliest_date = AsyncMock(return_value="2026-04-01")
#         yesterday = date(2026, 4, 15)
#
#         with patch(
#             "litellm.integrations.mavvrik.exporter.MAVVRIK_LOOKBACK_START_DATE",
#             "2026-01-01",
#         ):
#             result = await exporter._resolve_first_run_start_date(mock_db, yesterday)
#
#         assert result == date(2026, 4, 1)
#
#     @pytest.mark.asyncio
#     async def test_falls_back_to_earliest_db_date_when_no_lookback(self):
#         """Without LOOKBACK_START_DATE, use MIN(date) from DB."""
#         exporter = _make_exporter()
#         mock_db = MagicMock()
#         mock_db.get_earliest_date = AsyncMock(return_value="2026-02-15")
#         yesterday = date(2026, 4, 15)
#
#         with patch(
#             "litellm.integrations.mavvrik.exporter.MAVVRIK_LOOKBACK_START_DATE", None
#         ):
#             result = await exporter._resolve_first_run_start_date(mock_db, yesterday)
#
#         assert result == date(2026, 2, 15)
#
#     @pytest.mark.asyncio
#     async def test_falls_back_to_yesterday_when_db_empty(self):
#         """When DB is empty and no LOOKBACK_START_DATE, use yesterday."""
#         exporter = _make_exporter()
#         mock_db = MagicMock()
#         mock_db.get_earliest_date = AsyncMock(return_value=None)
#         yesterday = date(2026, 4, 15)
#
#         with patch(
#             "litellm.integrations.mavvrik.exporter.MAVVRIK_LOOKBACK_START_DATE", None
#         ):
#             result = await exporter._resolve_first_run_start_date(mock_db, yesterday)
#
#         assert result == yesterday
#
#     @pytest.mark.asyncio
#     async def test_invalid_lookback_date_falls_back_to_db(self):
#         """Invalid LOOKBACK_START_DATE string falls back to DB date."""
#         exporter = _make_exporter()
#         mock_db = MagicMock()
#         mock_db.get_earliest_date = AsyncMock(return_value="2026-03-10")
#         yesterday = date(2026, 4, 15)
#
#         with patch(
#             "litellm.integrations.mavvrik.exporter.MAVVRIK_LOOKBACK_START_DATE",
#             "not-a-date",
#         ):
#             result = await exporter._resolve_first_run_start_date(mock_db, yesterday)
#
#         assert result == date(2026, 3, 10)


# ---------------------------------------------------------------------------
# export_usage_data
# ---------------------------------------------------------------------------


class TestExportUsageData:
    @pytest.mark.asyncio
    async def test_returns_record_count_on_success(self):
        exporter = _make_exporter()
        mock_db = MagicMock()
        mock_db.get_usage_data = AsyncMock(return_value=_make_df(rows=5))
        mock_client = MagicMock()
        mock_client.upload = AsyncMock()

        # _mavvrik_client is built in __init__, so patch the instance attribute directly.
        exporter._mavvrik_client = mock_client

        with patch(
            "litellm.integrations.mavvrik.exporter.MavvrikDatabase",
            return_value=mock_db,
        ):
            count = await exporter.export_usage_data(date_str="2026-04-10")

        assert count == 5
        mock_client.upload.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_zero_when_no_data(self):
        exporter = _make_exporter()
        mock_db = MagicMock()
        mock_db.get_usage_data = AsyncMock(return_value=pl.DataFrame())
        mock_client = MagicMock()
        mock_client.upload = AsyncMock()

        with patch(
            "litellm.integrations.mavvrik.exporter.MavvrikDatabase",
            return_value=mock_db,
        ), patch(
            "litellm.integrations.mavvrik.exporter.MavvrikClient",
            return_value=mock_client,
        ):
            count = await exporter.export_usage_data(date_str="2026-04-10")

        assert count == 0
        mock_client.upload.assert_not_called()

    @pytest.mark.asyncio
    async def test_raises_value_error_when_config_missing(self):
        exporter = MavvrikExporter(api_key="", api_endpoint="", connection_id="")
        with pytest.raises(ValueError, match="missing required config fields"):
            await exporter.export_usage_data(date_str="2026-04-10")


# ---------------------------------------------------------------------------
# _scheduled_export
# (TODO: move to test_scheduler.py once MavvrikScheduler exists)
# ---------------------------------------------------------------------------

# class TestScheduledExport:
#     @pytest.mark.asyncio
#     async def test_exports_all_dates_since_marker(self):
#         """Scheduler should export each day from marker+1 to yesterday."""
#         exporter = _make_exporter()
#
#         mock_db = MagicMock()
#         mock_db.get_mavvrik_settings = AsyncMock(return_value={"marker": "2026-04-08"})
#         mock_db.advance_marker = AsyncMock()
#         mock_db.get_earliest_date = AsyncMock(return_value="2026-04-01")
#
#         exported_dates = []
#
#         async def fake_export(date_str, limit=None):
#             exported_dates.append(date_str)
#             return 7
#
#         mock_client = MagicMock()
#         mock_client.advance_marker = AsyncMock()
#
#         with patch(
#             "litellm.integrations.mavvrik.exporter.MavvrikDatabase", return_value=mock_db
#         ), patch(
#             "litellm.integrations.mavvrik.exporter.MavvrikClient",
#             return_value=mock_client,
#         ), patch.object(
#             exporter, "export_usage_data", side_effect=fake_export
#         ), patch(
#             "litellm.integrations.mavvrik.exporter._utc_now",
#             return_value=datetime(2026, 4, 11, 0, 0, 0, tzinfo=timezone.utc),
#         ):
#             await exporter._scheduled_export()
#
#         assert exported_dates == ["2026-04-09", "2026-04-10"]
#         assert mock_db.advance_marker.call_count == 2
#
#     @pytest.mark.asyncio
#     async def test_stops_on_config_error_but_continues_on_transient_error(self):
#         """ValueError stops the loop; other exceptions skip the date and continue."""
#         exporter = _make_exporter()
#
#         mock_db = MagicMock()
#         mock_db.get_mavvrik_settings = AsyncMock(return_value={"marker": "2026-04-07"})
#         mock_db.advance_marker = AsyncMock()
#
#         call_count = 0
#
#         async def fake_export(date_str, limit=None):
#             nonlocal call_count
#             call_count += 1
#             if date_str == "2026-04-08":
#                 raise Exception("network error")  # transient — should continue
#             if date_str == "2026-04-09":
#                 raise ValueError("missing credentials")  # config — should stop
#             return 5
#
#         mock_client = MagicMock()
#         mock_client.advance_marker = AsyncMock()
#
#         with patch(
#             "litellm.integrations.mavvrik.exporter.MavvrikDatabase", return_value=mock_db
#         ), patch(
#             "litellm.integrations.mavvrik.exporter.MavvrikClient",
#             return_value=mock_client,
#         ), patch.object(
#             exporter, "export_usage_data", side_effect=fake_export
#         ), patch(
#             "litellm.integrations.mavvrik.exporter._utc_now",
#             return_value=datetime(2026, 4, 11, 0, 0, 0, tzinfo=timezone.utc),
#         ):
#             await exporter._scheduled_export()
#
#         assert call_count == 2
#         assert mock_db.advance_marker.call_count == 0
#
#     @pytest.mark.asyncio
#     async def test_does_nothing_when_marker_up_to_date(self):
#         """No exports when marker is already at yesterday."""
#         exporter = _make_exporter()
#
#         mock_db = MagicMock()
#         mock_db.get_mavvrik_settings = AsyncMock(
#             return_value={"marker": "2026-04-10"}  # = yesterday
#         )
#         mock_db.advance_marker = AsyncMock()
#
#         with patch(
#             "litellm.integrations.mavvrik.exporter.MavvrikDatabase", return_value=mock_db
#         ), patch.object(exporter, "export_usage_data") as mock_export, patch(
#             "litellm.integrations.mavvrik.exporter._utc_now",
#             return_value=datetime(2026, 4, 11, 0, 0, 0, tzinfo=timezone.utc),
#         ):
#             await exporter._scheduled_export()
#
#         mock_export.assert_not_called()
#         mock_db.advance_marker.assert_not_called()
#
#     @pytest.mark.asyncio
#     async def test_first_run_uses_earliest_db_date(self):
#         """First run with no marker starts from MIN(date) in DB."""
#         exporter = _make_exporter()
#
#         mock_db = MagicMock()
#         mock_db.get_mavvrik_settings = AsyncMock(return_value={})  # no marker
#         mock_db.get_earliest_date = AsyncMock(return_value="2026-04-09")
#         mock_db.advance_marker = AsyncMock()
#
#         exported_dates = []
#
#         async def fake_export(date_str, limit=None):
#             exported_dates.append(date_str)
#             return 3
#
#         mock_client = MagicMock()
#         mock_client.advance_marker = AsyncMock()
#
#         with patch(
#             "litellm.integrations.mavvrik.exporter.MavvrikDatabase", return_value=mock_db
#         ), patch(
#             "litellm.integrations.mavvrik.exporter.MavvrikClient",
#             return_value=mock_client,
#         ), patch.object(
#             exporter, "export_usage_data", side_effect=fake_export
#         ), patch(
#             "litellm.integrations.mavvrik.exporter.MAVVRIK_LOOKBACK_START_DATE", None
#         ), patch(
#             "litellm.integrations.mavvrik.exporter._utc_now",
#             return_value=datetime(2026, 4, 11, 0, 0, 0, tzinfo=timezone.utc),
#         ):
#             await exporter._scheduled_export()
#
#         assert exported_dates == ["2026-04-09", "2026-04-10"]
