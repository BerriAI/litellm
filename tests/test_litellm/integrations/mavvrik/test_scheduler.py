"""Unit tests for Orchestrator — _resolve_first_run_start_date and run()."""

import os
import sys
from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.integrations.mavvrik.uploader import Uploader
from litellm.integrations.mavvrik.orchestrator import Orchestrator


def _make_uploader(**kwargs) -> Uploader:
    defaults = dict(
        api_key="mav_key",
        api_endpoint="https://api.mavvrik.dev/acme",
        connection_id="litellm-test",
    )
    defaults.update(kwargs)
    return Uploader(**defaults)


def _make_orchestrator(**kwargs) -> Orchestrator:
    return Orchestrator(uploader=_make_uploader(**kwargs))


# ---------------------------------------------------------------------------
# _resolve_first_run_start_date
# ---------------------------------------------------------------------------


class TestResolveFirstRunStartDate:
    @pytest.mark.asyncio
    async def test_uses_lookback_start_date_when_set(self):
        """MAVVRIK_LOOKBACK_START_DATE takes priority over MIN(date) in DB."""
        orc = _make_orchestrator()
        orc._exporter = MagicMock()
        orc._exporter.get_earliest_date = AsyncMock(return_value="2026-01-01")

        with patch(
            "litellm.integrations.mavvrik.orchestrator.MAVVRIK_LOOKBACK_START_DATE",
            "2026-03-01",
        ):
            result = await orc._resolve_first_run_start_date()

        assert result == date(2026, 3, 1)

    @pytest.mark.asyncio
    async def test_clamps_lookback_to_earliest_db_date(self):
        """If LOOKBACK_START_DATE is before MIN(date), use MIN(date)."""
        orc = _make_orchestrator()
        orc._exporter = MagicMock()
        orc._exporter.get_earliest_date = AsyncMock(return_value="2026-04-01")

        with patch(
            "litellm.integrations.mavvrik.orchestrator.MAVVRIK_LOOKBACK_START_DATE",
            "2026-01-01",
        ):
            result = await orc._resolve_first_run_start_date()

        assert result == date(2026, 4, 1)

    @pytest.mark.asyncio
    async def test_falls_back_to_earliest_db_date_when_no_lookback(self):
        """Without LOOKBACK_START_DATE, use MIN(date) from DB."""
        orc = _make_orchestrator()
        orc._exporter = MagicMock()
        orc._exporter.get_earliest_date = AsyncMock(return_value="2026-02-15")

        with patch(
            "litellm.integrations.mavvrik.orchestrator.MAVVRIK_LOOKBACK_START_DATE",
            None,
        ):
            result = await orc._resolve_first_run_start_date()

        assert result == date(2026, 2, 15)

    @pytest.mark.asyncio
    async def test_falls_back_to_yesterday_when_db_empty(self):
        """When DB is empty and no LOOKBACK_START_DATE, use yesterday."""
        orc = _make_orchestrator()
        orc._exporter = MagicMock()
        orc._exporter.get_earliest_date = AsyncMock(return_value=None)

        with patch(
            "litellm.integrations.mavvrik.orchestrator.MAVVRIK_LOOKBACK_START_DATE",
            None,
        ), patch.object(Orchestrator, "_utc_today", return_value=date(2026, 4, 16)):
            result = await orc._resolve_first_run_start_date()

        assert result == date(2026, 4, 15)

    @pytest.mark.asyncio
    async def test_invalid_lookback_date_falls_back_to_db(self):
        """Invalid LOOKBACK_START_DATE string falls back to DB date."""
        orc = _make_orchestrator()
        orc._exporter = MagicMock()
        orc._exporter.get_earliest_date = AsyncMock(return_value="2026-03-10")

        with patch(
            "litellm.integrations.mavvrik.orchestrator.MAVVRIK_LOOKBACK_START_DATE",
            "not-a-date",
        ):
            result = await orc._resolve_first_run_start_date()

        assert result == date(2026, 3, 10)


# ---------------------------------------------------------------------------
# run() / _run_pipeline()
# ---------------------------------------------------------------------------


class TestRunExportLoop:
    @pytest.mark.asyncio
    async def test_uploads_all_dates_since_marker(self):
        """Uploads each day from marker to yesterday (inclusive)."""
        orc = _make_orchestrator()
        uploaded_dates = []

        async def fake_upload(date_str, limit=None):  # noqa: ARG001
            uploaded_dates.append(date_str)
            return 7

        orc._client = MagicMock()
        orc._client.register = AsyncMock(return_value="2026-04-09")
        orc._client.advance_marker = AsyncMock()
        orc._client.report_error = AsyncMock()

        with patch.object(
            orc._uploader, "upload_usage_data", side_effect=fake_upload
        ), patch.object(
            Orchestrator, "_utc_today", return_value=date(2026, 4, 11)
        ), patch.object(
            Orchestrator, "_get_pod_lock_manager", return_value=None
        ):
            await orc.run()

        assert uploaded_dates == ["2026-04-09", "2026-04-10"]
        assert orc._client.advance_marker.call_count == 2

    @pytest.mark.asyncio
    async def test_advance_marker_uses_next_day(self):
        """advance_marker is called with (export_date + 1) epoch, not export_date."""
        from datetime import datetime, timezone

        orc = _make_orchestrator()

        async def fake_upload(date_str, limit=None):  # noqa: ARG001
            return 3

        orc._client = MagicMock()
        orc._client.register = AsyncMock(return_value="2026-04-09")
        orc._client.advance_marker = AsyncMock()
        orc._client.report_error = AsyncMock()

        with patch.object(
            orc._uploader, "upload_usage_data", side_effect=fake_upload
        ), patch.object(
            Orchestrator, "_utc_today", return_value=date(2026, 4, 10)
        ), patch.object(
            Orchestrator, "_get_pod_lock_manager", return_value=None
        ):
            await orc.run()

        expected_epoch = int(datetime(2026, 4, 10, tzinfo=timezone.utc).timestamp())
        orc._client.advance_marker.assert_called_once_with(expected_epoch)

    @pytest.mark.asyncio
    async def test_does_nothing_when_marker_up_to_date(self):
        """No uploads when marker is already at today (nothing to upload)."""
        orc = _make_orchestrator()

        orc._client = MagicMock()
        orc._client.register = AsyncMock(return_value="2026-04-11")  # = today
        orc._client.advance_marker = AsyncMock()
        orc._client.report_error = AsyncMock()

        with patch.object(
            orc._uploader, "upload_usage_data"
        ) as mock_upload, patch.object(
            Orchestrator, "_utc_today", return_value=date(2026, 4, 11)
        ), patch.object(
            Orchestrator, "_get_pod_lock_manager", return_value=None
        ):
            await orc.run()

        mock_upload.assert_not_called()
        orc._client.advance_marker.assert_not_called()

    @pytest.mark.asyncio
    async def test_first_run_uses_earliest_db_date(self):
        """First run with register() returning None starts from MIN(date) in DB."""
        orc = _make_orchestrator()
        uploaded_dates = []

        async def fake_upload(date_str, limit=None):  # noqa: ARG001
            uploaded_dates.append(date_str)
            return 3

        orc._client = MagicMock()
        orc._client.register = AsyncMock(return_value=None)
        orc._client.advance_marker = AsyncMock()
        orc._client.report_error = AsyncMock()
        orc._exporter = MagicMock()
        orc._exporter.get_earliest_date = AsyncMock(return_value="2026-04-09")

        with patch.object(
            orc._uploader, "upload_usage_data", side_effect=fake_upload
        ), patch(
            "litellm.integrations.mavvrik.orchestrator.MAVVRIK_LOOKBACK_START_DATE",
            None,
        ), patch.object(
            Orchestrator, "_utc_today", return_value=date(2026, 4, 11)
        ), patch.object(
            Orchestrator, "_get_pod_lock_manager", return_value=None
        ):
            await orc.run()

        assert uploaded_dates == ["2026-04-09", "2026-04-10"]
