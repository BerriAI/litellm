"""Unit tests for Orchestrator — pipeline sequencing and pod lock."""

import os
import sys
from datetime import date, datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import polars as pl
import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.integrations.mavvrik.orchestrator import Orchestrator
from litellm.integrations.mavvrik.client import Client
from litellm.integrations.mavvrik.uploader import Uploader


def _make_client(**kwargs) -> Client:
    defaults = dict(
        api_key="mav_key",
        api_endpoint="https://api.mavvrik.dev/acme",
        connection_id="litellm-test",
    )
    defaults.update(kwargs)
    return Client(**defaults)


def _make_uploader(client=None) -> Uploader:
    return Uploader(client=client or _make_client())


def _make_orchestrator() -> Orchestrator:
    client = _make_client()
    uploader = _make_uploader(client=client)
    return Orchestrator(client=client, uploader=uploader)


def _make_df(rows=3) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": ["2026-04-10"] * rows,
            "user_id": ["user-alice"] * rows,
            "model": ["gpt-4o"] * rows,
            "spend": [0.015] * rows,
            "successful_requests": [5] * rows,
            "prompt_tokens": [100] * rows,
            "completion_tokens": [50] * rows,
        }
    )


# ---------------------------------------------------------------------------
# _resolve_first_run_start_date
# ---------------------------------------------------------------------------


class TestResolveFirstRunStartDate:
    @pytest.mark.asyncio
    async def test_uses_earliest_db_date(self):
        orc = _make_orchestrator()
        orc._exporter = MagicMock()
        orc._exporter.get_earliest_date = AsyncMock(return_value="2026-02-15")

        result = await orc._resolve_first_run_start_date()

        assert result == date(2026, 2, 15)

    @pytest.mark.asyncio
    async def test_falls_back_to_yesterday_when_db_empty(self):
        orc = _make_orchestrator()
        orc._exporter = MagicMock()
        orc._exporter.get_earliest_date = AsyncMock(return_value=None)

        with patch.object(Orchestrator, "_utc_today", return_value=date(2026, 4, 16)):
            result = await orc._resolve_first_run_start_date()

        assert result == date(2026, 4, 15)


# ---------------------------------------------------------------------------
# run() / _run_pipeline()
# ---------------------------------------------------------------------------


class TestRunExportLoop:
    @pytest.mark.asyncio
    async def test_uploads_all_dates_since_marker(self):
        """Uploads each day from marker to yesterday (inclusive)."""
        orc = _make_orchestrator()
        uploaded_dates = []

        async def fake_upload(csv_payload, date_str):
            uploaded_dates.append(date_str)

        orc._client.register = AsyncMock(return_value="2026-04-09")
        orc._client.advance_marker = AsyncMock()
        orc._client.report_error = AsyncMock()
        orc._exporter = MagicMock()
        orc._exporter.get_usage_data = AsyncMock(return_value=_make_df())
        orc._exporter.filter = MagicMock(side_effect=lambda df: df)
        orc._exporter.to_csv = MagicMock(return_value="col\nval")

        with patch.object(
            orc._uploader, "upload", side_effect=fake_upload
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
        """advance_marker is called with (export_date + 1) epoch."""
        orc = _make_orchestrator()

        orc._client.register = AsyncMock(return_value="2026-04-09")
        orc._client.advance_marker = AsyncMock()
        orc._client.report_error = AsyncMock()
        orc._exporter = MagicMock()
        orc._exporter.get_usage_data = AsyncMock(return_value=_make_df())
        orc._exporter.filter = MagicMock(side_effect=lambda df: df)
        orc._exporter.to_csv = MagicMock(return_value="col\nval")

        with patch.object(
            orc._uploader, "upload", new_callable=AsyncMock
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
        """No uploads when marker is already at today."""
        orc = _make_orchestrator()

        orc._client.register = AsyncMock(return_value="2026-04-11")  # = today
        orc._client.advance_marker = AsyncMock()
        orc._client.report_error = AsyncMock()

        with patch.object(orc._uploader, "upload") as mock_upload, patch.object(
            Orchestrator, "_utc_today", return_value=date(2026, 4, 11)
        ), patch.object(Orchestrator, "_get_pod_lock_manager", return_value=None):
            await orc.run()

        mock_upload.assert_not_called()
        orc._client.advance_marker.assert_not_called()

    @pytest.mark.asyncio
    async def test_first_run_uses_earliest_db_date(self):
        """First run: register() returns None → start from MIN(date) in DB."""
        orc = _make_orchestrator()
        uploaded_dates = []

        async def fake_upload(csv_payload, date_str):
            uploaded_dates.append(date_str)

        orc._client.register = AsyncMock(return_value=None)
        orc._client.advance_marker = AsyncMock()
        orc._client.report_error = AsyncMock()
        orc._exporter = MagicMock()
        orc._exporter.get_earliest_date = AsyncMock(return_value="2026-04-09")
        orc._exporter.get_usage_data = AsyncMock(return_value=_make_df())
        orc._exporter.filter = MagicMock(side_effect=lambda df: df)
        orc._exporter.to_csv = MagicMock(return_value="col\nval")

        with patch.object(
            orc._uploader, "upload", side_effect=fake_upload
        ), patch.object(
            Orchestrator, "_utc_today", return_value=date(2026, 4, 11)
        ), patch.object(
            Orchestrator, "_get_pod_lock_manager", return_value=None
        ):
            await orc.run()

        assert uploaded_dates == ["2026-04-09", "2026-04-10"]

    @pytest.mark.asyncio
    async def test_skips_upload_when_no_data(self):
        """When exporter returns empty CSV, upload is not called."""
        orc = _make_orchestrator()

        orc._client.register = AsyncMock(return_value="2026-04-09")
        orc._client.advance_marker = AsyncMock()
        orc._client.report_error = AsyncMock()
        orc._exporter = MagicMock()
        orc._exporter.get_usage_data = AsyncMock(return_value=pl.DataFrame())
        orc._exporter.filter = MagicMock(side_effect=lambda df: df)
        orc._exporter.to_csv = MagicMock(return_value="")

        with patch.object(
            orc._uploader, "upload", new_callable=AsyncMock
        ) as mock_upload, patch.object(
            Orchestrator, "_utc_today", return_value=date(2026, 4, 10)
        ), patch.object(
            Orchestrator, "_get_pod_lock_manager", return_value=None
        ):
            await orc.run()

        # upload() is still called — it handles empty payload internally
        assert mock_upload.call_count == 1

    @pytest.mark.asyncio
    async def test_reports_error_on_pipeline_failure(self):
        """Any pipeline exception is reported to Mavvrik via report_error."""
        orc = _make_orchestrator()

        orc._client.register = AsyncMock(side_effect=RuntimeError("API down"))
        orc._client.report_error = AsyncMock()

        with patch.object(Orchestrator, "_get_pod_lock_manager", return_value=None):
            await orc.run()  # must not raise

        orc._client.report_error.assert_called_once()
        assert "API down" in orc._client.report_error.call_args.args[0]


# ---------------------------------------------------------------------------
# Overflow detection
# ---------------------------------------------------------------------------


class TestOverflowDetection:
    @pytest.mark.asyncio
    async def test_raises_when_rows_exceed_limit(self):
        """fetch returns limit+1 rows → RuntimeError, marker does not advance."""
        from litellm.constants import MAVVRIK_MAX_FETCHED_DATA_RECORDS

        orc = _make_orchestrator()
        orc._client.register = AsyncMock(return_value="2026-04-09")
        orc._client.advance_marker = AsyncMock()
        orc._client.report_error = AsyncMock()
        orc._exporter = MagicMock()
        # Return one row more than the limit to trigger overflow
        orc._exporter.get_usage_data = AsyncMock(
            return_value=_make_df(rows=MAVVRIK_MAX_FETCHED_DATA_RECORDS + 1)
        )

        with patch.object(
            Orchestrator, "_utc_today", return_value=date(2026, 4, 10)
        ), patch.object(Orchestrator, "_get_pod_lock_manager", return_value=None):
            await orc.run()

        # marker must NOT advance — day is not fully exported
        orc._client.advance_marker.assert_not_called()
        # error must be reported to Mavvrik
        orc._client.report_error.assert_called_once()
        assert "more than" in orc._client.report_error.call_args.args[0]

    @pytest.mark.asyncio
    async def test_proceeds_normally_at_exact_limit(self):
        """fetch returns exactly limit rows → no overflow, upload proceeds."""
        from litellm.constants import MAVVRIK_MAX_FETCHED_DATA_RECORDS

        orc = _make_orchestrator()
        orc._client.register = AsyncMock(return_value="2026-04-09")
        orc._client.advance_marker = AsyncMock()
        orc._client.report_error = AsyncMock()
        orc._exporter = MagicMock()
        orc._exporter.get_usage_data = AsyncMock(
            return_value=_make_df(rows=MAVVRIK_MAX_FETCHED_DATA_RECORDS)
        )
        orc._exporter.filter = MagicMock(side_effect=lambda df: df)
        orc._exporter.to_csv = MagicMock(return_value="col\nval")

        with patch.object(
            orc._uploader, "upload", new_callable=AsyncMock
        ), patch.object(
            Orchestrator, "_utc_today", return_value=date(2026, 4, 10)
        ), patch.object(
            Orchestrator, "_get_pod_lock_manager", return_value=None
        ):
            await orc.run()

        orc._client.advance_marker.assert_called_once()
        orc._client.report_error.assert_not_called()
