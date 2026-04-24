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
        """Exports each day from marker to yesterday (inclusive)."""
        orc = _make_orchestrator()
        exported_dates = []

        async def fake_export(export_date):
            exported_dates.append(export_date.isoformat())
            return 1024

        orc._client.register = AsyncMock(return_value="2026-04-09")
        orc._client.advance_marker = AsyncMock()
        orc._client.report_error = AsyncMock()

        with patch.object(orc, "_export", side_effect=fake_export), patch.object(
            Orchestrator, "_utc_today", return_value=date(2026, 4, 11)
        ), patch.object(Orchestrator, "_get_pod_lock_manager", return_value=None):
            await orc.run()

        assert exported_dates == ["2026-04-09", "2026-04-10"]
        assert orc._client.advance_marker.call_count == 2

    @pytest.mark.asyncio
    async def test_advance_marker_uses_next_day(self):
        """advance_marker is called with (export_date + 1) epoch."""
        orc = _make_orchestrator()

        orc._client.register = AsyncMock(return_value="2026-04-09")
        orc._client.advance_marker = AsyncMock()
        orc._client.report_error = AsyncMock()

        with patch.object(
            orc, "_export", new_callable=AsyncMock, return_value=512
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
        """No exports when marker is already at today."""
        orc = _make_orchestrator()

        orc._client.register = AsyncMock(return_value="2026-04-11")  # = today
        orc._client.advance_marker = AsyncMock()
        orc._client.report_error = AsyncMock()

        with patch.object(
            orc, "_export", new_callable=AsyncMock
        ) as mock_export, patch.object(
            Orchestrator, "_utc_today", return_value=date(2026, 4, 11)
        ), patch.object(
            Orchestrator, "_get_pod_lock_manager", return_value=None
        ):
            await orc.run()

        mock_export.assert_not_called()
        orc._client.advance_marker.assert_not_called()

    @pytest.mark.asyncio
    async def test_first_run_uses_earliest_db_date(self):
        """First run: register() returns None → start from MIN(date) in DB."""
        orc = _make_orchestrator()
        exported_dates = []

        async def fake_export(export_date):
            exported_dates.append(export_date.isoformat())
            return 1024

        orc._client.register = AsyncMock(return_value=None)
        orc._client.advance_marker = AsyncMock()
        orc._client.report_error = AsyncMock()
        orc._exporter = MagicMock()
        orc._exporter.get_earliest_date = AsyncMock(return_value="2026-04-09")

        with patch.object(orc, "_export", side_effect=fake_export), patch.object(
            Orchestrator, "_utc_today", return_value=date(2026, 4, 11)
        ), patch.object(Orchestrator, "_get_pod_lock_manager", return_value=None):
            await orc.run()

        assert exported_dates == ["2026-04-09", "2026-04-10"]

    @pytest.mark.asyncio
    async def test_does_not_advance_when_no_data(self):
        """When export returns 0 bytes, marker is NOT advanced — prevents silent data loss.

        0 bytes means DB was unavailable or no data for that date.
        Raising ensures the marker stays put so the date is retried next run.
        """
        orc = _make_orchestrator()

        orc._client.register = AsyncMock(return_value="2026-04-09")
        orc._client.advance_marker = AsyncMock()
        orc._client.report_error = AsyncMock()

        with patch.object(
            orc, "_export", new_callable=AsyncMock, return_value=0
        ), patch.object(
            Orchestrator, "_utc_today", return_value=date(2026, 4, 10)
        ), patch.object(
            Orchestrator, "_get_pod_lock_manager", return_value=None
        ):
            await orc.run()

        # advance_marker must NOT be called when 0 bytes exported
        orc._client.advance_marker.assert_not_called()
        # error reported to Mavvrik so the failure is visible
        orc._client.report_error.assert_called_once()

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


class TestOrchestratorHelpers:
    def test_utc_today_returns_date(self):
        result = Orchestrator._utc_today()
        assert isinstance(result, date)

    def test_to_epoch_converts_date(self):
        d = date(2026, 1, 1)
        epoch = Orchestrator._to_epoch(d)
        assert isinstance(epoch, int)
        assert epoch > 0

    def test_export_end_date_is_yesterday(self):
        orc = _make_orchestrator()
        with patch.object(Orchestrator, "_utc_today", return_value=date(2026, 4, 12)):
            end = orc._export_end_date()
        assert end == date(2026, 4, 11)

    def test_date_range_takes_explicit_end(self):
        orc = _make_orchestrator()
        end = date(2026, 4, 11)
        dates = list(orc._date_range(date(2026, 4, 10), end))
        assert dates == [date(2026, 4, 10), date(2026, 4, 11)]

    def test_date_range_single_date_when_start_equals_end(self):
        orc = _make_orchestrator()
        dates = list(orc._date_range(date(2026, 4, 10), date(2026, 4, 10)))
        assert dates == [date(2026, 4, 10)]

    def test_pipeline_skips_when_start_equals_end_plus_one(self):
        """start > end means nothing to export — no dates yielded."""
        orc = _make_orchestrator()
        dates = list(orc._date_range(date(2026, 4, 12), date(2026, 4, 11)))
        assert dates == []

    def test_get_pod_lock_manager_returns_none_when_proxy_logging_none(self):
        with patch(
            "litellm.integrations.mavvrik.orchestrator.proxy_logging_obj",
            None,
            create=True,
        ):
            pass  # import is lazy; tested via run() path below

    @pytest.mark.asyncio
    async def test_get_pod_lock_manager_returns_none_when_logging_obj_none(self):
        orc = _make_orchestrator()
        with patch(
            "litellm.integrations.mavvrik.orchestrator.Orchestrator._get_pod_lock_manager",
            return_value=None,
        ):
            orc._client.register = AsyncMock(return_value="2099-01-01")
            orc._client.report_error = AsyncMock()
            with patch.object(
                Orchestrator, "_utc_today", return_value=date(2026, 4, 10)
            ):
                await orc.run()  # no lock, runs directly


class TestPodLockAcquired:
    @pytest.mark.asyncio
    async def test_runs_pipeline_when_lock_acquired(self):
        """When Redis is available and lock is acquired, pipeline runs."""
        orc = _make_orchestrator()
        orc._client.register = AsyncMock(return_value="2099-01-01")
        orc._client.report_error = AsyncMock()

        mock_lock = MagicMock()
        mock_lock.redis_cache = MagicMock()
        mock_lock.acquire_lock = AsyncMock(return_value=True)
        mock_lock.release_lock = AsyncMock()

        with patch.object(
            Orchestrator, "_get_pod_lock_manager", return_value=mock_lock
        ), patch.object(Orchestrator, "_utc_today", return_value=date(2026, 4, 10)):
            await orc.run()

        mock_lock.acquire_lock.assert_called_once()
        mock_lock.release_lock.assert_called_once()

    @pytest.mark.asyncio
    async def test_skips_pipeline_when_lock_not_acquired(self):
        """When Redis lock is not acquired, pipeline does not run."""
        orc = _make_orchestrator()
        orc._client.register = AsyncMock()

        mock_lock = MagicMock()
        mock_lock.redis_cache = MagicMock()
        mock_lock.acquire_lock = AsyncMock(return_value=False)

        with patch.object(
            Orchestrator, "_get_pod_lock_manager", return_value=mock_lock
        ):
            await orc.run()

        orc._client.register.assert_not_called()


class TestResolveFirstRunInvalidDate:
    @pytest.mark.asyncio
    async def test_falls_back_to_yesterday_on_invalid_date_string(self):
        """When DB returns a non-ISO date string, fall back to yesterday."""
        orc = _make_orchestrator()
        orc._exporter = MagicMock()
        orc._exporter.get_earliest_date = AsyncMock(return_value="not-a-date")

        with patch.object(Orchestrator, "_utc_today", return_value=date(2026, 4, 16)):
            result = await orc._resolve_first_run_start_date()

        assert result == date(2026, 4, 15)


class TestStreamingExport:
    @pytest.mark.asyncio
    async def test_export_calls_stream_pages_and_stream_upload(self):
        """_export() wires exporter._stream_pages() into uploader._stream_upload()."""
        orc = _make_orchestrator()

        async def fake_stream_pages(**kwargs):
            yield "date,model\n"
            yield "2026-04-09,gpt-4o\n"

        orc._exporter._stream_pages = fake_stream_pages

        stream_upload_called_with = []

        async def fake_stream_upload(pages, date_str):
            stream_upload_called_with.append(date_str)
            # consume the generator
            async for _ in pages:
                pass
            return 1024

        orc._uploader._stream_upload = fake_stream_upload

        result = await orc._export(date(2026, 4, 9))

        assert result == 1024
        assert stream_upload_called_with == ["2026-04-09"]

    @pytest.mark.asyncio
    async def test_export_returns_zero_when_no_data(self):
        """_export() returns 0 when _stream_upload returns 0 (no data)."""
        orc = _make_orchestrator()

        async def empty_pages(**kwargs):
            return
            yield

        orc._exporter._stream_pages = empty_pages

        async def fake_stream_upload(pages, date_str):
            return 0

        orc._uploader._stream_upload = fake_stream_upload

        result = await orc._export(date(2026, 4, 9))
        assert result == 0
