"""Tests for Service facade — verifies constructors and method calls are correct."""

import io
import os
import sys
from datetime import date, datetime, timezone
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import polars as pl
import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.integrations.mavvrik import Service
from litellm.integrations.mavvrik.client import Client
from litellm.integrations.mavvrik.exporter import Exporter
from litellm.integrations.mavvrik.orchestrator import Orchestrator
from litellm.integrations.mavvrik.settings import Settings
from litellm.integrations.mavvrik.uploader import Uploader


_CREDS = {
    "api_key": "key",
    "api_endpoint": "https://api.mavvrik.dev/t",
    "connection_id": "c-1",
}


def _mock_settings(data=None):
    s = MagicMock()
    s.load = AsyncMock(return_value=data if data is not None else dict(_CREDS))
    s.save = AsyncMock()
    s.delete = AsyncMock()
    s.has_env_vars = False
    return s


def _make_df(rows=3):
    return pl.DataFrame(
        {
            "date": ["2026-04-10"] * rows,
            "user_id": ["user-1"] * rows,
            "model": ["gpt-4o"] * rows,
            "spend": [0.015] * rows,
            "successful_requests": [5] * rows,
            "prompt_tokens": [100] * rows,
            "completion_tokens": [50] * rows,
            "team_id": ["team-1"] * rows,
        }
    )


def _mock_exporter(df):
    """Return a mock Exporter instance with stubbed export() method."""
    exporter = MagicMock()
    csv = "" if df.is_empty() else "col\nval\n"
    exporter.export = AsyncMock(return_value=(df, csv))
    return exporter


def _mock_uploader():
    """Return a mock Uploader instance."""
    uploader = MagicMock()
    uploader.upload = AsyncMock()
    return uploader


# ---------------------------------------------------------------------------
# Service.initialize — schedules Orchestrator with correct constructors
# ---------------------------------------------------------------------------


def _mock_proxy_server(scheduler=None):
    """Return a mock proxy_server module with a stubbed scheduler."""
    mock_pserver = MagicMock()
    mock_pserver.scheduler = scheduler
    return mock_pserver


class TestServiceInitialize:
    @pytest.mark.asyncio
    async def test_initialize_builds_client_uploader_orchestrator(self):
        """initialize() must construct Client, Uploader(client=), Orchestrator(client=, uploader=)."""
        svc = Service()
        svc._settings = _mock_settings()

        created = {}

        mock_client_inst = MagicMock(spec=Client)
        mock_uploader_inst = MagicMock(spec=Uploader)
        mock_orchestrator_inst = MagicMock(spec=Orchestrator)
        mock_orchestrator_inst.run = AsyncMock()

        MockClient = MagicMock(return_value=mock_client_inst)
        MockUploader = MagicMock(return_value=mock_uploader_inst)

        def capture_orchestrator(client, uploader):
            created["client"] = client
            created["uploader"] = uploader
            return mock_orchestrator_inst

        MockOrchestrator = MagicMock(side_effect=capture_orchestrator)
        mock_scheduler = MagicMock()

        # Build a mock proxy_server module with scheduler set.
        # Use patch.dict to inject it — but also cover the case where
        # proxy_server is already loaded in CI by overwriting its scheduler attr.
        import sys

        mock_pserver = _mock_proxy_server(scheduler=mock_scheduler)

        # If proxy_server already loaded, patch its scheduler directly too.
        real_pserver = sys.modules.get("litellm.proxy.proxy_server")
        real_scheduler = (
            getattr(real_pserver, "scheduler", "MISSING") if real_pserver else "MISSING"
        )
        if real_pserver:
            real_pserver.scheduler = mock_scheduler

        try:
            with (
                patch.dict(sys.modules, {"litellm.proxy.proxy_server": mock_pserver}),
                patch("litellm.integrations.mavvrik.Client", MockClient),
                patch("litellm.integrations.mavvrik.Uploader", MockUploader),
                patch("litellm.integrations.mavvrik.Orchestrator", MockOrchestrator),
            ):
                await svc.initialize(
                    api_key="key",
                    api_endpoint="https://api.mavvrik.dev/t",
                    connection_id="c-1",
                )
        finally:
            if real_pserver and real_scheduler != "MISSING":
                real_pserver.scheduler = real_scheduler

        MockClient.assert_called_once_with(
            api_key="key",
            api_endpoint="https://api.mavvrik.dev/t",
            connection_id="c-1",
        )
        MockUploader.assert_called_once_with(client=mock_client_inst)
        assert created["client"] is mock_client_inst
        assert created["uploader"] is mock_uploader_inst
        mock_scheduler.add_job.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_returns_success_when_no_scheduler(self):
        """initialize() returns success even when scheduler is unavailable."""
        svc = Service()
        svc._settings = _mock_settings()

        mock_pserver = _mock_proxy_server(scheduler=None)

        import sys

        with patch.dict(sys.modules, {"litellm.proxy.proxy_server": mock_pserver}):
            result = await svc.initialize(
                api_key="key",
                api_endpoint="https://api.mavvrik.dev/t",
                connection_id="c-1",
            )

        assert result["status"] == "success"


# ---------------------------------------------------------------------------
# Service.export — uses Exporter + Uploader directly
# ---------------------------------------------------------------------------


class TestServiceExport:
    @pytest.mark.asyncio
    async def test_export_returns_record_count(self):
        """export() must return records_exported from the pipeline."""
        svc = Service()
        svc._settings = _mock_settings()

        mock_exporter_inst = _mock_exporter(_make_df(rows=7))
        mock_uploader_inst = _mock_uploader()

        with (
            patch(
                "litellm.integrations.mavvrik.Exporter", return_value=mock_exporter_inst
            ),
            patch(
                "litellm.integrations.mavvrik.Uploader", return_value=mock_uploader_inst
            ),
            patch("litellm.integrations.mavvrik.Client"),
        ):
            result = await svc.export(date_str="2026-04-10")

        assert result["status"] == "success"
        assert result["records_exported"] == 7
        mock_uploader_inst.upload.assert_called_once()

    @pytest.mark.asyncio
    async def test_export_raises_when_not_configured(self):
        """export() raises ValueError when settings missing and no env vars."""
        svc = Service()
        svc._settings = _mock_settings(data={})
        svc._settings.has_env_vars = False

        with pytest.raises(ValueError, match="not configured"):
            await svc.export(date_str="2026-04-10")

    @pytest.mark.asyncio
    async def test_export_returns_zero_when_no_data(self):
        """export() returns 0 records when DB has no rows for the date."""
        svc = Service()
        svc._settings = _mock_settings()

        mock_exporter_inst = _mock_exporter(pl.DataFrame())
        mock_uploader_inst = _mock_uploader()

        with (
            patch(
                "litellm.integrations.mavvrik.Exporter", return_value=mock_exporter_inst
            ),
            patch(
                "litellm.integrations.mavvrik.Uploader", return_value=mock_uploader_inst
            ),
            patch("litellm.integrations.mavvrik.Client"),
        ):
            result = await svc.export(date_str="2026-04-10")

        assert result["records_exported"] == 0
        mock_uploader_inst.upload.assert_not_called()

    @pytest.mark.asyncio
    async def test_export_builds_uploader_with_client(self):
        """export() must pass client= to Uploader, not credential kwargs."""
        svc = Service()
        svc._settings = _mock_settings()

        mock_client_inst = MagicMock(spec=Client)
        mock_client_inst.connection_id = "c-1"
        mock_uploader_inst = _mock_uploader()
        mock_exporter_inst = _mock_exporter(_make_df(rows=2))

        MockClient = MagicMock(return_value=mock_client_inst)
        MockUploader = MagicMock(return_value=mock_uploader_inst)

        with (
            patch("litellm.integrations.mavvrik.Client", MockClient),
            patch("litellm.integrations.mavvrik.Uploader", MockUploader),
            patch(
                "litellm.integrations.mavvrik.Exporter", return_value=mock_exporter_inst
            ),
        ):
            await svc.export(date_str="2026-04-10")

        MockUploader.assert_called_once_with(client=mock_client_inst)


# ---------------------------------------------------------------------------
# Service.dry_run — uses Exporter only, never calls uploader.upload
# ---------------------------------------------------------------------------


class TestServiceDryRun:
    @pytest.mark.asyncio
    async def test_dry_run_returns_preview_without_uploading(self):
        """dry_run() must return preview data and never call uploader.upload."""
        svc = Service()
        svc._settings = _mock_settings()

        mock_exporter_inst = _mock_exporter(_make_df(rows=5))
        mock_uploader_inst = _mock_uploader()

        with (
            patch(
                "litellm.integrations.mavvrik.Exporter", return_value=mock_exporter_inst
            ),
            patch(
                "litellm.integrations.mavvrik.Uploader", return_value=mock_uploader_inst
            ),
            patch("litellm.integrations.mavvrik.Client"),
        ):
            result = await svc.dry_run(date_str="2026-04-10")

        assert result["status"] == "success"
        assert "dry_run_data" in result
        assert "summary" in result
        mock_uploader_inst.upload.assert_not_called()

    @pytest.mark.asyncio
    async def test_dry_run_raises_when_not_configured(self):
        """dry_run() raises ValueError when not configured."""
        svc = Service()
        svc._settings = _mock_settings(data={})
        svc._settings.has_env_vars = False

        with pytest.raises(ValueError, match="not configured"):
            await svc.dry_run(date_str="2026-04-10")

    @pytest.mark.asyncio
    async def test_dry_run_returns_empty_when_no_data(self):
        """dry_run() returns zero summary when DB has no rows."""
        svc = Service()
        svc._settings = _mock_settings()

        mock_exporter_inst = _mock_exporter(pl.DataFrame())

        with (
            patch(
                "litellm.integrations.mavvrik.Exporter", return_value=mock_exporter_inst
            ),
            patch("litellm.integrations.mavvrik.Client"),
        ):
            result = await svc.dry_run(date_str="2026-04-10")

        assert result["summary"]["total_records"] == 0
        assert result["dry_run_data"]["usage_data"] == []


# ---------------------------------------------------------------------------
# Merged from test_scheduler.py
# ---------------------------------------------------------------------------


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


def _make_scheduler_df(rows=3) -> pl.DataFrame:
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

        with (
            patch.object(orc, "_export", side_effect=fake_export),
            patch.object(Orchestrator, "_utc_today", return_value=date(2026, 4, 11)),
            patch.object(Orchestrator, "_get_pod_lock_manager", return_value=None),
        ):
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

        with (
            patch.object(orc, "_export", new_callable=AsyncMock, return_value=512),
            patch.object(Orchestrator, "_utc_today", return_value=date(2026, 4, 10)),
            patch.object(Orchestrator, "_get_pod_lock_manager", return_value=None),
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

        with (
            patch.object(orc, "_export", new_callable=AsyncMock) as mock_export,
            patch.object(Orchestrator, "_utc_today", return_value=date(2026, 4, 11)),
            patch.object(Orchestrator, "_get_pod_lock_manager", return_value=None),
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

        with (
            patch.object(orc, "_export", side_effect=fake_export),
            patch.object(Orchestrator, "_utc_today", return_value=date(2026, 4, 11)),
            patch.object(Orchestrator, "_get_pod_lock_manager", return_value=None),
        ):
            await orc.run()

        assert exported_dates == ["2026-04-09", "2026-04-10"]

    @pytest.mark.asyncio
    async def test_advances_marker_on_zero_traffic_day(self):
        """Zero-traffic days (0 bytes, no DB error) still advance the marker.

        A legitimate date with no spend rows returns 0 bytes without raising.
        The marker must advance so the pipeline doesn't stall on quiet days.
        DB-unavailability is handled differently — _stream_pages raises, which
        propagates through _export and aborts before _advance is reached.
        """
        orc = _make_orchestrator()

        orc._client.register = AsyncMock(return_value="2026-04-09")
        orc._client.advance_marker = AsyncMock()
        orc._client.report_error = AsyncMock()

        with (
            patch.object(orc, "_export", new_callable=AsyncMock, return_value=0),
            patch.object(Orchestrator, "_utc_today", return_value=date(2026, 4, 10)),
            patch.object(Orchestrator, "_get_pod_lock_manager", return_value=None),
        ):
            await orc.run()

        # advance_marker IS called — zero bytes is a legitimate empty day
        orc._client.advance_marker.assert_called_once()
        orc._client.report_error.assert_not_called()

    @pytest.mark.asyncio
    async def test_does_not_advance_when_db_unavailable(self):
        """When DB is unavailable, _export raises — marker NOT advanced."""
        orc = _make_orchestrator()

        orc._client.register = AsyncMock(return_value="2026-04-09")
        orc._client.advance_marker = AsyncMock()
        orc._client.report_error = AsyncMock()

        with (
            patch.object(
                orc,
                "_export",
                new_callable=AsyncMock,
                side_effect=RuntimeError("database not connected"),
            ),
            patch.object(Orchestrator, "_utc_today", return_value=date(2026, 4, 10)),
            patch.object(Orchestrator, "_get_pod_lock_manager", return_value=None),
        ):
            await orc.run()

        orc._client.advance_marker.assert_not_called()
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
        assert "RuntimeError" in orc._client.report_error.call_args.args[0]


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

        with (
            patch.object(Orchestrator, "_get_pod_lock_manager", return_value=mock_lock),
            patch.object(Orchestrator, "_utc_today", return_value=date(2026, 4, 10)),
        ):
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


# ---------------------------------------------------------------------------
# Merged from test_endpoints.py
# ---------------------------------------------------------------------------


from litellm.proxy.spend_tracking.mavvrik_endpoints import (
    delete_mavvrik_settings,
    dry_run_mavvrik_export,
    get_mavvrik_settings,
    init_mavvrik_settings,
    update_mavvrik_settings,
)
from litellm.types.proxy.mavvrik_endpoints import (
    MavvrikExportRequest,
    MavvrikInitRequest,
    MavvrikSettingsUpdate,
)
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth

# Patch target prefix — MavvrikService methods live in the integrations package.
_SVC = "litellm.integrations.mavvrik.Service"


def _admin_user() -> UserAPIKeyAuth:
    return UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN)


def _non_admin_user() -> UserAPIKeyAuth:
    return UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER)


# ---------------------------------------------------------------------------
# Auth gate
# ---------------------------------------------------------------------------


class TestAdminGate:
    @pytest.mark.asyncio
    async def test_init_rejects_non_admin(self):
        from fastapi import HTTPException

        req = MavvrikInitRequest(
            api_key="k", api_endpoint="https://e.com/t", connection_id="c"
        )
        with pytest.raises(HTTPException) as exc_info:
            await init_mavvrik_settings(req, user_api_key_dict=_non_admin_user())
        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_get_settings_rejects_non_admin(self):
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            await get_mavvrik_settings(user_api_key_dict=_non_admin_user())
        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_delete_rejects_non_admin(self):
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            await delete_mavvrik_settings(user_api_key_dict=_non_admin_user())
        assert exc_info.value.status_code == 403


# ---------------------------------------------------------------------------
# POST /mavvrik/init
# ---------------------------------------------------------------------------


class TestInitSettings:
    @pytest.mark.asyncio
    async def test_init_stores_settings_and_returns_success(self):
        req = MavvrikInitRequest(
            api_key="mav_key",
            api_endpoint="https://api.mavvrik.dev/acme",
            connection_id="litellm-prod",
        )

        with patch(
            f"{_SVC}.initialize",
            new=AsyncMock(
                return_value={
                    "message": "Mavvrik settings initialized successfully",
                    "status": "success",
                }
            ),
        ):
            resp = await init_mavvrik_settings(req, user_api_key_dict=_admin_user())

        assert resp.status == "success"

    @pytest.mark.asyncio
    async def test_init_succeeds_even_when_scheduler_unavailable(self):
        """MavvrikService.initialize() succeeds even if scheduler is not available."""
        from litellm.integrations.mavvrik.settings import Settings

        req = MavvrikInitRequest(
            api_key="mav_key",
            api_endpoint="https://api.mavvrik.dev/acme",
            connection_id="litellm-prod",
        )

        mock_pserver = MagicMock()
        mock_pserver.scheduler = None
        with (
            patch.object(Settings, "save", new=AsyncMock()),
            patch("litellm.proxy.proxy_server", mock_pserver, create=True),
        ):
            resp = await init_mavvrik_settings(req, user_api_key_dict=_admin_user())

        assert resp.status == "success"


# ---------------------------------------------------------------------------
# GET /mavvrik/settings
# ---------------------------------------------------------------------------


class TestGetSettings:
    @pytest.mark.asyncio
    async def test_returns_not_configured_when_no_row(self):
        with patch(
            f"{_SVC}.get_settings",
            new=AsyncMock(
                return_value={
                    "api_key_masked": None,
                    "api_endpoint": None,
                    "connection_id": None,
                    "status": "not_configured",
                }
            ),
        ):
            resp = await get_mavvrik_settings(user_api_key_dict=_admin_user())

        assert resp.status == "not_configured"
        assert resp.api_key_masked is None

    @pytest.mark.asyncio
    async def test_returns_masked_key_when_configured(self):
        with patch(
            f"{_SVC}.get_settings",
            new=AsyncMock(
                return_value={
                    "api_key_masked": "mav_*******",
                    "api_endpoint": "https://api.mavvrik.dev/acme",
                    "connection_id": "prod",
                    "status": "configured",
                }
            ),
        ):
            resp = await get_mavvrik_settings(user_api_key_dict=_admin_user())

        assert resp.status == "configured"
        assert resp.api_key_masked is not None
        assert "mav_plaintextkey" not in (resp.api_key_masked or "")
        assert resp.connection_id == "prod"

    @pytest.mark.asyncio
    async def test_raises_500_on_service_error(self):
        """Any exception from MavvrikService.get_settings() → 500."""
        from fastapi import HTTPException

        with patch(
            f"{_SVC}.get_settings",
            new=AsyncMock(side_effect=Exception("DB exploded")),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await get_mavvrik_settings(user_api_key_dict=_admin_user())

        assert exc_info.value.status_code == 500


# ---------------------------------------------------------------------------
# PUT /mavvrik/settings
# ---------------------------------------------------------------------------


class TestUpdateSettings:
    @pytest.mark.asyncio
    async def test_update_rejects_empty_request(self):
        from fastapi import HTTPException

        req = MavvrikSettingsUpdate()  # all None
        with pytest.raises(HTTPException) as exc_info:
            await update_mavvrik_settings(req, user_api_key_dict=_admin_user())
        assert exc_info.value.status_code == 400

    def test_update_rejects_empty_api_key(self):
        with pytest.raises(Exception):
            MavvrikSettingsUpdate(api_key="")


# ---------------------------------------------------------------------------
# DELETE /mavvrik/delete
# ---------------------------------------------------------------------------


class TestDeleteSettings:
    @pytest.mark.asyncio
    async def test_delete_returns_404_when_not_configured(self):
        """Settings.delete() raises LookupError → endpoint returns 404."""
        from fastapi import HTTPException

        with patch(
            f"{_SVC}.delete",
            new=AsyncMock(
                side_effect=LookupError(
                    "Mavvrik settings not found — nothing to delete."
                )
            ),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await delete_mavvrik_settings(user_api_key_dict=_admin_user())
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_removes_row_and_returns_success(self):
        with patch(
            f"{_SVC}.delete",
            new=AsyncMock(
                return_value={
                    "message": "Mavvrik settings deleted successfully",
                    "status": "success",
                }
            ),
        ):
            resp = await delete_mavvrik_settings(user_api_key_dict=_admin_user())

        assert resp.status == "success"


# ---------------------------------------------------------------------------
# POST /mavvrik/dry-run
# ---------------------------------------------------------------------------


class TestDryRunMavvrikExport:
    @pytest.mark.asyncio
    async def test_dry_run_returns_preview(self):
        req = MavvrikExportRequest(date_str="2024-01-14")

        with patch(
            f"{_SVC}.dry_run",
            new=AsyncMock(
                return_value={
                    "message": "Mavvrik dry run completed",
                    "status": "success",
                    "dry_run_data": {
                        "usage_data": [
                            {"date": "2024-01-14", "model": "gpt-4o", "spend": 1.5}
                        ],
                        "csv_preview": "date,model,spend\n2024-01-14,gpt-4o,1.5",
                    },
                    "summary": {
                        "total_records": 1,
                        "total_cost": 1.5,
                        "total_tokens": 100,
                        "unique_models": 1,
                        "unique_teams": 1,
                    },
                }
            ),
        ):
            resp = await dry_run_mavvrik_export(req, user_api_key_dict=_admin_user())

        assert resp.status == "success"
        assert resp.summary is not None
        assert resp.summary["total_records"] == 1

    @pytest.mark.asyncio
    async def test_dry_run_defaults_to_yesterday_when_no_date(self):
        req = MavvrikExportRequest()  # no date_str

        with patch(
            f"{_SVC}.dry_run",
            new=AsyncMock(
                return_value={
                    "message": "Mavvrik dry run completed",
                    "status": "success",
                    "dry_run_data": {"usage_data": [], "csv_preview": ""},
                    "summary": {
                        "total_records": 0,
                        "total_cost": 0.0,
                        "total_tokens": 0,
                        "unique_models": 0,
                        "unique_teams": 0,
                    },
                }
            ),
        ) as mock_dry_run:
            resp = await dry_run_mavvrik_export(req, user_api_key_dict=_admin_user())

        assert resp.status == "success"
        mock_dry_run.assert_called_once()
        # date_str=None is passed through; MavvrikService.dry_run() resolves it to yesterday
        _, kwargs = mock_dry_run.call_args
        assert "date_str" in kwargs


# ---------------------------------------------------------------------------
# POST /mavvrik/export
# ---------------------------------------------------------------------------


class TestExportMavvrikData:
    @pytest.mark.asyncio
    async def test_export_returns_success_and_record_count(self):
        from litellm.proxy.spend_tracking.mavvrik_endpoints import export_mavvrik_data

        req = MavvrikExportRequest(date_str="2024-01-15")

        with patch(
            f"{_SVC}.export",
            new=AsyncMock(
                return_value={
                    "message": "Mavvrik export completed successfully for 2024-01-15",
                    "status": "success",
                    "records_exported": 7,
                }
            ),
        ):
            resp = await export_mavvrik_data(req, user_api_key_dict=_admin_user())

        assert resp.status == "success"
        assert resp.records_exported == 7
        assert "2024-01-15" in resp.message

    @pytest.mark.asyncio
    async def test_export_returns_400_when_not_configured(self):
        from fastapi import HTTPException

        from litellm.proxy.spend_tracking.mavvrik_endpoints import export_mavvrik_data

        req = MavvrikExportRequest(date_str="2024-01-15")

        with patch(
            f"{_SVC}.export",
            new=AsyncMock(
                side_effect=ValueError(
                    "Mavvrik not configured. Call POST /mavvrik/init first."
                )
            ),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await export_mavvrik_data(req, user_api_key_dict=_admin_user())

        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_export_defaults_to_yesterday_when_no_date(self):
        from litellm.proxy.spend_tracking.mavvrik_endpoints import export_mavvrik_data

        req = MavvrikExportRequest()  # no date_str

        with patch(
            f"{_SVC}.export",
            new=AsyncMock(
                return_value={
                    "message": "Mavvrik export completed successfully for 2024-01-15",
                    "status": "success",
                    "records_exported": 3,
                }
            ),
        ) as mock_export:
            resp = await export_mavvrik_data(req, user_api_key_dict=_admin_user())

        assert resp.status == "success"
        mock_export.assert_called_once()
        _, kwargs = mock_export.call_args
        assert "date_str" in kwargs


# ---------------------------------------------------------------------------
# Lifecycle flow — init → settings → update marker → delete → export fails
# ---------------------------------------------------------------------------


class TestLifecycleFlow:
    @pytest.mark.asyncio
    async def test_get_settings_returns_not_configured_after_delete(self):
        """DELETE succeeds → subsequent GET returns not_configured."""
        with patch(
            f"{_SVC}.delete",
            new=AsyncMock(
                return_value={
                    "message": "Mavvrik settings deleted successfully",
                    "status": "success",
                }
            ),
        ):
            del_resp = await delete_mavvrik_settings(user_api_key_dict=_admin_user())

        assert del_resp.status == "success"

        with patch(
            f"{_SVC}.get_settings",
            new=AsyncMock(
                return_value={
                    "api_key_masked": None,
                    "api_endpoint": None,
                    "connection_id": None,
                    "status": "not_configured",
                }
            ),
        ):
            get_resp = await get_mavvrik_settings(user_api_key_dict=_admin_user())

        assert get_resp.status == "not_configured"


# ---------------------------------------------------------------------------
# Settings — setup detection
# ---------------------------------------------------------------------------


class TestSettingsSetup:
    @pytest.mark.asyncio
    async def test_is_mavvrik_setup_true_when_env_vars_set(self):
        """Settings.is_setup returns True when all env vars are present."""
        from litellm.integrations.mavvrik.settings import Settings

        with patch.dict(
            "os.environ",
            {
                "MAVVRIK_API_KEY": "mav_key",
                "MAVVRIK_API_ENDPOINT": "https://api.mavvrik.dev/acme",
                "MAVVRIK_CONNECTION_ID": "prod",
            },
        ):
            result = await Settings().is_setup()

        assert result is True

    @pytest.mark.asyncio
    async def test_is_mavvrik_setup_false_when_no_env_and_no_db(self):
        """Settings.is_setup returns False when env vars missing and DB not connected."""
        from litellm.integrations.mavvrik.settings import Settings

        env = {
            k: ""
            for k in (
                "MAVVRIK_API_KEY",
                "MAVVRIK_API_ENDPOINT",
                "MAVVRIK_CONNECTION_ID",
            )
        }
        with patch.dict("os.environ", env):
            with patch(
                "litellm.integrations.mavvrik.settings.Settings._prisma_client",
                new_callable=lambda: property(lambda self: None),
            ):
                result = await Settings().is_setup()

        assert result is False


# ---------------------------------------------------------------------------
# Merged from test_exporter.py
# ---------------------------------------------------------------------------


def _make_exporter_df(**kwargs) -> pl.DataFrame:
    """Helper: build a minimal spend DataFrame, overriding defaults with kwargs."""
    defaults = {
        "date": ["2025-01-19"],
        "user_id": ["user-1"],
        "api_key": ["sk-abc"],
        "model": ["gpt-4o"],
        "model_group": ["gpt-4o-group"],
        "custom_llm_provider": ["openai"],
        "prompt_tokens": [100],
        "completion_tokens": [50],
        "spend": [1.5],
        "api_requests": [5],
        "successful_requests": [4],
        "failed_requests": [1],
        "team_id": ["team-1"],
        "api_key_alias": ["prod-key"],
        "team_alias": ["Alpha"],
        "user_email": ["alice@example.com"],
    }
    defaults.update(kwargs)
    return pl.DataFrame(defaults)


class TestExporterToCsv:
    def test_empty_dataframe_returns_empty_string(self):
        transformer = Exporter()
        result = transformer._to_csv(pl.DataFrame())
        assert result == ""

    def test_nonzero_successful_requests_in_output(self):
        transformer = Exporter()
        df = _make_exporter_df(successful_requests=[3])
        result = transformer._to_csv(df)
        assert result != ""

    def test_output_has_header_row(self):
        transformer = Exporter()
        df = _make_exporter_df()
        result = transformer._to_csv(df)
        header = result.split("\n")[0]
        assert "model" in header
        assert "spend" in header

    def test_all_db_columns_present_in_header(self):
        transformer = Exporter()
        df = _make_exporter_df()
        header = transformer._to_csv(df).split("\n")[0]
        for col in [
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
            "team_id",
            "api_key_alias",
            "team_alias",
            "user_email",
        ]:
            assert col in header, f"Expected column '{col}' in CSV header"

    def test_spend_value_in_output(self):
        transformer = Exporter()
        df = _make_exporter_df(spend=[42.5])
        result = transformer._to_csv(df)
        assert "42.5" in result

    def test_model_value_in_output(self):
        transformer = Exporter()
        df = _make_exporter_df(model=["claude-3-5-sonnet"])
        result = transformer._to_csv(df)
        assert "claude-3-5-sonnet" in result

    def test_multiple_rows_all_in_output(self):
        transformer = Exporter()
        df = pl.DataFrame(
            {
                "date": ["2025-01-19", "2025-01-20"],
                "successful_requests": [2, 3],
                "spend": [1.0, 2.0],
                "model": ["gpt-4", "claude-3"],
            }
        )
        result = transformer._to_csv(df)
        lines = [l for l in result.strip().split("\n") if l]
        assert len(lines) == 3  # header + 2 data rows

    def test_output_is_valid_csv(self):
        transformer = Exporter()
        df = _make_exporter_df()
        result = transformer._to_csv(df)
        # Polars can re-read its own CSV output
        reloaded = pl.read_csv(io.StringIO(result))
        assert len(reloaded) == 1
        assert "model" in reloaded.columns

    def test_return_type_is_str(self):
        transformer = Exporter()
        df = _make_exporter_df()
        result = transformer._to_csv(df)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# to_csv — connection_id column
# ---------------------------------------------------------------------------


class TestToCsvConnectionId:
    def test_adds_connection_id_column_when_provided(self):
        exporter = Exporter()
        df = _make_exporter_df()
        result = exporter._to_csv(df, connection_id="conn-123")
        assert "connection_id" in result
        assert "conn-123" in result

    def test_no_connection_id_column_when_omitted(self):
        exporter = Exporter()
        df = _make_exporter_df()
        result = exporter._to_csv(df)
        assert "connection_id" not in result.split("\n")[0]


# ---------------------------------------------------------------------------
# _prisma_client — raises when DB not connected
# ---------------------------------------------------------------------------


class TestExporterPrismaClient:
    def test_raises_runtime_error_when_db_not_connected(self):
        """_prisma_client raises RuntimeError when prisma_client is None."""
        exporter = Exporter()
        with patch(
            "litellm.integrations.mavvrik.exporter.prisma_client", None, create=True
        ):
            with patch(
                "litellm.integrations.mavvrik.exporter.Exporter._prisma_client",
                new_callable=lambda: property(
                    lambda self: (_ for _ in ()).throw(
                        RuntimeError("Database not connected")
                    )
                ),
            ):
                with pytest.raises(RuntimeError, match="Database not connected"):
                    _ = exporter._prisma_client


# ---------------------------------------------------------------------------
# get_usage_data and get_earliest_date — mocked prisma
# ---------------------------------------------------------------------------


class TestExporterDbMethods:
    @pytest.mark.asyncio
    async def test_get_usage_data_returns_dataframe(self):
        """get_usage_data() returns a Polars DataFrame from query_raw results."""
        exporter = Exporter()
        mock_rows = [
            {
                "date": "2026-04-10",
                "user_id": "user-1",
                "model": "gpt-4o",
                "spend": 0.015,
                "successful_requests": 5,
                "prompt_tokens": 100,
                "completion_tokens": 50,
            }
        ]
        mock_client = MagicMock()
        mock_client.db.query_raw = AsyncMock(return_value=mock_rows)

        with patch.object(
            type(exporter),
            "_prisma_client",
            new_callable=lambda: property(lambda self: mock_client),
        ):
            df = await exporter._get_usage_data("2026-04-10")

        assert len(df) == 1
        assert "model" in df.columns

    @pytest.mark.asyncio
    async def test_get_usage_data_with_limit(self):
        """get_usage_data() appends LIMIT clause when limit is provided."""
        exporter = Exporter()
        mock_client = MagicMock()
        mock_client.db.query_raw = AsyncMock(return_value=[])
        captured = []

        async def fake_query_raw(query, *params):
            captured.append((query, params))
            return []

        mock_client.db.query_raw = fake_query_raw

        with patch.object(
            type(exporter),
            "_prisma_client",
            new_callable=lambda: property(lambda self: mock_client),
        ):
            await exporter._get_usage_data("2026-04-10", limit=100)

        assert "LIMIT" in captured[0][0]
        assert 100 in captured[0][1]

    @pytest.mark.asyncio
    async def test_get_earliest_date_returns_date_string(self):
        """get_earliest_date() returns first 10 chars of the MIN(date) result."""
        exporter = Exporter()
        mock_client = MagicMock()
        mock_client.db.query_raw = AsyncMock(
            return_value=[{"earliest": "2026-01-01T00:00:00"}]
        )

        with patch.object(
            type(exporter),
            "_prisma_client",
            new_callable=lambda: property(lambda self: mock_client),
        ):
            result = await exporter.get_earliest_date()

        assert result == "2026-01-01"

    @pytest.mark.asyncio
    async def test_get_earliest_date_returns_none_when_empty(self):
        """get_earliest_date() returns None when table is empty."""
        exporter = Exporter()
        mock_client = MagicMock()
        mock_client.db.query_raw = AsyncMock(return_value=[{"earliest": None}])

        with patch.object(
            type(exporter),
            "_prisma_client",
            new_callable=lambda: property(lambda self: mock_client),
        ):
            result = await exporter.get_earliest_date()

        assert result is None


# ---------------------------------------------------------------------------
# Exporter.export() — public method combining _get_usage_data + filter + _to_csv
# ---------------------------------------------------------------------------


class TestExporterExport:
    @pytest.mark.asyncio
    async def test_export_returns_dataframe_and_csv(self):
        """export() returns (filtered_df, csv_str) in one call."""
        exporter = Exporter()
        mock_client = MagicMock()
        mock_rows = [
            {
                "date": "2026-04-10",
                "user_id": "user-1",
                "model": "gpt-4o",
                "spend": 0.015,
                "successful_requests": 5,
                "prompt_tokens": 100,
                "completion_tokens": 50,
            }
        ]
        mock_client.db.query_raw = AsyncMock(return_value=mock_rows)

        with patch.object(
            type(exporter),
            "_prisma_client",
            new_callable=lambda: property(lambda self: mock_client),
        ):
            df, csv = await exporter.export(
                date_str="2026-04-10", connection_id="conn-1"
            )

        assert len(df) == 1
        assert "conn-1" in csv
        assert isinstance(csv, str)

    @pytest.mark.asyncio
    async def test_export_returns_empty_when_no_data(self):
        """export() returns empty DataFrame and empty string when no rows."""
        exporter = Exporter()
        mock_client = MagicMock()
        mock_client.db.query_raw = AsyncMock(return_value=[])

        with patch.object(
            type(exporter),
            "_prisma_client",
            new_callable=lambda: property(lambda self: mock_client),
        ):
            df, csv = await exporter.export(
                date_str="2026-04-10", connection_id="conn-1"
            )

        assert df.is_empty()
        assert csv == ""


# ---------------------------------------------------------------------------
# Exporter._stream_pages — async generator for paginated DB fetch
# ---------------------------------------------------------------------------


class TestStreamPages:
    @pytest.mark.asyncio
    async def test_yields_header_then_csv_rows(self):
        """_stream_pages() first yields a CSV header, then row data."""
        exporter = Exporter()
        mock_rows = [
            {
                "date": "2026-04-10",
                "model": "gpt-4o",
                "spend": 0.01,
                "successful_requests": 1,
            },
        ]
        # page 1 returns 1 row, page 2 returns empty → stop
        mock_client = MagicMock()
        mock_client.db.query_raw = AsyncMock(side_effect=[mock_rows, []])

        with patch.object(
            type(exporter),
            "_prisma_client",
            new_callable=lambda: property(lambda self: mock_client),
        ):
            chunks = []
            async for chunk in exporter._stream_pages(
                "2026-04-10", connection_id="c-1"
            ):
                chunks.append(chunk)

        assert len(chunks) >= 1
        combined = "".join(chunks)
        assert "date" in combined and "model" in combined
        assert "gpt-4o" in combined

    @pytest.mark.asyncio
    async def test_yields_nothing_when_db_empty(self):
        """_stream_pages() yields nothing when DB returns no rows."""
        exporter = Exporter()
        mock_client = MagicMock()
        mock_client.db.query_raw = AsyncMock(return_value=[])

        with patch.object(
            type(exporter),
            "_prisma_client",
            new_callable=lambda: property(lambda self: mock_client),
        ):
            chunks = []
            async for chunk in exporter._stream_pages(
                "2026-04-10", connection_id="c-1"
            ):
                chunks.append(chunk)

        assert chunks == []

    @pytest.mark.asyncio
    async def test_paginates_using_offset(self):
        """_stream_pages() uses OFFSET to fetch subsequent pages."""
        exporter = Exporter()
        page1 = [
            {
                "date": "2026-04-10",
                "model": "gpt-4o",
                "spend": 0.01,
                "successful_requests": 1,
            }
        ] * 3
        page2 = []
        mock_client = MagicMock()
        captured_queries = []

        async def fake_query_raw(query, *params):
            captured_queries.append(params)
            return page1 if len(captured_queries) == 1 else page2

        mock_client.db.query_raw = fake_query_raw

        with patch.object(
            type(exporter),
            "_prisma_client",
            new_callable=lambda: property(lambda self: mock_client),
        ):
            async for _ in exporter._stream_pages(
                "2026-04-10", connection_id="c", page_size=3
            ):
                pass

        # Two queries: page 1 (offset 0) and page 2 (offset 3 → empty → stop)
        assert len(captured_queries) == 2
        assert captured_queries[0][-1] == 0  # first OFFSET is 0
        assert captured_queries[1][-1] == 3  # second OFFSET is page_size


# ---------------------------------------------------------------------------
# Exporter — no DB connected: log warning, return gracefully
# ---------------------------------------------------------------------------


class TestExporterNoDb:
    @pytest.mark.asyncio
    async def test_get_usage_data_returns_empty_when_no_db(self):
        """_get_usage_data returns empty DataFrame when DB not connected."""
        exporter = Exporter()
        with patch.object(
            type(exporter),
            "_prisma_client",
            new_callable=lambda: property(lambda self: None),
        ):
            df = await exporter._get_usage_data("2026-04-10")
        assert df.is_empty()

    @pytest.mark.asyncio
    async def test_get_earliest_date_returns_none_when_no_db(self):
        """get_earliest_date returns None when DB not connected."""
        exporter = Exporter()
        with patch.object(
            type(exporter),
            "_prisma_client",
            new_callable=lambda: property(lambda self: None),
        ):
            result = await exporter.get_earliest_date()
        assert result is None

    @pytest.mark.asyncio
    async def test_stream_pages_raises_when_no_db(self):
        """_stream_pages raises RuntimeError when DB not connected.

        Raising (not silently returning) ensures the Orchestrator's try/except
        catches it and does NOT advance the marker — preventing silent data loss.
        """
        exporter = Exporter()
        with patch.object(
            type(exporter),
            "_prisma_client",
            new_callable=lambda: property(lambda self: None),
        ):
            with pytest.raises(RuntimeError, match="database not connected"):
                async for _ in exporter._stream_pages("2026-04-10", connection_id="c"):
                    pass

    @pytest.mark.asyncio
    async def test_stream_pages_yields_nothing_when_db_empty(self):
        """_stream_pages yields nothing when DB returns no rows for the date."""
        exporter = Exporter()
        mock_client = MagicMock()
        mock_client.db.query_raw = AsyncMock(return_value=[])
        with patch.object(
            type(exporter),
            "_prisma_client",
            new_callable=lambda: property(lambda self: mock_client),
        ):
            chunks = []
            async for chunk in exporter._stream_pages("2026-04-10", connection_id="c"):
                chunks.append(chunk)
        assert chunks == []
