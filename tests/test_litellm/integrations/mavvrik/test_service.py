"""Tests for Service facade — verifies constructors and method calls are correct."""

import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import polars as pl
import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.integrations.mavvrik import Service
from litellm.integrations.mavvrik.client import Client
from litellm.integrations.mavvrik.uploader import Uploader
from litellm.integrations.mavvrik.orchestrator import Orchestrator


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
            with patch.dict(
                sys.modules, {"litellm.proxy.proxy_server": mock_pserver}
            ), patch("litellm.integrations.mavvrik.Client", MockClient), patch(
                "litellm.integrations.mavvrik.Uploader", MockUploader
            ), patch(
                "litellm.integrations.mavvrik.Orchestrator", MockOrchestrator
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

        with patch(
            "litellm.integrations.mavvrik.Exporter", return_value=mock_exporter_inst
        ), patch(
            "litellm.integrations.mavvrik.Uploader", return_value=mock_uploader_inst
        ), patch(
            "litellm.integrations.mavvrik.Client"
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

        with patch(
            "litellm.integrations.mavvrik.Exporter", return_value=mock_exporter_inst
        ), patch(
            "litellm.integrations.mavvrik.Uploader", return_value=mock_uploader_inst
        ), patch(
            "litellm.integrations.mavvrik.Client"
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

        with patch("litellm.integrations.mavvrik.Client", MockClient), patch(
            "litellm.integrations.mavvrik.Uploader", MockUploader
        ), patch(
            "litellm.integrations.mavvrik.Exporter", return_value=mock_exporter_inst
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

        with patch(
            "litellm.integrations.mavvrik.Exporter", return_value=mock_exporter_inst
        ), patch(
            "litellm.integrations.mavvrik.Uploader", return_value=mock_uploader_inst
        ), patch(
            "litellm.integrations.mavvrik.Client"
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

        with patch(
            "litellm.integrations.mavvrik.Exporter", return_value=mock_exporter_inst
        ), patch("litellm.integrations.mavvrik.Client"):
            result = await svc.dry_run(date_str="2026-04-10")

        assert result["summary"]["total_records"] == 0
        assert result["dry_run_data"]["usage_data"] == []
