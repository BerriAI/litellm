"""Unit tests for Mavvrik FastAPI admin endpoints."""

import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
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

# Patch target for LiteLLMDatabase used inside the endpoints module
_DB_PATH = "litellm.proxy.spend_tracking.mavvrik_endpoints.LiteLLMDatabase"


def _admin_user() -> UserAPIKeyAuth:
    return UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN)


def _non_admin_user() -> UserAPIKeyAuth:
    return UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER)


def _mock_db(settings: dict = None, set_ok: bool = True):
    """Return a mock LiteLLMDatabase instance with pre-configured returns."""
    db = MagicMock()
    db.get_mavvrik_settings = AsyncMock(return_value=settings or {})
    db.set_mavvrik_settings = AsyncMock()
    return db


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


class TestInitMavvrikSettings:
    @pytest.mark.asyncio
    async def test_init_stores_settings_and_returns_success(self):
        req = MavvrikInitRequest(
            api_key="mav_key",
            api_endpoint="https://api.mavvrik.dev/acme",
            connection_id="litellm-prod",
        )

        mock_streamer = MagicMock()
        mock_streamer.register = AsyncMock(return_value="2024-01-01T00:00:00+00:00")
        mock_db = _mock_db()

        with patch(
            "litellm.proxy.spend_tracking.mavvrik_endpoints.MavvrikUploader",
            return_value=mock_streamer,
        ), patch(
            "litellm.proxy.spend_tracking.mavvrik_endpoints.encrypt_value_helper",
            return_value="encrypted_key",
        ), patch(
            _DB_PATH, return_value=mock_db
        ), patch(
            "litellm.proxy.spend_tracking.mavvrik_endpoints._pserver",
            MagicMock(scheduler=None),
            create=True,
        ):
            resp = await init_mavvrik_settings(req, user_api_key_dict=_admin_user())

        assert resp.status == "success"
        mock_db.set_mavvrik_settings.assert_called_once()
        stored = mock_db.set_mavvrik_settings.call_args[0][0]
        assert stored["api_key"] == "encrypted_key"
        assert stored["marker"] == "2024-01-01T00:00:00+00:00"

    @pytest.mark.asyncio
    async def test_init_falls_back_to_first_of_month_when_register_fails(self):
        req = MavvrikInitRequest(
            api_key="mav_key",
            api_endpoint="https://api.mavvrik.dev/acme",
            connection_id="litellm-prod",
        )

        mock_streamer = MagicMock()
        mock_streamer.register = AsyncMock(side_effect=Exception("network error"))
        mock_db = _mock_db()

        with patch(
            "litellm.proxy.spend_tracking.mavvrik_endpoints.MavvrikUploader",
            return_value=mock_streamer,
        ), patch(
            "litellm.proxy.spend_tracking.mavvrik_endpoints.encrypt_value_helper",
            return_value="enc",
        ), patch(
            _DB_PATH, return_value=mock_db
        ), patch(
            "litellm.proxy.spend_tracking.mavvrik_endpoints._pserver",
            MagicMock(scheduler=None),
            create=True,
        ):
            resp = await init_mavvrik_settings(req, user_api_key_dict=_admin_user())

        assert resp.status == "success"
        mock_db.set_mavvrik_settings.assert_called_once()
        stored = mock_db.set_mavvrik_settings.call_args[0][0]
        assert "marker" in stored


# ---------------------------------------------------------------------------
# GET /mavvrik/settings
# ---------------------------------------------------------------------------


class TestGetMavvrikSettings:
    @pytest.mark.asyncio
    async def test_returns_not_configured_when_no_row(self):
        mock_db = _mock_db(settings={})

        with patch(_DB_PATH, return_value=mock_db):
            resp = await get_mavvrik_settings(user_api_key_dict=_admin_user())

        assert resp.status == "not_configured"
        assert resp.api_key_masked is None

    @pytest.mark.asyncio
    async def test_returns_masked_key_when_configured(self):
        mock_db = _mock_db(
            settings={
                "api_key": "enc_key",
                "api_endpoint": "https://api.mavvrik.dev/acme",
                "connection_id": "prod",
                "marker": "2024-01-14",
            }
        )

        with patch(_DB_PATH, return_value=mock_db), patch(
            "litellm.proxy.spend_tracking.mavvrik_endpoints.decrypt_value_helper",
            return_value="mav_plaintextkey",
        ):
            resp = await get_mavvrik_settings(user_api_key_dict=_admin_user())

        assert resp.status == "configured"
        assert resp.api_key_masked is not None
        assert "mav_plaintextkey" not in (resp.api_key_masked or "")
        assert resp.marker == "2024-01-14"
        assert resp.connection_id == "prod"

    @pytest.mark.asyncio
    async def test_raises_500_on_decrypt_failure(self):
        from fastapi import HTTPException

        mock_db = _mock_db(
            settings={"api_key": "enc_key", "api_endpoint": "e", "connection_id": "c"}
        )

        with patch(_DB_PATH, return_value=mock_db), patch(
            "litellm.proxy.spend_tracking.mavvrik_endpoints.decrypt_value_helper",
            return_value=None,  # decrypt failure returns None
        ):
            with pytest.raises(HTTPException) as exc_info:
                await get_mavvrik_settings(user_api_key_dict=_admin_user())

        assert exc_info.value.status_code == 500
        assert (
            "salt" in str(exc_info.value.detail).lower()
            or "decrypt" in str(exc_info.value.detail).lower()
        )


# ---------------------------------------------------------------------------
# PUT /mavvrik/settings
# ---------------------------------------------------------------------------


class TestUpdateMavvrikSettings:
    @pytest.mark.asyncio
    async def test_update_rejects_empty_request(self):
        from fastapi import HTTPException

        req = MavvrikSettingsUpdate()  # all None
        with pytest.raises(HTTPException) as exc_info:
            await update_mavvrik_settings(req, user_api_key_dict=_admin_user())
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_update_marker_only(self):
        current = {
            "api_key": "enc_key",
            "api_endpoint": "https://api.mavvrik.dev/acme",
            "connection_id": "prod",
            "marker": "2024-01-01",
        }
        mock_db = _mock_db(settings=current)
        req = MavvrikSettingsUpdate(marker="2024-06-01")

        with patch(_DB_PATH, return_value=mock_db), patch(
            "litellm.proxy.spend_tracking.mavvrik_endpoints.decrypt_value_helper",
            return_value="plain_key",
        ), patch(
            "litellm.proxy.spend_tracking.mavvrik_endpoints.encrypt_value_helper",
            return_value="enc_key",
        ):
            resp = await update_mavvrik_settings(req, user_api_key_dict=_admin_user())

        assert resp.status == "success"
        mock_db.set_mavvrik_settings.assert_called_once()
        stored = mock_db.set_mavvrik_settings.call_args[0][0]
        assert stored["marker"] == "2024-06-01"

    def test_update_rejects_invalid_marker_date(self):
        with pytest.raises(Exception, match="YYYY-MM-DD"):
            MavvrikSettingsUpdate(marker="not-a-date")

    def test_update_rejects_empty_api_key(self):
        with pytest.raises(Exception):
            MavvrikSettingsUpdate(api_key="")


# ---------------------------------------------------------------------------
# DELETE /mavvrik/delete
# ---------------------------------------------------------------------------


class TestDeleteMavvrikSettings:
    @pytest.mark.asyncio
    async def test_delete_returns_404_when_not_configured(self):
        from fastapi import HTTPException

        mock_prisma = MagicMock()
        mock_prisma.db.litellm_config.find_first = AsyncMock(return_value=None)

        with patch(
            "litellm.proxy.proxy_server.prisma_client",
            mock_prisma,
        ):
            with pytest.raises(HTTPException) as exc_info:
                await delete_mavvrik_settings(user_api_key_dict=_admin_user())
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_removes_row_and_returns_success(self):
        mock_prisma = MagicMock()
        mock_prisma.db.litellm_config.find_first = AsyncMock(
            return_value=MagicMock(param_value="{}")
        )
        mock_prisma.db.litellm_config.delete = AsyncMock()

        with patch(
            "litellm.proxy.proxy_server.prisma_client",
            mock_prisma,
        ):
            resp = await delete_mavvrik_settings(user_api_key_dict=_admin_user())

        assert resp.status == "success"
        mock_prisma.db.litellm_config.delete.assert_called_once()


# ---------------------------------------------------------------------------
# POST /mavvrik/dry-run
# ---------------------------------------------------------------------------


class TestDryRunMavvrikExport:
    @pytest.mark.asyncio
    async def test_dry_run_returns_preview(self):
        settings = {
            "api_key": "plain_key",
            "api_endpoint": "https://api.mavvrik.dev/acme",
            "connection_id": "prod",
        }
        mock_db = _mock_db(settings=settings)

        dry_run_result = {
            "usage_data": [{"date": "2024-01-14", "model": "gpt-4o", "spend": 1.5}],
            "csv_preview": "date,model,spend\n2024-01-14,gpt-4o,1.5",
            "summary": {
                "total_records": 1,
                "total_cost": 1.5,
                "total_tokens": 100,
                "unique_models": 1,
                "unique_teams": 1,
            },
        }

        mock_logger = MagicMock()
        mock_logger.dry_run_export_usage_data = AsyncMock(return_value=dry_run_result)

        req = MavvrikExportRequest(date_str="2024-01-14")

        with patch(_DB_PATH, return_value=mock_db), patch(
            "litellm.proxy.spend_tracking.mavvrik_endpoints.decrypt_value_helper",
            return_value="plain_key",
        ), patch(
            "litellm.proxy.spend_tracking.mavvrik_endpoints.MavvrikLogger",
            return_value=mock_logger,
        ):
            resp = await dry_run_mavvrik_export(req, user_api_key_dict=_admin_user())

        assert resp.status == "success"
        assert resp.summary is not None
        assert resp.summary["total_records"] == 1

    @pytest.mark.asyncio
    async def test_dry_run_defaults_to_yesterday_when_no_date(self):
        mock_db = _mock_db(
            settings={
                "api_key": "enc",
                "api_endpoint": "https://api.mavvrik.dev/acme",
                "connection_id": "prod",
            }
        )
        empty_result = {
            "usage_data": [],
            "csv_preview": "",
            "summary": {
                "total_records": 0,
                "total_cost": 0.0,
                "total_tokens": 0,
                "unique_models": 0,
                "unique_teams": 0,
            },
        }
        mock_logger = MagicMock()
        mock_logger.dry_run_export_usage_data = AsyncMock(return_value=empty_result)

        req = MavvrikExportRequest()  # no date_str

        with patch(_DB_PATH, return_value=mock_db), patch(
            "litellm.proxy.spend_tracking.mavvrik_endpoints.decrypt_value_helper",
            return_value="plain_key",
        ), patch(
            "litellm.proxy.spend_tracking.mavvrik_endpoints.MavvrikLogger",
            return_value=mock_logger,
        ):
            resp = await dry_run_mavvrik_export(req, user_api_key_dict=_admin_user())

        assert resp.status == "success"
        # Verify a date_str was passed (yesterday) rather than None
        called_date = (
            mock_logger.dry_run_export_usage_data.call_args[1].get("date_str")
            or mock_logger.dry_run_export_usage_data.call_args[0][0]
        )
        assert called_date is not None
        assert called_date != ""


# ---------------------------------------------------------------------------
# POST /mavvrik/export
# ---------------------------------------------------------------------------


class TestExportMavvrikData:
    @pytest.mark.asyncio
    async def test_export_returns_success_and_record_count(self):
        from litellm.proxy.spend_tracking.mavvrik_endpoints import export_mavvrik_data

        settings = {
            "api_key": "plain_key",
            "api_endpoint": "https://api.mavvrik.dev/acme",
            "connection_id": "prod",
        }
        mock_db = _mock_db(settings=settings)
        mock_logger = MagicMock()
        mock_logger.export_usage_data = AsyncMock(return_value=7)

        req = MavvrikExportRequest(date_str="2024-01-15")

        with patch(_DB_PATH, return_value=mock_db), patch(
            "litellm.proxy.spend_tracking.mavvrik_endpoints.decrypt_value_helper",
            return_value="plain_key",
        ), patch(
            "litellm.proxy.spend_tracking.mavvrik_endpoints.MavvrikLogger",
            return_value=mock_logger,
        ):
            resp = await export_mavvrik_data(req, user_api_key_dict=_admin_user())

        assert resp.status == "success"
        assert resp.records_exported == 7
        assert "2024-01-15" in resp.message

    @pytest.mark.asyncio
    async def test_export_returns_400_when_not_configured(self):
        from fastapi import HTTPException

        from litellm.proxy.spend_tracking.mavvrik_endpoints import export_mavvrik_data

        mock_db = _mock_db(settings={})  # empty = not configured
        req = MavvrikExportRequest(date_str="2024-01-15")

        with patch(_DB_PATH, return_value=mock_db), patch(
            "litellm.proxy.spend_tracking.mavvrik_endpoints.decrypt_value_helper",
            return_value=None,
        ):
            with pytest.raises(HTTPException) as exc_info:
                await export_mavvrik_data(req, user_api_key_dict=_admin_user())

        assert exc_info.value.status_code in (400, 500)

    @pytest.mark.asyncio
    async def test_export_defaults_to_yesterday_when_no_date(self):
        from litellm.proxy.spend_tracking.mavvrik_endpoints import export_mavvrik_data

        settings = {
            "api_key": "plain_key",
            "api_endpoint": "https://api.mavvrik.dev/acme",
            "connection_id": "prod",
        }
        mock_db = _mock_db(settings=settings)
        mock_logger = MagicMock()
        mock_logger.export_usage_data = AsyncMock(return_value=3)

        req = MavvrikExportRequest()  # no date_str

        with patch(_DB_PATH, return_value=mock_db), patch(
            "litellm.proxy.spend_tracking.mavvrik_endpoints.decrypt_value_helper",
            return_value="plain_key",
        ), patch(
            "litellm.proxy.spend_tracking.mavvrik_endpoints.MavvrikLogger",
            return_value=mock_logger,
        ):
            resp = await export_mavvrik_data(req, user_api_key_dict=_admin_user())

        assert resp.status == "success"
        mock_logger.export_usage_data.assert_called_once()
        called_date = (
            mock_logger.export_usage_data.call_args[1].get("date_str")
            or mock_logger.export_usage_data.call_args[0][0]
        )
        assert called_date is not None


# ---------------------------------------------------------------------------
# Lifecycle flow — init → settings → update marker → delete → export fails
# ---------------------------------------------------------------------------


class TestLifecycleFlow:
    @pytest.mark.asyncio
    async def test_get_settings_returns_not_configured_after_delete(self):
        """DELETE removes row → subsequent GET returns not_configured."""
        mock_prisma = MagicMock()
        mock_prisma.db.litellm_config.find_first = AsyncMock(
            return_value=MagicMock(param_value="{}")
        )
        mock_prisma.db.litellm_config.delete = AsyncMock()

        with patch("litellm.proxy.proxy_server.prisma_client", mock_prisma):
            del_resp = await delete_mavvrik_settings(user_api_key_dict=_admin_user())

        assert del_resp.status == "success"

        # Now GET should return not_configured (DB empty)
        mock_db_empty = _mock_db(settings={})
        with patch(_DB_PATH, return_value=mock_db_empty):
            get_resp = await get_mavvrik_settings(user_api_key_dict=_admin_user())

        assert get_resp.status == "not_configured"

    @pytest.mark.asyncio
    async def test_update_marker_visible_in_subsequent_get(self):
        """PUT marker → GET returns new marker value."""
        current = {
            "api_key": "enc_key",
            "api_endpoint": "https://api.mavvrik.dev/acme",
            "connection_id": "prod",
            "marker": "2024-01-01",
        }
        updated = {**current, "marker": "2024-06-01"}

        mock_db_current = _mock_db(settings=current)
        req = MavvrikSettingsUpdate(marker="2024-06-01")

        with patch(_DB_PATH, return_value=mock_db_current), patch(
            "litellm.proxy.spend_tracking.mavvrik_endpoints.decrypt_value_helper",
            return_value="plain_key",
        ), patch(
            "litellm.proxy.spend_tracking.mavvrik_endpoints.encrypt_value_helper",
            return_value="enc_key",
        ):
            put_resp = await update_mavvrik_settings(
                req, user_api_key_dict=_admin_user()
            )

        assert put_resp.status == "success"

        # Simulate GET after update
        mock_db_updated = _mock_db(settings=updated)
        with patch(_DB_PATH, return_value=mock_db_updated), patch(
            "litellm.proxy.spend_tracking.mavvrik_endpoints.decrypt_value_helper",
            return_value="plain_key",
        ):
            get_resp = await get_mavvrik_settings(user_api_key_dict=_admin_user())

        assert get_resp.marker == "2024-06-01"
        assert get_resp.status == "configured"


# ---------------------------------------------------------------------------
# register.py — scheduler registration
# ---------------------------------------------------------------------------


class TestRegisterModule:
    @pytest.mark.asyncio
    async def test_register_logger_and_job_adds_scheduler_job(self):
        """register_logger_and_job schedules the export job when scheduler is present."""
        from litellm.integrations.mavvrik.register import register_logger_and_job

        mock_scheduler = MagicMock()
        mock_scheduler.add_job = MagicMock()

        with patch("litellm.integrations.mavvrik.register.litellm") as mock_litellm:
            mock_litellm.logging_callback_manager = MagicMock()
            mock_litellm.success_callback = []
            mock_litellm._async_success_callback = []

            await register_logger_and_job(
                api_key="mav_key",
                api_endpoint="https://api.mavvrik.dev/acme",
                connection_id="prod",
                scheduler=mock_scheduler,
            )

        mock_scheduler.add_job.assert_called_once()
        # APScheduler add_job is called with positional + keyword args;
        # verify "interval" appears somewhere in the call
        call_args = mock_scheduler.add_job.call_args
        all_args = list(call_args.args) + list(call_args.kwargs.values())
        assert "interval" in all_args

    @pytest.mark.asyncio
    async def test_register_background_job_skips_when_no_loggers(self):
        """register_background_job does nothing if no MavvrikLogger instance exists."""
        from litellm.integrations.mavvrik.register import register_background_job

        mock_scheduler = MagicMock()
        mock_scheduler.add_job = MagicMock()

        with patch("litellm.integrations.mavvrik.register.litellm") as mock_litellm:
            mock_litellm.logging_callback_manager.get_custom_loggers_for_type = (
                MagicMock(return_value=[])
            )
            await register_background_job(scheduler=mock_scheduler)

        mock_scheduler.add_job.assert_not_called()

    @pytest.mark.asyncio
    async def test_is_mavvrik_setup_true_when_env_vars_set(self):
        """is_mavvrik_setup returns True when all env vars are present."""
        from litellm.integrations.mavvrik.register import is_mavvrik_setup

        with patch.dict(
            "os.environ",
            {
                "MAVVRIK_API_KEY": "mav_key",
                "MAVVRIK_API_ENDPOINT": "https://api.mavvrik.dev/acme",
                "MAVVRIK_CONNECTION_ID": "prod",
            },
        ):
            result = await is_mavvrik_setup()

        assert result is True

    @pytest.mark.asyncio
    async def test_is_mavvrik_setup_false_when_no_env_and_no_db(self):
        """is_mavvrik_setup returns False when env vars missing and DB not connected."""
        from litellm.integrations.mavvrik.register import is_mavvrik_setup

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
                "litellm.integrations.mavvrik.register.prisma_client",
                None,
                create=True,
            ):
                result = await is_mavvrik_setup()

        assert result is False

    @pytest.mark.asyncio
    async def test_register_logger_and_job_deduplicates_on_repeated_init(self):
        """Calling register_logger_and_job twice must not accumulate two loggers."""
        from litellm.integrations.mavvrik.exporter import MavvrikExporter as MavvrikLogger
        from litellm.integrations.mavvrik.register import register_logger_and_job

        mock_scheduler = MagicMock()
        mock_scheduler.add_job = MagicMock()

        # Simulate an existing logger already registered from a previous /mavvrik/init
        existing_logger = MagicMock(spec=MavvrikLogger)
        success_cbs = [existing_logger]
        async_success_cbs = [existing_logger]

        mock_litellm = MagicMock()
        mock_litellm.logging_callback_manager.success_callbacks = success_cbs
        mock_litellm.logging_callback_manager.failure_callbacks = []
        mock_litellm.logging_callback_manager.async_success_callbacks = (
            async_success_cbs
        )
        mock_litellm.success_callback = []
        mock_litellm._async_success_callback = []

        with patch("litellm.integrations.mavvrik.register.litellm", mock_litellm):
            await register_logger_and_job(
                api_key="new_key",
                api_endpoint="https://api.mavvrik.dev/acme",
                connection_id="prod",
                scheduler=mock_scheduler,
            )

        # The existing MavvrikLogger instance should have been removed
        assert (
            existing_logger
            not in mock_litellm.logging_callback_manager.success_callbacks
        )
        assert (
            existing_logger
            not in mock_litellm.logging_callback_manager.async_success_callbacks
        )
        # A new logger was added (the MavvrikLogger instance added by register_logger_and_job)
        mock_litellm.logging_callback_manager.add_litellm_success_callback.assert_called_once()
        mock_scheduler.add_job.assert_called_once()
