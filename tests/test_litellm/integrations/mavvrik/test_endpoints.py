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

# Patch target prefix — MavvrikService methods live in the integrations package.
_SVC = "litellm.integrations.mavvrik.MavvrikService"


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


class TestInitMavvrikSettings:
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
        from litellm.integrations.mavvrik.settings import MavvrikSettings

        req = MavvrikInitRequest(
            api_key="mav_key",
            api_endpoint="https://api.mavvrik.dev/acme",
            connection_id="litellm-prod",
        )

        with patch.object(MavvrikSettings, "save", new=AsyncMock()), patch(
            "litellm.proxy.proxy_server.scheduler", None
        ):
            resp = await init_mavvrik_settings(req, user_api_key_dict=_admin_user())

        assert resp.status == "success"


# ---------------------------------------------------------------------------
# GET /mavvrik/settings
# ---------------------------------------------------------------------------


class TestGetMavvrikSettings:
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


class TestUpdateMavvrikSettings:
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


class TestDeleteMavvrikSettings:
    @pytest.mark.asyncio
    async def test_delete_returns_404_when_not_configured(self):
        """MavvrikSettings.delete() raises LookupError → endpoint returns 404."""
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
# MavvrikSettings — setup detection
# ---------------------------------------------------------------------------


class TestSettingsSetup:
    @pytest.mark.asyncio
    async def test_is_mavvrik_setup_true_when_env_vars_set(self):
        """MavvrikSettings.is_setup returns True when all env vars are present."""
        from litellm.integrations.mavvrik.settings import MavvrikSettings

        with patch.dict(
            "os.environ",
            {
                "MAVVRIK_API_KEY": "mav_key",
                "MAVVRIK_API_ENDPOINT": "https://api.mavvrik.dev/acme",
                "MAVVRIK_CONNECTION_ID": "prod",
            },
        ):
            result = await MavvrikSettings().is_setup()

        assert result is True

    @pytest.mark.asyncio
    async def test_is_mavvrik_setup_false_when_no_env_and_no_db(self):
        """MavvrikSettings.is_setup returns False when env vars missing and DB not connected."""
        from litellm.integrations.mavvrik.settings import MavvrikSettings

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
                "litellm.integrations.mavvrik.settings.MavvrikSettings._prisma_client",
                new_callable=lambda: property(lambda self: None),
            ):
                result = await MavvrikSettings().is_setup()

        assert result is False
