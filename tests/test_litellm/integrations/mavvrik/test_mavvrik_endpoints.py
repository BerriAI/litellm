"""Unit tests for Mavvrik FastAPI admin endpoints."""

import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.spend_tracking.mavvrik_endpoints import (
    _get_mavvrik_settings,
    _set_mavvrik_settings,
    delete_mavvrik_settings,
    dry_run_mavvrik_export,
    export_mavvrik_data,
    get_mavvrik_settings,
    init_mavvrik_settings,
    update_mavvrik_settings,
)
from litellm.types.proxy.mavvrik_endpoints import (
    MavvrikExportRequest,
    MavvrikInitRequest,
    MavvrikSettingsUpdate,
)


def _admin_user() -> UserAPIKeyAuth:
    return UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN)


def _non_admin_user() -> UserAPIKeyAuth:
    return UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER)


def _make_prisma_row(settings: dict):
    row = MagicMock()
    row.param_value = json.dumps(settings)
    return row


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

        mock_prisma = MagicMock()
        mock_prisma.db.litellm_config.upsert = AsyncMock()

        with patch(
            "litellm.proxy.spend_tracking.mavvrik_endpoints.MavvrikStreamer",
            return_value=mock_streamer,
        ), patch(
            "litellm.proxy.spend_tracking.mavvrik_endpoints.encrypt_value_helper",
            return_value="encrypted_key",
        ), patch(
            "litellm.proxy.proxy_server.prisma_client",
            mock_prisma,
        ), patch(
            "litellm.proxy.spend_tracking.mavvrik_endpoints._pserver",
            MagicMock(scheduler=None),
            create=True,
        ):
            resp = await init_mavvrik_settings(req, user_api_key_dict=_admin_user())

        assert resp.status == "success"
        mock_prisma.db.litellm_config.upsert.assert_called_once()

    @pytest.mark.asyncio
    async def test_init_falls_back_to_first_of_month_when_register_fails(self):
        req = MavvrikInitRequest(
            api_key="mav_key",
            api_endpoint="https://api.mavvrik.dev/acme",
            connection_id="litellm-prod",
        )

        mock_streamer = MagicMock()
        mock_streamer.register = AsyncMock(side_effect=Exception("network error"))

        mock_prisma = MagicMock()
        mock_prisma.db.litellm_config.upsert = AsyncMock()

        with patch(
            "litellm.proxy.spend_tracking.mavvrik_endpoints.MavvrikStreamer",
            return_value=mock_streamer,
        ), patch(
            "litellm.proxy.spend_tracking.mavvrik_endpoints.encrypt_value_helper",
            return_value="enc",
        ), patch(
            "litellm.proxy.proxy_server.prisma_client",
            mock_prisma,
        ), patch(
            "litellm.proxy.spend_tracking.mavvrik_endpoints._pserver",
            MagicMock(scheduler=None),
            create=True,
        ):
            resp = await init_mavvrik_settings(req, user_api_key_dict=_admin_user())

        assert resp.status == "success"
        # marker should have been stored — check upsert was called
        mock_prisma.db.litellm_config.upsert.assert_called_once()
        stored_payload = json.loads(
            mock_prisma.db.litellm_config.upsert.call_args[1]["data"]["create"][
                "param_value"
            ]
        )
        assert "marker" in stored_payload


# ---------------------------------------------------------------------------
# GET /mavvrik/settings
# ---------------------------------------------------------------------------


class TestGetMavvrikSettings:
    @pytest.mark.asyncio
    async def test_returns_not_configured_when_no_row(self):
        mock_prisma = MagicMock()
        mock_prisma.db.litellm_config.find_first = AsyncMock(return_value=None)

        with patch(
            "litellm.proxy.proxy_server.prisma_client",
            mock_prisma,
        ):
            resp = await get_mavvrik_settings(user_api_key_dict=_admin_user())

        assert resp.status == "not_configured"
        assert resp.api_key_masked is None

    @pytest.mark.asyncio
    async def test_returns_masked_key_when_configured(self):
        row = _make_prisma_row(
            {
                "api_key": "mav_plaintextkey",
                "api_endpoint": "https://api.mavvrik.dev/acme",
                "connection_id": "prod",
                "marker": "2024-01-14",
            }
        )
        mock_prisma = MagicMock()
        mock_prisma.db.litellm_config.find_first = AsyncMock(return_value=row)

        with patch(
            "litellm.proxy.proxy_server.prisma_client",
            mock_prisma,
        ), patch(
            "litellm.proxy.spend_tracking.mavvrik_endpoints.decrypt_value_helper",
            return_value="mav_plaintextkey",
        ):
            resp = await get_mavvrik_settings(user_api_key_dict=_admin_user())

        assert resp.status == "configured"
        assert resp.api_key_masked is not None
        assert "mav_plaintextkey" not in (resp.api_key_masked or "")
        assert resp.marker == "2024-01-14"
        assert resp.connection_id == "prod"


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
        row = _make_prisma_row(current)
        mock_prisma = MagicMock()
        mock_prisma.db.litellm_config.find_first = AsyncMock(return_value=row)
        mock_prisma.db.litellm_config.upsert = AsyncMock()

        req = MavvrikSettingsUpdate(marker="2024-06-01")

        with patch(
            "litellm.proxy.proxy_server.prisma_client",
            mock_prisma,
        ), patch(
            "litellm.proxy.spend_tracking.mavvrik_endpoints.decrypt_value_helper",
            return_value="plain_key",
        ), patch(
            "litellm.proxy.spend_tracking.mavvrik_endpoints.encrypt_value_helper",
            return_value="enc_key",
        ):
            resp = await update_mavvrik_settings(req, user_api_key_dict=_admin_user())

        assert resp.status == "success"
        stored = json.loads(
            mock_prisma.db.litellm_config.upsert.call_args[1]["data"]["update"][
                "param_value"
            ]
        )
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
        row = _make_prisma_row(
            {"api_key": "enc", "api_endpoint": "e", "connection_id": "c"}
        )
        mock_prisma = MagicMock()
        mock_prisma.db.litellm_config.find_first = AsyncMock(return_value=row)
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
        row = _make_prisma_row(settings)
        mock_prisma = MagicMock()
        mock_prisma.db.litellm_config.find_first = AsyncMock(return_value=row)

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

        with patch(
            "litellm.proxy.proxy_server.prisma_client",
            mock_prisma,
        ), patch(
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
