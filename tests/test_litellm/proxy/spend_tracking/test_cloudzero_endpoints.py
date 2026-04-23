import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../../..")
)

import litellm.proxy.proxy_server as ps
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.proxy_server import app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.mark.asyncio
async def test_delete_cloudzero_settings_success(client, monkeypatch):
    mock_config = MagicMock()
    mock_config.param_name = "cloudzero_settings"
    mock_config.param_value = {"api_key": "encrypted_key", "connection_id": "conn_123", "timezone": "UTC"}

    mock_litellm_config = MagicMock()
    mock_litellm_config.find_first = AsyncMock(return_value=mock_config)
    mock_litellm_config.delete = AsyncMock(return_value=mock_config)

    mock_prisma = MagicMock()
    mock_prisma.db = MagicMock()
    mock_prisma.db.litellm_config = mock_litellm_config

    monkeypatch.setattr(ps, "prisma_client", mock_prisma)

    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN, user_id="admin_user"
    )

    try:
        response = client.delete("/cloudzero/delete")
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "CloudZero settings deleted successfully"
        assert data["status"] == "success"
        mock_litellm_config.find_first.assert_awaited_once()
        mock_litellm_config.delete.assert_awaited_once()
    finally:
        app.dependency_overrides.pop(ps.user_api_key_auth, None)


@pytest.mark.asyncio
async def test_delete_cloudzero_settings_not_found(client, monkeypatch):
    mock_litellm_config = MagicMock()
    mock_litellm_config.find_first = AsyncMock(return_value=None)

    mock_prisma = MagicMock()
    mock_prisma.db = MagicMock()
    mock_prisma.db.litellm_config = mock_litellm_config

    monkeypatch.setattr(ps, "prisma_client", mock_prisma)

    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN, user_id="admin_user"
    )

    try:
        response = client.delete("/cloudzero/delete")
        assert response.status_code == 404
        data = response.json()
        assert "error" in data["detail"]
        assert "CloudZero settings not found" in data["detail"]["error"]
        mock_litellm_config.find_first.assert_awaited_once()
        mock_litellm_config.delete.assert_not_called()
    finally:
        app.dependency_overrides.pop(ps.user_api_key_auth, None)


@pytest.mark.asyncio
async def test_get_cloudzero_settings_success(client, monkeypatch):
    """Test GET /cloudzero/settings returns settings when configured"""
    mock_config = MagicMock()
    mock_config.param_name = "cloudzero_settings"
    mock_config.param_value = {
        "api_key": "encrypted_key",
        "connection_id": "conn_123",
        "timezone": "UTC"
    }

    mock_litellm_config = MagicMock()
    mock_litellm_config.find_first = AsyncMock(return_value=mock_config)

    mock_prisma = MagicMock()
    mock_prisma.db = MagicMock()
    mock_prisma.db.litellm_config = mock_litellm_config

    monkeypatch.setattr(ps, "prisma_client", mock_prisma)

    # Mock the decrypt function to return a decrypted key
    with patch("litellm.proxy.spend_tracking.cloudzero_endpoints.decrypt_value_helper") as mock_decrypt:
        mock_decrypt.return_value = "decrypted_api_key"
        
        # Mock the masker
        with patch("litellm.proxy.spend_tracking.cloudzero_endpoints._sensitive_masker") as mock_masker:
            mock_masker.mask_dict.return_value = {"api_key": "test****key"}
            
            app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
                user_role=LitellmUserRoles.PROXY_ADMIN, user_id="admin_user"
            )

            try:
                response = client.get("/cloudzero/settings")
                assert response.status_code == 200
                data = response.json()
                assert data["connection_id"] == "conn_123"
                assert data["timezone"] == "UTC"
                assert data["status"] == "configured"
                assert data["api_key_masked"] == "test****key"
                mock_litellm_config.find_first.assert_awaited_once()
            finally:
                app.dependency_overrides.pop(ps.user_api_key_auth, None)


@pytest.mark.asyncio
async def test_get_cloudzero_settings_not_configured(client, monkeypatch):
    """Test GET /cloudzero/settings returns 200 with null values when not configured (consistent with other endpoints)"""
    mock_litellm_config = MagicMock()
    mock_litellm_config.find_first = AsyncMock(return_value=None)

    mock_prisma = MagicMock()
    mock_prisma.db = MagicMock()
    mock_prisma.db.litellm_config = mock_litellm_config

    monkeypatch.setattr(ps, "prisma_client", mock_prisma)

    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN, user_id="admin_user"
    )

    try:
        response = client.get("/cloudzero/settings")
        # Should return 200 with null values (not 404) - consistent with other settings endpoints
        assert response.status_code == 200
        data = response.json()
        assert data["api_key_masked"] is None
        assert data["connection_id"] is None
        assert data["timezone"] is None
        assert data["status"] is None
        mock_litellm_config.find_first.assert_awaited_once()
    finally:
        app.dependency_overrides.pop(ps.user_api_key_auth, None)


@pytest.mark.asyncio
async def test_get_cloudzero_settings_empty_param_value(client, monkeypatch):
    """Test GET /cloudzero/settings returns 200 with null values when param_value is None"""
    mock_config = MagicMock()
    mock_config.param_name = "cloudzero_settings"
    mock_config.param_value = None

    mock_litellm_config = MagicMock()
    mock_litellm_config.find_first = AsyncMock(return_value=mock_config)

    mock_prisma = MagicMock()
    mock_prisma.db = MagicMock()
    mock_prisma.db.litellm_config = mock_litellm_config

    monkeypatch.setattr(ps, "prisma_client", mock_prisma)

    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN, user_id="admin_user"
    )

    try:
        response = client.get("/cloudzero/settings")
        # Should return 200 with null values (not 404) - consistent with other settings endpoints
        assert response.status_code == 200
        data = response.json()
        assert data["api_key_masked"] is None
        assert data["connection_id"] is None
        assert data["timezone"] is None
        assert data["status"] is None
        mock_litellm_config.find_first.assert_awaited_once()
    finally:
        app.dependency_overrides.pop(ps.user_api_key_auth, None)

