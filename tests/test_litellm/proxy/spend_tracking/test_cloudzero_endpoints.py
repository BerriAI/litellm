import os
import sys
from unittest.mock import AsyncMock, MagicMock

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

