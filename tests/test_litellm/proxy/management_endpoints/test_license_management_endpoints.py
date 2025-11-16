import os
import sys
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path

from litellm.proxy._types import LitellmUserRoles
from litellm.proxy.auth.user_api_key_auth import UserAPIKeyAuth, user_api_key_auth
from litellm.proxy.proxy_server import app

# Create TestClient for making HTTP requests
client = TestClient(app)

def mock_user_auth(user_role: LitellmUserRoles):
    return UserAPIKeyAuth(
        user_role=user_role,
        api_key="sk-mock-user",
        user_id="mock-user",
    )

class TestGetLicenseInfoHTTP:

    @pytest.mark.asyncio
    async def test_license_exists(self):

        app.dependency_overrides[user_api_key_auth] = lambda: mock_user_auth(LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY)

        mock_premium_user_data = {
            "user_id": "full-license-123",
            "expiration_date": "2025-12-31",
            "allowed_features": ["feature1", "feature2"],
            "max_users": 100,
            "max_teams": 50,
        }

        with patch(
            "litellm.proxy.proxy_server.premium_user_data",
            mock_premium_user_data,
            create=True,
        ):
            response = client.get("/license/info")

            assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

            body = response.json()
            assert body["license_configured"] is True
            assert body["license_details"]["user_id"] == "full-license-123"
            assert body["license_details"]["expiration_date"] == "2025-12-31"
            assert body["license_details"]["allowed_features"] == ["feature1", "feature2"]
            assert body["license_details"]["max_users"] == 100
            assert body["license_details"]["max_teams"] == 50

        app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_license_does_not_exist(self):

        app.dependency_overrides[user_api_key_auth] = lambda: mock_user_auth(LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY)

        with patch(
            "litellm.proxy.proxy_server.premium_user_data",
            None,
            create=True,
        ):
            response = client.get("/license/info")

            assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

            body = response.json()
            assert body["license_configured"] is False
            assert body["license_details"] == {}

        app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_authorization(self):
        await self._test_role_authorization(mock_user_auth(LitellmUserRoles.PROXY_ADMIN), 200)
        await self._test_role_authorization(mock_user_auth(LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY), 200)

        await self._test_role_authorization(mock_user_auth(LitellmUserRoles.INTERNAL_USER), 403)
        await self._test_role_authorization(mock_user_auth(LitellmUserRoles.INTERNAL_USER_VIEW_ONLY), 403)

    async def _test_role_authorization(self, mock_user_auth, expected_status_code:int):
        app.dependency_overrides[user_api_key_auth] = lambda: mock_user_auth

        with patch(
                "litellm.proxy.proxy_server.premium_user_data",
                {"user_id": "test-license"},
                create=True,
        ):
            response = client.get("/license/info")

            assert response.status_code == expected_status_code, f"Role={mock_user_auth.user_role} Expected {expected_status_code}, got {response.status_code}: {response.text}"

        app.dependency_overrides.clear()