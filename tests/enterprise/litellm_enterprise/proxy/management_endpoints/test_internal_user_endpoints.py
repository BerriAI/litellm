from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from litellm_enterprise.proxy.management_endpoints.internal_user_endpoints import router


@pytest.fixture
def client():
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


@pytest.fixture
def mock_user_api_key_auth():
    """Mock the user_api_key_auth dependency"""
    with patch(
        "enterprise.litellm_enterprise.proxy.management_endpoints.internal_user_endpoints.user_api_key_auth"
    ) as mock_auth:
        mock_auth.return_value = {"user_id": "test_user", "api_key": "test_key"}
        yield mock_auth


class TestAvailableEnterpriseUsers:
    @pytest.mark.asyncio
    async def test_available_users_with_max_users_set(
        self, client, mock_user_api_key_auth
    ):
        """Test when max_users is set and user count is within limit"""
        with patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma, patch(
            "litellm.proxy.proxy_server.premium_user",
            True,
        ), patch(
            "litellm.proxy.proxy_server.premium_user_data",
            {"max_users": 10},
        ):
            # Mock database count
            mock_prisma.db.litellm_usertable.count = AsyncMock(return_value=5)
            mock_prisma.db.litellm_teamtable.count = AsyncMock(return_value=2)

            # Override the dependency
            client.app.dependency_overrides[mock_user_api_key_auth] = lambda: {
                "user_id": "test_user"
            }

            response = client.get("/user/available_users")

            assert response.status_code == 200
            data = response.json()

            assert data["total_users"] == 10
            assert data["total_users_used"] == 5
            assert data["total_users_remaining"] == 5
            assert data["total_teams"] is None
            assert data["total_teams_used"] == 2
            assert data["total_teams_remaining"] is None
            # Ensure no negative values
            assert data["total_users_remaining"] >= 0

    @pytest.mark.asyncio
    async def test_available_users_without_max_users_set(
        self, client, mock_user_api_key_auth
    ):
        """Test when max_users is not set (premium_user_data is None or doesn't contain max_users)"""
        with patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma, patch(
            "litellm.proxy.proxy_server.premium_user",
            True,
        ), patch(
            "litellm.proxy.proxy_server.premium_user_data",
            None,
        ):
            # Mock database count
            mock_prisma.db.litellm_usertable.count = AsyncMock(return_value=3)
            mock_prisma.db.litellm_teamtable.count = AsyncMock(return_value=1)

            # Override the dependency
            client.app.dependency_overrides[mock_user_api_key_auth] = lambda: {
                "user_id": "test_user"
            }

            response = client.get("/user/available_users")

            assert response.status_code == 200
            data = response.json()

            assert data["total_users"] is None
            assert data["total_users_used"] == 3
            assert data["total_users_remaining"] is None
            assert data["total_teams"] is None
            assert data["total_teams_used"] == 1
            assert data["total_teams_remaining"] is None

    @pytest.mark.asyncio
    async def test_available_users_negative_remaining_bug(
        self, client, mock_user_api_key_auth
    ):
        """Test the current bug where total_users_remaining can be negative"""
        with patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma, patch(
            "litellm.proxy.proxy_server.premium_user",
            True,
        ), patch(
            "litellm.proxy.proxy_server.premium_user_data",
            {"key": "value"},
        ):
            # Mock database count higher than max_users to trigger the bug
            mock_prisma.db.litellm_usertable.count = AsyncMock(return_value=8)
            mock_prisma.db.litellm_teamtable.count = AsyncMock(return_value=3)

            # Override the dependency
            client.app.dependency_overrides[mock_user_api_key_auth] = lambda: {
                "user_id": "test_user"
            }

            response = client.get("/user/available_users")

            assert response.status_code == 200
            data = response.json()

            print(f"data: {data}")

            assert data["total_users"] == None
            assert data["total_users_used"] == 8
            assert data["total_teams"] == None
            assert data["total_teams_used"] == 3
            # This assertion will fail due to the current bug - remaining is -3
            # TODO: Fix the bug to ensure remaining is never negative
            assert data["total_users_remaining"] == None  # Current buggy behavior
            assert data["total_teams_remaining"] == None  # Current buggy behavior
            # The following assertion would be the correct behavior:
            # assert data["total_users_remaining"] >= 0

    @pytest.mark.asyncio
    async def test_available_users_no_database_connection(
        self, client, mock_user_api_key_auth
    ):
        """Test when prisma_client is None (no database connection)"""
        from litellm.proxy._types import CommonProxyErrors

        with patch(
            "litellm.proxy.proxy_server.prisma_client",
            None,
        ), patch(
            "litellm.proxy.proxy_server.premium_user",
            True,
        ):
            # Override the dependency
            client.app.dependency_overrides[mock_user_api_key_auth] = lambda: {
                "user_id": "test_user"
            }

            response = client.get("/user/available_users")

            assert response.status_code == 500
            assert (
                CommonProxyErrors.db_not_connected_error.value
                in response.json()["detail"]["error"]
            )
