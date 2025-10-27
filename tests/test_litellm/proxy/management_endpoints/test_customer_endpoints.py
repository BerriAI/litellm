from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from litellm.proxy._types import (
    LiteLLM_BudgetTable,
    LiteLLM_EndUserTable,
    LitellmUserRoles,
)
from litellm.proxy.auth.user_api_key_auth import UserAPIKeyAuth
from litellm.proxy.management_endpoints.customer_endpoints import router
from litellm.proxy.proxy_server import ProxyException

app = FastAPI()
app.include_router(router)
client = TestClient(app)


@pytest.fixture
def mock_prisma_client():
    with patch("litellm.proxy.proxy_server.prisma_client") as mock:
        yield mock


@pytest.fixture
def mock_user_api_key_auth():
    with patch("litellm.proxy.proxy_server.user_api_key_auth") as mock:
        mock.return_value = UserAPIKeyAuth(
            user_id="test-user", user_role=LitellmUserRoles.PROXY_ADMIN
        )
        yield mock


def test_update_customer_success(mock_prisma_client, mock_user_api_key_auth):
    # Mock the database responses
    mock_end_user = LiteLLM_EndUserTable(
        user_id="test-user-1", alias="Test User", blocked=False
    )
    updated_mock_end_user = LiteLLM_EndUserTable(
        user_id="test-user-1", alias="Updated Test User", blocked=False
    )

    # Mock the find_first response
    mock_prisma_client.db.litellm_endusertable.find_first = AsyncMock(
        return_value=mock_end_user
    )

    # Mock the update response
    mock_prisma_client.db.litellm_endusertable.update = AsyncMock(
        return_value=updated_mock_end_user
    )

    # Test data
    test_data = {"user_id": "test-user-1", "alias": "Updated Test User"}

    # Make the request
    response = client.post(
        "/customer/update", json=test_data, headers={"Authorization": "Bearer test-key"}
    )

    # Assert response
    assert response.status_code == 200
    assert response.json()["user_id"] == "test-user-1"
    assert response.json()["alias"] == "Updated Test User"


def test_update_customer_not_found(mock_prisma_client, mock_user_api_key_auth):
    # Mock the database response to return None (user not found)
    mock_prisma_client.db.litellm_endusertable.find_first = AsyncMock(return_value=None)

    # Test data
    test_data = {"user_id": "non-existent-user", "alias": "Test User"}

    # Make the request
    try:
        response = client.post(
            "/customer/update",
            json=test_data,
            headers={"Authorization": "Bearer test-key"},
        )
    except Exception as e:
        print(e, type(e))
        assert isinstance(e, ProxyException)
        assert int(e.code) == 400
        assert "End User Id=non-existent-user does not exist in db" in e.message
