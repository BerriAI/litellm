from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from litellm.proxy._types import (
    LiteLLM_BudgetTable,
    LiteLLM_EndUserTable,
    LitellmUserRoles,
    ProxyException,
)
from litellm.proxy.auth.user_api_key_auth import UserAPIKeyAuth
from litellm.proxy.management_endpoints.customer_endpoints import router

app = FastAPI()


@app.exception_handler(ProxyException)
async def openai_exception_handler(request: Request, exc: ProxyException):
    headers = exc.headers
    error_dict = exc.to_dict()
    return JSONResponse(
        status_code=(
            int(exc.code) if exc.code else status.HTTP_500_INTERNAL_SERVER_ERROR
        ),
        content={"detail": {"error": error_dict["message"]}},
        headers=headers,
    )


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
    """
    Test that update_end_user raises a 404 ProxyException when user_id does not exist.
    """
    # Mock the database response to return None (user not found)
    mock_prisma_client.db.litellm_endusertable.find_first = AsyncMock(return_value=None)

    # Test data
    test_data = {"user_id": "non-existent-user", "alias": "Test User"}

    # Make the request
    response = client.post(
        "/customer/update",
        json=test_data,
        headers={"Authorization": "Bearer test-key"},
    )

    # Assert response
    assert response.status_code == 404
    assert "End User Id=non-existent-user does not exist in db" in response.json()["detail"]["error"]


def test_info_customer_not_found(mock_prisma_client, mock_user_api_key_auth):
    """
    Test that end_user_info raises a 404 ProxyException when end_user_id does not exist.
    """
    # Mock the database response to return None (user not found)
    mock_prisma_client.db.litellm_endusertable.find_first = AsyncMock(return_value=None)

    # Make the request
    response = client.get(
        "/customer/info?end_user_id=non-existent-user",
        headers={"Authorization": "Bearer test-key"},
    )

    # Assert response
    assert response.status_code == 404
    assert "End User Id=non-existent-user does not exist in db" in response.json()["detail"]["error"]
