from unittest.mock import AsyncMock, MagicMock, patch

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
        content={"error": error_dict},
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
    response_json = response.json()
    assert "error" in response_json
    assert response_json["error"]["message"] == "End User Id=non-existent-user does not exist in db"
    assert response_json["error"]["type"] == "not_found"
    assert response_json["error"]["param"] == "user_id"
    assert response_json["error"]["code"] == "404"


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
    response_json = response.json()
    assert "error" in response_json
    assert response_json["error"]["message"] == "End User Id=non-existent-user does not exist in db"
    assert response_json["error"]["type"] == "not_found"
    assert response_json["error"]["param"] == "end_user_id"
    assert response_json["error"]["code"] == "404"


def test_delete_customer_not_found(mock_prisma_client, mock_user_api_key_auth):
    """
    Test that delete_end_user raises a 404 ProxyException when user_ids do not exist.
    """
    # Mock the database response to return empty list (no users found)
    mock_prisma_client.db.litellm_endusertable.find_many = AsyncMock(return_value=[])

    # Test data
    test_data = {"user_ids": ["non-existent-user-1", "non-existent-user-2"]}

    # Make the request
    response = client.post(
        "/customer/delete",
        json=test_data,
        headers={"Authorization": "Bearer test-key"},
    )

    # Assert response
    assert response.status_code == 404
    response_json = response.json()
    assert "error" in response_json
    assert "do not exist in db" in response_json["error"]["message"]
    assert "non-existent-user-1" in response_json["error"]["message"]
    assert response_json["error"]["type"] == "not_found"
    assert response_json["error"]["param"] == "user_ids"
    assert response_json["error"]["code"] == "404"


def test_error_schema_consistency(mock_prisma_client, mock_user_api_key_auth):
    """
    Test that all customer endpoints return the same error schema format.
    All ProxyException errors should have: message, type, param, and code fields.
    """
    
    def validate_error_schema(response_json):
        assert "error" in response_json, "Response should have 'error' key"
        error = response_json["error"]
        assert "message" in error, "Error should have 'message' field"
        assert "type" in error, "Error should have 'type' field"
        assert "param" in error, "Error should have 'param' field"
        assert "code" in error, "Error should have 'code' field"
        assert isinstance(error["message"], str), "message should be a string"
        assert isinstance(error["type"], str), "type should be a string"
        assert isinstance(error["code"], str), "code should be a string"
        return error

    # Test /customer/info - not found error
    mock_prisma_client.db.litellm_endusertable.find_first = AsyncMock(return_value=None)
    response = client.get(
        "/customer/info?end_user_id=non-existent",
        headers={"Authorization": "Bearer test-key"},
    )
    error = validate_error_schema(response.json())
    assert error["type"] == "not_found"
    assert error["code"] == "404"

    # Test /customer/update - not found error
    mock_prisma_client.db.litellm_endusertable.find_first = AsyncMock(return_value=None)
    response = client.post(
        "/customer/update",
        json={"user_id": "non-existent", "alias": "Test"},
        headers={"Authorization": "Bearer test-key"},
    )
    error = validate_error_schema(response.json())
    assert error["type"] == "not_found"
    assert error["code"] == "404"

    # Test /customer/delete - not found error
    mock_prisma_client.db.litellm_endusertable.find_many = AsyncMock(return_value=[])
    response = client.post(
        "/customer/delete",
        json={"user_ids": ["non-existent"]},
        headers={"Authorization": "Bearer test-key"},
    )
    error = validate_error_schema(response.json())
    assert error["type"] == "not_found"
    assert error["code"] == "404"

    # Test /customer/new - duplicate user error
    from unittest.mock import MagicMock
    
    mock_end_user = LiteLLM_EndUserTable(
        user_id="existing-user", alias="Existing User", blocked=False
    )
    mock_prisma_client.db.litellm_endusertable.create = AsyncMock(
        side_effect=Exception("Unique constraint failed on the fields: (`user_id`)")
    )
    response = client.post(
        "/customer/new",
        json={"user_id": "existing-user"},
        headers={"Authorization": "Bearer test-key"},
    )
    error = validate_error_schema(response.json())
    assert error["type"] == "bad_request"
    assert error["code"] == "400"


def test_customer_endpoints_error_schema_consistency(mock_prisma_client, mock_user_api_key_auth):
    """
    Test the exact scenarios from the curl examples provided.
    
    Scenario 1: GET /end_user/info with non-existent user
    OLD (incorrect): {"detail":{"error":"End User Id=... does not exist in db"}}
    NEW (correct):   {"error":{"message":"...","type":"not_found","param":"end_user_id","code":"404"}}
    
    Scenario 2: POST /end_user/new with existing user
    Expected:        {"error":{"message":"...","type":"bad_request","param":"user_id","code":"400"}}
    
    Both should use the same error format structure.
    """
    
    # Scenario 1: GET /end_user/info with non-existent user
    # Should return 404 with proper error schema
    mock_prisma_client.db.litellm_endusertable.find_first = AsyncMock(return_value=None)
    
    response1 = client.get(
        "/end_user/info?end_user_id=fake-test-end-user-michaels-local-testng",
        headers={"Authorization": "Bearer test-key"},
    )
    
    assert response1.status_code == 404, "Should return 404 for non-existent user"
    response1_json = response1.json()

    
    # Should have the correct format with {"error": {...}}
    assert "error" in response1_json, "Should have top-level 'error' key"
    error1 = response1_json["error"]
    assert "message" in error1, "Error should have 'message' field"
    assert "type" in error1, "Error should have 'type' field"
    assert "param" in error1, "Error should have 'param' field"
    assert "code" in error1, "Error should have 'code' field"
    assert error1["type"] == "not_found"
    assert error1["code"] == "404"
    assert "does not exist in db" in error1["message"]
    
    # Scenario 2: POST /end_user/new with existing user
    # Should return 400 with proper error schema
    mock_prisma_client.db.litellm_endusertable.create = AsyncMock(
        side_effect=Exception("Unique constraint failed on the fields: (`user_id`)")
    )
    
    response2 = client.post(
        "/end_user/new",
        json={"user_id": "fake-test-end-user-michaels-local-testing", "budget_id": "Tier0"},
        headers={"Authorization": "Bearer test-key"},
    )
    
    assert response2.status_code == 400, "Should return 400 for duplicate user"
    response2_json = response2.json()
    
    # Should have the same error structure as Scenario 1
    assert "error" in response2_json, "Should have top-level 'error' key"
    error2 = response2_json["error"]
    assert "message" in error2, "Error should have 'message' field"
    assert "type" in error2, "Error should have 'type' field"
    assert "param" in error2, "Error should have 'param' field"
    assert "code" in error2, "Error should have 'code' field"
    assert error2["type"] == "bad_request"
    assert error2["code"] == "400"
    assert "Customer already exists" in error2["message"]
    
    # Verify both errors have the same schema structure
    assert set(error1.keys()) == set(error2.keys()), \
        "Both errors should have the same top-level keys"
    
    # Both should have string values for all fields
    for key in ["message", "type", "code"]:
        assert isinstance(error1[key], str), f"error1[{key}] should be a string"
        assert isinstance(error2[key], str), f"error2[{key}] should be a string"


@pytest.mark.asyncio
async def test_get_customer_daily_activity_admin_param_passing(monkeypatch):
    from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
    from litellm.proxy.management_endpoints import customer_endpoints
    from litellm.proxy.management_endpoints.customer_endpoints import (
        get_customer_daily_activity,
    )

    mock_prisma_client = AsyncMock()
    mock_prisma_client.db.litellm_endusertable.find_many = AsyncMock(
        return_value=[]
    )
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)

    mocked_response = MagicMock(name="SpendAnalyticsPaginatedResponse")
    get_daily_activity_mock = AsyncMock(return_value=mocked_response)
    monkeypatch.setattr(
        customer_endpoints, "get_daily_activity", get_daily_activity_mock
    )

    auth = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN, user_id="admin1")
    result = await get_customer_daily_activity(
        end_user_ids="end-user-1,end-user-2",
        start_date="2024-01-01",
        end_date="2024-01-31",
        model="gpt-4",
        api_key="test-key",
        page=2,
        page_size=5,
        exclude_end_user_ids="end-user-3",
        user_api_key_dict=auth,
    )

    get_daily_activity_mock.assert_awaited_once()
    kwargs = get_daily_activity_mock.call_args.kwargs
    assert kwargs["table_name"] == "litellm_dailyenduserspend"
    assert kwargs["entity_id_field"] == "end_user_id"
    assert kwargs["entity_id"] == ["end-user-1", "end-user-2"]
    assert kwargs["exclude_entity_ids"] == ["end-user-3"]
    assert kwargs["start_date"] == "2024-01-01"
    assert kwargs["end_date"] == "2024-01-31"
    assert kwargs["model"] == "gpt-4"
    assert kwargs["api_key"] == "test-key"
    assert kwargs["page"] == 2
    assert kwargs["page_size"] == 5

    assert result is mocked_response


@pytest.mark.asyncio
async def test_get_customer_daily_activity_with_end_user_aliases(monkeypatch):
    from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
    from litellm.proxy.management_endpoints import customer_endpoints
    from litellm.proxy.management_endpoints.customer_endpoints import (
        get_customer_daily_activity,
    )

    mock_prisma_client = AsyncMock()
    mock_end_user1 = MagicMock()
    mock_end_user1.user_id = "end-user-1"
    mock_end_user1.alias = "Customer One"
    mock_end_user2 = MagicMock()
    mock_end_user2.user_id = "end-user-2"
    mock_end_user2.alias = "Customer Two"
    
    mock_prisma_client.db.litellm_endusertable.find_many = AsyncMock(
        return_value=[mock_end_user1, mock_end_user2]
    )
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)

    mocked_response = MagicMock(name="SpendAnalyticsPaginatedResponse")
    get_daily_activity_mock = AsyncMock(return_value=mocked_response)
    monkeypatch.setattr(
        customer_endpoints, "get_daily_activity", get_daily_activity_mock
    )

    auth = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN, user_id="admin1")
    await get_customer_daily_activity(
        end_user_ids="end-user-1,end-user-2",
        start_date="2024-01-01",
        end_date="2024-01-31",
        model=None,
        api_key=None,
        page=1,
        page_size=10,
        exclude_end_user_ids=None,
        user_api_key_dict=auth,
    )

    kwargs = get_daily_activity_mock.call_args.kwargs
    assert kwargs["entity_metadata_field"] == {
        "end-user-1": {"alias": "Customer One"},
        "end-user-2": {"alias": "Customer Two"},
    }
