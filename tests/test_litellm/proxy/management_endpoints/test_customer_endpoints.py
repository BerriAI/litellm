from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from litellm.proxy._types import (
    LiteLLM_EndUserTable,
    LitellmUserRoles,
    ProxyException,
)
from litellm.proxy.auth.user_api_key_auth import UserAPIKeyAuth, user_api_key_auth
from litellm.proxy.management_endpoints.customer_endpoints import router
from litellm.types.proxy.management_endpoints.common_daily_activity import (
    SpendAnalyticsPaginatedResponse,
)
from litellm.types.proxy.management_endpoints.customer_endpoints import (
    BlockUsersResponse,
    CustomerResponse,
    DeleteCustomersResponse,
    UnblockUsersResponse,
)

app = FastAPI()


@app.exception_handler(ProxyException)
async def openai_exception_handler(request: Request, exc: ProxyException):
    headers = exc.headers
    error_dict = exc.to_dict()
    return JSONResponse(
        status_code=(int(exc.code) if exc.code else status.HTTP_500_INTERNAL_SERVER_ERROR),
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
    original_overrides = app.dependency_overrides.copy()
    app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_id="test-user", user_role=LitellmUserRoles.PROXY_ADMIN
    )
    try:
        yield
    finally:
        app.dependency_overrides = original_overrides


def test_update_customer_success(mock_prisma_client, mock_user_api_key_auth):
    # Mock the database responses
    mock_end_user = LiteLLM_EndUserTable(user_id="test-user-1", alias="Test User", blocked=False)
    updated_mock_end_user = LiteLLM_EndUserTable(user_id="test-user-1", alias="Updated Test User", blocked=False)

    # Mock the find_first response
    mock_prisma_client.db.litellm_endusertable.find_first = AsyncMock(return_value=mock_end_user)

    # Mock the update response
    mock_prisma_client.db.litellm_endusertable.update = AsyncMock(return_value=updated_mock_end_user)

    # Test data
    test_data = {"user_id": "test-user-1", "alias": "Updated Test User"}

    # Make the request
    response = client.post("/customer/update", json=test_data, headers={"Authorization": "Bearer test-key"})

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
        json={
            "user_id": "fake-test-end-user-michaels-local-testing",
            "budget_id": "Tier0",
        },
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
    assert set(error1.keys()) == set(error2.keys()), "Both errors should have the same top-level keys"

    # Both should have string values for all fields
    for key in ["message", "type", "code"]:
        assert isinstance(error1[key], str), f"error1[{key}] should be a string"
        assert isinstance(error2[key], str), f"error2[{key}] should be a string"


EXPECTED_RESPONSE_MODELS = {
    "/customer/block": BlockUsersResponse,
    "/customer/unblock": UnblockUsersResponse,
    "/customer/new": CustomerResponse,
    "/customer/update": CustomerResponse,
    "/customer/delete": DeleteCustomersResponse,
    "/customer/info": CustomerResponse,
    "/customer/list": List[CustomerResponse],
    "/customer/daily/activity": SpendAnalyticsPaginatedResponse,
}


@pytest.mark.parametrize("path, expected_model", EXPECTED_RESPONSE_MODELS.items())
def test_customer_routes_declare_response_model(path, expected_model):
    """
    Every public /customer/* operation must declare a typed response_model so
    the generated OpenAPI schema documents the response body. Regression for the
    OpenAPI response-type coverage goal: drop a response_model and this fails.
    """
    route = next(r for r in router.routes if isinstance(r, APIRoute) and r.path == path)
    assert route.response_model == expected_model


def test_customer_new_documented_in_openapi_schema():
    """
    The response_model must surface in the OpenAPI schema as a concrete ref, not
    an empty/default response. This is what the coverage metric measures.
    """
    schema = app.openapi()["paths"]["/customer/new"]["post"]
    json_schema = schema["responses"]["200"]["content"]["application/json"]["schema"]
    assert json_schema["$ref"].endswith("/CustomerResponse")


def test_update_customer_response_preserves_budget_id(mock_prisma_client, mock_user_api_key_auth):
    """
    Regression for the response_model field-stripping concern: budget_id is a real
    column on the end-user table that /customer/update echoes. response_model=
    LiteLLM_EndUserTable must NOT drop it, so budget_id stays in LiteLLM_EndUserTable.
    """
    existing = LiteLLM_EndUserTable(user_id="cust-1", blocked=False)
    updated = LiteLLM_EndUserTable(user_id="cust-1", blocked=False, budget_id="budget-123")
    mock_prisma_client.db.litellm_endusertable.find_first = AsyncMock(return_value=existing)
    mock_prisma_client.db.litellm_endusertable.update = AsyncMock(return_value=updated)

    response = client.post(
        "/customer/update",
        json={"user_id": "cust-1", "budget_id": "budget-123"},
        headers={"Authorization": "Bearer test-key"},
    )

    assert response.status_code == 200
    assert response.json()["budget_id"] == "budget-123"


def test_update_customer_can_set_blocked_false(mock_prisma_client, mock_user_api_key_auth):
    existing = LiteLLM_EndUserTable(user_id="cust-1", blocked=True)
    updated = LiteLLM_EndUserTable(user_id="cust-1", blocked=False)
    mock_prisma_client.db.litellm_endusertable.find_first = AsyncMock(return_value=existing)
    mock_prisma_client.db.litellm_endusertable.update = AsyncMock(return_value=updated)

    response = client.post(
        "/customer/update",
        json={"user_id": "cust-1", "blocked": False},
        headers={"Authorization": "Bearer test-key"},
    )

    assert response.status_code == 200
    update_data = mock_prisma_client.db.litellm_endusertable.update.call_args.kwargs["data"]
    assert update_data["blocked"] is False
    assert response.json()["blocked"] is False


def test_update_customer_does_not_send_default_blocked_false(mock_prisma_client, mock_user_api_key_auth):
    existing = LiteLLM_EndUserTable(user_id="cust-1", alias="old", blocked=True)
    updated = LiteLLM_EndUserTable(user_id="cust-1", alias="new", blocked=True)
    mock_prisma_client.db.litellm_endusertable.find_first = AsyncMock(return_value=existing)
    mock_prisma_client.db.litellm_endusertable.update = AsyncMock(return_value=updated)

    response = client.post(
        "/customer/update",
        json={"user_id": "cust-1", "alias": "new"},
        headers={"Authorization": "Bearer test-key"},
    )

    assert response.status_code == 200
    update_data = mock_prisma_client.db.litellm_endusertable.update.call_args.kwargs["data"]
    assert "blocked" not in update_data


def test_update_customer_response_keeps_nested_budget_server_fields(mock_prisma_client, mock_user_api_key_auth):
    """
    Faithfulness regression: /customer/update embeds the full budget row. The
    response_model must keep the server-managed budget fields the endpoint used
    to return (budget_reset_at, created_at) instead of the narrow write-allowlist
    shape. The intentionally-internal audit fields (created_by/updated_by) stay out.
    """
    existing = LiteLLM_EndUserTable(user_id="cust-1", blocked=False)
    raw_row = MagicMock()
    raw_row.model_dump.return_value = {
        "user_id": "cust-1",
        "blocked": False,
        "alias": "renamed",
        "spend": 0.0,
        "allowed_model_region": None,
        "default_model": None,
        "budget_id": "b-1",
        "object_permission_id": None,
        "object_permission": None,
        "litellm_budget_table": {
            "budget_id": "b-1",
            "max_budget": 10.0,
            "budget_duration": "30d",
            "budget_reset_at": "2024-02-01T00:00:00",
            "created_at": "2024-01-01T00:00:00",
            "created_by": "admin",
            "updated_at": "2024-01-02T00:00:00",
            "updated_by": "admin",
        },
    }
    mock_prisma_client.db.litellm_endusertable.find_first = AsyncMock(return_value=existing)
    mock_prisma_client.db.litellm_endusertable.update = AsyncMock(return_value=raw_row)

    response = client.post(
        "/customer/update",
        json={"user_id": "cust-1", "alias": "renamed"},
        headers={"Authorization": "Bearer test-key"},
    )

    assert response.status_code == 200
    budget = response.json()["litellm_budget_table"]
    assert budget["budget_reset_at"] == "2024-02-01T00:00:00"
    assert budget["created_at"] == "2024-01-01T00:00:00"
    assert "created_by" not in budget
    assert "updated_by" not in budget


def test_block_customer_success_serializes_through_response_model(mock_prisma_client, mock_user_api_key_auth):
    """
    /customer/block returns {"blocked_users": [<end user rows>]}. With
    response_model=BlockUsersResponse, a shape mismatch would raise a 500
    ResponseValidationError, so a clean 200 proves the model matches runtime output.
    """
    blocked_row = LiteLLM_EndUserTable(user_id="blocked-1", blocked=True)
    mock_prisma_client.db.litellm_endusertable.upsert = AsyncMock(return_value=blocked_row)

    response = client.post(
        "/customer/block",
        json={"user_ids": ["blocked-1"]},
        headers={"Authorization": "Bearer test-key"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["blocked_users"][0]["user_id"] == "blocked-1"
    assert body["blocked_users"][0]["blocked"] is True


def test_delete_customer_success_serializes_through_response_model(mock_prisma_client, mock_user_api_key_auth):
    """
    /customer/delete returns {"deleted_customers": <int>, "message": <str>}.
    response_model=DeleteCustomersResponse enforces that exact shape.
    """
    existing = [
        LiteLLM_EndUserTable(user_id="u1", blocked=False),
        LiteLLM_EndUserTable(user_id="u2", blocked=False),
    ]
    mock_prisma_client.db.litellm_endusertable.find_many = AsyncMock(return_value=existing)
    mock_prisma_client.db.litellm_endusertable.delete_many = AsyncMock(return_value=2)

    response = client.post(
        "/customer/delete",
        json={"user_ids": ["u1", "u2"]},
        headers={"Authorization": "Bearer test-key"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "deleted_customers": 2,
        "message": "Successfully deleted customers with ids: ['u1', 'u2']",
    }


@pytest.mark.asyncio
async def test_get_customer_daily_activity_admin_param_passing(monkeypatch):
    from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
    from litellm.proxy.management_endpoints import customer_endpoints
    from litellm.proxy.management_endpoints.customer_endpoints import (
        get_customer_daily_activity,
    )

    mock_prisma_client = AsyncMock()
    mock_prisma_client.db.litellm_endusertable.find_many = AsyncMock(return_value=[])
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)

    mocked_response = MagicMock(name="SpendAnalyticsPaginatedResponse")
    get_daily_activity_mock = AsyncMock(return_value=mocked_response)
    monkeypatch.setattr(customer_endpoints, "get_daily_activity", get_daily_activity_mock)

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

    mock_prisma_client.db.litellm_endusertable.find_many = AsyncMock(return_value=[mock_end_user1, mock_end_user2])
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)

    mocked_response = MagicMock(name="SpendAnalyticsPaginatedResponse")
    get_daily_activity_mock = AsyncMock(return_value=mocked_response)
    monkeypatch.setattr(customer_endpoints, "get_daily_activity", get_daily_activity_mock)

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


@pytest.mark.asyncio
async def test_get_customer_daily_activity_non_admin_is_rejected(monkeypatch):
    """
    Security regression: any non-admin caller must receive 401 from
    /customer/daily/activity and /end_user/daily/activity.

    Before this fix, the endpoint performed no role check. A caller with
    user_role=INTERNAL_USER could omit end_user_ids, causing entity_id=None
    to flow into get_daily_activity where the SQL builder treats it as no
    filter — returning every tenant's spend across the full
    LiteLLM_DailyEndUserSpend table.

    LiteLLM_EndUserTable has no per-tenant ownership column, so non-admin
    scoping is not possible. The correct fix is admin-only, matching the
    existing /customer/list gate.
    """
    from litellm.proxy.management_endpoints import customer_endpoints
    from litellm.proxy.management_endpoints.customer_endpoints import (
        get_customer_daily_activity,
    )

    mock_prisma_client = MagicMock()
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)

    get_daily_activity_mock = AsyncMock()
    monkeypatch.setattr(customer_endpoints, "get_daily_activity", get_daily_activity_mock)

    non_admin_key = UserAPIKeyAuth(
        user_id="regular-user-abc",
        user_role=LitellmUserRoles.INTERNAL_USER,
    )

    with pytest.raises(HTTPException) as exc_info:
        await get_customer_daily_activity(
            end_user_ids=None,
            start_date="2025-01-01",
            end_date="2025-01-31",
            model=None,
            api_key=None,
            page=1,
            page_size=10,
            exclude_end_user_ids=None,
            user_api_key_dict=non_admin_key,
        )

    assert exc_info.value.status_code == 401
    assert "Admin-only endpoint" in str(exc_info.value.detail)
    get_daily_activity_mock.assert_not_called()


@pytest.mark.asyncio
async def test_get_customer_daily_activity_service_account_key_is_rejected(monkeypatch):
    """
    Security regression: service-account keys (user_id=None, role=INTERNAL_USER)
    must be rejected at the admin gate before reaching get_daily_activity.

    A service-account key with end_user_ids omitted is the worst-case caller:
    entity_id=None and no user identity to scope by — the SQL builder would
    return the full LiteLLM_DailyEndUserSpend table with no WHERE clause.
    """
    from litellm.proxy.management_endpoints import customer_endpoints
    from litellm.proxy.management_endpoints.customer_endpoints import (
        get_customer_daily_activity,
    )

    mock_prisma_client = MagicMock()
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)

    get_daily_activity_mock = AsyncMock()
    monkeypatch.setattr(customer_endpoints, "get_daily_activity", get_daily_activity_mock)

    service_account_key = UserAPIKeyAuth(
        user_id=None,
        user_role=LitellmUserRoles.INTERNAL_USER,
    )

    with pytest.raises(HTTPException) as exc_info:
        await get_customer_daily_activity(
            end_user_ids=None,
            start_date="2025-01-01",
            end_date="2025-01-31",
            model=None,
            api_key=None,
            page=1,
            page_size=10,
            exclude_end_user_ids=None,
            user_api_key_dict=service_account_key,
        )

    assert exc_info.value.status_code == 401
    assert "Admin-only endpoint" in str(exc_info.value.detail)
    get_daily_activity_mock.assert_not_called()


# ---------------------------------------------------------------------------
# Characterization (golden-master) tests.
#
# These lock the EXACT JSON body every customer-object endpoint returns today,
# so a type-safety refactor of the handlers is only allowed to land if it
# reproduces these byte for byte. The input below is what a Prisma row's
# .model_dump() yields (full nested budget incl. audit fields + object_permission
# incl. reverse relations); the expected output is what the live endpoint emits.
# ---------------------------------------------------------------------------

_FULL_DB_ROW = {
    "user_id": "c1",
    "blocked": False,
    "alias": "Acme",
    "spend": 1.5,
    "allowed_model_region": None,
    "default_model": None,
    "budget_id": "b1",
    "object_permission_id": "p1",
    "litellm_budget_table": {
        "budget_id": "b1",
        "max_budget": 10.0,
        "soft_budget": None,
        "max_parallel_requests": None,
        "tpm_limit": None,
        "rpm_limit": None,
        "model_max_budget": None,
        "budget_duration": "30d",
        "allowed_models": [],
        "budget_reset_at": "2024-02-01T00:00:00",
        "created_at": "2024-01-01T00:00:00",
        "created_by": "admin",
        "updated_at": "2024-01-02T00:00:00",
        "updated_by": "admin",
    },
    "object_permission": {
        "object_permission_id": "p1",
        "mcp_servers": ["s1"],
        "mcp_access_groups": [],
        "mcp_tool_permissions": None,
        "vector_stores": [],
        "agents": [],
        "agent_access_groups": [],
        "models": [],
        "mcp_toolsets": None,
        "blocked_tools": [],
        "search_tools": [],
        "teams": [{"team_id": "t1"}],
        "users": [{"user_id": "x"}],
        "end_users": [],
        "organizations": [],
        "verification_tokens": [],
    },
}

_EXPECTED_CUSTOMER = {
    "user_id": "c1",
    "blocked": False,
    "alias": "Acme",
    "spend": 1.5,
    "allowed_model_region": None,
    "default_model": None,
    "budget_id": "b1",
    "litellm_budget_table": {
        "budget_id": "b1",
        "soft_budget": None,
        "max_budget": 10.0,
        "max_parallel_requests": None,
        "tpm_limit": None,
        "rpm_limit": None,
        "model_max_budget": None,
        "budget_duration": "30d",
        "allowed_models": [],
        "budget_reset_at": "2024-02-01T00:00:00",
        "created_at": "2024-01-01T00:00:00",
    },
    "object_permission_id": "p1",
    "object_permission": {
        "object_permission_id": "p1",
        "mcp_servers": ["s1"],
        "mcp_access_groups": [],
        "mcp_tool_permissions": None,
        "vector_stores": [],
        "agents": [],
        "agent_access_groups": [],
        "models": [],
        "mcp_toolsets": None,
        "blocked_tools": [],
        "search_tools": [],
        "mcp_tool_search_enabled": None,
    },
}


def _row(dump: dict) -> MagicMock:
    row = MagicMock()
    row.model_dump.return_value = dump
    return row


def test_char_info_body(mock_prisma_client, mock_user_api_key_auth):
    mock_prisma_client.db.litellm_endusertable.find_first = AsyncMock(return_value=_row(_FULL_DB_ROW))
    response = client.get("/customer/info?end_user_id=c1", headers={"Authorization": "Bearer k"})
    assert response.status_code == 200
    assert response.json() == _EXPECTED_CUSTOMER


def test_char_list_body(mock_prisma_client, mock_user_api_key_auth):
    mock_prisma_client.db.litellm_endusertable.find_many = AsyncMock(return_value=[_row(_FULL_DB_ROW)])
    response = client.get("/customer/list", headers={"Authorization": "Bearer k"})
    assert response.status_code == 200
    assert response.json() == [_EXPECTED_CUSTOMER]


def test_char_new_body(mock_prisma_client, mock_user_api_key_auth):
    mock_prisma_client.db.litellm_endusertable.create = AsyncMock(return_value=_row(_FULL_DB_ROW))
    response = client.post("/customer/new", json={"user_id": "c1"}, headers={"Authorization": "Bearer k"})
    assert response.status_code == 200
    assert response.json() == _EXPECTED_CUSTOMER


def test_char_update_body(mock_prisma_client, mock_user_api_key_auth):
    mock_prisma_client.db.litellm_endusertable.find_first = AsyncMock(
        return_value=_row({"user_id": "c1", "blocked": False})
    )
    mock_prisma_client.db.litellm_endusertable.update = AsyncMock(return_value=_row(_FULL_DB_ROW))
    response = client.post(
        "/customer/update",
        json={"user_id": "c1", "alias": "Acme"},
        headers={"Authorization": "Bearer k"},
    )
    assert response.status_code == 200
    assert response.json() == _EXPECTED_CUSTOMER


def test_char_delete_body(mock_prisma_client, mock_user_api_key_auth):
    mock_prisma_client.db.litellm_endusertable.find_many = AsyncMock(
        return_value=[
            LiteLLM_EndUserTable(user_id="c1", blocked=False),
            LiteLLM_EndUserTable(user_id="c2", blocked=False),
        ]
    )
    mock_prisma_client.db.litellm_endusertable.delete_many = AsyncMock(return_value=2)
    response = client.post(
        "/customer/delete",
        json={"user_ids": ["c1", "c2"]},
        headers={"Authorization": "Bearer k"},
    )
    assert response.status_code == 200
    assert response.json() == {
        "deleted_customers": 2,
        "message": "Successfully deleted customers with ids: ['c1', 'c2']",
    }
