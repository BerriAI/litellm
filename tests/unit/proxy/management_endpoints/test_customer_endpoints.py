import asyncio
import json
import sqlite3
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager, contextmanager
from types import MappingProxyType, SimpleNamespace
from typing import Final, Literal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from prisma.models import LiteLLM_EndUserTable as PrismaEndUserRow
from pydantic import TypeAdapter

from litellm.proxy._types import (
    LiteLLM_BudgetTable,
    LiteLLM_EndUserTable,
    LiteLLM_ObjectPermissionBase,
    LiteLLM_ObjectPermissionTable,
    LiteLLM_UserTable,
    LitellmUserRoles,
    ProxyException,
)
from litellm.proxy.auth.route_checks import RouteChecks
from litellm.proxy.auth.user_api_key_auth import UserAPIKeyAuth, user_api_key_auth
from litellm.proxy.common_utils.user_api_key_cache import (
    UserApiKeyCache,
    end_user_cache_key,
    end_user_restricted_registry_cache_key,
)
from litellm.proxy.management_endpoints.customer_endpoints import _should_update_field, _update_single_customer, router
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


@pytest.mark.parametrize(
    ("field", "value", "sent_fields", "expected"),
    (
        ("models", None, frozenset({"models"}), False),
        ("blocked", False, frozenset({"blocked"}), True),
        ("blocked", False, frozenset(), False),
        ("models", [], frozenset({"models"}), True),
        ("models", [], frozenset(), False),
        ("metadata", {}, frozenset({"metadata"}), False),
        ("object_permission", {"mcp_servers": ["s1"]}, frozenset(), True),
        ("object_permission", {"mcp_servers": ["s1"]}, frozenset({"object_permission"}), True),
        ("metadata", ["m1"], frozenset(), True),
        ("models", ["m1"], frozenset(), True),
        ("max_budget", 0, frozenset({"max_budget"}), False),
        ("max_budget", 5.0, frozenset(), True),
        ("alias", "a", frozenset(), True),
    ),
    ids=(
        "null-models",
        "explicit-false",
        "omitted-false",
        "clear-models",
        "omitted-empty-models",
        "empty-metadata",
        "nonempty-object-permission-omitted",
        "nonempty-object-permission-sent",
        "nonempty-list-other-field",
        "nonempty-models",
        "zero-budget",
        "nonzero-budget",
        "alias",
    ),
)
def test_should_update_field(
    field: str,
    value: object,
    sent_fields: frozenset[str],
    expected: bool,
) -> None:
    assert _should_update_field(field, value, sent_fields) is expected


@pytest.fixture
def mock_prisma_client():
    with patch("litellm.proxy.proxy_server.prisma_client") as mock:

        async def lock_customer(_query: str, user_id: str) -> tuple[Mapping[str, str], ...]:
            customer: Final = mock.db.litellm_endusertable.find_first.return_value
            if customer is None:
                return ()
            row: Final = LiteLLM_EndUserTable.model_validate(customer.model_dump())
            return ({"user_id": user_id},) if row.user_id == user_id else ()

        mock.configure_mock(
            **{
                "db.query_raw": AsyncMock(side_effect=lock_customer),
                "tx.return_value.__aenter__": AsyncMock(side_effect=lambda: mock.db),
            }
        )
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


def test_update_customer_unblock(mock_prisma_client, mock_user_api_key_auth):
    mock_end_user = LiteLLM_EndUserTable(user_id="test-user-1", blocked=True)
    updated_mock_end_user = LiteLLM_EndUserTable(user_id="test-user-1", blocked=False)

    mock_prisma_client.db.litellm_endusertable.find_first = AsyncMock(return_value=mock_end_user)
    mock_prisma_client.db.litellm_endusertable.update = AsyncMock(return_value=updated_mock_end_user)

    response = client.post(
        "/customer/update",
        json={"user_id": "test-user-1", "blocked": False},
        headers={"Authorization": "Bearer test-key"},
    )

    assert response.status_code == 200
    assert response.json()["blocked"] is False
    update_mock = mock_prisma_client.db.litellm_endusertable.update
    update_mock.assert_called_once()
    assert update_mock.call_args.kwargs["data"]["blocked"] is False


def test_update_customer_keeps_blocked_when_omitted(mock_prisma_client, mock_user_api_key_auth):
    """
    Regression test: updating a blocked customer without supplying `blocked`
    must NOT reset it to unblocked. `blocked=False` is the model default and
    should only be applied when explicitly provided by the caller.
    """
    mock_end_user = LiteLLM_EndUserTable(user_id="test-user-1", blocked=True)
    updated_mock_end_user = LiteLLM_EndUserTable(user_id="test-user-1", blocked=True)

    mock_prisma_client.db.litellm_endusertable.find_first = AsyncMock(return_value=mock_end_user)
    mock_prisma_client.db.litellm_endusertable.update = AsyncMock(return_value=updated_mock_end_user)

    response = client.post(
        "/customer/update",
        json={"user_id": "test-user-1", "alias": "Updated Test User"},
        headers={"Authorization": "Bearer test-key"},
    )

    assert response.status_code == 200
    update_mock = mock_prisma_client.db.litellm_endusertable.update
    update_mock.assert_called_once()
    assert "blocked" not in update_mock.call_args.kwargs["data"]


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
    "/customer/list": list[CustomerResponse],
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


@pytest.mark.parametrize(
    "budget_payload",
    [{"max_budget": None}, {}],
    ids=["explicit-null", "omitted"],
)
def test_update_customer_budget_omission_and_null_preserve_existing_budget(
    mock_prisma_client, mock_user_api_key_auth, budget_payload
):
    from litellm.proxy._types import LiteLLM_BudgetTable

    class BudgetState:
        def __init__(self) -> None:
            self.max_budget: float | None = 100.0

        def store(self, data) -> None:
            self.max_budget = data.get("max_budget", self.max_budget)

    budget_state = BudgetState()

    def end_user_row():
        return LiteLLM_EndUserTable(
            user_id="cust-1",
            blocked=False,
            budget_id="budget-1",
            litellm_budget_table=LiteLLM_BudgetTable(budget_id="budget-1", max_budget=budget_state.max_budget),
        )

    def response_row():
        row = MagicMock()
        row.model_dump.return_value = {
            "user_id": "cust-1",
            "blocked": False,
            "budget_id": "budget-1",
            "litellm_budget_table": {
                "budget_id": "budget-1",
                "max_budget": budget_state.max_budget,
                "created_at": "2024-01-01T00:00:00",
            },
        }
        return row

    async def update_budget(*, where, data):
        budget_state.store(data)
        return LiteLLM_BudgetTable(budget_id="budget-1", max_budget=budget_state.max_budget)

    mock_prisma_client.db.litellm_endusertable.find_first = AsyncMock(return_value=end_user_row())
    mock_prisma_client.db.litellm_budgettable.update = AsyncMock(side_effect=update_budget)
    mock_prisma_client.db.litellm_endusertable.update = AsyncMock(side_effect=lambda **_: response_row())

    response = client.post(
        "/customer/update",
        json={"user_id": "cust-1", **budget_payload},
        headers={"Authorization": "Bearer test-key"},
    )

    assert response.status_code == 200, response.text
    assert response.json()["litellm_budget_table"]["max_budget"] == 100.0


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
    "models": ["allowed-model"],
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
    "models": ["allowed-model"],
    "budget_id": "b1",
    "litellm_budget_table": {
        "budget_id": "b1",
        "soft_budget": None,
        "max_budget": 10.0,
        "max_parallel_requests": None,
        "tpm_limit": None,
        "rpm_limit": None,
        "tpd_limit": None,
        "model_max_budget": None,
        "budget_duration": "30d",
        "allowed_models": [],
        "temp_budget_increase": None,
        "temp_budget_expiry": None,
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
        "skills": None,
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


def test_customer_new_forwards_models_to_db(mock_prisma_client, mock_user_api_key_auth):
    mock_prisma_client.db.litellm_endusertable.create = AsyncMock(return_value=_row(_FULL_DB_ROW))

    response = client.post(
        "/customer/new",
        json={"user_id": "c1", "models": ["allowed-model"]},
        headers={"Authorization": "Bearer k"},
    )

    assert response.status_code == 200, response.text
    assert mock_prisma_client.db.litellm_endusertable.create.call_args.kwargs["data"]["models"] == ["allowed-model"]


@pytest.mark.parametrize("bad_duration", ["0s", "-5m"])
def test_customer_new_rejects_a_duration_that_never_advances(mock_prisma_client, mock_user_api_key_auth, bad_duration):
    """A zero-length window resets to "now", leaving the customer's budget row
    permanently due for the reset job to re-read every tick."""
    mock_prisma_client.db.litellm_endusertable.create = AsyncMock(return_value=_row(_FULL_DB_ROW))

    response = client.post(
        "/customer/new",
        json={"user_id": "c1", "max_budget": 10.0, "budget_duration": bad_duration},
        headers={"Authorization": "Bearer k"},
    )

    assert response.status_code == 400, response.text
    assert "Invalid budget_duration" in response.text
    mock_prisma_client.db.litellm_endusertable.create.assert_not_awaited()


def test_customer_new_accepts_a_normal_duration(mock_prisma_client, mock_user_api_key_auth):
    mock_prisma_client.db.litellm_endusertable.create = AsyncMock(return_value=_row(_FULL_DB_ROW))
    mock_prisma_client.db.litellm_budgettable.create = AsyncMock(
        return_value=_row({"budget_id": "b1", "max_budget": 10.0})
    )

    response = client.post(
        "/customer/new",
        json={"user_id": "c1", "max_budget": 10.0, "budget_duration": "30d"},
        headers={"Authorization": "Bearer k"},
    )

    assert response.status_code == 200, response.text


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
    assert "models" not in mock_prisma_client.db.litellm_endusertable.update.call_args.kwargs["data"]


def test_customer_update_clears_models_allowlist(mock_prisma_client, mock_user_api_key_auth):
    mock_prisma_client.db.litellm_endusertable.find_first = AsyncMock(
        return_value=_row({"user_id": "c1", "blocked": False})
    )
    mock_prisma_client.db.litellm_endusertable.update = AsyncMock(return_value=_row(_FULL_DB_ROW))

    response = client.post(
        "/customer/update",
        json={"user_id": "c1", "models": []},
        headers={"Authorization": "Bearer k"},
    )

    assert response.status_code == 200, response.text
    assert mock_prisma_client.db.litellm_endusertable.update.call_args.kwargs["data"]["models"] == []


def test_customer_update_applies_nonempty_object_permission(mock_prisma_client, mock_user_api_key_auth):
    mock_prisma_client.db.litellm_endusertable.find_first = AsyncMock(
        return_value=_row({"user_id": "c1", "blocked": False, "object_permission_id": None})
    )
    mock_prisma_client.db.litellm_objectpermissiontable.find_unique = AsyncMock(return_value=None)
    updated_permission = MagicMock()
    updated_permission.object_permission_id = "permission-1"
    mock_prisma_client.db.litellm_objectpermissiontable.upsert = AsyncMock(return_value=updated_permission)
    mock_prisma_client.db.litellm_endusertable.update = AsyncMock(
        return_value=LiteLLM_EndUserTable(user_id="c1", blocked=False, object_permission_id="permission-1")
    )

    response = client.post(
        "/customer/update",
        json={"user_id": "c1", "object_permission": {"mcp_servers": ["s1"]}},
        headers={"Authorization": "Bearer k"},
    )

    assert response.status_code == 200, response.text
    mock_prisma_client.db.litellm_objectpermissiontable.upsert.assert_awaited_once()
    permission_upsert = mock_prisma_client.db.litellm_objectpermissiontable.upsert.call_args.kwargs
    assert permission_upsert["data"]["create"]["mcp_servers"] == ["s1"]
    assert mock_prisma_client.db.litellm_endusertable.update.call_args.kwargs["data"]["object_permission_id"] == (
        "permission-1"
    )


def test_customer_update_creates_partial_toolset_permission(
    mock_prisma_client: MagicMock, mock_user_api_key_auth: None
) -> None:
    async def persist_permission(
        *, where: Mapping[str, str], data: Mapping[str, Mapping[str, object]]
    ) -> LiteLLM_ObjectPermissionTable:
        assert data["create"] == {
            "object_permission_id": where["object_permission_id"],
            "mcp_toolsets": ["toolset-id"],
        }
        assert data["update"] == {"mcp_toolsets": ["toolset-id"]}
        return LiteLLM_ObjectPermissionTable.model_validate(data["create"])

    mock_prisma_client.configure_mock(
        **{
            "db.litellm_endusertable.find_first": AsyncMock(
                return_value=LiteLLM_EndUserTable(user_id="customer-id", blocked=False)
            ),
            "db.litellm_objectpermissiontable.find_unique": AsyncMock(return_value=None),
            "db.litellm_objectpermissiontable.upsert": AsyncMock(side_effect=persist_permission),
            "db.litellm_endusertable.update": AsyncMock(
                return_value=LiteLLM_EndUserTable(user_id="customer-id", blocked=False)
            ),
        }
    )

    response: Final = client.post(
        "/customer/update",
        json={"user_id": "customer-id", "object_permission": {"mcp_toolsets": ["toolset-id"]}},
        headers={"Authorization": "Bearer test-key"},
    )

    assert response.status_code == 200, response.text
    assert response.json()["user_id"] == "customer-id"


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


class _RecordingAuthCache:
    """Captures the keys an endpoint evicts, so tests assert on cache keys not mock plumbing."""

    def __init__(self):
        self.deleted: list[str] = []

    async def async_delete_cache(self, key: str) -> None:
        self.deleted.append(key)


@contextmanager
def _end_user_cache_doubles():
    """Swaps in the auth cache and the cross-worker publisher a customer mutation is expected to hit."""
    recording_cache = _RecordingAuthCache()
    mock_publish = AsyncMock()
    with (
        patch("litellm.proxy.proxy_server.user_api_key_cache", recording_cache),
        patch(
            "litellm.proxy.common_utils.auth_cache_invalidation_pubsub.publish_auth_cache_invalidation",
            mock_publish,
        ),
    ):
        yield recording_cache, mock_publish


def _published_keys(mock_publish) -> list[str]:
    return [call.kwargs["cache_key"] for call in mock_publish.call_args_list]


def test_customer_new_invalidates_end_user_and_registry_caches(mock_prisma_client, mock_user_api_key_auth):
    """
    A customer created on one worker must be visible to every worker's auth path immediately.

    Auth serves end users cache-first, and the cached restricted-id registry is what decides whether
    the row is read at all, so a create that leaves both entries stale means the new customer's
    budget or block goes unenforced until the TTL expires.
    """
    mock_prisma_client.db.litellm_endusertable.create = AsyncMock(return_value=_row(_FULL_DB_ROW))

    with _end_user_cache_doubles() as (recording_cache, mock_publish):
        response = client.post(
            "/customer/new",
            json={"user_id": "c1", "blocked": True},
            headers={"Authorization": "Bearer k"},
        )

    assert response.status_code == 200, response.text
    assert recording_cache.deleted == ["end_user_id:c1", "end_user_restricted_registry"]
    assert _published_keys(mock_publish) == ["end_user_id:c1", "end_user_restricted_registry"]


def test_customer_update_invalidates_end_user_and_registry_caches(mock_prisma_client, mock_user_api_key_auth):
    """An update can add or drop a budget, block, region or permission, moving the id in the registry."""
    mock_prisma_client.db.litellm_endusertable.find_first = AsyncMock(
        return_value=_row({"user_id": "c1", "blocked": False})
    )
    mock_prisma_client.db.litellm_endusertable.update = AsyncMock(return_value=_row(_FULL_DB_ROW))

    with _end_user_cache_doubles() as (recording_cache, mock_publish):
        response = client.post(
            "/customer/update",
            json={"user_id": "c1", "budget_id": "b1"},
            headers={"Authorization": "Bearer k"},
        )

    assert response.status_code == 200, response.text
    assert recording_cache.deleted == ["end_user_id:c1", "end_user_restricted_registry"]
    assert _published_keys(mock_publish) == ["end_user_id:c1", "end_user_restricted_registry"]


def test_customer_block_invalidates_end_user_and_registry_caches(mock_prisma_client, mock_user_api_key_auth):
    """Blocking is the one mutation that must take effect instantly; a stale registry keeps serving it."""
    mock_prisma_client.db.litellm_endusertable.upsert = AsyncMock(
        return_value=LiteLLM_EndUserTable(user_id="c1", blocked=True)
    )

    with _end_user_cache_doubles() as (recording_cache, mock_publish):
        response = client.post(
            "/customer/block",
            json={"user_ids": ["c1", "c2"]},
            headers={"Authorization": "Bearer k"},
        )

    assert response.status_code == 200, response.text
    assert recording_cache.deleted == [
        "end_user_id:c1",
        "end_user_id:c2",
        "end_user_restricted_registry",
    ]
    assert _published_keys(mock_publish) == [
        "end_user_id:c1",
        "end_user_id:c2",
        "end_user_restricted_registry",
    ]


def test_customer_delete_invalidates_end_user_and_registry_caches(mock_prisma_client, mock_user_api_key_auth):
    """Without this a deleted customer keeps its cached budget and block enforced until the TTL expires."""
    mock_prisma_client.db.litellm_endusertable.find_many = AsyncMock(
        return_value=[
            LiteLLM_EndUserTable(user_id="c1", blocked=False),
            LiteLLM_EndUserTable(user_id="c2", blocked=False),
        ]
    )
    mock_prisma_client.db.litellm_endusertable.delete_many = AsyncMock(return_value=2)

    with _end_user_cache_doubles() as (recording_cache, mock_publish):
        response = client.post(
            "/customer/delete",
            json={"user_ids": ["c1", "c2"]},
            headers={"Authorization": "Bearer k"},
        )

    assert response.status_code == 200, response.text
    assert recording_cache.deleted == [
        "end_user_id:c1",
        "end_user_id:c2",
        "end_user_restricted_registry",
    ]
    assert _published_keys(mock_publish) == [
        "end_user_id:c1",
        "end_user_id:c2",
        "end_user_restricted_registry",
    ]


class _CustomerBatchDatabase:
    def __init__(
        self,
        customers: tuple[LiteLLM_EndUserTable, ...],
        permissions: tuple[LiteLLM_ObjectPermissionTable, ...] = (),
        budgets: tuple[LiteLLM_BudgetTable, ...] = (),
        failed_customer_ids: frozenset[str] = frozenset(),
        budget_references: tuple[tuple[str, str], ...] = (),
    ) -> None:
        self.customers: Mapping[str, LiteLLM_EndUserTable] = MappingProxyType(
            {customer.user_id: customer for customer in customers}
        )
        self.permissions: Mapping[str, LiteLLM_ObjectPermissionTable] = MappingProxyType(
            {permission.object_permission_id: permission for permission in permissions}
        )
        self.budgets: Mapping[str | None, LiteLLM_BudgetTable] = MappingProxyType(
            {budget.budget_id: budget for budget in budgets}
        )
        self.failed_customer_ids = failed_customer_ids
        self.budget_references = budget_references
        self.locked_budget_ids: frozenset[str] = frozenset()

    async def query_raw(self, query: str, identifier: str, *parameters: str) -> tuple[Mapping[str, object], ...]:
        if 'FROM "LiteLLM_EndUserTable"' in query and query.rstrip().endswith("FOR UPDATE"):
            return ({"user_id": identifier},) if identifier in self.customers else ()
        if query.rstrip().endswith("FOR UPDATE"):
            self.locked_budget_ids = self.locked_budget_ids | {identifier}
            return ({"budget_id": identifier},) if identifier in self.budgets else ()
        assert identifier in self.locked_budget_ids
        return self.budget_reference_rows(query, (identifier, *parameters))

    def budget_reference_rows(self, query: str, parameters: tuple[str, ...]) -> tuple[Mapping[str, object], ...]:
        table_names: Final = (
            "LiteLLM_BudgetTable",
            "LiteLLM_EndUserTable",
            "LiteLLM_VerificationToken",
            "LiteLLM_OrganizationTable",
            "LiteLLM_ProjectTable",
            "LiteLLM_TagTable",
            "LiteLLM_ModelAccessGroupBudgetTable",
            "LiteLLM_TeamMembership",
            "LiteLLM_OrganizationMembership",
            "LiteLLM_TeamTable",
        )
        with sqlite3.connect(":memory:") as database:
            database.row_factory = sqlite3.Row
            for table in table_names:
                database.execute(f'CREATE TABLE "{table}" (budget_id TEXT, user_id TEXT, metadata TEXT)')
            database.executemany(
                'INSERT INTO "LiteLLM_BudgetTable" (budget_id) VALUES (?)',
                tuple((budget_id,) for budget_id in self.budgets),
            )
            database.executemany(
                'INSERT INTO "LiteLLM_EndUserTable" (budget_id, user_id) VALUES (?, ?)',
                tuple((customer.budget_id, customer.user_id) for customer in self.customers.values()),
            )
            for reference, budget_id in self.budget_references:
                table: Final = (
                    "LiteLLM_VerificationToken"
                    if reference == "key_default"
                    else "LiteLLM_TeamTable"
                    if reference == "team_default"
                    else reference
                )
                assert table in table_names
                metadata_key: Final = "end_user_budget_id" if reference == "key_default" else "team_member_budget_id"
                metadata: Final = json.dumps({metadata_key: budget_id}) if reference.endswith("_default") else "{}"
                database.execute(
                    f'INSERT INTO "{table}" (budget_id, metadata) VALUES (?, ?)',
                    (None if reference.endswith("_default") else budget_id, metadata),
                )
            return TypeAdapter(tuple[dict[str, object], ...]).validate_python(
                [
                    dict(row)
                    for row in database.execute(
                        query, {str(index): value for index, value in enumerate(parameters, 1)}
                    ).fetchall()
                ]
            )

    async def find_customer(
        self, *, where: Mapping[str, str], include: Mapping[str, bool]
    ) -> LiteLLM_EndUserTable | None:
        return self.customers.get(where["user_id"])

    async def update_customer(
        self, *, where: Mapping[str, str], data: Mapping[str, object], include: Mapping[str, bool]
    ) -> LiteLLM_EndUserTable:
        if where["user_id"] in self.failed_customer_ids:
            raise ValueError("Customer write rejected")
        if data.get("budget_id") and data["budget_id"] not in self.budgets:
            raise ValueError("Budget does not exist")
        existing: Final = self.customers[where["user_id"]]
        updated: Final = LiteLLM_EndUserTable.model_validate({**existing.model_dump(), **data})
        permission: Final = self.permissions.get(updated.object_permission_id or "")
        saved: Final = updated.model_copy(update={"object_permission": permission})
        self.customers = MappingProxyType({**self.customers, saved.user_id: saved})
        budget: Final = self.budgets.get(saved.budget_id)
        return (
            CustomerResponse.model_validate(
                {
                    **saved.model_dump(),
                    "litellm_budget_table": {**budget.model_dump(), "created_at": "2026-01-01T00:00:00Z"},
                }
            )
            if budget is not None
            else saved
        )

    async def find_permission(self, *, where: Mapping[str, str]) -> LiteLLM_ObjectPermissionTable | None:
        return self.permissions.get(where["object_permission_id"])

    async def upsert_permission(
        self, *, where: Mapping[str, str], data: Mapping[str, Mapping[str, object]]
    ) -> LiteLLM_ObjectPermissionTable:
        existing: Final = self.permissions.get(where["object_permission_id"])
        stored: Final = {**existing.model_dump(), **data["update"]} if existing is not None else data["create"]
        tool_permissions: Final = stored.get("mcp_tool_permissions")
        decoded: Final = (
            TypeAdapter(dict[str, list[str]]).validate_json(tool_permissions)
            if isinstance(tool_permissions, str)
            else tool_permissions
        )
        permission: Final = LiteLLM_ObjectPermissionTable.model_validate({**stored, "mcp_tool_permissions": decoded})
        self.permissions = MappingProxyType({**self.permissions, permission.object_permission_id: permission})
        return permission

    async def create_budget(self, *, data: Mapping[str, object], include: Mapping[str, bool]) -> LiteLLM_BudgetTable:
        budget: Final = LiteLLM_BudgetTable.model_validate({**data, "budget_id": "created-budget"})
        self.budgets = MappingProxyType({**self.budgets, budget.budget_id: budget})
        return budget

    async def update_budget(self, *, where: Mapping[str, str], data: Mapping[str, object]) -> LiteLLM_BudgetTable:
        existing: Final = self.budgets[where["budget_id"]]
        budget: Final = LiteLLM_BudgetTable.model_validate({**existing.model_dump(), **data})
        self.budgets = MappingProxyType({**self.budgets, budget.budget_id: budget})
        return budget

    def attach(self, prisma_client: MagicMock) -> None:
        @asynccontextmanager
        async def transaction() -> AsyncIterator[MagicMock]:
            customers: Final = self.customers
            permissions: Final = self.permissions
            budgets: Final = self.budgets
            try:
                yield prisma_client.db
            except Exception:
                self.customers = customers
                self.permissions = permissions
                self.budgets = budgets
                raise

        prisma_client.configure_mock(
            **{
                "tx": MagicMock(side_effect=transaction),
                "db.query_raw": AsyncMock(side_effect=self.query_raw),
                "db.litellm_endusertable.find_first": AsyncMock(side_effect=self.find_customer),
                "db.litellm_endusertable.update": AsyncMock(side_effect=self.update_customer),
                "db.litellm_objectpermissiontable.find_unique": AsyncMock(side_effect=self.find_permission),
                "db.litellm_objectpermissiontable.upsert": AsyncMock(side_effect=self.upsert_permission),
                "db.litellm_budgettable.create": AsyncMock(side_effect=self.create_budget),
                "db.litellm_budgettable.update": AsyncMock(side_effect=self.update_budget),
            }
        )


def test_customer_update_does_not_write_row_created_after_empty_lock_result(
    mock_prisma_client: MagicMock, mock_user_api_key_auth: None
) -> None:
    appearing: Final = LiteLLM_EndUserTable(user_id="created-after-lock", alias="Original", blocked=False)
    database: Final = _CustomerBatchDatabase(())
    database.attach(mock_prisma_client)

    async def lock_missing_customer(_query: str, _user_id: str) -> tuple[Mapping[str, object], ...]:
        database.customers = MappingProxyType({appearing.user_id: appearing})
        return ()

    @asynccontextmanager
    async def transaction() -> AsyncIterator[MagicMock]:
        yield mock_prisma_client.db

    mock_prisma_client.configure_mock(
        **{
            "tx": MagicMock(side_effect=transaction),
            "db.query_raw": AsyncMock(side_effect=lock_missing_customer),
        }
    )
    with _end_user_cache_doubles() as (recording_cache, mock_publish):
        response: Final = client.post(
            "/customer/update",
            json={"user_id": appearing.user_id, "alias": "Updated"},
            headers={"Authorization": "Bearer test-key"},
        )

    assert response.status_code == 404, response.text
    assert response.json()["error"]["type"] == "not_found"
    assert database.customers == {appearing.user_id: appearing}
    mock_prisma_client.db.litellm_endusertable.find_first.assert_not_awaited()
    mock_prisma_client.db.litellm_endusertable.update.assert_not_awaited()
    mock_prisma_client.db.litellm_objectpermissiontable.upsert.assert_not_awaited()
    mock_prisma_client.db.litellm_budgettable.create.assert_not_awaited()
    mock_prisma_client.db.litellm_budgettable.update.assert_not_awaited()
    assert recording_cache.deleted == []
    assert _published_keys(mock_publish) == []


class _ConcurrentCustomerTransaction:
    def __init__(
        self, database: _CustomerBatchDatabase, row_lock: asyncio.Lock, unlocked_reads: asyncio.Barrier
    ) -> None:
        self.database = database
        self.row_lock = row_lock
        self.unlocked_reads = unlocked_reads
        self.has_lock = False
        self.litellm_endusertable = SimpleNamespace(
            find_first=AsyncMock(side_effect=self.find_customer),
            update=AsyncMock(side_effect=database.update_customer),
        )
        self.litellm_objectpermissiontable = SimpleNamespace(
            find_unique=AsyncMock(side_effect=database.find_permission),
            upsert=AsyncMock(side_effect=database.upsert_permission),
        )
        self.litellm_mcpservertable = SimpleNamespace(find_many=AsyncMock(return_value=[]))
        self.litellm_budgettable = SimpleNamespace(
            create=AsyncMock(side_effect=database.create_budget),
            update=AsyncMock(side_effect=database.update_budget),
        )

    async def query_raw(self, query: str, user_id: object) -> tuple[Mapping[str, object], ...]:
        if query.rstrip().upper().endswith("FOR UPDATE"):
            await self.row_lock.acquire()
            self.has_lock = True
        return ({"user_id": user_id},) if user_id in self.database.customers else ()

    async def find_customer(
        self, *, where: Mapping[str, str], include: Mapping[str, bool]
    ) -> LiteLLM_EndUserTable | None:
        snapshot: Final = await self.database.find_customer(where=where, include=include)
        if not self.has_lock:
            await self.unlocked_reads.wait()
        return snapshot


@pytest.mark.asyncio
@pytest.mark.parametrize("toolsets_first", (False, True))
async def test_concurrent_customer_updates_share_new_permission_without_losing_unsent_grants(
    mock_prisma_client: MagicMock, toolsets_first: bool
) -> None:
    from litellm.proxy._types import UpdateCustomerRequest

    original: Final = LiteLLM_EndUserTable(user_id="customer-id", blocked=True, models=["existing-model"])
    database: Final = _CustomerBatchDatabase((original,))
    row_lock: Final = asyncio.Lock()
    unlocked_reads: Final = asyncio.Barrier(2)

    @asynccontextmanager
    async def transaction() -> AsyncIterator[_ConcurrentCustomerTransaction]:
        tx: Final = _ConcurrentCustomerTransaction(database, row_lock, unlocked_reads)
        try:
            yield tx
        finally:
            if tx.has_lock:
                row_lock.release()

    mock_prisma_client.configure_mock(
        **{
            "db": _ConcurrentCustomerTransaction(database, row_lock, unlocked_reads),
            "tx": MagicMock(side_effect=transaction),
        }
    )
    server_patch: Final = UpdateCustomerRequest(
        user_id=original.user_id, object_permission=LiteLLM_ObjectPermissionBase(mcp_servers=["new-server"])
    )
    toolset_patch: Final = UpdateCustomerRequest(
        user_id=original.user_id, object_permission=LiteLLM_ObjectPermissionBase(mcp_toolsets=["new-toolset"])
    )
    patches: Final = (toolset_patch, server_patch) if toolsets_first else (server_patch, toolset_patch)
    auth: Final = UserAPIKeyAuth(user_id="admin", user_role=LitellmUserRoles.PROXY_ADMIN)

    results: Final = await asyncio.gather(
        *(_update_single_customer(data=customer, user_api_key_dict=auth) for customer in patches)
    )

    assert all(isinstance(result, CustomerResponse) for result in results)
    permission_id: Final = database.customers[original.user_id].object_permission_id
    assert permission_id is not None
    expected_permission: Final = LiteLLM_ObjectPermissionTable(
        object_permission_id=permission_id, mcp_servers=["new-server"], mcp_toolsets=["new-toolset"]
    )
    assert database.permissions == {permission_id: expected_permission}
    assert database.customers == {
        original.user_id: original.model_copy(
            update={"object_permission_id": permission_id, "object_permission": expected_permission}
        )
    }


@pytest.mark.parametrize("route", ("/customer/bulk_update", "/end_user/bulk_update"))
def test_customer_bulk_update_applies_each_customers_fields(
    mock_prisma_client: MagicMock, mock_user_api_key_auth: None, route: str
) -> None:
    database: Final = _CustomerBatchDatabase(
        (
            LiteLLM_EndUserTable(user_id="bulk-a", alias="Original A", blocked=True, models=["model-a"]),
            LiteLLM_EndUserTable(user_id="bulk-b", alias="Original B", blocked=False, models=["model-b"]),
        )
    )
    database.attach(mock_prisma_client)

    response: Final = client.post(
        route,
        json={
            "customers": [
                {"user_id": "bulk-a", "alias": "Updated A", "models": ["replacement-model"]},
                {"user_id": "bulk-b", "blocked": True, "models": []},
            ]
        },
        headers={"Authorization": "Bearer test-key"},
    )

    assert response.status_code == 200, response.text
    assert TypeAdapter(dict[str, object]).validate_json(response.content) == {
        "total_requested": 2,
        "successful_updates": 2,
        "failed_updates": 0,
        "results": [
            {
                "user_id": "bulk-a",
                "success": True,
                "updated_customer": CustomerResponse(
                    user_id="bulk-a", alias="Updated A", blocked=True, models=["replacement-model"]
                ).model_dump(mode="json"),
                "error": None,
            },
            {
                "user_id": "bulk-b",
                "success": True,
                "updated_customer": CustomerResponse(user_id="bulk-b", alias="Original B", blocked=True).model_dump(
                    mode="json"
                ),
                "error": None,
            },
        ],
    }
    assert database.customers["bulk-a"].alias == "Updated A"
    assert database.customers["bulk-a"].models == ["replacement-model"]
    assert database.customers["bulk-b"].models == []


@pytest.mark.parametrize("route", ("/customer/bulk_update", "/end_user/bulk_update"))
def test_customer_bulk_update_reports_unknown_middle_item_and_updates_later_customer(
    mock_prisma_client: MagicMock, mock_user_api_key_auth: None, route: str
) -> None:
    database: Final = _CustomerBatchDatabase(
        (
            LiteLLM_EndUserTable(user_id="bulk-a", blocked=False),
            LiteLLM_EndUserTable(user_id="bulk-b", blocked=False),
        )
    )
    database.attach(mock_prisma_client)

    response: Final = client.post(
        route,
        json={
            "customers": [
                {"user_id": "bulk-a", "alias": "Updated A"},
                {"user_id": "bulk-missing", "alias": "Unknown"},
                {"user_id": "bulk-b", "alias": "Updated B"},
            ]
        },
        headers={"Authorization": "Bearer test-key"},
    )

    assert response.status_code == 200, response.text
    assert TypeAdapter(dict[str, object]).validate_json(response.content) == {
        "total_requested": 3,
        "successful_updates": 2,
        "failed_updates": 1,
        "results": [
            {
                "user_id": "bulk-a",
                "success": True,
                "updated_customer": CustomerResponse(user_id="bulk-a", alias="Updated A", blocked=False).model_dump(
                    mode="json"
                ),
                "error": None,
            },
            {
                "user_id": "bulk-missing",
                "success": False,
                "updated_customer": None,
                "error": "End User Id=bulk-missing does not exist in db",
            },
            {
                "user_id": "bulk-b",
                "success": True,
                "updated_customer": CustomerResponse(user_id="bulk-b", alias="Updated B", blocked=False).model_dump(
                    mode="json"
                ),
                "error": None,
            },
        ],
    }
    assert database.customers["bulk-a"].alias == "Updated A"
    assert database.customers["bulk-b"].alias == "Updated B"
    assert "bulk-missing" not in database.customers


@pytest.mark.parametrize("route", ("/customer/bulk_update", "/end_user/bulk_update"))
@pytest.mark.parametrize("budget_edit", ("link", "create", "update"))
def test_customer_bulk_update_rolls_back_failed_item_permissions_and_budgets(
    mock_prisma_client: MagicMock,
    mock_user_api_key_auth: None,
    route: str,
    budget_edit: Literal["link", "create", "update"],
) -> None:
    permissions: Final = tuple(
        LiteLLM_ObjectPermissionTable(
            object_permission_id=f"permission-{suffix}",
            mcp_servers=[f"server-{suffix}"],
            mcp_toolsets=[f"old-toolset-{suffix}"],
        )
        for suffix in ("a", "b", "c")
    )
    original_budget: Final = LiteLLM_BudgetTable(budget_id="budget-b", max_budget=10)
    budgets: Final = (original_budget,) if budget_edit == "update" else ()
    customers: Final = tuple(
        LiteLLM_EndUserTable(
            user_id=f"bulk-{suffix}",
            blocked=False,
            alias=f"Original {suffix}",
            object_permission_id=f"permission-{suffix}",
            budget_id=original_budget.budget_id if suffix == "b" and budgets else None,
            litellm_budget_table=original_budget if suffix == "b" and budgets else None,
        )
        for suffix in ("a", "b", "c")
    )
    database: Final = _CustomerBatchDatabase(
        customers,
        permissions,
        budgets,
        failed_customer_ids=frozenset() if budget_edit == "link" else frozenset({"bulk-b"}),
    )
    database.attach(mock_prisma_client)
    failed_budget_patch: Final = {"budget_id": "missing-budget"} if budget_edit == "link" else {"max_budget": 25}
    cache: Final = UserApiKeyCache()
    for customer in customers:
        cache.set_cache(key=end_user_cache_key(customer.user_id), value=f"cached-{customer.user_id}")

    with patch.multiple("litellm.proxy.proxy_server", user_api_key_cache=cache, redis_usage_cache=None):
        response: Final = client.post(
            route,
            json={
                "customers": [
                    {"user_id": "bulk-a", "object_permission": {"mcp_toolsets": ["new-toolset"]}},
                    {
                        "user_id": "bulk-b",
                        "object_permission": {"mcp_toolsets": ["new-toolset"]},
                        **failed_budget_patch,
                    },
                    {"user_id": "bulk-c", "object_permission": {"mcp_toolsets": ["new-toolset"]}},
                ]
            },
            headers={"Authorization": "Bearer test-key"},
        )

    assert response.status_code == 200, response.text
    body: Final = TypeAdapter(dict[str, object]).validate_json(response.content)
    assert (body["total_requested"], body["successful_updates"], body["failed_updates"]) == (3, 2, 1)
    results: Final = TypeAdapter(tuple[dict[str, object], ...]).validate_python(body["results"])
    assert tuple((result["user_id"], result["success"]) for result in results) == (
        ("bulk-a", True),
        ("bulk-b", False),
        ("bulk-c", True),
    )
    assert results[1]["error"] == ("Budget does not exist" if budget_edit == "link" else "Customer write rejected")
    assert database.customers["bulk-b"] == customers[1]
    assert database.permissions == {
        "permission-a": permissions[0].model_copy(update={"mcp_toolsets": ["new-toolset"]}),
        "permission-b": permissions[1],
        "permission-c": permissions[2].model_copy(update={"mcp_toolsets": ["new-toolset"]}),
    }
    assert database.budgets == {budget.budget_id: budget for budget in budgets}
    assert cache.get_cache(key=end_user_cache_key("bulk-a")) is None
    assert cache.get_cache(key=end_user_cache_key("bulk-b")) == "cached-bulk-b"
    assert cache.get_cache(key=end_user_cache_key("bulk-c")) is None


@pytest.mark.parametrize("route", ("/customer/update", "/end_user/update"))
def test_customer_update_rolls_back_created_permission_and_budget_after_customer_write_failure(
    mock_prisma_client: MagicMock, mock_user_api_key_auth: None, route: str
) -> None:
    original: Final = LiteLLM_EndUserTable(user_id="customer-id", blocked=True, models=["old-model"])
    database: Final = _CustomerBatchDatabase((original,), failed_customer_ids=frozenset({original.user_id}))
    database.attach(mock_prisma_client)

    response: Final = client.post(
        route,
        json={
            "user_id": original.user_id,
            "max_budget": 25,
            "object_permission": {"mcp_toolsets": ["new-toolset"]},
        },
        headers={"Authorization": "Bearer test-key"},
    )

    assert response.status_code == 500, response.text
    assert response.json()["error"]["message"] == "Customer write rejected"
    assert database.customers == {original.user_id: original}
    assert database.permissions == {}
    assert database.budgets == {}


@pytest.mark.parametrize("route", ("/customer/update", "/end_user/update"))
def test_customer_update_rolls_back_writes_when_saved_response_cannot_be_validated(
    mock_prisma_client: MagicMock, mock_user_api_key_auth: None, route: str
) -> None:
    permission: Final = LiteLLM_ObjectPermissionTable(
        object_permission_id="permission-id", mcp_toolsets=["old-toolset"]
    )
    original: Final = LiteLLM_EndUserTable(
        user_id="customer-id", blocked=True, models=["old-model"], object_permission_id=permission.object_permission_id
    )
    database: Final = _CustomerBatchDatabase((original,), (permission,))
    database.attach(mock_prisma_client)

    async def save_invalid_response(
        *, where: Mapping[str, str], data: Mapping[str, object], include: Mapping[str, bool]
    ) -> PrismaEndUserRow:
        saved: Final = await database.update_customer(where=where, data=data, include=include)
        saved_permission: Final = database.permissions[permission.object_permission_id]
        return PrismaEndUserRow.model_validate(
            {
                **saved.model_dump(),
                "object_permission": {
                    **saved_permission.model_dump(),
                    "mcp_tool_permissions": '{"server-id": 7}',
                },
            }
        )

    mock_prisma_client.configure_mock(
        **{"db.litellm_endusertable.update": AsyncMock(side_effect=save_invalid_response)}
    )

    response: Final = client.post(
        route,
        json={
            "user_id": original.user_id,
            "models": ["new-model"],
            "object_permission": {"mcp_toolsets": ["new-toolset"]},
        },
        headers={"Authorization": "Bearer test-key"},
    )

    assert response.status_code == 500, response.text
    assert database.customers == {original.user_id: original}
    assert database.permissions == {permission.object_permission_id: permission}


@pytest.mark.parametrize("route", ("/customer/bulk_update", "/end_user/bulk_update"))
def test_customer_bulk_update_toolsets_only_preserves_each_customers_other_permissions(
    mock_prisma_client: MagicMock, mock_user_api_key_auth: None, route: str
) -> None:
    permissions: Final = tuple(
        LiteLLM_ObjectPermissionTable(
            object_permission_id=f"permission-{suffix}",
            mcp_servers=[f"server-{suffix}"],
            mcp_access_groups=[f"group-{suffix}"],
            mcp_tool_permissions={f"server-{suffix}": [f"read-{suffix}"]},
            vector_stores=[f"vector-store-{suffix}"],
            agents=[f"agent-{suffix}"],
            agent_access_groups=[f"agent-group-{suffix}"],
            models=[f"permission-model-{suffix}"],
            mcp_toolsets=[f"old-toolset-{suffix}"],
            blocked_tools=[f"blocked-tool-{suffix}"],
            search_tools=[f"search-{suffix}"],
            mcp_tool_search_enabled=suffix == "a",
            skills=[f"skill-{suffix}"],
        )
        for suffix in ("a", "b")
    )
    database: Final = _CustomerBatchDatabase(
        tuple(
            LiteLLM_EndUserTable(user_id=f"bulk-{suffix}", blocked=False, object_permission_id=f"permission-{suffix}")
            for suffix in ("a", "b")
        ),
        permissions,
    )
    database.attach(mock_prisma_client)

    response: Final = client.post(
        route,
        json={
            "customers": [
                {"user_id": "bulk-a", "object_permission": {"mcp_toolsets": ["new-toolset"]}},
                {"user_id": "bulk-b", "object_permission": {"mcp_toolsets": ["new-toolset"]}},
            ]
        },
        headers={"Authorization": "Bearer test-key"},
    )

    assert response.status_code == 200, response.text
    body: Final = TypeAdapter(dict[str, object]).validate_json(response.content)
    assert (body["total_requested"], body["successful_updates"], body["failed_updates"]) == (2, 2, 0)
    results: Final = TypeAdapter(tuple[dict[str, object], ...]).validate_python(body["results"])
    assert len(results) == 2
    for original, result in zip(permissions, results, strict=True):
        expected: Final = original.model_copy(update={"mcp_toolsets": ["new-toolset"]})
        customer: Final = CustomerResponse.model_validate(result["updated_customer"])
        assert result["success"] is True
        assert result["error"] is None
        assert customer.object_permission == expected
        assert database.permissions[original.object_permission_id] == expected
        assert database.customers[customer.user_id].object_permission == expected


@pytest.mark.parametrize("route", ("/customer/bulk_update", "/end_user/bulk_update"))
@pytest.mark.parametrize("role", (LitellmUserRoles.INTERNAL_USER, LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY))
def test_customer_bulk_update_rejects_non_admins_without_writing(
    mock_prisma_client: MagicMock, route: str, role: LitellmUserRoles
) -> None:
    original: Final = LiteLLM_EndUserTable(user_id="bulk-a", alias="Original", blocked=False)
    database: Final = _CustomerBatchDatabase((original,))
    database.attach(mock_prisma_client)
    payload: Final[dict[str, object]] = {"customers": [{"user_id": "bulk-a", "alias": "Updated"}]}

    def authorize_role(request: Request) -> UserAPIKeyAuth:
        auth: Final = UserAPIKeyAuth(user_id="bulk-non-admin", user_role=role)
        try:
            RouteChecks.non_proxy_admin_allowed_routes_check(
                user_obj=LiteLLM_UserTable(user_id="bulk-non-admin", user_role=role),
                _user_role=role,
                route=request.url.path,
                request=request,
                valid_token=auth,
                request_data=payload,
            )
        except HTTPException:
            raise
        except Exception as error:
            raise HTTPException(status_code=403, detail=str(error)) from error
        return auth

    with patch.dict(app.dependency_overrides, {user_api_key_auth: authorize_role}):
        response: Final = client.post(route, json=payload, headers={"Authorization": "Bearer test-key"})

    assert response.status_code == 403, response.text
    assert route in response.text
    assert database.customers == {"bulk-a": original}
    mock_prisma_client.db.litellm_endusertable.update.assert_not_awaited()
    mock_prisma_client.db.litellm_objectpermissiontable.upsert.assert_not_awaited()


@pytest.mark.parametrize("route", ("/customer/bulk_update", "/end_user/bulk_update"))
def test_customer_bulk_update_evicts_every_updated_id_and_registry_only(
    mock_prisma_client: MagicMock, mock_user_api_key_auth: None, route: str
) -> None:
    database: Final = _CustomerBatchDatabase(
        (
            LiteLLM_EndUserTable(user_id="bulk-a", blocked=False),
            LiteLLM_EndUserTable(user_id="bulk-b", blocked=False),
            LiteLLM_EndUserTable(user_id="bulk-untouched", blocked=False),
        )
    )
    database.attach(mock_prisma_client)
    cache: Final = UserApiKeyCache()
    for user_id in ("bulk-a", "bulk-b", "bulk-missing", "bulk-untouched"):
        cache.set_cache(key=end_user_cache_key(user_id), value=f"cached-{user_id}")
    cache.set_cache(key=end_user_restricted_registry_cache_key(), value="cached-registry")

    with patch.multiple("litellm.proxy.proxy_server", user_api_key_cache=cache, redis_usage_cache=None):
        response: Final = client.post(
            route,
            json={
                "customers": [
                    {"user_id": "bulk-a", "alias": "Updated A"},
                    {"user_id": "bulk-missing", "alias": "Unknown"},
                    {"user_id": "bulk-b", "alias": "Updated B"},
                ]
            },
            headers={"Authorization": "Bearer test-key"},
        )

    assert response.status_code == 200, response.text
    body: Final = TypeAdapter(dict[str, object]).validate_json(response.content)
    assert (body["successful_updates"], body["failed_updates"]) == (2, 1)
    assert cache.get_cache(key=end_user_cache_key("bulk-a")) is None
    assert cache.get_cache(key=end_user_cache_key("bulk-b")) is None
    assert cache.get_cache(key=end_user_restricted_registry_cache_key()) is None
    assert cache.get_cache(key=end_user_cache_key("bulk-missing")) == "cached-bulk-missing"
    assert cache.get_cache(key=end_user_cache_key("bulk-untouched")) == "cached-bulk-untouched"


@pytest.mark.parametrize("route", ("/customer/bulk_update", "/end_user/bulk_update"))
@pytest.mark.parametrize(
    "customers",
    ([], [{"user_id": "bulk-a"}] * 501, [{"user_id": "bulk-a"}, {"alias": "Missing ID"}]),
    ids=("empty", "over-limit", "malformed-item"),
)
def test_customer_bulk_update_rejects_invalid_batches_before_writing(
    mock_prisma_client: MagicMock, mock_user_api_key_auth: None, route: str, customers: list[dict[str, object]]
) -> None:
    original: Final = LiteLLM_EndUserTable(user_id="bulk-a", alias="Original", blocked=False)
    database: Final = _CustomerBatchDatabase((original,))
    database.attach(mock_prisma_client)

    response: Final = client.post(
        route,
        json={"customers": customers},
        headers={"Authorization": "Bearer test-key"},
    )

    assert response.status_code == 422, response.text
    assert database.customers == {"bulk-a": original}
    mock_prisma_client.db.litellm_endusertable.find_first.assert_not_awaited()
    mock_prisma_client.db.litellm_endusertable.update.assert_not_awaited()


@pytest.mark.parametrize("route", ("/customer/bulk_update", "/end_user/bulk_update"))
@pytest.mark.parametrize("role", (LitellmUserRoles.INTERNAL_USER, LitellmUserRoles.ORG_ADMIN))
def test_customer_bulk_update_rejects_org_admin_context_without_writing(
    mock_prisma_client: MagicMock, route: str, role: LitellmUserRoles
) -> None:
    original: Final = LiteLLM_EndUserTable(user_id="bulk-a", alias="Original", blocked=False)
    database: Final = _CustomerBatchDatabase((original,))
    database.attach(mock_prisma_client)
    payload: Final[dict[str, object]] = {
        "organization_id": "bulk-organization",
        "customers": [{"user_id": "bulk-a", "alias": "Updated"}],
    }
    user: Final = LiteLLM_UserTable.model_validate(
        {
            "user_id": "bulk-org-admin",
            "user_role": role,
            "organization_memberships": [
                {
                    "user_id": "bulk-org-admin",
                    "organization_id": "bulk-organization",
                    "user_role": LitellmUserRoles.ORG_ADMIN.value,
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "updated_at": "2026-01-01T00:00:00+00:00",
                }
            ],
        }
    )

    def authorize_role(request: Request) -> UserAPIKeyAuth:
        auth: Final = UserAPIKeyAuth(user_id=user.user_id, user_role=role)
        try:
            RouteChecks.non_proxy_admin_allowed_routes_check(
                user_obj=user,
                _user_role=role,
                route=request.url.path,
                request=request,
                valid_token=auth,
                request_data=payload,
            )
        except HTTPException:
            raise
        except Exception as error:
            raise HTTPException(status_code=403, detail=str(error)) from error
        return auth

    with patch.dict(app.dependency_overrides, {user_api_key_auth: authorize_role}):
        response: Final = client.post(route, json=payload, headers={"Authorization": "Bearer test-key"})

    assert response.status_code == 403, response.text
    assert route in response.text
    assert database.customers == {"bulk-a": original}
    mock_prisma_client.db.litellm_endusertable.update.assert_not_awaited()
    mock_prisma_client.db.litellm_objectpermissiontable.upsert.assert_not_awaited()


_SHARED_BUDGET_REFERENCES: Final = (
    "customer",
    "LiteLLM_VerificationToken",
    "LiteLLM_OrganizationTable",
    "LiteLLM_ProjectTable",
    "LiteLLM_TagTable",
    "LiteLLM_ModelAccessGroupBudgetTable",
    "LiteLLM_TeamMembership",
    "LiteLLM_OrganizationMembership",
    "key_default",
    "team_default",
    "proxy_default",
)


@pytest.mark.parametrize(
    "route", ("/customer/update", "/end_user/update", "/customer/bulk_update", "/end_user/bulk_update")
)
@pytest.mark.parametrize("reference", _SHARED_BUDGET_REFERENCES)
def test_customer_budget_update_rejects_shared_references_without_partial_writes(
    mock_prisma_client: MagicMock, mock_user_api_key_auth: None, route: str, reference: str
) -> None:
    budget: Final = LiteLLM_BudgetTable(budget_id="shared-budget", max_budget=50)
    permission: Final = LiteLLM_ObjectPermissionTable(object_permission_id="shared-permission", mcp_toolsets=["old"])
    original: Final = LiteLLM_EndUserTable(
        user_id="shared-customer",
        blocked=False,
        alias="Original",
        budget_id=budget.budget_id,
        litellm_budget_table=budget,
        object_permission_id=permission.object_permission_id,
    )
    other: Final = LiteLLM_EndUserTable(
        user_id="other-customer", blocked=False, budget_id=budget.budget_id if reference == "customer" else None
    )
    references: Final = () if reference in {"customer", "proxy_default"} else ((reference, "shared-budget"),)
    database: Final = _CustomerBatchDatabase((original, other), (permission,), (budget,), budget_references=references)
    database.attach(mock_prisma_client)
    update: Final = {
        "user_id": original.user_id,
        "max_budget": 100,
        "alias": "Changed",
        "object_permission": {"mcp_toolsets": ["new"]},
    }
    bulk: Final = route.endswith("bulk_update")
    payload: Final = {"customers": [update, {"user_id": other.user_id, "alias": "Independent"}]} if bulk else update
    with patch("litellm.max_end_user_budget_id", "shared-budget" if reference == "proxy_default" else None):
        response: Final = client.post(route, json=payload, headers={"Authorization": "Bearer test-key"})

    expected_error: Final = (
        "Cannot update max_budget for a shared budget. Update the budget directly or assign a different budget."
    )
    assert response.status_code == (200 if bulk else 400), response.text
    if bulk:
        body: Final = response.json()
        assert (body["total_requested"], body["successful_updates"], body["failed_updates"]) == (2, 1, 1)
        assert body["results"][0] == {
            "user_id": original.user_id,
            "success": False,
            "updated_customer": None,
            "error": expected_error,
        }
        assert body["results"][1]["success"] is True
        assert database.customers[other.user_id].alias == "Independent"
    else:
        assert response.json()["error"]["message"] == expected_error
        assert database.customers[other.user_id] == other
    assert database.customers[original.user_id] == original
    assert database.permissions == {permission.object_permission_id: permission}
    assert database.budgets == {budget.budget_id: budget}
    assert database.locked_budget_ids == {budget.budget_id}


@pytest.mark.parametrize(
    "route", ("/customer/update", "/end_user/update", "/customer/bulk_update", "/end_user/bulk_update")
)
def test_customer_budget_update_allows_own_budget_with_unrelated_shared_references(
    mock_prisma_client: MagicMock, mock_user_api_key_auth: None, route: str
) -> None:
    own_budget: Final = LiteLLM_BudgetTable(budget_id="own-budget", max_budget=50)
    other_budget: Final = LiteLLM_BudgetTable(budget_id="other-budget", max_budget=75)
    own_customer: Final = LiteLLM_EndUserTable(
        user_id="own-customer",
        blocked=False,
        budget_id=own_budget.budget_id,
        litellm_budget_table=own_budget,
    )
    other: Final = LiteLLM_EndUserTable(user_id="other-customer", blocked=False, budget_id=other_budget.budget_id)
    references: Final = tuple(
        (reference, "other-budget")
        for reference in _SHARED_BUDGET_REFERENCES
        if reference not in {"customer", "proxy_default"}
    )
    database: Final = _CustomerBatchDatabase(
        (own_customer, other), budgets=(own_budget, other_budget), budget_references=references
    )
    database.attach(mock_prisma_client)
    update: Final = {"user_id": own_customer.user_id, "max_budget": 100}
    payload: Final = {"customers": [update]} if route.endswith("bulk_update") else update
    with patch("litellm.max_end_user_budget_id", "other-budget"):
        response: Final = client.post(route, json=payload, headers={"Authorization": "Bearer test-key"})

    assert response.status_code == 200, response.text
    assert database.budgets[own_budget.budget_id].max_budget == 100
    assert database.budgets[other_budget.budget_id] == other_budget
    assert database.customers[own_customer.user_id].budget_id == own_budget.budget_id
    assert database.locked_budget_ids == {own_budget.budget_id}


@pytest.mark.parametrize(
    "route", ("/customer/update", "/end_user/update", "/customer/bulk_update", "/end_user/bulk_update")
)
def test_customer_budget_update_rejects_missing_linked_budget_without_writing(
    mock_prisma_client: MagicMock, mock_user_api_key_auth: None, route: str
) -> None:
    original: Final = LiteLLM_EndUserTable(user_id="customer", alias="Original", blocked=False, budget_id="gone")
    database: Final = _CustomerBatchDatabase((original,))
    database.attach(mock_prisma_client)
    update: Final = {"user_id": original.user_id, "max_budget": 100, "alias": "Changed"}
    bulk: Final = route.endswith("bulk_update")
    payload: Final = {"customers": [update]} if bulk else update
    response: Final = client.post(route, json=payload, headers={"Authorization": "Bearer test-key"})

    assert response.status_code == (200 if bulk else 404), response.text
    if bulk:
        assert response.json()["results"] == [
            {"user_id": original.user_id, "success": False, "updated_customer": None, "error": "Budget does not exist"}
        ]
    else:
        assert response.json()["error"]["message"] == "Budget does not exist"
    assert database.customers == {original.user_id: original}
    assert database.budgets == {}
    assert database.locked_budget_ids == {"gone"}


@pytest.mark.parametrize(
    "route", ("/customer/update", "/end_user/update", "/customer/bulk_update", "/end_user/bulk_update")
)
@pytest.mark.parametrize("max_budget", (None, 100))
@pytest.mark.parametrize("budget_id", (None, "other-budget"))
def test_customer_update_rejects_both_budget_fields_before_writing(
    mock_prisma_client: MagicMock,
    mock_user_api_key_auth: None,
    route: str,
    max_budget: int | None,
    budget_id: str | None,
) -> None:
    original: Final = LiteLLM_EndUserTable(user_id="customer", alias="Original", blocked=False)
    database: Final = _CustomerBatchDatabase((original,))
    database.attach(mock_prisma_client)
    invalid: Final = {"user_id": original.user_id, "max_budget": max_budget, "budget_id": budget_id}
    payload: Final = (
        {"customers": [{"user_id": original.user_id, "alias": "Changed"}, invalid]}
        if route.endswith("bulk_update")
        else invalid
    )
    response: Final = client.post(route, json=payload, headers={"Authorization": "Bearer test-key"})

    assert response.status_code == 422, response.text
    assert "Set either 'max_budget' or 'budget_id', not both." in response.text
    assert database.customers == {original.user_id: original}
    assert database.budgets == {}
    mock_prisma_client.db.query_raw.assert_not_awaited()
    mock_prisma_client.db.litellm_endusertable.update.assert_not_awaited()
