# tests/test_budget_endpoints.py

import os
import sys
import types
import pytest
from unittest.mock import AsyncMock, MagicMock
from fastapi.testclient import TestClient

import litellm.proxy.proxy_server as ps
from litellm.proxy.proxy_server import app
from litellm.proxy._types import UserAPIKeyAuth, LitellmUserRoles, CommonProxyErrors


sys.path.insert(
    0, os.path.abspath("../../../")
)  # Adds the parent directory to the system path


@pytest.fixture
def admin_client_and_mocks(monkeypatch):
    mock_prisma = MagicMock()
    mock_table = MagicMock()
    mock_table.create = AsyncMock(side_effect=lambda *, data: data)
    mock_table.update = AsyncMock(side_effect=lambda *, where, data: {**where, **data})
    mock_table.delete = AsyncMock(side_effect=lambda *, where: where)
    mock_table.find_many = AsyncMock(return_value=[])
    mock_table.find_first = AsyncMock(return_value=None)

    mock_prisma.db = types.SimpleNamespace(
        litellm_budgettable=mock_table,
        litellm_dailyspend=mock_table,
    )

    monkeypatch.setattr(ps, "prisma_client", mock_prisma)

    fake_user = UserAPIKeyAuth(
        user_id="admin_user",
        user_role=LitellmUserRoles.PROXY_ADMIN,
    )
    app.dependency_overrides[ps.user_api_key_auth] = lambda: fake_user

    client = TestClient(app)

    yield client, mock_prisma, mock_table

    app.dependency_overrides.clear()


@pytest.fixture
def client_and_mocks(monkeypatch):
    # Setup MagicMock Prisma
    mock_prisma = MagicMock()
    mock_table = MagicMock()
    mock_table.create = AsyncMock(side_effect=lambda *, data: data)
    mock_table.update = AsyncMock(side_effect=lambda *, where, data: {**where, **data})

    mock_prisma.db = types.SimpleNamespace(
        litellm_budgettable=mock_table,
        litellm_dailyspend=mock_table,
    )

    # Monkeypatch Mocked Prisma client into the server module
    monkeypatch.setattr(ps, "prisma_client", mock_prisma)

    # override returned auth user
    fake_user = UserAPIKeyAuth(
        user_id="test_user",
        user_role=LitellmUserRoles.INTERNAL_USER,
    )
    app.dependency_overrides[ps.user_api_key_auth] = lambda: fake_user

    client = TestClient(app)

    yield client, mock_prisma, mock_table

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_new_budget_success(client_and_mocks):
    client, _, mock_table = client_and_mocks

    # Call /budget/new endpoint
    payload = {
        "budget_id": "budget_123",
        "max_budget": 42.0,
        "budget_duration": "30d",
    }
    resp = client.post("/budget/new", json=payload)
    assert resp.status_code == 200, resp.text

    body = resp.json()
    assert body["budget_id"] == payload["budget_id"]
    assert body["max_budget"] == payload["max_budget"]
    assert body["budget_duration"] == payload["budget_duration"]
    assert body["created_by"] == "test_user"
    assert body["updated_by"] == "test_user"

    mock_table.create.assert_awaited_once()


@pytest.mark.asyncio
async def test_new_budget_db_not_connected(client_and_mocks, monkeypatch):
    client, mock_prisma, mock_table = client_and_mocks

    # override the prisma_client that the handler imports at runtime
    import litellm.proxy.proxy_server as ps

    monkeypatch.setattr(ps, "prisma_client", None)

    # Call /budget/new endpoint
    resp = client.post("/budget/new", json={"budget_id": "no_db", "max_budget": 1.0})
    assert resp.status_code == 500
    detail = resp.json()["detail"]
    assert detail["error"] == CommonProxyErrors.db_not_connected_error.value


@pytest.mark.asyncio
async def test_update_budget_success(client_and_mocks, monkeypatch):
    client, mock_prisma, mock_table = client_and_mocks

    payload = {
        "budget_id": "budget_456",
        "max_budget": 99.0,
        "soft_budget": 50.0,
    }
    resp = client.post("/budget/update", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["budget_id"] == payload["budget_id"]
    assert body["max_budget"] == payload["max_budget"]
    assert body["soft_budget"] == payload["soft_budget"]
    assert body["updated_by"] == "test_user"


@pytest.mark.asyncio
async def test_update_budget_missing_id(client_and_mocks, monkeypatch):
    client, mock_prisma, mock_table = client_and_mocks

    payload = {"max_budget": 10.0}
    resp = client.post("/budget/update", json=payload)
    assert resp.status_code == 400, resp.text
    detail = resp.json()["detail"]
    assert detail["error"] == "budget_id is required"


@pytest.mark.asyncio
async def test_update_budget_db_not_connected(client_and_mocks, monkeypatch):
    client, mock_prisma, mock_table = client_and_mocks

    # override the prisma_client that the handler imports at runtime
    import litellm.proxy.proxy_server as ps

    monkeypatch.setattr(ps, "prisma_client", None)

    payload = {"budget_id": "any", "max_budget": 1.0}
    resp = client.post("/budget/update", json=payload)
    assert resp.status_code == 500
    detail = resp.json()["detail"]
    assert detail["error"] == CommonProxyErrors.db_not_connected_error.value


@pytest.mark.asyncio
async def test_update_budget_allows_null_max_budget(client_and_mocks):
    """
    Test that /budget/update allows setting max_budget to null.

    Previously, using exclude_none=True would drop null values,
    making it impossible to remove a budget limit. With exclude_unset=True,
    explicitly setting max_budget to null should include it in the update.
    """
    client, _, mock_table = client_and_mocks

    captured_data = {}

    async def capture_update(*, where, data):
        captured_data.update(data)
        return {**where, **data}

    mock_table.update = AsyncMock(side_effect=capture_update)

    payload = {
        "budget_id": "budget_789",
        "max_budget": None,  # Explicitly setting to null to remove budget limit
    }
    resp = client.post("/budget/update", json=payload)
    assert resp.status_code == 200, resp.text

    # Verify that max_budget=None was included in the update data
    assert (
        "max_budget" in captured_data
    ), "max_budget should be included when explicitly set to null"
    assert captured_data["max_budget"] is None, "max_budget should be None"

    mock_table.update.assert_awaited_once()


@pytest.mark.asyncio
async def test_new_budget_negative_max_budget(client_and_mocks):
    """
    Test that /budget/new rejects negative max_budget values.

    This prevents the issue where negative budgets would always trigger
    budget exceeded errors.
    """
    client, _, _ = client_and_mocks

    payload = {
        "budget_id": "budget_negative",
        "max_budget": -7.0,
    }
    resp = client.post("/budget/new", json=payload)
    assert resp.status_code == 400, resp.text

    detail = resp.json()["detail"]
    assert "max_budget must be a non-negative finite number" in str(detail)


@pytest.mark.asyncio
async def test_new_budget_negative_soft_budget(client_and_mocks):
    """
    Test that /budget/new rejects negative soft_budget values.
    """
    client, _, _ = client_and_mocks

    payload = {
        "budget_id": "budget_negative_soft",
        "soft_budget": -10.0,
    }
    resp = client.post("/budget/new", json=payload)
    assert resp.status_code == 400, resp.text

    detail = resp.json()["detail"]
    assert "soft_budget must be a non-negative finite number" in str(detail)


@pytest.mark.asyncio
async def test_update_budget_negative_max_budget(client_and_mocks):
    """
    Test that /budget/update rejects negative max_budget values.
    """
    client, _, _ = client_and_mocks

    payload = {
        "budget_id": "budget_update_negative",
        "max_budget": -5.0,
    }
    resp = client.post("/budget/update", json=payload)
    assert resp.status_code == 400, resp.text

    detail = resp.json()["detail"]
    assert "max_budget must be a non-negative finite number" in str(detail)


@pytest.mark.asyncio
async def test_update_budget_negative_soft_budget(client_and_mocks):
    """
    Test that /budget/update rejects negative soft_budget values.
    """
    client, _, _ = client_and_mocks

    payload = {
        "budget_id": "budget_update_negative_soft",
        "soft_budget": -15.0,
    }
    resp = client.post("/budget/update", json=payload)
    assert resp.status_code == 400, resp.text

    detail = resp.json()["detail"]
    assert "soft_budget must be a non-negative finite number" in str(detail)


@pytest.mark.asyncio
async def test_new_budget_invalid_model_max_budget(client_and_mocks, monkeypatch):
    """
    Test that /budget/new validates model_max_budget and returns 400 for invalid structure.
    Per-model budget implementation: validate_model_max_budget is called in new_budget.
    """
    import litellm.proxy.proxy_server as ps

    monkeypatch.setattr(ps, "premium_user", True)

    client, _, _ = client_and_mocks

    payload = {
        "budget_id": "budget_invalid_mmb",
        "max_budget": 10.0,
        "model_max_budget": {"gpt-4": "not-a-dict"},
    }
    resp = client.post("/budget/new", json=payload)
    # Pydantic may reject invalid structure with 422 before our validator runs
    assert resp.status_code in (400, 422), resp.text
    detail = resp.json()["detail"]
    assert "model_max_budget" in str(detail) or "dictionary" in str(detail).lower()


@pytest.mark.asyncio
async def test_info_budget_success(admin_client_and_mocks):
    client, _, mock_table = admin_client_and_mocks

    mock_table.find_many = AsyncMock(
        return_value=[
            {
                "budget_id": "budget-info-1",
                "max_budget": 10.0,
                "budget_duration": "30d",
            }
        ]
    )

    resp = client.post("/budget/info", json={"budgets": ["budget-info-1"]})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert len(body) == 1
    assert body[0]["budget_id"] == "budget-info-1"
    mock_table.find_many.assert_awaited_once()


@pytest.mark.asyncio
async def test_info_budget_empty_list_rejected(admin_client_and_mocks):
    client, _, _ = admin_client_and_mocks

    resp = client.post("/budget/info", json={"budgets": []})
    assert resp.status_code == 400, resp.text
    assert "Specify list of budget id" in str(resp.json()["detail"])


@pytest.mark.asyncio
async def test_info_budget_db_not_connected(admin_client_and_mocks, monkeypatch):
    client, _, _ = admin_client_and_mocks
    monkeypatch.setattr(ps, "prisma_client", None)

    resp = client.post("/budget/info", json={"budgets": ["budget-info-1"]})
    assert resp.status_code == 500
    assert resp.json()["detail"]["error"] == "No db connected"


@pytest.mark.asyncio
async def test_list_budget_success(admin_client_and_mocks):
    client, _, mock_table = admin_client_and_mocks

    mock_table.find_many = AsyncMock(
        return_value=[
            {"budget_id": "budget-a", "max_budget": 1.0},
            {"budget_id": "budget-b", "max_budget": 2.0},
        ]
    )

    resp = client.get("/budget/list")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert len(body) == 2
    mock_table.find_many.assert_awaited_once()


@pytest.mark.asyncio
async def test_list_budget_rejects_non_admin(client_and_mocks):
    client, _, _ = client_and_mocks

    resp = client.get("/budget/list")
    assert resp.status_code == 400, resp.text
    assert CommonProxyErrors.not_allowed_access.value in str(resp.json()["detail"])


@pytest.mark.asyncio
async def test_budget_settings_success(admin_client_and_mocks):
    client, _, mock_table = admin_client_and_mocks

    mock_row = types.SimpleNamespace(
        model_dump=lambda exclude_none=True: {
            "budget_id": "budget-settings-1",
            "max_budget": 25.0,
            "soft_budget": 20.0,
            "budget_duration": "7d",
        }
    )
    mock_table.find_first = AsyncMock(return_value=mock_row)

    resp = client.get("/budget/settings", params={"budget_id": "budget-settings-1"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    field_names = {item["field_name"] for item in body}
    assert "max_budget" in field_names
    assert "soft_budget" in field_names
    max_budget_field = next(item for item in body if item["field_name"] == "max_budget")
    assert max_budget_field["field_value"] == 25.0


@pytest.mark.asyncio
async def test_budget_settings_rejects_non_admin(client_and_mocks):
    client, _, _ = client_and_mocks

    resp = client.get("/budget/settings", params={"budget_id": "budget-settings-1"})
    assert resp.status_code == 400, resp.text
    assert CommonProxyErrors.not_allowed_access.value in str(resp.json()["detail"])


@pytest.mark.asyncio
async def test_delete_budget_success(admin_client_and_mocks):
    client, _, mock_table = admin_client_and_mocks

    mock_table.delete = AsyncMock(return_value={"budget_id": "budget-delete-1"})

    resp = client.post("/budget/delete", json={"id": "budget-delete-1"})
    assert resp.status_code == 200, resp.text
    assert resp.json()["budget_id"] == "budget-delete-1"
    mock_table.delete.assert_awaited_once_with(
        where={"budget_id": "budget-delete-1"}
    )


@pytest.mark.asyncio
async def test_delete_budget_rejects_non_admin(client_and_mocks):
    client, _, mock_table = client_and_mocks

    fake_viewer = UserAPIKeyAuth(
        user_id="viewer_user",
        user_role=LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY,
    )
    app.dependency_overrides[ps.user_api_key_auth] = lambda: fake_viewer

    try:
        resp = client.post("/budget/delete", json={"id": "budget-delete-1"})
        assert resp.status_code == 400, resp.text
        assert CommonProxyErrors.not_allowed_access.value in str(resp.json()["detail"])
        mock_table.delete.assert_not_called()
    finally:
        app.dependency_overrides.clear()
