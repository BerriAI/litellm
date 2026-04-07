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
def client_and_mocks(monkeypatch):
    # Setup MagicMock Prisma
    mock_prisma = MagicMock()
    mock_table = MagicMock()
    mock_table.create = AsyncMock(side_effect=lambda *, data: data)
    mock_table.update = AsyncMock(side_effect=lambda *, where, data: {**where, **data})
    # Default find_unique: return a budget whose duration differs from "7d" so
    # duration-change tests see a change.  Individual tests override this as needed.
    mock_table.find_unique = AsyncMock(return_value=types.SimpleNamespace(
        budget_duration=None,
        budget_reset_at=None,
    ))

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

    # teardown
    app.dependency_overrides.clear()
    monkeypatch.setattr(ps, "prisma_client", ps.prisma_client)


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
    assert "max_budget cannot be negative" in str(detail)


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
    assert "soft_budget cannot be negative" in str(detail)


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
    assert "max_budget cannot be negative" in str(detail)


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
    assert "soft_budget cannot be negative" in str(detail)


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
async def test_update_budget_duration_recalculates_budget_reset_at(client_and_mocks):
    """
    Regression test: changing budget_duration via /budget/update must also update
    budget_reset_at to reflect the new period.

    Previously, update_budget() stored the new duration but left the old
    budget_reset_at untouched, so the next reset shown in the UI was still in the
    past (the old period's reset time).
    """
    from datetime import datetime, timezone

    client, _, mock_table = client_and_mocks

    captured_data: dict = {}

    async def capture_update(*, where, data):
        captured_data.update(data)
        return {**where, **data}

    mock_table.update = AsyncMock(side_effect=capture_update)

    payload = {
        "budget_id": "budget_duration_change",
        "budget_duration": "7d",
    }
    resp = client.post("/budget/update", json=payload)
    assert resp.status_code == 200, resp.text

    # budget_reset_at must be present and in the future
    assert "budget_reset_at" in captured_data, (
        "budget_reset_at must be recalculated when budget_duration changes"
    )
    reset_at = captured_data["budget_reset_at"]
    if isinstance(reset_at, str):
        reset_at = datetime.fromisoformat(reset_at)
    if reset_at.tzinfo is None:
        reset_at = reset_at.replace(tzinfo=timezone.utc)
    assert reset_at > datetime.now(timezone.utc), (
        "budget_reset_at must be in the future after a duration change"
    )


@pytest.mark.asyncio
async def test_update_budget_duration_respects_explicit_budget_reset_at(client_and_mocks):
    """
    If the caller explicitly provides budget_reset_at alongside budget_duration,
    the supplied value must be used as-is and not overwritten.
    """
    from datetime import datetime, timezone, timedelta

    client, _, mock_table = client_and_mocks

    captured_data: dict = {}

    async def capture_update(*, where, data):
        captured_data.update(data)
        return {**where, **data}

    mock_table.update = AsyncMock(side_effect=capture_update)

    explicit_reset = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()
    payload = {
        "budget_id": "budget_explicit_reset",
        "budget_duration": "7d",
        "budget_reset_at": explicit_reset,
    }
    resp = client.post("/budget/update", json=payload)
    assert resp.status_code == 200, resp.text

    stored = captured_data.get("budget_reset_at")
    if isinstance(stored, str):
        stored = datetime.fromisoformat(stored)
    if isinstance(stored, datetime) and stored.tzinfo is None:
        stored = stored.replace(tzinfo=timezone.utc)

    explicit_dt = datetime.fromisoformat(explicit_reset)
    # Should be within a second of the explicit value
    assert abs((stored - explicit_dt).total_seconds()) < 1, (
        "Explicit budget_reset_at must not be overwritten by the duration recalculation"
    )


@pytest.mark.asyncio
async def test_update_budget_duration_via_internal_call_path(client_and_mocks, monkeypatch):
    """
    Regression test for the internal call path used by upsert_team_member_budget_table.

    When the team member budget duration is changed via the team settings UI, the
    backend constructs a BudgetNewRequest by first calling BudgetNewRequest(budget_id=...)
    and then setting budget_duration via attribute assignment AFTER construction:

        budget_request = BudgetNewRequest(budget_id=some_id)
        budget_request.budget_duration = "7d"   # ← attribute assignment, not __init__
        await update_budget(budget_obj=budget_request, ...)

    In Pydantic v2 without validate_assignment=True, attribute assignment does NOT
    reliably update model_fields_set.  The previous fix checked model_fields_set and
    therefore silently skipped the budget_reset_at recalculation on this path.

    The new fix checks budget_obj.budget_duration directly and writes budget_reset_at
    explicitly into the update dict, bypassing model_fields_set entirely.
    """
    import litellm.proxy.proxy_server as ps
    from datetime import datetime, timezone
    from litellm.proxy._types import BudgetNewRequest, UserAPIKeyAuth, LitellmUserRoles
    from litellm.proxy.management_endpoints.budget_management_endpoints import update_budget

    import types as _types

    _, mock_prisma, mock_table = client_and_mocks

    # find_unique: existing budget has no duration (None → "7d" is a real change)
    mock_table.find_unique = AsyncMock(return_value=_types.SimpleNamespace(
        budget_duration=None,
        budget_reset_at=None,
    ))

    captured_data: dict = {}

    async def capture_update(*, where, data):
        captured_data.update(data)
        return {**where, **data}

    mock_table.update = AsyncMock(side_effect=capture_update)

    fake_user = UserAPIKeyAuth(
        user_id="internal_caller",
        user_role=LitellmUserRoles.PROXY_ADMIN,
    )

    # Simulate exactly what upsert_team_member_budget_table does:
    # construct with budget_id only, then set budget_duration via attribute assignment.
    budget_request = BudgetNewRequest(budget_id="internal-budget-id")
    budget_request.budget_duration = "7d"  # attribute assignment — not via __init__

    await update_budget(budget_obj=budget_request, user_api_key_dict=fake_user)

    assert "budget_reset_at" in captured_data, (
        "budget_reset_at must be recalculated even when budget_duration was set "
        "via attribute assignment after BudgetNewRequest construction"
    )
    assert "budget_duration" in captured_data, (
        "budget_duration must be present in the DB update"
    )
    reset_at = captured_data["budget_reset_at"]
    if isinstance(reset_at, str):
        reset_at = datetime.fromisoformat(reset_at)
    if reset_at.tzinfo is None:
        reset_at = reset_at.replace(tzinfo=timezone.utc)
    assert reset_at > datetime.now(timezone.utc), (
        "budget_reset_at must be a future datetime"
    )


@pytest.mark.asyncio
async def test_update_budget_same_duration_does_not_overwrite_reset_at(client_and_mocks):
    """
    Regression test: proxy startup calls update_team → upsert_team_member_budget_table
    → update_budget with the SAME budget_duration that's already stored in the DB.

    Before this fix, update_budget always recalculated budget_reset_at when
    budget_duration was set, so every proxy restart would stamp the template
    budget's reset time to "now + duration", making the UI display proxy-start
    time as the "Next Budget Reset" for all members.

    When the duration hasn't changed and budget_reset_at is already set to a
    future value, update_budget must NOT touch budget_reset_at.
    """
    import types as _types
    from datetime import datetime, timezone, timedelta

    client, _, mock_table = client_and_mocks

    future_reset = datetime.now(timezone.utc) + timedelta(days=5)

    # Simulate existing budget with the SAME duration and a valid future reset time
    mock_table.find_unique = AsyncMock(return_value=_types.SimpleNamespace(
        budget_duration="7d",
        budget_reset_at=future_reset,
    ))

    captured_data: dict = {}

    async def capture_update(*, where, data):
        captured_data.update(data)
        return {**where, **data}

    mock_table.update = AsyncMock(side_effect=capture_update)

    payload = {
        "budget_id": "template-budget-startup",
        "budget_duration": "7d",   # same as what's already in DB
        "max_budget": 50.0,
    }
    resp = client.post("/budget/update", json=payload)
    assert resp.status_code == 200, resp.text

    assert "budget_reset_at" not in captured_data, (
        "budget_reset_at must NOT be overwritten when budget_duration is unchanged "
        "and a valid future reset time already exists — proxy-start-time bug"
    )


@pytest.mark.asyncio
async def test_update_budget_changed_duration_recalculates_reset_at(client_and_mocks):
    """
    Complementary to the no-op test: when budget_duration genuinely changes
    (e.g. user switches from 7d to 1mo in the UI), budget_reset_at must be
    recalculated to reflect the new period.
    """
    import types as _types
    from datetime import datetime, timezone, timedelta

    client, _, mock_table = client_and_mocks

    # Existing budget has "7d"; request changes it to "30d"
    mock_table.find_unique = AsyncMock(return_value=_types.SimpleNamespace(
        budget_duration="7d",
        budget_reset_at=datetime.now(timezone.utc) + timedelta(days=3),
    ))

    captured_data: dict = {}

    async def capture_update(*, where, data):
        captured_data.update(data)
        return {**where, **data}

    mock_table.update = AsyncMock(side_effect=capture_update)

    payload = {
        "budget_id": "template-budget-change",
        "budget_duration": "30d",   # different from DB value
    }
    resp = client.post("/budget/update", json=payload)
    assert resp.status_code == 200, resp.text

    assert "budget_reset_at" in captured_data, (
        "budget_reset_at must be recalculated when budget_duration changes"
    )
    reset_at = captured_data["budget_reset_at"]
    if isinstance(reset_at, str):
        reset_at = datetime.fromisoformat(reset_at)
    if reset_at.tzinfo is None:
        reset_at = reset_at.replace(tzinfo=timezone.utc)
    assert reset_at > datetime.now(timezone.utc)
