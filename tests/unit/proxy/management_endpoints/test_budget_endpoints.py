# tests/test_budget_endpoints.py

import json
import types
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from typing import Final
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from prisma import Json

import litellm.proxy.proxy_server as ps
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.proxy._types import CommonProxyErrors, LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.proxy_server import app


@pytest.fixture
def client_and_mocks(monkeypatch):
    # Setup MagicMock Prisma
    mock_prisma = MagicMock()
    mock_table = MagicMock()

    def created(*, data: Mapping[str, object]) -> dict[str, object]:
        return {key: value.data if isinstance(value, Json) else value for key, value in data.items()}

    def updated(*, where: Mapping[str, object], data: Mapping[str, object]) -> dict[str, object]:
        return created(data={**where, **data})

    mock_table.create = AsyncMock(side_effect=created)
    mock_table.update = AsyncMock(side_effect=updated)

    mock_prisma.db = types.SimpleNamespace(
        litellm_budgettable=mock_table,
        litellm_customermembertable=types.SimpleNamespace(find_many=AsyncMock(return_value=[])),
        litellm_customerteamtable=types.SimpleNamespace(find_many=AsyncMock(return_value=[])),
        litellm_customerorganizationtable=types.SimpleNamespace(find_many=AsyncMock(return_value=[])),
        litellm_dailyspend=mock_table,
        query_raw=AsyncMock(return_value=[{"referenced": False}]),
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


@pytest.mark.parametrize("path", ("/budget/new", "/budget/update"))
def test_budget_windows_round_trip_with_existing_scalar_cap(
    client_and_mocks: tuple[TestClient, MagicMock, MagicMock], path: str
) -> None:
    client, _, _ = client_and_mocks
    response: Final = client.post(path, json={
        "budget_id": "combined", "max_budget": 25, "budget_duration": "30d",
        "budget_limits": [
            {"budget_duration": "1mo", "max_budget": 0, "reset_at": "2027-02-01T00:00:00Z"},
            {"budget_duration": "12mo", "max_budget": 1000, "reset_at": "2028-01-01T00:00:00Z"},
        ],
    })
    assert response.status_code == 200, response.text
    result: Final = response.json()
    assert result["max_budget"] == 25
    assert result["budget_duration"] == "30d"
    assert {entry["budget_duration"]: entry["max_budget"] for entry in result["budget_limits"]} == {
        "1mo": 0, "12mo": 1000,
    }
    assert result["budget_limits"][0]["reset_at"] == "2027-02-01T00:00:00+00:00"


@pytest.mark.parametrize("clear", (None, []))
def test_budget_update_can_clear_windows(
    client_and_mocks: tuple[TestClient, MagicMock, MagicMock], clear: object
) -> None:
    client, _, _ = client_and_mocks
    response: Final = client.post("/budget/update", json={"budget_id": "combined", "budget_limits": clear})
    assert response.status_code == 200, response.text
    assert response.json()["budget_limits"] == clear


@pytest.mark.parametrize("path", ("/budget/new", "/budget/update"))
@pytest.mark.parametrize("duration", ("2d", "14d", "30d", "1h"))
def test_concurrent_windows_reject_noncanonical_durations_but_scalar_budgets_preserve_them(
    client_and_mocks: tuple[TestClient, MagicMock, MagicMock], path: str, duration: str,
) -> None:
    client, _, table = client_and_mocks
    rejected: Final = client.post(path, json={
        "budget_id": "duration-test",
        "budget_limits": [{"budget_duration": duration, "max_budget": 10}],
    })
    assert rejected.status_code == 422, rejected.text
    table.create.assert_not_awaited()
    table.update.assert_not_awaited()

    accepted: Final = client.post(path, json={
        "budget_id": "duration-test", "max_budget": 10, "budget_duration": duration,
    })
    assert accepted.status_code == 200, accepted.text
    assert accepted.json()["budget_duration"] == duration


def test_budget_update_rejects_windows_on_an_existing_scalar_consumer(
    client_and_mocks: tuple[TestClient, MagicMock, MagicMock],
) -> None:
    client, database, table = client_and_mocks
    database.db.query_raw.return_value = [{"referenced": True}]
    response: Final = client.post("/budget/update", json={
        "budget_id": "used-by-key", "budget_limits": [{"budget_duration": "12mo", "max_budget": 1000}],
    })
    assert response.status_code == 400, response.text
    table.update.assert_not_awaited()


def test_budget_edit_evicts_members_and_organization_aliases(
    client_and_mocks: tuple[TestClient, MagicMock, MagicMock], monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, database, _ = client_and_mocks
    cache: Final = UserApiKeyCache(in_memory_cache=InMemoryCache())
    member_key: Final = "customer_member_id:member"
    alias_key: Final = "customer_org_alias:organization"
    cache.set_cache(member_key, {"stale": True}, local_only=True)
    cache.set_cache(alias_key, {"stale": True}, local_only=True)
    monkeypatch.setattr(ps, "user_api_key_cache", cache)
    database.db.litellm_customermembertable.find_many.return_value = [types.SimpleNamespace(customer_member_id="member")]
    database.db.litellm_customerorganizationtable.find_many.return_value = [types.SimpleNamespace(alias="organization")]
    response: Final = client.post("/budget/update", json={"budget_id": "shared", "max_budget": 3})
    assert response.status_code == 200, response.text
    assert cache.get_cache(member_key, local_only=True) is None
    assert cache.get_cache(alias_key, local_only=True) is None


def test_budget_edit_evicts_a_members_inherited_team_budget(
    client_and_mocks: tuple[TestClient, MagicMock, MagicMock], monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, database, _ = client_and_mocks
    cache: Final = UserApiKeyCache(in_memory_cache=InMemoryCache())
    member_key: Final = "customer_member_id:through-team"
    other_key: Final = "customer_member_id:unrelated"
    cache.set_cache(member_key, {"max_budget": 10}, local_only=True)
    cache.set_cache(other_key, {"max_budget": 20}, local_only=True)
    monkeypatch.setattr(ps, "user_api_key_cache", cache)
    database.db.litellm_customerteamtable.find_many.return_value = [types.SimpleNamespace(customer_team_id="team")]
    database.db.litellm_customermembertable.find_many.side_effect = [
        [], [types.SimpleNamespace(customer_member_id="through-team")],
    ]
    response: Final = client.post("/budget/update", json={"budget_id": "team-budget", "max_budget": 3})
    assert response.status_code == 200, response.text
    assert cache.get_cache(member_key, local_only=True) is None
    assert cache.get_cache(other_key, local_only=True) == {"max_budget": 20}


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


@pytest.mark.parametrize("bad_duration", ["0s", "-5m"])
@pytest.mark.asyncio
async def test_new_budget_rejects_a_duration_that_never_advances(
    client_and_mocks, bad_duration
):
    """A zero-length window resets to "now", so the row is due again the moment
    it is written and the reset job re-reads it on every tick forever."""
    client, _, mock_table = client_and_mocks

    resp = client.post(
        "/budget/new",
        json={"budget_id": "budget_bad", "max_budget": 10.0, "budget_duration": bad_duration},
    )

    assert resp.status_code == 400, resp.text
    assert "Invalid budget_duration" in resp.json()["detail"]["error"]
    mock_table.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_update_budget_rejects_a_duration_that_never_advances(client_and_mocks):
    client, _, mock_table = client_and_mocks

    resp = client.post(
        "/budget/update",
        json={"budget_id": "budget_456", "budget_duration": "0s"},
    )

    assert resp.status_code == 400, resp.text
    assert "Invalid budget_duration" in resp.json()["detail"]["error"]
    mock_table.update.assert_not_awaited()


@pytest.mark.asyncio
async def test_update_budget_evicts_local_cache_when_redis_delete_fails(
    client_and_mocks, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, _, _ = client_and_mocks
    cache_key = "team_member_default_budget:budget_456"
    cache = UserApiKeyCache(in_memory_cache=InMemoryCache())
    cache.set_cache(cache_key, {"budget_id": "budget_456"}, local_only=True)
    redis_cache = MagicMock()
    redis_cache.async_delete_cache = AsyncMock(side_effect=RuntimeError("redis unavailable"))
    cache.redis_cache = redis_cache
    monkeypatch.setattr(ps, "user_api_key_cache", cache)

    response = client.post("/budget/update", json={"budget_id": "budget_456", "max_budget": 10.0})

    assert response.status_code == 200, response.text
    assert cache.get_cache(cache_key, local_only=True) is None


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
async def test_update_budget_evicts_team_member_default_budget_cache(client_and_mocks, monkeypatch):
    client, _, _ = client_and_mocks
    cache: Final = UserApiKeyCache(in_memory_cache=InMemoryCache())
    monkeypatch.setattr(ps, "user_api_key_cache", cache)
    cache_key: Final = "team_member_default_budget:budget_456"
    await cache.async_set_cache(key=cache_key, value={"budget_id": "budget_456"})
    assert await cache.async_get_cache(key=cache_key) == {"budget_id": "budget_456"}

    try:
        response: Final = client.post(
            "/budget/update",
            json={"budget_id": "budget_456", "max_budget": 10.0},
        )
        assert response.status_code == 200, response.text
        assert await cache.async_get_cache(key=cache_key) is None
    finally:
        await cache.async_delete_cache(key=cache_key)


@pytest.mark.asyncio
async def test_new_and_update_budget_persist_tpd_limit(client_and_mocks):
    client, _, mock_table = client_and_mocks

    resp = client.post("/budget/new", json={"budget_id": "budget_tpd", "tpd_limit": 250000})
    assert resp.status_code == 200, resp.text
    assert resp.json()["tpd_limit"] == 250000
    assert mock_table.create.await_args.kwargs["data"]["tpd_limit"] == 250000

    resp = client.post("/budget/update", json={"budget_id": "budget_tpd", "tpd_limit": 500000})
    assert resp.status_code == 200, resp.text
    assert resp.json()["tpd_limit"] == 500000
    assert mock_table.update.await_args.kwargs["data"]["tpd_limit"] == 500000


@pytest.mark.asyncio
async def test_new_and_update_budget_persist_per_model_tpm_and_rpm_limits(client_and_mocks):
    client, _, mock_table = client_and_mocks

    resp = client.post(
        "/budget/new",
        json={"budget_id": "budget_models", "model_tpm_limit": {"gpt-4o": 400000}, "model_rpm_limit": {"gpt-4o": 200}},
    )
    assert resp.status_code == 200, resp.text
    created: Final = mock_table.create.await_args.kwargs["data"]
    assert json.loads(created["model_tpm_limit"]) == {"gpt-4o": 400000}, created
    assert json.loads(created["model_rpm_limit"]) == {"gpt-4o": 200}, created

    resp = client.post("/budget/update", json={"budget_id": "budget_models", "model_tpm_limit": {"gpt-4o": 0}})
    assert resp.status_code == 200, resp.text
    updated: Final = mock_table.update.await_args.kwargs["data"]
    assert json.loads(updated["model_tpm_limit"]) == {"gpt-4o": 0}, updated
    assert "model_rpm_limit" not in updated, updated


@pytest.mark.parametrize("route", ["/budget/new", "/budget/update"])
@pytest.mark.parametrize("field", ["model_tpm_limit", "model_rpm_limit"])
@pytest.mark.parametrize("value", [-1, 1.5])
@pytest.mark.asyncio
async def test_budget_rejects_a_per_model_limit_that_is_not_a_whole_number_of_zero_or_more(
    client_and_mocks, route, field, value
):
    client, _, mock_table = client_and_mocks

    resp = client.post(route, json={"budget_id": "budget_models", field: {"gpt-4o": value}})

    assert resp.status_code == 422, resp.text
    assert field in resp.text
    mock_table.create.assert_not_awaited()
    mock_table.update.assert_not_awaited()


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


def _capture_update_data(mock_table):
    captured = {}

    async def capture(*, where, data):
        captured.update(data)
        return {**where, **data}

    mock_table.update = AsyncMock(side_effect=capture)
    return captured


@pytest.mark.asyncio
async def test_update_budget_recomputes_reset_at_when_duration_changes(
    client_and_mocks,
):
    """
    Regression for LIT-3362: shortening budget_duration without an explicit
    budget_reset_at must bring the reset forward instead of leaving it pinned
    to the previous (longer) schedule.
    """
    client, _, mock_table = client_and_mocks
    captured = _capture_update_data(mock_table)

    before = datetime.now(timezone.utc)
    resp = client.post(
        "/budget/update",
        json={"budget_id": "budget_reset_recompute", "budget_duration": "1d"},
    )
    assert resp.status_code == 200, resp.text

    assert (
        "budget_reset_at" in captured
    ), "duration change must recompute budget_reset_at"
    reset_at = captured["budget_reset_at"]
    assert isinstance(reset_at, datetime)
    assert reset_at > before, "recomputed reset must be in the future"
    # "1d" resets at the next standardized day boundary, always within ~24h
    assert reset_at <= before + timedelta(days=1, hours=1), reset_at
    # and it must be far closer than a stale 30d schedule would have left it
    assert reset_at < before + timedelta(days=29)


@pytest.mark.asyncio
@pytest.mark.parametrize("budget_duration", ["1d", None])
async def test_update_budget_preserves_explicit_reset_at(client_and_mocks, budget_duration):
    """An explicit budget_reset_at from the caller always wins over recompute."""
    client, _, mock_table = client_and_mocks
    captured = _capture_update_data(mock_table)

    explicit = datetime(2027, 1, 1, tzinfo=timezone.utc)
    resp = client.post(
        "/budget/update",
        json={
            "budget_id": "budget_explicit_reset",
            "budget_duration": budget_duration,
            "budget_reset_at": explicit.isoformat(),
        },
    )
    assert resp.status_code == 200, resp.text

    assert captured["budget_reset_at"] == explicit


@pytest.mark.asyncio
async def test_update_budget_without_duration_leaves_reset_at_untouched(
    client_and_mocks,
):
    """Updates that do not touch budget_duration must not introduce budget_reset_at."""
    client, _, mock_table = client_and_mocks
    captured = _capture_update_data(mock_table)

    resp = client.post(
        "/budget/update",
        json={"budget_id": "budget_other_field", "max_budget": 200.0},
    )
    assert resp.status_code == 200, resp.text

    assert "budget_reset_at" not in captured


@pytest.mark.asyncio
async def test_update_budget_duration_none_clears_obsolete_reset(client_and_mocks):
    client, _, mock_table = client_and_mocks
    captured = _capture_update_data(mock_table)

    resp = client.post(
        "/budget/update",
        json={"budget_id": "budget_clear_duration", "budget_duration": None},
    )
    assert resp.status_code == 200, resp.text

    assert "budget_duration" in captured and captured["budget_duration"] is None
    assert captured["budget_reset_at"] is None


@pytest.mark.asyncio
async def test_update_budget_serializes_model_max_budget_for_prisma(
    client_and_mocks, monkeypatch
):
    monkeypatch.setattr(ps, "premium_user", True)

    client, _, mock_table = client_and_mocks
    captured: Final = _capture_update_data(mock_table)

    resp: Final = client.post(
        "/budget/update",
        json={
            "budget_id": "budget_per_model",
            "model_max_budget": {
                "gpt4o": {"budget_limit": 5.0, "time_period": "1d"},
                "glm-5.2": {"budget_limit": 7.5, "time_period": "30d"},
            },
        },
    )
    assert resp.status_code == 200, resp.text

    stored: Final = captured["model_max_budget"]
    assert isinstance(stored, str), (
        f"model_max_budget must reach prisma as a JSON string, got {type(stored).__name__}"
    )
    assert json.loads(stored) == {
        "gpt4o": {"max_budget": 5.0, "budget_duration": "1d"},
        "glm-5.2": {"max_budget": 7.5, "budget_duration": "30d"},
    }


@pytest.fixture
def admin_client(client_and_mocks):
    client, mock_prisma, mock_table = client_and_mocks
    mock_table.delete = AsyncMock(return_value={"budget_id": "tpl-1"})
    mock_prisma.db.litellm_customerorganizationtable = MagicMock()
    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_id="admin", user_role=LitellmUserRoles.PROXY_ADMIN
    )
    return client, mock_prisma, mock_table


@pytest.mark.asyncio
async def test_delete_budget_refuses_customer_organization_default_budget(admin_client):
    client, mock_prisma, mock_table = admin_client
    mock_prisma.db.litellm_customerorganizationtable.count = AsyncMock(return_value=1)

    resp: Final = client.post("/budget/delete", json={"id": "tpl-1"})

    assert resp.status_code == 409, resp.text
    assert "/default_budgets/" in resp.json()["detail"]["error"]
    mock_table.delete.assert_not_awaited()


@pytest.mark.asyncio
async def test_delete_budget_deletes_budget_no_customer_organization_uses(admin_client):
    client, mock_prisma, mock_table = admin_client
    mock_prisma.db.litellm_customerorganizationtable.count = AsyncMock(return_value=0)

    resp: Final = client.post("/budget/delete", json={"id": "tpl-1"})

    assert resp.status_code == 200, resp.text
    mock_table.delete.assert_awaited_once_with(where={"budget_id": "tpl-1"})
