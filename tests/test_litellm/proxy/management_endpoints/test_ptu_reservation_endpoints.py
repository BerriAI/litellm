import os
import sys
import types
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

import litellm.proxy.proxy_server as ps
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.proxy_server import app

sys.path.insert(0, os.path.abspath("../../../"))


def _dt(days: int = 0, hours: int = 0) -> datetime:
    return datetime(2026, 7, 1, tzinfo=timezone.utc) + timedelta(days=days, hours=hours)


def _iso(days: int = 0, hours: int = 0) -> str:
    return _dt(days=days, hours=hours).isoformat()


def _row(
    *,
    id: str = "res_1",
    team_id: str = "team_x",
    model: str = "gpt-4",
    cost_source: str = "manual",
    ptu_count: int | None = 1,
    cost_per_ptu: float | None = 200.0,
    azure_resource_id: str | None = None,
    effective_from: datetime | None = None,
    effective_to: datetime | None = None,
    created_by: str = "admin",
    updated_by: str = "admin",
):
    row = MagicMock()
    row.id = id
    row.team_id = team_id
    row.model = model
    row.cost_source = cost_source
    row.ptu_count = ptu_count
    row.cost_per_ptu = cost_per_ptu
    row.azure_resource_id = azure_resource_id
    row.effective_from = effective_from or _dt()
    row.effective_to = effective_to
    row.created_by = created_by
    row.updated_by = updated_by
    return row


@pytest.fixture
def client_and_mocks(monkeypatch):
    mock_prisma = MagicMock()
    mock_table = MagicMock()
    mock_table.create = AsyncMock(side_effect=lambda *, data: _row(**{k: v for k, v in data.items() if k != "created_at" and k != "updated_at"}))
    mock_table.update = AsyncMock(side_effect=lambda *, where, data: _row(id=where["id"], **{k: v for k, v in data.items() if k != "updated_at"}))
    mock_table.find_many = AsyncMock(return_value=[])
    mock_table.find_unique = AsyncMock(return_value=None)

    mock_prisma.db = types.SimpleNamespace(litellm_ptureservation=mock_table)
    monkeypatch.setattr(ps, "prisma_client", mock_prisma)

    monkeypatch.setitem(ps.general_settings, "enable_ptu_cost_attribution", True)

    admin_user = UserAPIKeyAuth(user_id="test_admin", user_role=LitellmUserRoles.PROXY_ADMIN)
    app.dependency_overrides[ps.user_api_key_auth] = lambda: admin_user

    client = TestClient(app)
    yield client, mock_prisma, mock_table

    app.dependency_overrides.clear()


def _switch_to_internal_user() -> None:
    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_id="not_admin", user_role=LitellmUserRoles.INTERNAL_USER
    )


def _valid_new_payload(**overrides):
    payload = {
        "team_id": "team_x",
        "model": "gpt-4",
        "ptu_count": 1,
        "cost_per_ptu": 200.0,
        "effective_from": _iso(),
    }
    payload.update(overrides)
    return payload


@pytest.mark.asyncio
async def test_new_reservation_success(client_and_mocks):
    client, _, mock_table = client_and_mocks

    resp = client.post("/ptu_reservation/new", json=_valid_new_payload())
    assert resp.status_code == 200, resp.text

    mock_table.find_many.assert_awaited_once()
    mock_table.create.assert_awaited_once()
    body = mock_table.create.await_args.kwargs["data"]
    assert body["team_id"] == "team_x"
    assert body["model"] == "gpt-4"
    assert body["ptu_count"] == 1
    assert body["cost_per_ptu"] == 200.0
    assert body["cost_source"] == "manual"
    assert body["created_by"] == "test_admin"
    assert body["updated_by"] == "test_admin"
    assert "id" not in body


@pytest.mark.asyncio
async def test_new_reservation_rejects_manual_without_ptu_count(client_and_mocks):
    client, _, _ = client_and_mocks

    payload = _valid_new_payload()
    del payload["ptu_count"]
    resp = client.post("/ptu_reservation/new", json=payload)
    assert resp.status_code == 400, resp.text
    assert "ptu_count" in resp.json()["detail"]["error"]


@pytest.mark.asyncio
async def test_new_reservation_rejects_manual_without_cost_per_ptu(client_and_mocks):
    client, _, _ = client_and_mocks

    payload = _valid_new_payload()
    del payload["cost_per_ptu"]
    resp = client.post("/ptu_reservation/new", json=payload)
    assert resp.status_code == 400, resp.text
    assert "cost_per_ptu" in resp.json()["detail"]["error"]


@pytest.mark.asyncio
async def test_new_reservation_rejects_non_positive_ptu_count(client_and_mocks):
    client, _, _ = client_and_mocks
    resp = client.post("/ptu_reservation/new", json=_valid_new_payload(ptu_count=0))
    assert resp.status_code == 400, resp.text


@pytest.mark.asyncio
async def test_new_reservation_rejects_negative_cost_per_ptu(client_and_mocks):
    client, _, _ = client_and_mocks
    resp = client.post("/ptu_reservation/new", json=_valid_new_payload(cost_per_ptu=-1.0))
    assert resp.status_code == 400, resp.text


@pytest.mark.asyncio
async def test_new_reservation_rejects_effective_to_before_from(client_and_mocks):
    client, _, _ = client_and_mocks
    resp = client.post(
        "/ptu_reservation/new",
        json=_valid_new_payload(effective_from=_iso(days=10), effective_to=_iso(days=1)),
    )
    assert resp.status_code == 400, resp.text
    assert "effective_to" in resp.json()["detail"]["error"]


@pytest.mark.asyncio
async def test_new_reservation_rejects_effective_to_equal_from(client_and_mocks):
    client, _, _ = client_and_mocks
    t = _iso()
    resp = client.post(
        "/ptu_reservation/new",
        json=_valid_new_payload(effective_from=t, effective_to=t),
    )
    assert resp.status_code == 400, resp.text


@pytest.mark.asyncio
async def test_new_reservation_rejects_azure_billing_mode(client_and_mocks):
    client, _, mock_table = client_and_mocks
    resp = client.post(
        "/ptu_reservation/new",
        json={
            "team_id": "team_x",
            "model": "gpt-4",
            "cost_source": "azure_billing",
            "azure_resource_id": "/subscriptions/x/deployments/gpt-4-ptu",
            "effective_from": _iso(),
        },
    )
    assert resp.status_code == 400, resp.text
    mock_table.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_new_reservation_rejects_overlap(client_and_mocks):
    client, _, mock_table = client_and_mocks
    mock_table.find_many = AsyncMock(return_value=[_row(id="existing_1")])

    resp = client.post("/ptu_reservation/new", json=_valid_new_payload())
    assert resp.status_code == 409, resp.text
    body = resp.json()
    assert body["detail"]["overlapping_ids"] == ["existing_1"]
    mock_table.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_new_reservation_allows_non_overlapping_same_team_model(client_and_mocks):
    client, _, mock_table = client_and_mocks
    mock_table.find_many = AsyncMock(return_value=[])

    resp = client.post(
        "/ptu_reservation/new",
        json=_valid_new_payload(effective_from=_iso(days=40)),
    )
    assert resp.status_code == 200, resp.text
    mock_table.create.assert_awaited_once()


@pytest.mark.asyncio
async def test_feature_flag_off_rejects_new(client_and_mocks, monkeypatch):
    client, _, mock_table = client_and_mocks
    monkeypatch.setitem(ps.general_settings, "enable_ptu_cost_attribution", False)

    resp = client.post("/ptu_reservation/new", json=_valid_new_payload())
    assert resp.status_code == 403, resp.text
    mock_table.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_feature_flag_off_rejects_list(client_and_mocks, monkeypatch):
    client, _, _ = client_and_mocks
    monkeypatch.setitem(ps.general_settings, "enable_ptu_cost_attribution", False)

    resp = client.get("/ptu_reservation/list")
    assert resp.status_code == 403, resp.text


@pytest.mark.asyncio
async def test_feature_flag_off_rejects_info(client_and_mocks, monkeypatch):
    client, _, _ = client_and_mocks
    monkeypatch.setitem(ps.general_settings, "enable_ptu_cost_attribution", False)

    resp = client.get("/ptu_reservation/info?id=res_1")
    assert resp.status_code == 403, resp.text


@pytest.mark.asyncio
async def test_feature_flag_off_rejects_close(client_and_mocks, monkeypatch):
    client, _, _ = client_and_mocks
    monkeypatch.setitem(ps.general_settings, "enable_ptu_cost_attribution", False)

    resp = client.post("/ptu_reservation/close", json={"id": "res_1"})
    assert resp.status_code == 403, resp.text


@pytest.mark.asyncio
async def test_non_admin_forbidden_from_new(client_and_mocks):
    client, _, mock_table = client_and_mocks
    _switch_to_internal_user()

    resp = client.post("/ptu_reservation/new", json=_valid_new_payload())
    assert resp.status_code == 403, resp.text
    mock_table.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_non_admin_forbidden_from_list(client_and_mocks):
    client, _, _ = client_and_mocks
    _switch_to_internal_user()

    resp = client.get("/ptu_reservation/list")
    assert resp.status_code == 403, resp.text


@pytest.mark.asyncio
async def test_non_admin_forbidden_from_close(client_and_mocks):
    client, _, _ = client_and_mocks
    _switch_to_internal_user()

    resp = client.post("/ptu_reservation/close", json={"id": "res_1"})
    assert resp.status_code == 403, resp.text


@pytest.mark.asyncio
async def test_list_no_filters(client_and_mocks):
    client, _, mock_table = client_and_mocks
    mock_table.find_many = AsyncMock(return_value=[_row(id="a"), _row(id="b")])

    resp = client.get("/ptu_reservation/list")
    assert resp.status_code == 200, resp.text
    mock_table.find_many.assert_awaited_once_with(where={})


@pytest.mark.asyncio
async def test_list_with_team_and_model_filters(client_and_mocks):
    client, _, mock_table = client_and_mocks
    mock_table.find_many = AsyncMock(return_value=[])

    resp = client.get("/ptu_reservation/list?team_id=team_x&model=gpt-4")
    assert resp.status_code == 200, resp.text
    mock_table.find_many.assert_awaited_once_with(where={"team_id": "team_x", "model": "gpt-4"})


@pytest.mark.asyncio
async def test_list_active_only(client_and_mocks):
    client, _, mock_table = client_and_mocks
    mock_table.find_many = AsyncMock(return_value=[])

    resp = client.get("/ptu_reservation/list?active_only=true&team_id=team_x")
    assert resp.status_code == 200, resp.text
    where = mock_table.find_many.await_args.kwargs["where"]
    assert where["team_id"] == "team_x"
    assert "effective_from" in where
    assert "OR" in where


@pytest.mark.asyncio
async def test_info_not_found(client_and_mocks):
    client, _, mock_table = client_and_mocks
    mock_table.find_unique = AsyncMock(return_value=None)

    resp = client.get("/ptu_reservation/info?id=missing")
    assert resp.status_code == 404, resp.text


@pytest.mark.asyncio
async def test_info_returns_row(client_and_mocks):
    client, _, mock_table = client_and_mocks
    mock_table.find_unique = AsyncMock(return_value=_row(id="res_9"))

    resp = client.get("/ptu_reservation/info?id=res_9")
    assert resp.status_code == 200, resp.text


@pytest.mark.asyncio
async def test_close_sets_effective_to(client_and_mocks):
    client, _, mock_table = client_and_mocks
    existing = _row(id="res_1", effective_from=_dt(), effective_to=None)
    mock_table.find_unique = AsyncMock(return_value=existing)

    resp = client.post("/ptu_reservation/close", json={"id": "res_1"})
    assert resp.status_code == 200, resp.text

    mock_table.update.assert_awaited_once()
    update_kwargs = mock_table.update.await_args.kwargs
    assert update_kwargs["where"] == {"id": "res_1"}
    assert "effective_to" in update_kwargs["data"]
    assert update_kwargs["data"]["updated_by"] == "test_admin"


@pytest.mark.asyncio
async def test_close_with_explicit_effective_to(client_and_mocks):
    client, _, mock_table = client_and_mocks
    existing = _row(id="res_1", effective_from=_dt(), effective_to=None)
    mock_table.find_unique = AsyncMock(return_value=existing)

    when = _iso(days=15)
    resp = client.post("/ptu_reservation/close", json={"id": "res_1", "effective_to": when})
    assert resp.status_code == 200, resp.text

    update_kwargs = mock_table.update.await_args.kwargs
    assert update_kwargs["data"]["effective_to"].isoformat() == when


@pytest.mark.asyncio
async def test_close_refuses_already_closed(client_and_mocks):
    client, _, mock_table = client_and_mocks
    existing = _row(id="res_1", effective_from=_dt(), effective_to=_dt(days=5))
    mock_table.find_unique = AsyncMock(return_value=existing)

    resp = client.post("/ptu_reservation/close", json={"id": "res_1"})
    assert resp.status_code == 400, resp.text
    mock_table.update.assert_not_awaited()


@pytest.mark.asyncio
async def test_close_refuses_effective_to_before_from(client_and_mocks):
    client, _, mock_table = client_and_mocks
    existing = _row(id="res_1", effective_from=_dt(days=10), effective_to=None)
    mock_table.find_unique = AsyncMock(return_value=existing)

    resp = client.post(
        "/ptu_reservation/close",
        json={"id": "res_1", "effective_to": _iso(days=1)},
    )
    assert resp.status_code == 400, resp.text
    mock_table.update.assert_not_awaited()


@pytest.mark.asyncio
async def test_close_not_found(client_and_mocks):
    client, _, mock_table = client_and_mocks
    mock_table.find_unique = AsyncMock(return_value=None)

    resp = client.post("/ptu_reservation/close", json={"id": "missing"})
    assert resp.status_code == 404, resp.text


@pytest.mark.asyncio
async def test_create_after_close_of_same_team_model(client_and_mocks):
    client, _, mock_table = client_and_mocks

    closed = _row(id="res_old", effective_from=_dt(), effective_to=_dt(days=15))
    mock_table.find_unique = AsyncMock(return_value=closed)
    resp = client.post("/ptu_reservation/close", json={"id": "res_old", "effective_to": _iso(days=14)})
    assert resp.status_code == 400, resp.text

    mock_table.find_unique = AsyncMock(return_value=_row(id="res_active", effective_from=_dt(), effective_to=None))
    close_time = _iso(days=15)
    resp = client.post("/ptu_reservation/close", json={"id": "res_active", "effective_to": close_time})
    assert resp.status_code == 200

    mock_table.find_many = AsyncMock(return_value=[])
    resp = client.post(
        "/ptu_reservation/new",
        json=_valid_new_payload(effective_from=close_time, ptu_count=100),
    )
    assert resp.status_code == 200, resp.text
    mock_table.create.assert_awaited_once()
