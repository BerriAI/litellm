"""`PATCH /management/v1/access-groups/{id}` against the real proxy app, so the problem+json handlers
registered in proxy_server are the ones rendering every error."""

import types
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

import litellm.proxy.proxy_server as ps
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.management_endpoints.management_v1.common import (
    MANAGEMENT_V1_PREFIX,
    PROBLEM_CONTENT_TYPE,
    PROBLEM_TYPE_BASE,
)

PATH = f"{MANAGEMENT_V1_PREFIX}/access-groups/ag-1"


def _record(**overrides):
    data = {
        "access_group_id": "ag-1",
        "access_group_name": "prod",
        "description": "Production models",
        "access_model_names": ["gpt-5.2"],
        "access_mcp_server_ids": [],
        "access_agent_ids": ["agent-1"],
        "assigned_team_ids": ["team-a"],
        "assigned_key_ids": [],
        "created_at": datetime(2026, 8, 1, tzinfo=timezone.utc),
        "created_by": "admin",
        "updated_at": datetime(2026, 8, 2, tzinfo=timezone.utc),
        "updated_by": "admin",
        **overrides,
    }
    record = MagicMock()
    for key, value in data.items():
        setattr(record, key, value)
    record.dict = lambda: data
    return record


@pytest.fixture
def access_group_table(monkeypatch):
    """A prisma double whose transaction hands back the same table mocks; `update` echoes the written data."""
    table = MagicMock()
    table.find_unique = AsyncMock(return_value=_record())
    table.update = AsyncMock(side_effect=lambda *, where, data: _record(**{k: v for k, v in data.items()}))

    team_table = MagicMock()
    team_table.find_unique = AsyncMock(return_value=None)
    team_table.update = AsyncMock(return_value=None)
    key_table = MagicMock()
    key_table.find_unique = AsyncMock(return_value=None)
    key_table.update = AsyncMock(return_value=None)

    @asynccontextmanager
    async def tx():
        yield types.SimpleNamespace(
            litellm_accessgrouptable=table, litellm_teamtable=team_table, litellm_verificationtoken=key_table
        )

    prisma = MagicMock()
    prisma.db = types.SimpleNamespace(
        litellm_accessgrouptable=table, litellm_teamtable=team_table, litellm_verificationtoken=key_table, tx=tx
    )
    monkeypatch.setattr(ps, "prisma_client", prisma)

    cache = MagicMock()
    cache.async_set_cache = AsyncMock(return_value=None)
    cache.async_get_cache = AsyncMock(return_value=None)
    monkeypatch.setattr(ps, "user_api_key_cache", cache)
    logging_obj = MagicMock()
    logging_obj.internal_usage_cache.dual_cache.async_get_cache = AsyncMock(return_value=None)
    logging_obj.internal_usage_cache.dual_cache.async_set_cache = AsyncMock(return_value=None)
    logging_obj.internal_usage_cache.dual_cache.async_delete_cache = AsyncMock(return_value=None)
    monkeypatch.setattr(ps, "proxy_logging_obj", logging_obj)
    return table


@pytest.fixture
def client():
    ps.app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_id="admin", user_role=LitellmUserRoles.PROXY_ADMIN
    )
    yield TestClient(ps.app)
    ps.app.dependency_overrides.clear()


def _as_role(role: LitellmUserRoles) -> None:
    ps.app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(user_id="u", user_role=role)


def _assert_problem(resp, status: int, problem_type: str) -> dict:
    assert resp.status_code == status
    assert resp.headers["content-type"] == PROBLEM_CONTENT_TYPE
    body = resp.json()
    assert body["type"] == f"{PROBLEM_TYPE_BASE}{problem_type}"
    assert body["status"] == status
    assert isinstance(body["detail"], str)
    return body


def test_writes_only_the_sent_fields_and_answers_in_the_data_envelope(client, access_group_table):
    resp = client.patch(PATH, json={"description": "Only this changes"})

    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/json"
    assert access_group_table.update.call_args.kwargs == {
        "where": {"access_group_id": "ag-1"},
        "data": {"updated_by": "admin", "description": "Only this changes"},
    }
    body = resp.json()
    assert set(body) == {"data"}
    assert body["data"]["access_group_id"] == "ag-1"
    assert body["data"]["description"] == "Only this changes"


def test_null_clears_a_scalar_and_a_list(client, access_group_table):
    resp = client.patch(PATH, json={"description": None, "access_model_names": None})

    assert resp.status_code == 200
    assert access_group_table.update.call_args.kwargs["data"] == {
        "updated_by": "admin",
        "description": None,
        "access_model_names": [],
    }


def test_syncs_team_membership_from_the_assigned_team_delta(client, access_group_table):
    team_table = ps.prisma_client.db.litellm_teamtable
    team_table.find_unique = AsyncMock(
        side_effect=lambda *, where: types.SimpleNamespace(team_id=where["team_id"], access_group_ids=[])
    )

    resp = client.patch(PATH, json={"assigned_team_ids": ["team-b"]})

    assert resp.status_code == 200
    updates = {call.kwargs["where"]["team_id"]: call.kwargs["data"] for call in team_table.update.call_args_list}
    assert updates == {"team-b": {"access_group_ids": ["ag-1"]}}


def test_an_unknown_body_key_is_a_422_problem_not_a_silent_no_op(client, access_group_table):
    resp = client.patch(PATH, json={"descripton": "typo"})

    body = _assert_problem(resp, 422, "invalid-request-body")
    assert "descripton" in body["detail"]
    access_group_table.update.assert_not_awaited()


def test_a_wrongly_typed_field_is_a_422_problem(client, access_group_table):
    resp = client.patch(PATH, json={"access_model_names": "gpt-5.2"})

    body = _assert_problem(resp, 422, "invalid-request-body")
    assert "access_model_names" in body["detail"]
    access_group_table.update.assert_not_awaited()


def test_a_missing_group_is_a_404_problem(client, access_group_table):
    access_group_table.find_unique = AsyncMock(return_value=None)

    body = _assert_problem(client.patch(PATH, json={"description": "x"}), 404, "not-found")
    assert "ag-1" in body["detail"]
    access_group_table.update.assert_not_awaited()


def test_a_taken_name_is_a_409_problem(client, access_group_table):
    access_group_table.update = AsyncMock(side_effect=Exception("P2002: Unique constraint failed"))

    body = _assert_problem(client.patch(PATH, json={"access_group_name": "taken"}), 409, "conflict")
    assert "taken" in body["detail"]


@pytest.mark.parametrize(
    "role",
    [LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY, LitellmUserRoles.INTERNAL_USER, LitellmUserRoles.TEAM],
)
def test_a_caller_who_is_not_a_proxy_admin_is_refused_with_a_403_problem(client, access_group_table, role):
    _as_role(role)

    _assert_problem(client.patch(PATH, json={"description": "x"}), 403, "forbidden")
    access_group_table.update.assert_not_awaited()


def test_a_driver_error_is_a_500_problem_not_the_openai_error_shape(client, access_group_table):
    access_group_table.update = AsyncMock(side_effect=RuntimeError("connection reset"))

    body = _assert_problem(client.patch(PATH, json={"description": "x"}), 500, "internal-server-error")
    assert "connection reset" not in body["detail"]


def test_no_database_is_a_503_problem(client, access_group_table, monkeypatch):
    monkeypatch.setattr(ps, "prisma_client", None)

    _assert_problem(client.patch(PATH, json={"description": "x"}), 503, "database-not-connected")


def test_the_route_advertises_its_errors_as_problem_documents():
    operation = ps.app.openapi()["paths"][f"{MANAGEMENT_V1_PREFIX}/access-groups/{{access_group_id}}"]["patch"]

    assert sorted(operation["responses"]) == ["200", "403", "404", "409", "422", "500", "503"]
    assert operation["responses"]["422"]["content"] == {
        PROBLEM_CONTENT_TYPE: {"schema": {"$ref": "#/components/schemas/ProblemDetail"}}
    }
    assert "ProblemDetail" in ps.app.openapi()["components"]["schemas"]
