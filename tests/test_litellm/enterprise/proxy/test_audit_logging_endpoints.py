"""
Tests for the denormalized audit log alias columns and the object_team filter (LIT-4997).
"""

from datetime import datetime, timezone
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from litellm_enterprise.proxy.audit_logging_endpoints import (
    _build_json_field_or_condition,
    _build_object_team_condition,
)
from litellm_enterprise.proxy.audit_logging_endpoints import router as audit_router
from litellm_enterprise.types.proxy.audit_logging_endpoints import AuditLogResponse

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth


class FakeAuditLogTable:
    def __init__(self, rows=()):
        self.rows = list(rows)
        self.find_many_calls = []

    async def find_many(self, where=None, **kwargs):
        self.find_many_calls.append(where)
        return self.rows

    async def count(self, where=None):
        return len(self.rows)

    async def find_unique(self, where):
        return next((row for row in self.rows if row.id == where["id"]), None)


class FakeDb:
    def __init__(self, audit_logs=()):
        self.litellm_auditlog = FakeAuditLogTable(audit_logs)


class FakePrismaClient:
    def __init__(self, db: FakeDb):
        self.db = db


def make_log(**overrides) -> AuditLogResponse:
    defaults = {
        "id": "log-1",
        "updated_at": datetime(2026, 8, 1, tzinfo=timezone.utc),
        "changed_by": "",
        "changed_by_api_key": "",
        "action": "updated",
        "table_name": "LiteLLM_TeamTable",
        "object_id": "obj-1",
        "before_value": None,
        "updated_values": None,
    }
    return AuditLogResponse(**{**defaults, **overrides})


def _client_for(db: FakeDb) -> TestClient:
    app = FastAPI()
    app.include_router(audit_router)
    app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(user_role="proxy_admin")
    return TestClient(app)


def test_build_object_team_condition_matches_id_and_alias_columns():
    """object_team ORs an exact object_team_id match with an object_team_alias contains match."""
    assert _build_object_team_condition("prod") == {
        "OR": [
            {"object_team_id": "prod"},
            {"object_team_alias": {"contains": "prod"}},
        ]
    }


def test_get_audit_logs_returns_denormalized_columns_verbatim():
    """GET /audit passes the alias columns straight through from the DB row."""
    audit_row = make_log(
        id="l1",
        action="deleted",
        object_id="team-1",
        object_alias="ml-team",
        object_team_id="team-1",
        object_team_alias="ml-team",
        changed_by="admin-user",
        changed_by_api_key="hash-admin",
        changed_by_user_email="admin@example.com",
        changed_by_key_alias="admin-key",
    )
    db = FakeDb(audit_logs=[audit_row])

    with patch("litellm.proxy.proxy_server.prisma_client", FakePrismaClient(db)):
        response = _client_for(db).get("/audit")

    assert response.status_code == 200
    log = response.json()["audit_logs"][0]
    assert log["object_alias"] == "ml-team"
    assert log["object_team_id"] == "team-1"
    assert log["object_team_alias"] == "ml-team"
    assert log["changed_by_user_email"] == "admin@example.com"
    assert log["changed_by_key_alias"] == "admin-key"


def test_get_audit_logs_object_team_filter_uses_columns():
    """GET /audit?object_team=... ANDs in the column-based id-or-alias condition."""
    db = FakeDb()

    with patch("litellm.proxy.proxy_server.prisma_client", FakePrismaClient(db)):
        response = _client_for(db).get("/audit?object_team=prod")

    assert response.status_code == 200
    where = db.litellm_auditlog.find_many_calls[0]
    assert where["AND"] == [
        {
            "OR": [
                {"object_team_id": "prod"},
                {"object_team_alias": {"contains": "prod"}},
            ]
        }
    ]


def test_get_audit_logs_object_team_id_filter_unchanged():
    """The pre-existing object_team_id param still builds its exact JSON-blob condition."""
    db = FakeDb()

    with patch("litellm.proxy.proxy_server.prisma_client", FakePrismaClient(db)):
        response = _client_for(db).get("/audit?object_team_id=team-1")

    assert response.status_code == 200
    where = db.litellm_auditlog.find_many_calls[0]
    assert where["AND"] == [_build_json_field_or_condition("team_id", "team-1")]
    assert where["AND"] == [
        {
            "OR": [
                {"before_value": {"path": ["team_id"], "string_contains": "team-1"}},
                {"updated_values": {"path": ["team_id"], "string_contains": "team-1"}},
            ]
        }
    ]


def test_get_audit_log_by_id_returns_denormalized_columns():
    """GET /audit/{id} carries the same alias columns as the list endpoint."""
    audit_row = make_log(
        id="l1",
        table_name="LiteLLM_VerificationToken",
        object_id="gone-hash",
        action="deleted",
        object_alias="deleted-key",
        changed_by_user_email="admin@example.com",
        changed_by_key_alias="admin-key",
    )
    db = FakeDb(audit_logs=[audit_row])

    with patch("litellm.proxy.proxy_server.prisma_client", FakePrismaClient(db)):
        response = _client_for(db).get("/audit/l1")

    assert response.status_code == 200
    body = response.json()
    assert body["object_alias"] == "deleted-key"
    assert body["changed_by_user_email"] == "admin@example.com"
    assert body["changed_by_key_alias"] == "admin-key"


def test_alias_columns_default_to_none_for_legacy_rows():
    """Rows written before the migration serialize with null alias columns, not errors."""
    db = FakeDb(audit_logs=[make_log(id="l1")])

    with patch("litellm.proxy.proxy_server.prisma_client", FakePrismaClient(db)):
        response = _client_for(db).get("/audit")

    log = response.json()["audit_logs"][0]
    assert log["object_alias"] is None
    assert log["object_team_id"] is None
    assert log["object_team_alias"] is None
    assert log["changed_by_user_email"] is None
    assert log["changed_by_key_alias"] is None
