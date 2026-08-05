"""
Tests for audit log alias enrichment and the combined object_team filter (LIT-4997).
"""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from litellm_enterprise.proxy.audit_logging_endpoints import (
    _build_json_field_or_condition,
    _build_object_team_condition,
    _enrich_audit_logs,
)
from litellm_enterprise.proxy.audit_logging_endpoints import router as audit_router
from litellm_enterprise.types.proxy.audit_logging_endpoints import AuditLogResponse

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth


class FakeTable:
    def __init__(self, rows=()):
        self.rows = list(rows)
        self.find_many_calls = []

    async def find_many(self, where=None, **kwargs):
        self.find_many_calls.append(where)
        return self.rows


class FakeAuditLogTable(FakeTable):
    def __init__(self, rows=()):
        super().__init__(rows)
        self.count_calls = []

    async def count(self, where=None):
        self.count_calls.append(where)
        return len(self.rows)

    async def find_unique(self, where):
        return next((row for row in self.rows if row.id == where["id"]), None)


class FakeDb:
    def __init__(
        self,
        audit_logs=(),
        keys=(),
        users=(),
        teams=(),
        orgs=(),
        models=(),
        team_id_rows=(),
    ):
        self.litellm_auditlog = FakeAuditLogTable(audit_logs)
        self.litellm_verificationtoken = FakeTable(keys)
        self.litellm_usertable = FakeTable(users)
        self.litellm_teamtable = FakeTable(teams)
        self.litellm_organizationtable = FakeTable(orgs)
        self.litellm_proxymodeltable = FakeTable(models)
        self.team_id_rows = list(team_id_rows)
        self.query_raw_calls = []

    async def query_raw(self, sql, *args):
        self.query_raw_calls.append((sql, *args))
        return self.team_id_rows


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


async def test_enrichment_resolves_object_alias_per_table_name():
    """object_alias comes from the right table per table_name; changed_by fields resolve too."""
    logs = [
        make_log(id="l1", table_name="LiteLLM_VerificationToken", object_id="hash-1"),
        make_log(id="l2", table_name="LiteLLM_TeamTable", object_id="team-1"),
        make_log(id="l3", table_name="LiteLLM_UserTable", object_id="user-1"),
        make_log(id="l4", table_name="LiteLLM_UserTable", object_id="user-2"),
        make_log(id="l5", table_name="LiteLLM_OrganizationTable", object_id="org-1"),
        make_log(id="l6", table_name="LiteLLM_ProxyModelTable", object_id="model-1"),
        make_log(
            id="l7",
            table_name="LiteLLM_TeamTable",
            object_id="team-1",
            changed_by="admin-user",
            changed_by_api_key="hash-admin",
        ),
    ]
    db = FakeDb(
        keys=[
            SimpleNamespace(token="hash-1", key_alias="prod-key"),
            SimpleNamespace(token="hash-admin", key_alias="admin-key"),
        ],
        users=[
            SimpleNamespace(user_id="user-1", user_alias="Alice", user_email="alice@example.com"),
            SimpleNamespace(user_id="user-2", user_alias=None, user_email="bob@example.com"),
            SimpleNamespace(user_id="admin-user", user_alias=None, user_email="admin@example.com"),
        ],
        teams=[SimpleNamespace(team_id="team-1", team_alias="ml-team")],
        orgs=[SimpleNamespace(organization_id="org-1", organization_alias="acme-org")],
        models=[SimpleNamespace(model_id="model-1", model_name="gpt-5.2")],
    )

    enriched = await _enrich_audit_logs(FakePrismaClient(db), logs)
    by_id = {log.id: log for log in enriched}

    assert by_id["l1"].object_alias == "prod-key"
    assert by_id["l2"].object_alias == "ml-team"
    assert by_id["l3"].object_alias == "Alice"
    assert by_id["l4"].object_alias == "bob@example.com"
    assert by_id["l5"].object_alias == "acme-org"
    assert by_id["l6"].object_alias == "gpt-5.2"
    assert by_id["l7"].changed_by_user_email == "admin@example.com"
    assert by_id["l7"].changed_by_key_alias == "admin-key"
    assert by_id["l1"].changed_by_user_email is None
    assert by_id["l1"].changed_by_key_alias is None


async def test_enrichment_runs_one_query_per_entity_type():
    """A page with many rows triggers at most one find_many per entity table, ids batched via `in`."""
    logs = [
        make_log(id="l1", table_name="LiteLLM_VerificationToken", object_id="hash-1", changed_by="u1"),
        make_log(id="l2", table_name="LiteLLM_VerificationToken", object_id="hash-2", changed_by="u2"),
        make_log(
            id="l3",
            table_name="LiteLLM_TeamTable",
            object_id="team-1",
            changed_by="u1",
            changed_by_api_key="hash-caller",
        ),
    ]
    db = FakeDb()

    await _enrich_audit_logs(FakePrismaClient(db), logs)

    assert len(db.litellm_verificationtoken.find_many_calls) == 1
    assert len(db.litellm_usertable.find_many_calls) == 1
    assert len(db.litellm_teamtable.find_many_calls) == 1
    assert len(db.litellm_organizationtable.find_many_calls) == 0
    assert len(db.litellm_proxymodeltable.find_many_calls) == 0
    assert set(db.litellm_verificationtoken.find_many_calls[0]["token"]["in"]) == {
        "hash-1",
        "hash-2",
        "hash-caller",
    }
    assert set(db.litellm_usertable.find_many_calls[0]["user_id"]["in"]) == {"u1", "u2"}
    assert db.litellm_teamtable.find_many_calls[0] == {"team_id": {"in": ["team-1"]}}


async def test_enrichment_falls_back_to_blobs_for_deleted_objects():
    """When DB lookups miss (deleted objects), aliases come from updated_values then before_value."""
    logs = [
        make_log(
            id="l1",
            table_name="LiteLLM_TeamTable",
            object_id="gone-team",
            action="deleted",
            before_value={"team_id": "gone-team", "team_alias": "old-team"},
        ),
        make_log(
            id="l2",
            table_name="LiteLLM_VerificationToken",
            object_id="gone-hash",
            before_value={"key_alias": "old-alias"},
            updated_values={"key_alias": "new-alias"},
        ),
        make_log(
            id="l3",
            table_name="LiteLLM_UserTable",
            object_id="gone-user",
            updated_values={"user_email": "gone@example.com"},
        ),
        make_log(
            id="l4",
            table_name="LiteLLM_OrganizationTable",
            object_id="gone-org",
            before_value={"organization_alias": "old-org"},
        ),
        make_log(id="l5", table_name="SomeUnknownTable", object_id="x", updated_values={"team_alias": "nope"}),
    ]

    enriched = await _enrich_audit_logs(FakePrismaClient(FakeDb()), logs)
    by_id = {log.id: log for log in enriched}

    assert by_id["l1"].object_alias == "old-team"
    assert by_id["l2"].object_alias == "new-alias"
    assert by_id["l3"].object_alias == "gone@example.com"
    assert by_id["l4"].object_alias == "old-org"
    assert by_id["l5"].object_alias is None


async def test_enrichment_db_lookup_wins_over_blob():
    """A live DB row beats a stale alias captured in the audit blobs."""
    logs = [
        make_log(
            id="l1",
            table_name="LiteLLM_TeamTable",
            object_id="team-1",
            before_value={"team_alias": "stale-alias"},
        )
    ]
    db = FakeDb(teams=[SimpleNamespace(team_id="team-1", team_alias="current-alias")])

    enriched = await _enrich_audit_logs(FakePrismaClient(db), logs)

    assert enriched[0].object_alias == "current-alias"


async def test_build_object_team_condition_matches_id_and_alias():
    """object_team ORs the raw value with every team_id whose team_alias contains it,
    via a projected and capped query so one request cannot load the whole team table."""
    db = FakeDb(team_id_rows=[{"team_id": "team-1"}, {"team_id": "team-2"}])

    condition = await _build_object_team_condition(FakePrismaClient(db), "prod")

    assert db.query_raw_calls == [
        ('SELECT team_id FROM "LiteLLM_TeamTable" WHERE team_alias LIKE $1 LIMIT 100', "%prod%")
    ]
    assert db.litellm_teamtable.find_many_calls == []
    assert condition == {
        "OR": [
            _build_json_field_or_condition("team_alias", "prod"),
            _build_json_field_or_condition("team_id", "prod"),
            _build_json_field_or_condition("team_id", "team-1"),
            _build_json_field_or_condition("team_id", "team-2"),
        ]
    }


async def test_build_object_team_condition_escapes_like_wildcards():
    """LIKE wildcards in the user-supplied value are escaped, not treated as patterns."""
    db = FakeDb()

    await _build_object_team_condition(FakePrismaClient(db), "pr_od%te\\am")

    assert db.query_raw_calls[0][1] == "%pr\\_od\\%te\\\\am%"


async def test_build_object_team_condition_deleted_team_matches_blob_alias():
    """With no live team rows the condition still matches blob team_alias and the raw value as team_id."""
    condition = await _build_object_team_condition(FakePrismaClient(FakeDb()), "gone-team")

    assert condition == {
        "OR": [
            _build_json_field_or_condition("team_alias", "gone-team"),
            _build_json_field_or_condition("team_id", "gone-team"),
        ]
    }


def _client_for(db: FakeDb) -> TestClient:
    app = FastAPI()
    app.include_router(audit_router)
    app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(user_role="proxy_admin")
    return TestClient(app)


def test_get_audit_logs_object_team_filter_and_enrichment():
    """GET /audit?object_team=... ANDs in the combined id/alias condition and returns enriched rows."""
    audit_row = make_log(
        id="l1",
        table_name="LiteLLM_TeamTable",
        object_id="team-1",
        changed_by="admin-user",
        updated_values={"team_id": "team-1"},
    )
    db = FakeDb(
        audit_logs=[audit_row],
        users=[SimpleNamespace(user_id="admin-user", user_alias=None, user_email="admin@example.com")],
        teams=[SimpleNamespace(team_id="team-1", team_alias="prod-team")],
        team_id_rows=[{"team_id": "team-1"}],
    )
    client = _client_for(db)

    with patch("litellm.proxy.proxy_server.prisma_client", FakePrismaClient(db)):
        response = client.get("/audit?object_team=prod")

    assert response.status_code == 200
    where = db.litellm_auditlog.find_many_calls[0]
    assert where["AND"] == [
        {
            "OR": [
                _build_json_field_or_condition("team_alias", "prod"),
                _build_json_field_or_condition("team_id", "prod"),
                _build_json_field_or_condition("team_id", "team-1"),
            ]
        }
    ]
    log = response.json()["audit_logs"][0]
    assert log["object_alias"] == "prod-team"
    assert log["changed_by_user_email"] == "admin@example.com"
    assert log["changed_by_key_alias"] is None


def test_get_audit_logs_object_team_id_filter_unchanged():
    """The pre-existing object_team_id param still builds its exact condition, no alias lookup."""
    db = FakeDb()
    client = _client_for(db)

    with patch("litellm.proxy.proxy_server.prisma_client", FakePrismaClient(db)):
        response = client.get("/audit?object_team_id=team-1")

    assert response.status_code == 200
    where = db.litellm_auditlog.find_many_calls[0]
    assert where["AND"] == [_build_json_field_or_condition("team_id", "team-1")]
    assert db.litellm_teamtable.find_many_calls == []
    assert db.query_raw_calls == []


def test_get_audit_log_by_id_is_enriched():
    """GET /audit/{id} carries the same alias enrichment as the list endpoint."""
    audit_row = make_log(
        id="l1",
        table_name="LiteLLM_VerificationToken",
        object_id="gone-hash",
        action="deleted",
        before_value={"key_alias": "deleted-key"},
        changed_by="admin-user",
        changed_by_api_key="hash-admin",
    )
    db = FakeDb(
        audit_logs=[audit_row],
        keys=[SimpleNamespace(token="hash-admin", key_alias="admin-key")],
        users=[SimpleNamespace(user_id="admin-user", user_alias=None, user_email="admin@example.com")],
    )
    client = _client_for(db)

    with patch("litellm.proxy.proxy_server.prisma_client", FakePrismaClient(db)):
        response = client.get("/audit/l1")

    assert response.status_code == 200
    body = response.json()
    assert body["object_alias"] == "deleted-key"
    assert body["changed_by_user_email"] == "admin@example.com"
    assert body["changed_by_key_alias"] == "admin-key"
