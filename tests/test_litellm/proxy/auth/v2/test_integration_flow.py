"""End-to-end auth_v2 flow through the real FastAPI dependency.

Exercises flag dispatch -> user_api_key_auth -> v2 entry -> authenticator chain ->
casbin authorization -> 403/200, plus live policy CRUD, with an in-memory Prisma
stand-in. No running proxy or provider keys: the allow/deny decision happens at
auth time, before any model call.
"""

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

import litellm.proxy.proxy_server as ps
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.auth.v2.authz import policy_store
from litellm.proxy.auth.v2.management_endpoints import router as auth_v2_router

MASTER_KEY = "sk-master-1234"


class _Row:
    def __init__(self, data):
        self.ptype = data["ptype"]
        for i in range(6):
            setattr(self, f"v{i}", data.get(f"v{i}"))


class _CasbinTable:
    def __init__(self):
        self.rows = []

    async def create(self, data):
        self.rows.append(_Row(data))
        return data

    async def find_many(self):
        return list(self.rows)

    async def delete_many(self, where):
        before = len(self.rows)
        self.rows = [
            r
            for r in self.rows
            if not all(getattr(r, k, None) == v for k, v in where.items())
        ]
        return before - len(self.rows)


class _MockPrisma:
    def __init__(self):
        self.db = type("_DB", (), {"litellm_casbinrule": _CasbinTable()})()


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(ps, "general_settings", {"auth_version": "v2"})
    monkeypatch.setattr(ps, "master_key", MASTER_KEY)
    monkeypatch.setattr(ps, "prisma_client", _MockPrisma())
    policy_store.reset_cache()

    # Non-admin virtual keys resolve to a fixed identity (no DB key row needed).
    async def fake_get_key_object(hashed_token, **kwargs):
        return UserAPIKeyAuth(user_id="u_test", token=hashed_token)

    monkeypatch.setattr(
        "litellm.proxy.auth.auth_checks.get_key_object", fake_get_key_object
    )

    app = FastAPI()
    app.include_router(auth_v2_router)

    @app.post("/model/new")
    async def create_model(auth: UserAPIKeyAuth = Depends(user_api_key_auth)):
        return {"user_id": auth.user_id}

    @app.post("/chat/completions")
    async def chat(auth: UserAPIKeyAuth = Depends(user_api_key_auth)):
        return {"user_id": auth.user_id}

    yield TestClient(app, raise_server_exceptions=False)
    policy_store.reset_cache()


def _h(key):
    return {"Authorization": f"Bearer {key}"}


def test_master_key_is_admin_and_passes_governed_route(client):
    # Full chain: flag -> v2 entry -> master-key authn -> proxy_admin -> bootstrap
    # policy allows -> 200.
    r = client.post("/model/new", headers=_h(MASTER_KEY))
    assert r.status_code == 200


def test_non_admin_without_policy_is_denied(client):
    # u_test has no casbin grant for model:write -> 403.
    r = client.post("/model/new", headers=_h("sk-user-test"))
    assert r.status_code == 403


def test_granting_a_role_then_assigning_it_unlocks_the_route(client):
    # 1. Admin grants role:writer write on models, assigns it to u_test.
    add_perm = client.post(
        "/auth/v2/policy/permission/add",
        headers=_h(MASTER_KEY),
        json={"role": "writer", "resource": "model", "action": "write"},
    )
    assert add_perm.status_code == 200
    add_assign = client.post(
        "/auth/v2/policy/assignment/add",
        headers=_h(MASTER_KEY),
        json={"subject_type": "user", "subject_id": "u_test", "role": "writer"},
    )
    assert add_assign.status_code == 200

    # 2. u_test can now write models (cache was reset on the policy write).
    r = client.post("/model/new", headers=_h("sk-user-test"))
    assert r.status_code == 200


def test_non_admin_cannot_administer_policies(client):
    # The _require_admin gate, over the wire.
    r = client.post(
        "/auth/v2/policy/permission/add",
        headers=_h("sk-user-test"),
        json={"role": "writer", "resource": "model", "action": "write"},
    )
    assert r.status_code == 403


def test_model_call_requires_call_permission(client):
    # Inference is the `call` action on model:<id>. No grant -> 403.
    denied = client.post(
        "/chat/completions", headers=_h("sk-user-test"), json={"model": "gpt-4o"}
    )
    assert denied.status_code == 403

    # Grant call on gpt-* to u_test, then it is allowed.
    client.post(
        "/auth/v2/policy/permission/add",
        headers=_h(MASTER_KEY),
        json={
            "role": "caller",
            "resource": "model",
            "action": "call",
            "resource_id": "gpt-*",
        },
    )
    client.post(
        "/auth/v2/policy/assignment/add",
        headers=_h(MASTER_KEY),
        json={"subject_type": "user", "subject_id": "u_test", "role": "caller"},
    )
    allowed = client.post(
        "/chat/completions", headers=_h("sk-user-test"), json={"model": "gpt-4o"}
    )
    assert allowed.status_code == 200
