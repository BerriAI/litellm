"""Regression tests for the backend admin teams router (``backend/routers/teams.py``).

These pin the two properties the router is responsible for: every route is gated
through the ``auth_v2`` ``AuthSecurity`` Security layer (role-checked, not the
legacy ``user_api_key_auth``), and team CRUD plus membership round-trips through
the injected identity store.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from scim2_models import Group as ScimGroup

# backend/ lives at the repo root, not inside litellm/.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from backend.routers.teams import router as teams_router  # noqa: E402

from litellm.proxy.auth_v2 import errors  # noqa: E402
from litellm.proxy.auth_v2.authenticators import APIKeyAuthenticator  # noqa: E402
from litellm.proxy.auth_v2.config import ApiKeySchemeConfig, AuthConfig  # noqa: E402
from litellm.proxy.auth_v2.authorization import Role  # noqa: E402
from litellm.proxy.auth_v2.models import (  # noqa: E402
    AuthMethod,
    Credential,
    Principal,
    PrincipalType,
)
from litellm.proxy.auth_v2.security import AuthSecurity  # noqa: E402
from litellm.proxy.auth_v2.utils import hash_api_key  # noqa: E402

ADMIN_KEY = "sk-admin"
READER_KEY = "sk-reader"


def _principal(subject: str, roles: List[Role]) -> Principal:
    return Principal(
        principal_type=PrincipalType.HUMAN,
        subject=subject,
        auth_method=AuthMethod.API_KEY,
        roles=roles,
    )


class _FakeStore:
    """IdentityResolver + team-group ProvisioningStore backed by a dict.

    Resolution maps API keys to fully-formed Principals (so role gating can be
    driven directly), and the group methods store the exact ScimGroup handed in
    so membership round-trips without a database.
    """

    def __init__(self, principals: Dict[str, Principal]) -> None:
        self._by_key = principals
        self._groups: Dict[str, ScimGroup] = {}
        self._seq = 0

    async def resolve(self, credential: Credential) -> Principal:
        raw = credential.claims.get("_raw_api_key")
        principal = (
            self._by_key.get(hash_api_key(raw)) if isinstance(raw, str) else None
        )
        if principal is None:
            raise errors.invalid_token()
        return principal.model_copy()

    async def upsert_group(self, group: ScimGroup) -> ScimGroup:
        if not group.id:
            self._seq += 1
            group.id = f"team-{self._seq}"
        self._groups[group.id] = group
        return group

    async def get_group(self, resource_id: str) -> Optional[ScimGroup]:
        return self._groups.get(resource_id)

    async def delete_group(self, resource_id: str) -> None:
        self._groups.pop(resource_id, None)

    async def list_groups(self, filter_expr: Optional[str]) -> List[ScimGroup]:
        return list(self._groups.values())


@pytest.fixture
def client() -> TestClient:
    resolver = _FakeStore(
        {
            hash_api_key(ADMIN_KEY): _principal("admin", [Role.ORG_ADMIN]),
            hash_api_key(READER_KEY): _principal("reader", []),
        }
    )
    auth = AuthSecurity(
        AuthConfig(),
        resolver,
        authenticators=[APIKeyAuthenticator(ApiKeySchemeConfig())],
    )
    app = FastAPI()
    app.state.auth_v2 = auth
    app.include_router(teams_router)
    return TestClient(app)


def _admin(headers: Optional[dict] = None) -> dict:
    return {"x-litellm-api-key": ADMIN_KEY, **(headers or {})}


def test_create_requires_authentication(client):
    response = client.post("/admin/teams", json={"name": "eng"})
    assert response.status_code == 401
    assert response.json()["detail"] == "Not authenticated"


def test_create_rejects_unknown_key(client):
    response = client.post(
        "/admin/teams",
        json={"name": "eng"},
        headers={"x-litellm-api-key": "sk-bogus"},
    )
    assert response.status_code == 401


def test_create_denied_without_admin_role(client):
    response = client.post(
        "/admin/teams",
        json={"name": "eng"},
        headers={"x-litellm-api-key": READER_KEY},
    )
    assert response.status_code == 403
    assert response.json()["detail"] == "Insufficient role"


def test_create_then_get_round_trips_membership(client):
    created = client.post(
        "/admin/teams",
        json={"name": "eng", "members": ["u-1", "u-2"]},
        headers=_admin(),
    )
    assert created.status_code == 201
    body = created.json()
    assert body["name"] == "eng"
    assert body["members"] == ["u-1", "u-2"]
    team_id = body["id"]

    fetched = client.get(f"/admin/teams/{team_id}", headers=_admin())
    assert fetched.status_code == 200
    assert fetched.json() == {"id": team_id, "name": "eng", "members": ["u-1", "u-2"]}


def test_list_returns_created_team(client):
    client.post("/admin/teams", json={"name": "eng"}, headers=_admin())
    listed = client.get("/admin/teams", headers=_admin())
    assert listed.status_code == 200
    assert [team["name"] for team in listed.json()] == ["eng"]


def test_update_replaces_membership(client):
    team_id = client.post(
        "/admin/teams", json={"name": "eng", "members": ["u-1"]}, headers=_admin()
    ).json()["id"]

    updated = client.put(
        f"/admin/teams/{team_id}",
        json={"name": "eng", "members": ["u-2", "u-3"]},
        headers=_admin(),
    )
    assert updated.status_code == 200
    assert updated.json()["members"] == ["u-2", "u-3"]


def test_delete_removes_team(client):
    team_id = client.post(
        "/admin/teams", json={"name": "eng"}, headers=_admin()
    ).json()["id"]

    assert client.delete(f"/admin/teams/{team_id}", headers=_admin()).status_code == 204
    assert client.get(f"/admin/teams/{team_id}", headers=_admin()).status_code == 404


def test_delete_missing_team_is_404(client):
    assert client.delete("/admin/teams/nope", headers=_admin()).status_code == 404
