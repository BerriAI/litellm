from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from litellm.proxy.auth_v2.config import AuthConfig
from litellm.proxy.auth_v2.models import AuthMethod, Principal, PrincipalType
from litellm.proxy.auth_v2.resolver import InMemoryIdentityStore, _hash_api_key
from litellm.proxy.auth_v2.security import install_auth

USER_SCHEMA = "urn:ietf:params:scim:schemas:core:2.0:User"
GROUP_SCHEMA = "urn:ietf:params:scim:schemas:core:2.0:Group"
ERROR_SCHEMA = "urn:ietf:params:scim:api:messages:2.0:Error"

SCIM_KEY = "sk-scim-writer"
NOSCOPE_KEY = "sk-no-scim-scope"


def _principal(subject: str, scopes: list) -> Principal:
    return Principal(
        principal_type=PrincipalType.HUMAN,
        subject=subject,
        auth_method=AuthMethod.API_KEY,
        scopes=scopes,
    )


def _app() -> FastAPI:
    app = FastAPI()
    install_auth(
        app,
        AuthConfig(),
        InMemoryIdentityStore(
            api_keys={
                _hash_api_key(SCIM_KEY): _principal("scim-writer", ["scim:write"]),
                _hash_api_key(NOSCOPE_KEY): _principal("no-scope", []),
            }
        ),
        mount_scim=True,
        mount_oidc=False,
        mount_saml=False,
    )
    return app


@pytest.fixture
def client() -> TestClient:
    # SCIM routes require scim:write; authenticate every request with a scoped key
    return TestClient(_app(), headers={"x-litellm-api-key": SCIM_KEY})


def _create_user(client: TestClient, user_name="alice@example.com", display="Alice"):
    return client.post(
        "/scim/v2/Users",
        json={"schemas": [USER_SCHEMA], "userName": user_name, "displayName": display},
    )


def test_create_user_returns_201_with_id(client):
    response = _create_user(client)
    assert response.status_code == 201
    body = response.json()
    assert body["id"]
    assert body["userName"] == "alice@example.com"
    assert USER_SCHEMA in body["schemas"]


def test_get_user_round_trips(client):
    user_id = _create_user(client).json()["id"]
    response = client.get(f"/scim/v2/Users/{user_id}")
    assert response.status_code == 200
    assert response.json()["userName"] == "alice@example.com"


def test_get_unknown_user_returns_scim_404(client):
    response = client.get("/scim/v2/Users/does-not-exist")
    assert response.status_code == 404
    assert ERROR_SCHEMA in response.json()["schemas"]


def test_patch_replace_display_name(client):
    user_id = _create_user(client).json()["id"]
    response = client.patch(
        f"/scim/v2/Users/{user_id}",
        json={
            "schemas": ["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
            "Operations": [
                {"op": "replace", "path": "displayName", "value": "Alice B"}
            ],
        },
    )
    assert response.status_code == 200
    assert response.json()["displayName"] == "Alice B"
    # persisted
    assert client.get(f"/scim/v2/Users/{user_id}").json()["displayName"] == "Alice B"


def test_list_users_returns_list_response(client):
    _create_user(client, user_name="a@example.com")
    _create_user(client, user_name="b@example.com")
    response = client.get("/scim/v2/Users")
    assert response.status_code == 200
    body = response.json()
    assert body["totalResults"] == 2
    user_names = {r["userName"] for r in body["Resources"]}
    assert user_names == {"a@example.com", "b@example.com"}


def test_deactivate_user_sets_active_false(client):
    user_id = _create_user(client).json()["id"]
    assert client.delete(f"/scim/v2/Users/{user_id}").status_code == 204
    assert client.get(f"/scim/v2/Users/{user_id}").json()["active"] is False


def test_malformed_user_returns_scim_400_error(client):
    # userName is required for a SCIM User creation request
    response = client.post(
        "/scim/v2/Users", json={"schemas": [USER_SCHEMA], "displayName": "No Username"}
    )
    assert response.status_code == 400
    body = response.json()
    assert ERROR_SCHEMA in body["schemas"]
    assert body["status"] == "400"


def test_group_membership_round_trips(client):
    response = client.post(
        "/scim/v2/Groups",
        json={
            "schemas": [GROUP_SCHEMA],
            "displayName": "Engineering",
            "members": [{"value": "user-1", "display": "Alice"}],
        },
    )
    assert response.status_code == 201
    group_id = response.json()["id"]

    fetched = client.get(f"/scim/v2/Groups/{group_id}").json()
    assert fetched["displayName"] == "Engineering"
    assert fetched["members"][0]["value"] == "user-1"


def test_delete_group_removes_it(client):
    group_id = client.post(
        "/scim/v2/Groups",
        json={"schemas": [GROUP_SCHEMA], "displayName": "Temp"},
    ).json()["id"]
    assert client.delete(f"/scim/v2/Groups/{group_id}").status_code == 204
    assert client.get(f"/scim/v2/Groups/{group_id}").status_code == 404


def test_service_provider_config_advertises_patch(client):
    response = client.get("/scim/v2/ServiceProviderConfig")
    assert response.status_code == 200
    assert response.json()["patch"]["supported"] is True


def test_resource_types_lists_user_and_group(client):
    response = client.get("/scim/v2/ResourceTypes")
    assert response.status_code == 200
    names = {r["name"] for r in response.json()["Resources"]}
    assert names == {"User", "Group"}


def test_schemas_endpoint_returns_user_and_group(client):
    response = client.get("/scim/v2/Schemas")
    assert response.status_code == 200
    assert response.json()["totalResults"] == 2


# --------------------------------------------------------------------------- #
# SCIM routes are gated by scim:write (design section 11)
# --------------------------------------------------------------------------- #


def test_scim_requires_authentication():
    unauth = TestClient(_app())
    response = unauth.post(
        "/scim/v2/Users",
        json={"schemas": [USER_SCHEMA], "userName": "x@example.com"},
    )
    assert response.status_code == 401
    assert "WWW-Authenticate" in response.headers


def test_scim_requires_scim_write_scope():
    underscoped = TestClient(_app(), headers={"x-litellm-api-key": NOSCOPE_KEY})
    response = underscoped.get("/scim/v2/Users")
    assert response.status_code == 403
    assert "insufficient_scope" in response.headers.get("WWW-Authenticate", "")
