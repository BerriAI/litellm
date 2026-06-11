from __future__ import annotations

from typing import Annotated, Any, Tuple

import pytest
from fastapi import FastAPI, Security
from fastapi.testclient import TestClient

from litellm.auth_v2.authenticators import (
    ApiKeyAuthenticator,
    HttpAuthenticator,
    JwtVerifier,
)
from litellm.auth_v2.config import (
    ApiKeySchemeConfig,
    AuthConfig,
    HttpBasicConfig,
    OidcProviderConfig,
)
from litellm.auth_v2.models import AuthMethod, Principal, PrincipalType
from litellm.auth_v2.rbac import Role
from litellm.auth_v2.resolver import InMemoryIdentityStore, _hash_api_key
from litellm.auth_v2.security import (
    AuthContext,
    get_current_principal,
    require_roles,
)

from auth_v2_helpers import TEST_AUDIENCE, TEST_ISSUER, FakeJwksClient

ADMIN_KEY = "sk-admin-key"
READER_KEY = "sk-reader-key"
NOSCOPE_KEY = "sk-noscope-key"


def _principal(subject: str, *, scopes=None, roles=None) -> Principal:
    return Principal(
        principal_type=PrincipalType.HUMAN,
        subject=subject,
        auth_method=AuthMethod.API_KEY,
        scopes=scopes or [],
        roles=roles or [],
    )


def _build_app(public_key: Any) -> Tuple[FastAPI, InMemoryIdentityStore]:
    verifier = JwtVerifier(
        OidcProviderConfig(issuer=TEST_ISSUER, audience=[TEST_AUDIENCE]),
        jwks_client=FakeJwksClient(public_key),
    )
    authenticators = [
        ApiKeyAuthenticator(ApiKeySchemeConfig()),
        HttpAuthenticator(HttpBasicConfig(), [verifier]),
    ]
    resolver = InMemoryIdentityStore(
        api_keys={
            _hash_api_key(ADMIN_KEY): _principal(
                "admin-principal", scopes=["models:read"], roles=[Role.ORG_ADMIN]
            ),
            _hash_api_key(READER_KEY): _principal(
                "reader-principal", scopes=["models:read"]
            ),
            _hash_api_key(NOSCOPE_KEY): _principal("noscope-principal"),
        }
    )
    ctx = AuthContext(AuthConfig(), authenticators, resolver)

    app = FastAPI()
    app.state.auth_v2 = ctx

    @app.get("/open")
    async def open_route(
        principal: Annotated[Principal, Security(get_current_principal)],
    ):
        return {
            "subject": principal.subject,
            "auth_method": principal.auth_method.value,
            "network_host": principal.network.host,
        }

    @app.get("/scoped")
    async def scoped_route(
        principal: Annotated[
            Principal, Security(get_current_principal, scopes=["models:read"])
        ],
    ):
        return {"subject": principal.subject}

    @app.get("/admin")
    async def admin_route(
        principal: Annotated[Principal, Security(require_roles(Role.ORG_ADMIN))],
    ):
        return {"subject": principal.subject}

    return app, resolver


@pytest.fixture
def client(rsa_keypair) -> TestClient:
    _, public_key = rsa_keypair
    app, _ = _build_app(public_key)
    return TestClient(app)


def _bearer(token_factory, **claims) -> dict:
    return {"Authorization": f"Bearer {token_factory.mint(**claims)}"}


# --------------------------------------------------------------------------- #
# Missing credential
# --------------------------------------------------------------------------- #


def test_no_credential_returns_401_with_challenge(client):
    response = client.get("/open")
    assert response.status_code == 401
    assert "WWW-Authenticate" in response.headers
    assert "Bearer" in response.headers["WWW-Authenticate"]


# --------------------------------------------------------------------------- #
# Single-scheme success + network wiring
# --------------------------------------------------------------------------- #


def test_valid_api_key_authenticates(client):
    response = client.get("/open", headers={"x-litellm-api-key": READER_KEY})
    assert response.status_code == 200
    body = response.json()
    assert body["subject"] == "reader-principal"
    assert body["network_host"]  # resolve_network_context wired into principal


def test_valid_bearer_authenticates_from_claims(client, token_factory):
    response = client.get(
        "/open", headers=_bearer(token_factory, subject="jwt-user", scope="models:read")
    )
    assert response.status_code == 200
    assert response.json()["subject"] == "jwt-user"


# --------------------------------------------------------------------------- #
# OR precedence: first match wins, present-but-invalid fails fast
# --------------------------------------------------------------------------- #


def test_first_match_wins_api_key_before_bearer(client, token_factory):
    response = client.get(
        "/open",
        headers={
            "x-litellm-api-key": READER_KEY,
            "Authorization": f"Bearer {token_factory.mint(subject='jwt-user')}",
        },
    )
    assert response.status_code == 200
    # api key is earlier in scheme_order, so it resolves; bearer is never consulted
    assert response.json()["subject"] == "reader-principal"


def test_present_but_invalid_api_key_does_not_fall_through_to_bearer(
    client, token_factory
):
    response = client.get(
        "/open",
        headers={
            "x-litellm-api-key": "sk-totally-unknown",
            "Authorization": f"Bearer {token_factory.mint(subject='jwt-user', scope='models:read')}",
        },
    )
    assert response.status_code == 401
    assert "jwt-user" not in response.text


# --------------------------------------------------------------------------- #
# Scope enforcement (RFC 6750: 403 insufficient_scope)
# --------------------------------------------------------------------------- #


def test_scope_satisfied_returns_200(client):
    response = client.get("/scoped", headers={"x-litellm-api-key": READER_KEY})
    assert response.status_code == 200


def test_missing_scope_returns_403_insufficient_scope(client):
    response = client.get("/scoped", headers={"x-litellm-api-key": NOSCOPE_KEY})
    assert response.status_code == 403
    assert "insufficient_scope" in response.headers.get("WWW-Authenticate", "")


# --------------------------------------------------------------------------- #
# Role enforcement
# --------------------------------------------------------------------------- #


def test_required_role_present_returns_200(client):
    response = client.get("/admin", headers={"x-litellm-api-key": ADMIN_KEY})
    assert response.status_code == 200
    assert response.json()["subject"] == "admin-principal"


def test_required_role_missing_returns_403(client):
    response = client.get("/admin", headers={"x-litellm-api-key": READER_KEY})
    assert response.status_code == 403
