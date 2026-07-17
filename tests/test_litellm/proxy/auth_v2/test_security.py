from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pytest
from fastapi import FastAPI, Request, Security
from fastapi.security import SecurityScopes
from fastapi.testclient import TestClient

from litellm.proxy.auth_v2.authenticators import (
    APIKeyAuthenticator,
    Carrier,
    CredentialLocation,
    HttpAuthenticator,
    JWTVerifier,
)
from litellm.proxy.auth_v2.config import (
    ApiKeySchemeConfig,
    AuthConfig,
    HttpBasicConfig,
)
from litellm.proxy.auth_v2 import OIDCProviderConfig, errors
from litellm.proxy.auth_v2.models import (
    AuthMethod,
    Credential,
    Principal,
    PrincipalType,
    SecuritySchemeType,
)
from litellm.proxy.auth_v2.authorization import RBACEngine, Role
from litellm.proxy.auth_v2.utils import hash_api_key
from litellm.proxy.auth_v2.security import AuthSecurity

from auth_v2_helpers import TEST_AUDIENCE, TEST_ISSUER, FakeJwksClient

ADMIN_KEY = "sk-admin-key"
READER_KEY = "sk-reader-key"
NOSCOPE_KEY = "sk-noscope-key"
PLATFORM_ADMIN_KEY = "sk-platform-admin-key"
PLATFORM_VIEWER_KEY = "sk-platform-viewer-key"


def _principal(subject: str, *, scopes=None, roles=None) -> Principal:
    return Principal(
        principal_type=PrincipalType.HUMAN,
        subject=subject,
        auth_method=AuthMethod.API_KEY,
        scopes=scopes or [],
        roles=roles or [],
    )


class _FakeResolver:
    """Resolver double for the security-layer tests.

    These tests inject fully-formed Principals (arbitrary scopes/roles) keyed by
    API key, which the production DbResolver cannot express; DbResolver
    has its own coverage in test_resolver.py. An API-key credential is looked up
    by its raw-key claim; anything else echoes the credential's subject. Returns a
    fresh Principal per the Resolver contract.
    """

    def __init__(self, by_key: Dict[str, Principal]) -> None:
        self._by_key = by_key

    async def resolve(self, credential: Credential) -> Principal:
        raw = credential.claims.get("_raw_api_key")
        if isinstance(raw, str):
            principal = self._by_key.get(hash_api_key(raw))
            if principal is None:
                raise errors.invalid_token()
            return principal.model_copy()
        return Principal(
            principal_type=PrincipalType.HUMAN,
            subject=credential.subject,
            auth_method=credential.method,
            scopes=list(credential.scopes),
        )


def _build_app(
    public_key: Any, *, rbac: RBACEngine = None
) -> Tuple[FastAPI, _FakeResolver]:
    verifier = JWTVerifier(
        OIDCProviderConfig(issuer=TEST_ISSUER, audience=[TEST_AUDIENCE]),
        jwks_client=FakeJwksClient(public_key),
    )
    authenticators = [
        APIKeyAuthenticator(ApiKeySchemeConfig()),
        HttpAuthenticator(HttpBasicConfig(), [verifier]),
    ]
    resolver = _FakeResolver(
        {
            hash_api_key(ADMIN_KEY): _principal(
                "admin-principal", scopes=["models:read"], roles=[Role.ORG_ADMIN]
            ),
            hash_api_key(READER_KEY): _principal(
                "reader-principal", scopes=["models:read"]
            ),
            hash_api_key(NOSCOPE_KEY): _principal("noscope-principal"),
            hash_api_key(PLATFORM_ADMIN_KEY): _principal(
                "platform-admin-principal", roles=[Role.PLATFORM_ADMIN]
            ),
            hash_api_key(PLATFORM_VIEWER_KEY): _principal(
                "platform-viewer-principal", roles=[Role.PLATFORM_VIEWER]
            ),
        }
    )
    auth = AuthSecurity(
        AuthConfig(), resolver, authorizer=rbac, authenticators=authenticators
    )

    app = FastAPI()

    # default-value Security() style: the dependency marker stays a real object even
    # under `from __future__ import annotations`, where Annotated[...] would be a string
    # that FastAPI re-evaluates in module globals (the closure-local `auth` is invisible)
    @app.get("/open")
    async def open_route(principal: Principal = Security(auth.principal)):
        return {
            "subject": principal.subject,
            "auth_method": principal.auth_method.value,
            "network_host": principal.network.host,
        }

    @app.get("/scoped")
    async def scoped_route(
        principal: Principal = Security(auth.principal, scopes=["models:read"]),
    ):
        return {"subject": principal.subject}

    @app.get("/admin")
    async def admin_route(
        principal: Principal = Security(auth.require_roles(Role.ORG_ADMIN)),
    ):
        return {"subject": principal.subject}

    @app.post("/perm-widgets")
    async def widgets_route(
        principal: Principal = Security(auth.require_permission("/widgets", "POST")),
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


def test_required_role_honors_hierarchy(client):
    # platform_admin inherits org_admin via the Casbin g-rules, so it passes a
    # require_roles(ORG_ADMIN) gate without holding org_admin explicitly
    response = client.get("/admin", headers={"x-litellm-api-key": PLATFORM_ADMIN_KEY})
    assert response.status_code == 200
    assert response.json()["subject"] == "platform-admin-principal"


# --------------------------------------------------------------------------- #
# Permission enforcement (require_permission -> RBACEngine.enforce)
# --------------------------------------------------------------------------- #


def test_require_permission_allows_platform_admin(client):
    response = client.post(
        "/perm-widgets", headers={"x-litellm-api-key": PLATFORM_ADMIN_KEY}
    )
    assert response.status_code == 200


def test_require_permission_denies_viewer_on_write(client):
    # platform_viewer is GET-only in the default policy -> POST /widgets is denied
    response = client.post(
        "/perm-widgets", headers={"x-litellm-api-key": PLATFORM_VIEWER_KEY}
    )
    assert response.status_code == 403
    assert response.json()["detail"] == "Forbidden"


def test_require_permission_unauthenticated_returns_401(client):
    response = client.post("/perm-widgets")
    assert response.status_code == 401
    assert "WWW-Authenticate" in response.headers


def test_injected_rbac_engine_overrides_default_policy(rsa_keypair, tmp_path):
    # operator CSV grants only platform_viewer POST /widgets and drops the
    # built-in platform_admin "/*" grant; the injected engine governs enforce
    policy = tmp_path / "policy.csv"
    policy.write_text("p, platform_viewer, /widgets, POST\n")

    _, public_key = rsa_keypair
    app, _ = _build_app(public_key, rbac=RBACEngine(policy_path=str(policy)))
    client = TestClient(app)

    # viewer now passes, platform_admin (default grant removed) now fails
    assert (
        client.post(
            "/perm-widgets", headers={"x-litellm-api-key": PLATFORM_VIEWER_KEY}
        ).status_code
        == 200
    )
    assert (
        client.post(
            "/perm-widgets", headers={"x-litellm-api-key": PLATFORM_ADMIN_KEY}
        ).status_code
        == 403
    )


# --------------------------------------------------------------------------- #
# Carrier dispatch: only authenticators whose credential is present are run
# --------------------------------------------------------------------------- #


def _request(headers=None, cookies=None, tls=None) -> Request:
    raw = [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()]
    if cookies:
        cookie = "; ".join(f"{name}={value}" for name, value in cookies.items())
        raw.append((b"cookie", cookie.encode()))
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "query_string": b"",
        "headers": raw,
        "client": ("1.2.3.4", 0),
    }
    if tls is not None:
        scope["extensions"] = {"tls": tls}
    return Request(scope)


class _SpyAuthenticator:
    def __init__(self, carriers, subject):
        self._carriers = tuple(carriers)
        self._subject = subject
        self.calls = 0

    async def authenticate(self, request: Request) -> Optional[Credential]:
        self.calls += 1
        return Credential(
            scheme=SecuritySchemeType.API_KEY,
            method=AuthMethod.API_KEY,
            subject=self._subject,
        )

    def carriers(self) -> Sequence[Carrier]:
        return self._carriers

    def challenge(self) -> str:
        return "spy"


def _auth(authenticators: List[_SpyAuthenticator]) -> AuthSecurity:
    return AuthSecurity(AuthConfig(), _FakeResolver({}), authenticators=authenticators)


_BEARER = Carrier(CredentialLocation.AUTHORIZATION_SCHEME, "bearer")
_API_HEADER = Carrier(CredentialLocation.HEADER, "x-litellm-api-key")
_COOKIE = Carrier(CredentialLocation.COOKIE, "litellm_session")


def test_dispatch_runs_only_the_matching_authenticator():
    # the bearer authenticator is earlier in order, but the request carries only
    # an api key: dispatch must select the api-key authenticator and never touch
    # the bearer one
    bearer = _SpyAuthenticator([_BEARER], "bearer-subject")
    api_key = _SpyAuthenticator([_API_HEADER], "api-subject")
    auth = _auth([bearer, api_key])

    request = _request(headers={"x-litellm-api-key": "sk-x"})
    principal = asyncio.run(auth.principal(SecurityScopes(scopes=[]), request))

    assert principal.subject == "api-subject"
    assert api_key.calls == 1
    assert bearer.calls == 0


def test_first_claiming_authenticator_owns_a_shared_carrier():
    # bearer is read by several schemes; the earliest in scheme_order owns it and
    # is the only one ever consulted
    first = _SpyAuthenticator([_BEARER], "first")
    second = _SpyAuthenticator([_BEARER], "second")
    auth = _auth([first, second])

    request = _request(headers={"authorization": "Bearer abc"})
    principal = asyncio.run(auth.principal(SecurityScopes(scopes=[]), request))

    assert principal.subject == "first"
    assert first.calls == 1
    assert second.calls == 0


def test_untrusted_forwarded_cert_does_not_shadow_session():
    # a spoofed client-cert header from an untrusted peer must not be selected;
    # dispatch falls through to the valid session cookie
    cert = _SpyAuthenticator(
        [
            Carrier(
                CredentialLocation.CLIENT_CERTIFICATE,
                "x-forwarded-client-cert",
                ("9.9.9.9/32",),
            )
        ],
        "cert",
    )
    session = _SpyAuthenticator([_COOKIE], "session")
    auth = _auth([cert, session])

    request = _request(
        headers={"x-forwarded-client-cert": "CN=svc"},
        cookies={"litellm_session": "sid"},
    )
    principal = asyncio.run(auth.principal(SecurityScopes(scopes=[]), request))

    assert principal.subject == "session"
    assert cert.calls == 0
    assert session.calls == 1


def test_no_matching_carrier_is_unauthenticated():
    api_key = _SpyAuthenticator([_API_HEADER], "api-subject")
    auth = _auth([api_key])

    with pytest.raises(Exception) as exc_info:
        asyncio.run(auth.principal(SecurityScopes(scopes=[]), _request()))

    assert getattr(exc_info.value, "status_code", None) == 401
    assert api_key.calls == 0


def test_authorization_scheme_carrier_discriminates_bearer_from_basic():
    request = _request(headers={"authorization": "Bearer abc"})
    assert _BEARER.present(request) is True
    assert (
        Carrier(CredentialLocation.AUTHORIZATION_SCHEME, "basic").present(request)
        is False
    )


def test_header_carrier_requires_nonempty_value():
    assert _API_HEADER.present(_request(headers={"x-litellm-api-key": "sk"})) is True
    assert _API_HEADER.present(_request()) is False


def test_cookie_carrier_detects_named_cookie():
    assert _COOKIE.present(_request(cookies={"litellm_session": "sid"})) is True
    assert _COOKIE.present(_request(cookies={"other": "x"})) is False


def test_client_certificate_carrier_is_trusted_proxy_aware():
    direct = Carrier(CredentialLocation.CLIENT_CERTIFICATE, "x-forwarded-client-cert")
    assert direct.present(_request(tls={"client_cert_name": "CN=svc"})) is True

    # request client ip is 1.2.3.4; only a matching trusted CIDR accepts the header
    trusted = Carrier(
        CredentialLocation.CLIENT_CERTIFICATE,
        "x-forwarded-client-cert",
        ("1.2.3.4/32",),
    )
    untrusted = Carrier(
        CredentialLocation.CLIENT_CERTIFICATE,
        "x-forwarded-client-cert",
        ("9.9.9.9/32",),
    )
    header = {"x-forwarded-client-cert": "CN=svc"}
    assert trusted.present(_request(headers=header)) is True
    assert untrusted.present(_request(headers=header)) is False
    assert trusted.present(_request()) is False
