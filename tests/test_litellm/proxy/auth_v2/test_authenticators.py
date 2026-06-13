from __future__ import annotations

import base64
import hashlib
import hmac
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.proxy.auth_v2.authenticators import (
    APIKeyAuthenticator,
    HttpAuthenticator,
    JWTVerifier,
    MutualTLSAuthenticator,
    OAuth2Authenticator,
    OIDCAuthenticator,
    build_authenticators,
    hash_basic_password,
)
from litellm.proxy.auth_v2.config import (
    ApiKeySchemeConfig,
    AuthConfig,
    HttpBasicConfig,
    MutualTLSConfig,
    OAuth2IntrospectionConfig,
    TrustedProxyConfig,
)
from litellm.proxy.auth_v2 import OIDCProviderConfig
from litellm.proxy.auth_v2.errors import AuthError
from litellm.proxy.auth_v2.models import AuthMethod

from auth_v2_helpers import (
    TEST_AUDIENCE,
    TEST_ISSUER,
    FakeJwksClient,
    make_request,
)


class _BasicAuthStore:
    """A minimal in-memory BasicAuthVerifier injected into HttpAuthenticator.

    Verifies passwords against the pbkdf2_sha256$iterations$salt$digest format
    produced by the production hash_basic_password helper."""

    def __init__(self, credentials: Dict[str, str]) -> None:
        self._credentials = credentials

    def verify(self, username: str, password: str) -> bool:
        stored = self._credentials.get(username)
        if stored is None:
            return False
        try:
            _algorithm, iterations, salt, expected = stored.split("$")
            candidate = hashlib.pbkdf2_hmac(
                "sha256", password.encode(), bytes.fromhex(salt), int(iterations)
            ).hex()
        except ValueError:
            return False
        return hmac.compare_digest(candidate, expected)


# --------------------------------------------------------------------------- #
# JWTVerifier: every RFC 7519 check must be enforced.
# --------------------------------------------------------------------------- #


def test_jwt_verifier_accepts_valid_token(jwt_verifier, token_factory):
    claims = jwt_verifier.verify(token_factory.mint(subject="alice", scope="a b"))
    assert claims["sub"] == "alice"
    assert claims["aud"] == TEST_AUDIENCE


def test_jwt_verifier_rejects_bad_signature(
    rsa_keypair, other_rsa_keypair, oidc_provider, token_factory
):
    _, public_key = rsa_keypair
    verifier = JWTVerifier(oidc_provider, jwks_client=FakeJwksClient(public_key))
    other_pem, _ = other_rsa_keypair
    forged = token_factory.mint(private_pem=other_pem)
    with pytest.raises(AuthError) as exc:
        verifier.verify(forged)
    assert exc.value.status_code == 401


def test_jwt_verifier_rejects_wrong_audience(jwt_verifier, token_factory):
    with pytest.raises(AuthError) as exc:
        jwt_verifier.verify(token_factory.mint(audience="some-other-app"))
    assert exc.value.status_code == 401


def test_jwt_verifier_rejects_wrong_issuer(jwt_verifier, token_factory):
    with pytest.raises(AuthError) as exc:
        jwt_verifier.verify(token_factory.mint(issuer="https://evil.example.com"))
    assert exc.value.status_code == 401


def test_jwt_verifier_rejects_expired(jwt_verifier, token_factory):
    with pytest.raises(AuthError) as exc:
        jwt_verifier.verify(token_factory.mint(expires_in=-30))
    assert exc.value.status_code == 401


def test_jwt_verifier_requires_exp_iss_aud(jwt_verifier, rsa_keypair):
    import jwt as pyjwt

    private_pem, _ = rsa_keypair
    # token deliberately missing exp/iss/aud
    token = pyjwt.encode({"sub": "x"}, private_pem, algorithm="RS256")
    with pytest.raises(AuthError) as exc:
        jwt_verifier.verify(token)
    assert exc.value.status_code == 401


def test_jwt_verifier_enforces_at_jwt_typ(jwt_verifier, token_factory):
    without_typ = token_factory.mint()
    with pytest.raises(AuthError):
        jwt_verifier.verify(without_typ, require_at_jwt=True)

    with_typ = token_factory.mint(headers={"typ": "at+jwt"})
    claims = jwt_verifier.verify(with_typ, require_at_jwt=True)
    assert claims["sub"] == "user-1"


# --------------------------------------------------------------------------- #
# APIKeyAuthenticator
# --------------------------------------------------------------------------- #


async def test_api_key_authenticator_extracts_header():
    auth = APIKeyAuthenticator(ApiKeySchemeConfig(header_name="x-litellm-api-key"))
    request = make_request(headers={"x-litellm-api-key": "sk-secret-value"})
    credential = await auth.authenticate(request)
    assert credential is not None
    assert credential.method == AuthMethod.API_KEY
    assert credential.subject == "sk-secret-value"
    assert credential.claims["_raw_api_key"] == "sk-secret-value"
    assert credential.credential_ref.key_id == "sk-secret-"


async def test_api_key_authenticator_returns_none_when_absent():
    auth = APIKeyAuthenticator(ApiKeySchemeConfig())
    assert await auth.authenticate(make_request()) is None


# --------------------------------------------------------------------------- #
# HttpAuthenticator (bearer-JWT + basic)
# --------------------------------------------------------------------------- #


def _http_auth(
    public_key: Any, *, basic: HttpBasicConfig = None, basic_verifier=None
) -> HttpAuthenticator:
    verifier = JWTVerifier(
        OIDCProviderConfig(issuer=TEST_ISSUER, audience=[TEST_AUDIENCE]),
        jwks_client=FakeJwksClient(public_key),
    )
    return HttpAuthenticator(
        basic or HttpBasicConfig(), [verifier], basic_verifier=basic_verifier
    )


async def test_http_bearer_valid_token_resolves_credential(rsa_keypair, token_factory):
    _, public_key = rsa_keypair
    auth = _http_auth(public_key)
    token = token_factory.mint(subject="bob", scope="models:read chat:write")
    request = make_request(headers={"authorization": f"Bearer {token}"})
    credential = await auth.authenticate(request)
    assert credential is not None
    assert credential.method == AuthMethod.BEARER_JWT
    assert credential.subject == "bob"
    assert credential.issuer == TEST_ISSUER
    assert credential.audience == [TEST_AUDIENCE]
    assert credential.scopes == ["models:read", "chat:write"]


async def test_http_bearer_present_but_invalid_fails_fast(rsa_keypair, token_factory):
    _, public_key = rsa_keypair
    auth = _http_auth(public_key)
    # issuer with no configured verifier -> must raise, not return None
    token = token_factory.mint(issuer="https://unconfigured.example.com")
    request = make_request(headers={"authorization": f"Bearer {token}"})
    with pytest.raises(AuthError) as exc:
        await auth.authenticate(request)
    assert exc.value.status_code == 401


async def test_http_no_authorization_header_returns_none(rsa_keypair):
    _, public_key = rsa_keypair
    assert await _http_auth(public_key).authenticate(make_request()) is None


async def test_http_basic_disabled_ignores_basic_scheme(rsa_keypair):
    _, public_key = rsa_keypair
    auth = _http_auth(public_key, basic=HttpBasicConfig(enabled=False))
    creds = base64.b64encode(b"alice:pw").decode()
    request = make_request(headers={"authorization": f"Basic {creds}"})
    assert await auth.authenticate(request) is None


def _basic_store() -> _BasicAuthStore:
    return _BasicAuthStore({"alice": hash_basic_password("supersecret")})


async def test_http_basic_verifies_correct_credentials(rsa_keypair):
    _, public_key = rsa_keypair
    auth = _http_auth(
        public_key, basic=HttpBasicConfig(enabled=True), basic_verifier=_basic_store()
    )
    creds = base64.b64encode(b"alice:supersecret").decode()
    request = make_request(headers={"authorization": f"Basic {creds}"})
    credential = await auth.authenticate(request)
    assert credential is not None
    assert credential.method == AuthMethod.HTTP_BASIC
    assert credential.subject == "alice"
    # the password must never be carried on the credential (leak regression)
    assert "_basic_password" not in credential.claims
    assert "supersecret" not in str(credential.claims)


async def test_http_basic_wrong_password_rejected(rsa_keypair):
    _, public_key = rsa_keypair
    auth = _http_auth(
        public_key, basic=HttpBasicConfig(enabled=True), basic_verifier=_basic_store()
    )
    creds = base64.b64encode(b"alice:WRONG").decode()
    request = make_request(headers={"authorization": f"Basic {creds}"})
    with pytest.raises(AuthError) as exc:
        await auth.authenticate(request)
    assert exc.value.status_code == 401


async def test_http_basic_unknown_user_rejected(rsa_keypair):
    _, public_key = rsa_keypair
    auth = _http_auth(
        public_key, basic=HttpBasicConfig(enabled=True), basic_verifier=_basic_store()
    )
    creds = base64.b64encode(b"mallory:supersecret").decode()
    request = make_request(headers={"authorization": f"Basic {creds}"})
    with pytest.raises(AuthError) as exc:
        await auth.authenticate(request)
    assert exc.value.status_code == 401


async def test_http_basic_without_verifier_fails_closed(rsa_keypair):
    # basic enabled but no verifier wired -> must never accept (fail closed)
    _, public_key = rsa_keypair
    auth = _http_auth(public_key, basic=HttpBasicConfig(enabled=True))
    creds = base64.b64encode(b"alice:supersecret").decode()
    request = make_request(headers={"authorization": f"Basic {creds}"})
    with pytest.raises(AuthError) as exc:
        await auth.authenticate(request)
    assert exc.value.status_code == 401


async def test_http_basic_malformed_payload_raises(rsa_keypair):
    _, public_key = rsa_keypair
    auth = _http_auth(public_key, basic=HttpBasicConfig(enabled=True))
    request = make_request(headers={"authorization": "Basic !!!not-base64!!!"})
    with pytest.raises(AuthError) as exc:
        await auth.authenticate(request)
    assert exc.value.status_code == 401


def test_http_challenge_advertises_basic_only_when_enabled(rsa_keypair):
    _, public_key = rsa_keypair
    assert "Basic" not in _http_auth(public_key).challenge()
    enabled = _http_auth(public_key, basic=HttpBasicConfig(enabled=True))
    assert "Basic" in enabled.challenge()
    assert "Bearer" in enabled.challenge()


def test_hash_basic_password_is_salted_and_verifiable():
    # the stored hash is never the plaintext, and re-hashing yields a fresh salt
    first = hash_basic_password("supersecret")
    second = hash_basic_password("supersecret")
    assert "supersecret" not in first
    assert first != second  # random salt per call

    store = _BasicAuthStore({"alice": first})
    assert store.verify("alice", "supersecret")
    assert not store.verify("alice", "supersecre")
    assert not store.verify("unknown", "supersecret")


# --------------------------------------------------------------------------- #
# OAuth2Authenticator (at+jwt enforcement + opaque token path)
# --------------------------------------------------------------------------- #


def _oauth2(public_key: Any) -> OAuth2Authenticator:
    verifier = JWTVerifier(
        OIDCProviderConfig(issuer=TEST_ISSUER, audience=[TEST_AUDIENCE]),
        jwks_client=FakeJwksClient(public_key),
    )
    return OAuth2Authenticator([verifier], introspection=None)


async def test_oauth2_rejects_jwt_without_at_jwt_typ(rsa_keypair, token_factory):
    _, public_key = rsa_keypair
    token = token_factory.mint()  # no typ header
    request = make_request(headers={"authorization": f"Bearer {token}"})
    with pytest.raises(AuthError) as exc:
        await _oauth2(public_key).authenticate(request)
    assert exc.value.status_code == 401


async def test_oauth2_accepts_at_jwt(rsa_keypair, token_factory):
    _, public_key = rsa_keypair
    token = token_factory.mint(headers={"typ": "at+jwt"}, subject="svc-1")
    request = make_request(headers={"authorization": f"Bearer {token}"})
    credential = await _oauth2(public_key).authenticate(request)
    assert credential is not None
    assert credential.subject == "svc-1"


async def test_oauth2_opaque_token_without_introspection_raises(rsa_keypair):
    _, public_key = rsa_keypair
    request = make_request(headers={"authorization": "Bearer opaque-not-a-jwt"})
    with pytest.raises(AuthError) as exc:
        await _oauth2(public_key).authenticate(request)
    assert exc.value.status_code == 401


async def test_oauth2_no_bearer_returns_none(rsa_keypair):
    _, public_key = rsa_keypair
    assert await _oauth2(public_key).authenticate(make_request()) is None


def _introspecting_oauth2(client_factory) -> OAuth2Authenticator:
    return OAuth2Authenticator(
        [],
        introspection=OAuth2IntrospectionConfig(
            introspection_endpoint="https://idp.example.com/introspect",
            client_id="rs-client",
            client_secret="rs-secret",
            subject_field="sub",
        ),
        client_factory=client_factory,
    )


def _introspection_client_factory(status_code: int, body: dict) -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = body
    client = MagicMock()
    client.post = AsyncMock(return_value=response)
    return MagicMock(return_value=client)


async def test_oauth2_opaque_token_introspects_active_to_credential():
    factory = _introspection_client_factory(
        200,
        {"active": True, "sub": "svc-9", "scope": "models:read tools:run", "aud": "rs"},
    )
    request = make_request(headers={"authorization": "Bearer opaque-xyz"})
    credential = await _introspecting_oauth2(factory).authenticate(request)

    assert credential is not None
    assert credential.method == AuthMethod.OAUTH2_INTROSPECTION
    assert credential.subject == "svc-9"
    assert credential.scopes == ["models:read", "tools:run"]
    assert credential.audience == ["rs"]

    client = factory.return_value
    _, kwargs = client.post.call_args
    assert kwargs["data"] == {"token": "opaque-xyz"}
    expected_basic = base64.b64encode(b"rs-client:rs-secret").decode()
    assert kwargs["headers"]["Authorization"] == f"Basic {expected_basic}"
    assert client.post.call_args.args[0] == "https://idp.example.com/introspect"


async def test_oauth2_introspection_inactive_token_raises():
    factory = _introspection_client_factory(200, {"active": False})
    request = make_request(headers={"authorization": "Bearer opaque-xyz"})
    with pytest.raises(AuthError) as exc:
        await _introspecting_oauth2(factory).authenticate(request)
    assert exc.value.status_code == 401


async def test_oauth2_introspection_non_200_raises():
    factory = _introspection_client_factory(500, {})
    request = make_request(headers={"authorization": "Bearer opaque-xyz"})
    with pytest.raises(AuthError) as exc:
        await _introspecting_oauth2(factory).authenticate(request)
    assert exc.value.status_code == 401


# --------------------------------------------------------------------------- #
# OIDCAuthenticator
# --------------------------------------------------------------------------- #


def _oidc(public_key: Any) -> OIDCAuthenticator:
    verifier = JWTVerifier(
        OIDCProviderConfig(issuer=TEST_ISSUER, audience=[TEST_AUDIENCE]),
        jwks_client=FakeJwksClient(public_key),
    )
    return OIDCAuthenticator([verifier])


async def test_oidc_valid_token_sets_oidc_method(rsa_keypair, token_factory):
    _, public_key = rsa_keypair
    token = token_factory.mint(subject="carol", email="carol@example.com")
    request = make_request(headers={"authorization": f"Bearer {token}"})
    credential = await _oidc(public_key).authenticate(request)
    assert credential is not None
    assert credential.method == AuthMethod.OIDC
    assert credential.subject == "carol"
    assert credential.claims["email"] == "carol@example.com"


async def test_oidc_unknown_issuer_raises(rsa_keypair, token_factory):
    _, public_key = rsa_keypair
    token = token_factory.mint(issuer="https://other.example.com")
    request = make_request(headers={"authorization": f"Bearer {token}"})
    with pytest.raises(AuthError):
        await _oidc(public_key).authenticate(request)


# --------------------------------------------------------------------------- #
# MutualTLSAuthenticator
# --------------------------------------------------------------------------- #


# make_request's default peer is 203.0.113.7; trust that /24 for the proxy path
_TRUSTED_NET = TrustedProxyConfig(trusted_proxy_cidrs=["203.0.113.0/24"])


def _mtls(config: MutualTLSConfig, network: TrustedProxyConfig = None):
    return MutualTLSAuthenticator(config, network or _TRUSTED_NET)


async def test_mtls_reads_forwarded_subject_header_from_trusted_peer():
    auth = _mtls(MutualTLSConfig(enabled=True, forwarded_subject_header="x-client-dn"))
    request = make_request(headers={"x-client-dn": "CN=svc-a,O=Co,C=US"})
    credential = await auth.authenticate(request)
    assert credential is not None
    assert credential.method == AuthMethod.MUTUAL_TLS
    assert credential.subject == "CN=svc-a,O=Co,C=US"
    assert credential.client_certificate.subject_dn == "CN=svc-a,O=Co,C=US"


async def test_mtls_forwarded_header_from_untrusted_peer_is_ignored():
    # spoofing guard: a client that is not a trusted proxy cannot forge the DN header
    auth = _mtls(MutualTLSConfig(enabled=True, forwarded_subject_header="x-client-dn"))
    request = make_request(
        headers={"x-client-dn": "CN=attacker"}, client=("8.8.8.8", 4444)
    )
    assert await auth.authenticate(request) is None


async def test_mtls_forwarded_header_gate_ignores_spoofed_xff():
    # the gate keys on the raw socket peer, not X-Forwarded-For: an untrusted peer
    # cannot claim a trusted address via XFF to smuggle a forged DN header
    auth = _mtls(MutualTLSConfig(enabled=True, forwarded_subject_header="x-client-dn"))
    request = make_request(
        headers={"x-client-dn": "CN=attacker", "x-forwarded-for": "10.0.0.5"},
        client=("8.8.8.8", 4444),
    )
    assert await auth.authenticate(request) is None


async def test_mtls_prefers_verified_asgi_cert_over_forwarded_header():
    # a genuinely verified client cert from the TLS layer wins over a proxy header
    auth = _mtls(MutualTLSConfig(enabled=True, forwarded_subject_header="x-client-dn"))
    request = make_request(
        headers={"x-client-dn": "CN=from-header"},
        client=("10.0.0.9", 1),
        scope_extra={"extensions": {"tls": {"client_cert_name": "CN=from-tls"}}},
    )
    credential = await auth.authenticate(request)
    assert credential is not None
    assert credential.subject == "CN=from-tls"


async def test_mtls_forwarded_header_absent_returns_none():
    auth = _mtls(MutualTLSConfig(enabled=True, forwarded_subject_header="x-client-dn"))
    assert await auth.authenticate(make_request()) is None


async def test_mtls_reads_asgi_tls_extension():
    auth = _mtls(MutualTLSConfig(enabled=True))
    request = make_request(
        scope_extra={"extensions": {"tls": {"client_cert_name": "CN=from-asgi"}}}
    )
    credential = await auth.authenticate(request)
    assert credential is not None
    assert credential.subject == "CN=from-asgi"


async def test_mtls_no_cert_returns_none():
    auth = _mtls(MutualTLSConfig(enabled=True))
    assert await auth.authenticate(make_request()) is None


# --------------------------------------------------------------------------- #
# build_authenticators: ordering and inclusion follow config
# --------------------------------------------------------------------------- #


def test_build_authenticators_follows_scheme_order():
    config = AuthConfig()
    types = [type(a) for a in build_authenticators(config)]
    # mutual_tls disabled by default -> excluded
    assert types == [
        APIKeyAuthenticator,
        HttpAuthenticator,
        OIDCAuthenticator,
        OAuth2Authenticator,
    ]


def test_build_authenticators_omits_api_key_when_unconfigured():
    config = AuthConfig(api_key=None)
    types = [type(a) for a in build_authenticators(config)]
    assert APIKeyAuthenticator not in types


def test_build_authenticators_includes_mtls_when_enabled():
    config = AuthConfig(mutual_tls=MutualTLSConfig(enabled=True))
    types = [type(a) for a in build_authenticators(config)]
    assert types[-1] is MutualTLSAuthenticator


# --------------------------------------------------------------------------- #
# H1: token role claims are gated by the provider allowlist (privilege escalation)
# --------------------------------------------------------------------------- #


def _bearer_roles(public_key, token_factory, roles, **provider_kwargs):
    provider = OIDCProviderConfig(
        issuer=TEST_ISSUER, audience=[TEST_AUDIENCE], **provider_kwargs
    )
    auth = HttpAuthenticator(
        HttpBasicConfig(),
        [JWTVerifier(provider, jwks_client=FakeJwksClient(public_key))],
    )
    token = token_factory.mint(roles=roles)
    return auth, make_request(headers={"authorization": f"Bearer {token}"})


async def test_token_roles_are_dropped_without_allowlist(rsa_keypair, token_factory):
    _, public_key = rsa_keypair
    auth, request = _bearer_roles(
        public_key, token_factory, ["platform_admin", "org_admin"]
    )
    credential = await auth.authenticate(request)
    # default allowed_roles=[] -> a self-asserted role grants nothing
    assert credential.claims["roles"] == []


async def test_token_roles_filtered_to_allowlist(rsa_keypair, token_factory):
    _, public_key = rsa_keypair
    auth, request = _bearer_roles(
        public_key,
        token_factory,
        ["platform_admin", "org_admin"],
        allowed_roles=["org_admin"],
    )
    credential = await auth.authenticate(request)
    # org_admin is allowed; platform_admin is dropped (and not in the allowlist anyway)
    assert credential.claims["roles"] == ["org_admin"]


async def test_platform_role_requires_explicit_gate(rsa_keypair, token_factory):
    _, public_key = rsa_keypair
    gated_auth, gated_req = _bearer_roles(
        public_key, token_factory, ["platform_admin"], allowed_roles=["platform_admin"]
    )
    # allowed but platform gate off -> still dropped
    assert (await gated_auth.authenticate(gated_req)).claims["roles"] == []

    open_auth, open_req = _bearer_roles(
        public_key,
        token_factory,
        ["platform_admin"],
        allowed_roles=["platform_admin"],
        allow_platform_roles=True,
    )
    assert (await open_auth.authenticate(open_req)).claims["roles"] == [
        "platform_admin"
    ]
