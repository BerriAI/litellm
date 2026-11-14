import threading
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from typing import Final

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from litellm.proxy.control_plane.auth.jwt import (
    SESSION_AUDIENCE,
    OidcTokenVerifier,
    PyJwkClientLookup,
    Rejected,
    SessionTokenCodec,
    TokenVerifier,
    master_key_principal,
)
from litellm.proxy.control_plane.auth.rbac import (
    ADMIN_ROLE,
    ALL_SCOPES,
    VIEWER_ROLE,
    Principal,
    RoleDefinition,
    RolePolicy,
    Scope,
    build_policy,
    build_principal,
    default_policy,
)
from litellm.proxy.control_plane.settings import OidcSettings

MASTER_KEY: Final = "sk-master-test"
ISSUER: Final = "https://idp.example.test/"
AUDIENCE: Final = "litellm-control-plane"
TOKEN_IAT: Final = 1_700_000_000
TOKEN_EXP: Final = 4_102_444_800
EXPIRED_TOKEN_EXP: Final = 1_600_000_000

_PRIVATE_KEY: Final = rsa.generate_private_key(public_exponent=65537, key_size=2048)
_PUBLIC_PEM: Final = (
    _PRIVATE_KEY.public_key()
    .public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
    .decode()
)
_PRIVATE_PEM: Final = _PRIVATE_KEY.private_bytes(
    serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
).decode()


class StaticKeys:
    def __init__(self, key: str | Rejected) -> None:
        self._key = key

    def signing_key(self, token: str) -> str | Rejected:  # str is a valid SigningKey
        return self._key


class StaticJwksClient:
    def get_signing_key_from_jwt(self, _token: str) -> str:
        return _PUBLIC_PEM


class _Clock:
    def __init__(self) -> None:
        self.current = 100.0

    def __call__(self) -> float:
        return self.current

    def advance(self, seconds: float) -> None:
        self.current += seconds


class _FetchCounter:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._calls = 0

    @property
    def count(self) -> int:
        with self._lock:
            return self._calls

    def record(self) -> int:
        with self._lock:
            self._calls += 1
            return self._calls


def _oidc_settings(**overrides: str) -> OidcSettings:
    return OidcSettings(
        issuer=ISSUER,
        audience=AUDIENCE,
        jwks_url=None,
        roles_claim=overrides.get("roles_claim", "roles"),
        scope_claim="scope",
        dataplanes_claim="litellm_dataplanes",
        email_claim="email",
        leeway_seconds=0,
    )


def _oidc_token(claims: dict[str, object]) -> str:
    return jwt.encode(
        {"iss": ISSUER, "aud": AUDIENCE, "sub": "user-1", "exp": TOKEN_EXP, "iat": TOKEN_IAT, **claims},
        _PRIVATE_PEM,
        algorithm="RS256",
    )


POLICY: Final = default_policy()


def _sessions(
    ttl: timedelta = timedelta(hours=1),
    master_key: str = MASTER_KEY,
    issuer: str = "cp",
    audience: str = SESSION_AUDIENCE,
) -> SessionTokenCodec:
    return SessionTokenCodec(master_key=master_key, issuer=issuer, ttl=ttl, policy=POLICY, audience=audience)


def _viewer(ids: frozenset[str]) -> Principal:
    return build_principal(POLICY, "alice", "a@x.test", frozenset({VIEWER_ROLE}), None, ids, "session")


def _oidc_verifier(settings: OidcSettings, key: str | Rejected) -> OidcTokenVerifier:
    return OidcTokenVerifier(settings, StaticKeys(key), POLICY)


def test_session_round_trip_preserves_subject_roles_scopes_and_binding() -> None:
    codec: Final = _sessions()
    original: Final = _viewer(frozenset({"eu", "us"}))
    verified: Final = codec.verify(codec.mint(original))
    assert isinstance(verified, Principal)
    assert (verified.subject, verified.email, verified.roles) == ("alice", "a@x.test", original.roles)
    assert verified.scopes == original.scopes
    assert verified.dataplanes == original.dataplanes
    assert verified.token_kind == "session"


def test_fleet_wide_session_round_trips_as_fleet_wide() -> None:
    codec: Final = _sessions()
    admin: Final = build_principal(POLICY, "root", None, frozenset({ADMIN_ROLE}), None, frozenset(), "session")
    verified: Final = codec.verify(codec.mint(admin))
    assert isinstance(verified, Principal) and verified.dataplanes.all


def test_expired_session_is_rejected() -> None:
    codec: Final = _sessions(ttl=timedelta(seconds=-1))
    assert isinstance(codec.verify(codec.mint(_viewer(frozenset()))), Rejected)


def test_session_signed_with_another_key_is_rejected() -> None:
    other: Final = _sessions(master_key="sk-other")
    assert isinstance(_sessions().verify(other.mint(_viewer(frozenset()))), Rejected)


def test_session_from_another_issuer_is_rejected() -> None:
    other: Final = _sessions(issuer="not-cp")
    assert isinstance(_sessions().verify(other.mint(_viewer(frozenset()))), Rejected)


def test_session_minted_for_another_audience_is_rejected() -> None:
    """Same master key and issuer, but a token for another component never opens the control plane."""
    other: Final = _sessions(audience="litellm-proxy-ui")
    assert isinstance(_sessions().verify(other.mint(_viewer(frozenset()))), Rejected)


def test_session_without_an_audience_claim_is_rejected() -> None:
    no_aud: Final = jwt.encode(
        {"iss": "cp", "sub": "alice", "roles": ["viewer"], "iat": TOKEN_IAT, "exp": TOKEN_EXP},
        MASTER_KEY,
        algorithm="HS256",
    )
    assert isinstance(_sessions().verify(no_aud), Rejected)
    with_aud: Final = jwt.encode(
        {
            "iss": "cp",
            "aud": SESSION_AUDIENCE,
            "sub": "alice",
            "roles": ["viewer"],
            "iat": TOKEN_IAT,
            "exp": TOKEN_EXP,
        },
        MASTER_KEY,
        algorithm="HS256",
    )
    assert isinstance(_sessions().verify(with_aud), Principal)


def test_oidc_scope_claim_with_only_foreign_scopes_grants_no_control_plane_scope() -> None:
    verifier: Final = _oidc_verifier(_oidc_settings(), _PUBLIC_PEM)
    scoped: Final = verifier.verify(_oidc_token({"roles": ["admin"], "scope": "openid profile email"}))
    assert isinstance(scoped, Principal) and scoped.scopes == frozenset()
    unscoped: Final = verifier.verify(_oidc_token({"roles": ["admin"]}))
    assert isinstance(unscoped, Principal) and unscoped.scopes == POLICY.scopes_for(ADMIN_ROLE)


def test_oidc_token_maps_claims_to_a_clamped_principal() -> None:
    verifier: Final = _oidc_verifier(_oidc_settings(), _PUBLIC_PEM)
    token: Final = _oidc_token(
        {
            "roles": ["viewer"],
            "scope": "openid usage:read proxy:write",
            "litellm_dataplanes": ["eu"],
            "email": "u@x.test",
        }
    )
    principal: Final = verifier.verify(token)
    assert isinstance(principal, Principal)
    assert principal.roles == frozenset({VIEWER_ROLE})
    assert principal.scopes == frozenset({Scope.USAGE_READ})
    assert principal.dataplanes.ids == frozenset({"eu"}) and not principal.dataplanes.all
    assert (principal.email, principal.token_kind) == ("u@x.test", "oidc")


@pytest.mark.parametrize("dataplanes", [1, ["eu", 1]])
def test_oidc_token_with_malformed_dataplanes_claim_is_rejected(dataplanes: object) -> None:
    verifier: Final = _oidc_verifier(_oidc_settings(), _PUBLIC_PEM)
    assert isinstance(verifier.verify(_oidc_token({"roles": ["admin"], "litellm_dataplanes": dataplanes})), Rejected)


def test_oidc_token_without_dataplanes_claim_keeps_fleet_wide_access() -> None:
    verifier: Final = _oidc_verifier(_oidc_settings(), _PUBLIC_PEM)
    principal: Final = verifier.verify(_oidc_token({"roles": ["admin"]}))
    assert isinstance(principal, Principal) and principal.dataplanes.all


def test_oidc_token_with_a_fixed_past_expiration_is_rejected() -> None:
    verifier: Final = _oidc_verifier(_oidc_settings(), _PUBLIC_PEM)
    assert isinstance(verifier.verify(_oidc_token({"roles": ["admin"], "exp": EXPIRED_TOKEN_EXP})), Rejected)


def test_oidc_roles_claim_name_is_configurable_and_space_delimited_roles_parse() -> None:
    verifier: Final = _oidc_verifier(_oidc_settings(roles_claim="groups"), _PUBLIC_PEM)
    principal: Final = verifier.verify(_oidc_token({"groups": "viewer other"}))
    assert isinstance(principal, Principal) and principal.roles == frozenset({VIEWER_ROLE})


@pytest.mark.parametrize(
    "claims",
    [
        {"roles": ["admin"], "aud": "someone-else"},
        {"roles": ["admin"], "iss": "https://other.test/"},
        {"roles": ["proxy_admin"]},
        {},
    ],
)
def test_oidc_token_with_wrong_audience_issuer_or_no_role_is_rejected(claims: dict[str, object]) -> None:
    verifier: Final = _oidc_verifier(_oidc_settings(), _PUBLIC_PEM)
    assert isinstance(verifier.verify(_oidc_token(claims)), Rejected)


def test_oidc_token_signed_by_an_unknown_key_is_rejected() -> None:
    other_key: Final = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    other_pem: Final = (
        other_key.public_key()
        .public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
        .decode()
    )
    verifier: Final = _oidc_verifier(_oidc_settings(), other_pem)
    assert isinstance(verifier.verify(_oidc_token({"roles": ["admin"]})), Rejected)


def test_oidc_token_is_rejected_when_the_jwks_lookup_fails() -> None:
    verifier: Final = _oidc_verifier(_oidc_settings(), Rejected("jwks down"))
    assert verifier.verify(_oidc_token({"roles": ["admin"]})) == Rejected("jwks down")


def test_oidc_discovery_is_lazy_cached_and_uses_the_discovered_jwks_uri() -> None:
    configurations: Final = iter(({"jwks_uri": "https://idp.example.test/keys"},))
    clients: Final = iter((StaticJwksClient(),))

    def fetch(url: str) -> Mapping[str, object]:
        assert url == "https://idp.example.test/.well-known/openid-configuration"
        return next(configurations)

    def client_factory(url: str) -> StaticJwksClient:
        assert url == "https://idp.example.test/keys"
        return next(clients)

    lookup: Final = PyJwkClientLookup(None, ISSUER, fetch=fetch, client_factory=client_factory)
    assert lookup.signing_key("token") == _PUBLIC_PEM
    assert lookup.signing_key("token") == _PUBLIC_PEM


def test_oidc_discovery_failures_are_retried_on_the_next_lookup() -> None:
    clock: Final = _Clock()
    configurations: Final = iter((OSError("discovery unavailable"), {"jwks_uri": "https://idp.example.test/keys"}))

    def fetch(_url: str) -> Mapping[str, object]:
        outcome: Final = next(configurations)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    lookup: Final = PyJwkClientLookup(
        None,
        ISSUER,
        fetch=fetch,
        client_factory=lambda _url: StaticJwksClient(),
        clock=clock,
    )
    assert isinstance(lookup.signing_key("token"), Rejected)
    clock.advance(30)
    assert lookup.signing_key("token") == _PUBLIC_PEM


def test_concurrent_oidc_discovery_failures_are_single_flight_and_cool_down() -> None:
    clock: Final = _Clock()
    fetch_calls: Final = _FetchCounter()
    fetch_started: Final = threading.Event()
    release_fetch: Final = threading.Event()

    def fetch(_url: str) -> Mapping[str, object]:
        call_number: Final = fetch_calls.record()
        fetch_started.set()
        if call_number == 1:
            assert release_fetch.wait(timeout=5)
            raise OSError("discovery unavailable")
        return {"jwks_uri": "https://idp.example.test/keys"}

    lookup: Final = PyJwkClientLookup(
        None,
        ISSUER,
        fetch=fetch,
        client_factory=lambda _url: StaticJwksClient(),
        clock=clock,
    )
    oidc: Final = OidcTokenVerifier(_oidc_settings(), lookup, POLICY)
    verifier: Final = TokenVerifier(MASTER_KEY, _sessions(), oidc, POLICY)
    forged_token: Final = jwt.encode(
        {
            "iss": ISSUER,
            "aud": AUDIENCE,
            "sub": "forged",
            "roles": ["admin"],
            "iat": TOKEN_IAT,
            "exp": TOKEN_EXP,
        },
        "not-the-idp-key",
        algorithm="HS256",
    )
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures: Final = tuple(executor.submit(verifier.verify, forged_token) for _ in range(8))
        assert fetch_started.wait(timeout=5)
        release_fetch.set()
        assert all(isinstance(future.result(timeout=5), Rejected) for future in futures)
    assert fetch_calls.count == 1
    assert isinstance(verifier.verify(forged_token), Rejected)
    assert fetch_calls.count == 1
    clock.advance(30)
    assert isinstance(verifier.verify(forged_token), Rejected)
    assert fetch_calls.count == 2


def test_explicit_jwks_url_takes_precedence_over_oidc_discovery() -> None:
    lookup: Final = PyJwkClientLookup(
        "https://idp.example.test/custom-keys",
        ISSUER,
        fetch=lambda _url: pytest.fail("discovery should not run when jwks_url is configured"),
        client_factory=lambda url: (
            StaticJwksClient() if url == "https://idp.example.test/custom-keys" else pytest.fail()
        ),
    )
    assert lookup.signing_key("token") == _PUBLIC_PEM


@pytest.mark.parametrize(
    "error",
    [OSError("network unavailable"), TimeoutError("request timed out"), ValueError("malformed JWKS"), jwt.PyJWTError()],
)
def test_jwks_client_errors_are_rejected(error: Exception) -> None:
    class FailingJwksClient:
        def get_signing_key_from_jwt(self, _token: str) -> str:
            raise error

    lookup: Final = PyJwkClientLookup(
        "https://idp.example.test/keys",
        ISSUER,
        client_factory=lambda _url: FailingJwksClient(),
    )
    assert isinstance(lookup.signing_key("token"), Rejected)


def test_hs256_token_claiming_the_oidc_issuer_cannot_pass_as_oidc() -> None:
    verifier: Final = _oidc_verifier(_oidc_settings(), _PUBLIC_PEM)
    forged: Final = jwt.encode(
        {
            "iss": ISSUER,
            "aud": AUDIENCE,
            "sub": "x",
            "roles": ["admin"],
            "iat": TOKEN_IAT,
            "exp": TOKEN_EXP,
        },
        "a" * 32,
        algorithm="HS256",
    )
    assert isinstance(verifier.verify(forged), Rejected)


def test_token_verifier_routes_master_key_session_and_oidc_credentials() -> None:
    sessions: Final = _sessions()
    verifier: Final = TokenVerifier(MASTER_KEY, sessions, _oidc_verifier(_oidc_settings(), _PUBLIC_PEM), POLICY)
    assert verifier.verify(MASTER_KEY) == master_key_principal()
    session: Final = verifier.verify(sessions.mint(_viewer(frozenset({"eu"}))))
    assert isinstance(session, Principal) and session.token_kind == "session"
    oidc: Final = verifier.verify(_oidc_token({"roles": ["viewer"]}))
    assert isinstance(oidc, Principal) and oidc.token_kind == "oidc"
    assert isinstance(verifier.verify("not-a-token"), Rejected)
    assert isinstance(verifier.verify("sk-master-tes"), Rejected)
    assert isinstance(verifier.verify("café"), Rejected)


def test_oidc_issued_token_is_not_accepted_as_a_session_when_oidc_is_off() -> None:
    verifier: Final = TokenVerifier(MASTER_KEY, _sessions(), None, POLICY)
    assert isinstance(verifier.verify(_oidc_token({"roles": ["admin"]})), Rejected)


def test_master_key_principal_is_a_fleet_wide_admin() -> None:
    principal: Final = master_key_principal()
    assert principal.roles == frozenset({ADMIN_ROLE}) and principal.dataplanes.all
    assert principal.token_kind == "master_key"
    assert principal.scopes == ALL_SCOPES


def test_master_key_keeps_every_scope_when_the_default_roles_are_switched_off() -> None:
    custom_only: Final = build_policy({"finops": RoleDefinition(frozenset({Scope.USAGE_READ}), frozenset())}, False)
    assert isinstance(custom_only, RolePolicy) and custom_only.scopes_for(ADMIN_ROLE) == frozenset()
    verifier: Final = TokenVerifier(MASTER_KEY, _sessions(), None, custom_only)
    verified: Final = verifier.verify(MASTER_KEY)
    assert isinstance(verified, Principal) and verified.scopes == ALL_SCOPES
