from datetime import datetime, timedelta, timezone
from typing import Final

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from litellm.proxy.control_plane.auth.jwt import (
    OidcTokenVerifier,
    Rejected,
    SESSION_AUDIENCE,
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


def _oidc_token(claims: dict[str, object], exp_delta: timedelta = timedelta(hours=1)) -> str:
    now: Final = datetime.now(timezone.utc)
    return jwt.encode(
        {"iss": ISSUER, "aud": AUDIENCE, "sub": "user-1", "exp": now + exp_delta, "iat": now, **claims},
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
    codec = _sessions()
    original = _viewer(frozenset({"eu", "us"}))
    verified = codec.verify(codec.mint(original))
    assert isinstance(verified, Principal)
    assert (verified.subject, verified.email, verified.roles) == ("alice", "a@x.test", original.roles)
    assert verified.scopes == original.scopes
    assert verified.dataplanes == original.dataplanes
    assert verified.token_kind == "session"


def test_fleet_wide_session_round_trips_as_fleet_wide() -> None:
    codec = _sessions()
    admin = build_principal(POLICY, "root", None, frozenset({ADMIN_ROLE}), None, frozenset(), "session")
    verified = codec.verify(codec.mint(admin))
    assert isinstance(verified, Principal) and verified.dataplanes.all


def test_expired_session_is_rejected() -> None:
    codec = _sessions(ttl=timedelta(seconds=-1))
    assert isinstance(codec.verify(codec.mint(_viewer(frozenset()))), Rejected)


def test_session_signed_with_another_key_is_rejected() -> None:
    other = _sessions(master_key="sk-other")
    assert isinstance(_sessions().verify(other.mint(_viewer(frozenset()))), Rejected)


def test_session_from_another_issuer_is_rejected() -> None:
    other = _sessions(issuer="not-cp")
    assert isinstance(_sessions().verify(other.mint(_viewer(frozenset()))), Rejected)


def test_session_minted_for_another_audience_is_rejected() -> None:
    """Same master key and issuer, but a token for another component never opens the control plane."""
    other = _sessions(audience="litellm-proxy-ui")
    assert isinstance(_sessions().verify(other.mint(_viewer(frozenset()))), Rejected)


def test_session_without_an_audience_claim_is_rejected() -> None:
    now = datetime.now(timezone.utc)
    no_aud = jwt.encode(
        {"iss": "cp", "sub": "alice", "roles": ["viewer"], "exp": now + timedelta(hours=1)},
        MASTER_KEY,
        algorithm="HS256",
    )
    assert isinstance(_sessions().verify(no_aud), Rejected)
    with_aud = jwt.encode(
        {"iss": "cp", "aud": SESSION_AUDIENCE, "sub": "alice", "roles": ["viewer"], "exp": now + timedelta(hours=1)},
        MASTER_KEY,
        algorithm="HS256",
    )
    assert isinstance(_sessions().verify(with_aud), Principal)


def test_oidc_scope_claim_with_only_foreign_scopes_grants_no_control_plane_scope() -> None:
    verifier = _oidc_verifier(_oidc_settings(), _PUBLIC_PEM)
    scoped = verifier.verify(_oidc_token({"roles": ["admin"], "scope": "openid profile email"}))
    assert isinstance(scoped, Principal) and scoped.scopes == frozenset()
    unscoped = verifier.verify(_oidc_token({"roles": ["admin"]}))
    assert isinstance(unscoped, Principal) and unscoped.scopes == POLICY.scopes_for(ADMIN_ROLE)


def test_oidc_token_maps_claims_to_a_clamped_principal() -> None:
    verifier = _oidc_verifier(_oidc_settings(), _PUBLIC_PEM)
    token = _oidc_token(
        {
            "roles": ["viewer"],
            "scope": "openid usage:read proxy:write",
            "litellm_dataplanes": ["eu"],
            "email": "u@x.test",
        }
    )
    principal = verifier.verify(token)
    assert isinstance(principal, Principal)
    assert principal.roles == frozenset({VIEWER_ROLE})
    assert principal.scopes == frozenset({Scope.USAGE_READ})
    assert principal.dataplanes.ids == frozenset({"eu"}) and not principal.dataplanes.all
    assert (principal.email, principal.token_kind) == ("u@x.test", "oidc")


def test_oidc_roles_claim_name_is_configurable_and_space_delimited_roles_parse() -> None:
    verifier = _oidc_verifier(_oidc_settings(roles_claim="groups"), _PUBLIC_PEM)
    principal = verifier.verify(_oidc_token({"groups": "viewer other"}))
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
    verifier = _oidc_verifier(_oidc_settings(), _PUBLIC_PEM)
    assert isinstance(verifier.verify(_oidc_token(claims)), Rejected)


def test_oidc_token_signed_by_an_unknown_key_is_rejected() -> None:
    other_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    other_pem = (
        other_key.public_key()
        .public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
        .decode()
    )
    verifier = _oidc_verifier(_oidc_settings(), other_pem)
    assert isinstance(verifier.verify(_oidc_token({"roles": ["admin"]})), Rejected)


def test_oidc_token_is_rejected_when_the_jwks_lookup_fails() -> None:
    verifier = _oidc_verifier(_oidc_settings(), Rejected("jwks down"))
    assert verifier.verify(_oidc_token({"roles": ["admin"]})) == Rejected("jwks down")


def test_hs256_token_claiming_the_oidc_issuer_cannot_pass_as_oidc() -> None:
    verifier = _oidc_verifier(_oidc_settings(), _PUBLIC_PEM)
    forged = jwt.encode(
        {
            "iss": ISSUER,
            "aud": AUDIENCE,
            "sub": "x",
            "roles": ["admin"],
            "exp": datetime.now(timezone.utc) + timedelta(hours=1),
        },
        "a" * 32,
        algorithm="HS256",
    )
    assert isinstance(verifier.verify(forged), Rejected)


def test_token_verifier_routes_master_key_session_and_oidc_credentials() -> None:
    sessions = _sessions()
    verifier = TokenVerifier(MASTER_KEY, sessions, _oidc_verifier(_oidc_settings(), _PUBLIC_PEM), POLICY)
    assert verifier.verify(MASTER_KEY) == master_key_principal()
    session = verifier.verify(sessions.mint(_viewer(frozenset({"eu"}))))
    assert isinstance(session, Principal) and session.token_kind == "session"
    oidc = verifier.verify(_oidc_token({"roles": ["viewer"]}))
    assert isinstance(oidc, Principal) and oidc.token_kind == "oidc"
    assert isinstance(verifier.verify("not-a-token"), Rejected)
    assert isinstance(verifier.verify("sk-master-tes"), Rejected)


def test_oidc_issued_token_is_not_accepted_as_a_session_when_oidc_is_off() -> None:
    verifier = TokenVerifier(MASTER_KEY, _sessions(), None, POLICY)
    assert isinstance(verifier.verify(_oidc_token({"roles": ["admin"]})), Rejected)


def test_master_key_principal_is_a_fleet_wide_admin() -> None:
    principal = master_key_principal()
    assert principal.roles == frozenset({ADMIN_ROLE}) and principal.dataplanes.all
    assert principal.token_kind == "master_key"
    assert principal.scopes == ALL_SCOPES


def test_master_key_keeps_every_scope_when_the_default_roles_are_switched_off() -> None:
    custom_only = build_policy({"finops": RoleDefinition(frozenset({Scope.USAGE_READ}), frozenset())}, False)
    assert isinstance(custom_only, RolePolicy) and custom_only.scopes_for(ADMIN_ROLE) == frozenset()
    verifier = TokenVerifier(MASTER_KEY, _sessions(), None, custom_only)
    verified = verifier.verify(MASTER_KEY)
    assert isinstance(verified, Principal) and verified.scopes == ALL_SCOPES
