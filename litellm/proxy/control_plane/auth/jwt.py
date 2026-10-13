"""Token verification, independent of FastAPI. Session JWTs are HS256 under the master key, OIDC JWTs are
verified against the IdP's JWKS. Failures are values so the HTTP layer decides the status code."""

import secrets
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Final, Literal, Protocol, TypeAlias

import jwt
from jwt import PyJWK, PyJWKClient
from pydantic import TypeAdapter, ValidationError

from litellm.proxy.control_plane.auth.rbac import (
    ADMIN_ROLE,
    ALL_SCOPES,
    DataplaneBinding,
    Principal,
    RolePolicy,
    build_principal,
    parse_scopes,
)
from litellm.proxy.control_plane.settings import ControlPlaneSettings, OidcSettings

SESSION_ALGORITHM: Final = "HS256"
SESSION_AUDIENCE: Final = "litellm-control-plane-api"
OIDC_ALGORITHMS: Final = ("RS256", "RS384", "RS512", "ES256", "ES384", "ES512", "PS256", "PS384", "PS512")

_STRING_LIST: Final = TypeAdapter(list[str])
_STRING: Final = TypeAdapter(str)
_CLAIMS: Final = TypeAdapter(Mapping[str, object])


@dataclass(frozen=True, slots=True)
class Rejected:
    reason: str


VerifiedToken: TypeAlias = Principal | Rejected


SigningKey: TypeAlias = PyJWK | str
JWKS_FETCH_TIMEOUT_SECONDS: Final = 10


class JwksKeyLookup(Protocol):
    def signing_key(self, token: str) -> SigningKey | Rejected: ...


class PyJwkClientLookup:
    def __init__(self, jwks_url: str) -> None:
        self._client = PyJWKClient(jwks_url, cache_keys=False, timeout=JWKS_FETCH_TIMEOUT_SECONDS)

    def signing_key(self, token: str) -> SigningKey | Rejected:
        try:
            return self._client.get_signing_key_from_jwt(token)
        except jwt.PyJWKClientError as error:
            return Rejected(f"jwks lookup failed: {error}")


def _as_string_tuple(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return tuple(value.split())
    try:
        return tuple(_STRING_LIST.validate_python(value))
    except ValidationError:
        return ()


def _as_optional_string(value: object) -> str | None:
    try:
        return _STRING.validate_python(value)
    except ValidationError:
        return None


def _decode(
    token: str, key: SigningKey, algorithms: tuple[str, ...], audience: str | None, issuer: str, leeway: int
) -> Mapping[str, object] | Rejected:
    try:
        raw: Final = jwt.decode(
            token,
            key,
            algorithms=list(algorithms),
            audience=audience,
            issuer=issuer,
            leeway=leeway,
            options={"require": ["exp", "sub"]},
        )
        return _CLAIMS.validate_python(raw)
    except jwt.PyJWTError as error:
        return Rejected(f"invalid token: {error}")
    except ValidationError:
        return Rejected("invalid token: claims are not an object")


class SessionTokenCodec:
    """Mints and verifies the control plane's own session JWTs. The audience pins a token to this API, so a
    token minted under the same master key by another component never authorizes a control plane route."""

    def __init__(
        self, master_key: str, issuer: str, ttl: timedelta, policy: RolePolicy, audience: str = SESSION_AUDIENCE
    ) -> None:
        self._master_key = master_key
        self._issuer = issuer
        self._audience = audience
        self._ttl = ttl
        self._policy = policy

    def mint(self, principal: Principal, now: datetime | None = None) -> str:
        issued_at: Final = now or datetime.now(timezone.utc)
        claims: Final = {
            "iss": self._issuer,
            "aud": self._audience,
            "sub": principal.subject,
            "email": principal.email,
            "roles": sorted(principal.roles),
            "scope": " ".join(sorted(scope.value for scope in principal.scopes)),
            "litellm_dataplanes": ["*"] if principal.dataplanes.all else sorted(principal.dataplanes.ids),
            "iat": int(issued_at.timestamp()),
            "exp": int((issued_at + self._ttl).timestamp()),
            "jti": secrets.token_urlsafe(16),
        }
        return jwt.encode(claims, self._master_key, algorithm=SESSION_ALGORITHM)

    def verify(self, token: str) -> VerifiedToken:
        claims: Final = _decode(token, self._master_key, (SESSION_ALGORITHM,), self._audience, self._issuer, 0)
        if isinstance(claims, Rejected):
            return claims
        return _principal_from_claims(self._policy, claims, "roles", "scope", "litellm_dataplanes", "email", "session")


class OidcTokenVerifier:
    def __init__(self, settings: OidcSettings, keys: JwksKeyLookup, policy: RolePolicy) -> None:
        self._settings = settings
        self._keys = keys
        self._policy = policy

    @property
    def issuer(self) -> str:
        return self._settings.issuer

    def verify(self, token: str) -> VerifiedToken:
        key: Final = self._keys.signing_key(token)
        if isinstance(key, Rejected):
            return key
        claims: Final = _decode(
            token, key, OIDC_ALGORITHMS, self._settings.audience, self._settings.issuer, self._settings.leeway_seconds
        )
        if isinstance(claims, Rejected):
            return claims
        return _principal_from_claims(
            self._policy,
            claims,
            self._settings.roles_claim,
            self._settings.scope_claim,
            self._settings.dataplanes_claim,
            self._settings.email_claim,
            "oidc",
        )


def _principal_from_claims(
    policy: RolePolicy,
    claims: Mapping[str, object],
    roles_claim: str,
    scope_claim: str,
    dataplanes_claim: str,
    email_claim: str,
    token_kind: Literal["session", "oidc"],
) -> VerifiedToken:
    subject: Final = _as_optional_string(claims.get("sub"))
    if subject is None:
        return Rejected("invalid token: sub is not a string")
    roles: Final = policy.known(_as_string_tuple(claims.get(roles_claim)))
    if not roles:
        return Rejected("token carries no control plane role")
    raw_scopes: Final = claims.get(scope_claim)
    return build_principal(
        policy=policy,
        subject=subject,
        email=_as_optional_string(claims.get(email_claim)),
        roles=roles,
        claimed_scopes=None if raw_scopes is None else parse_scopes(_as_string_tuple(raw_scopes)),
        dataplane_ids=frozenset(_as_string_tuple(claims.get(dataplanes_claim))),
        token_kind=token_kind,
    )


MASTER_KEY_SUBJECT: Final = "master_key"


def master_key_principal() -> Principal:
    """The master key is the break-glass credential and holds every scope even when operators drop the default
    ``admin`` role from the policy."""
    return Principal(
        subject=MASTER_KEY_SUBJECT,
        email=None,
        roles=frozenset({ADMIN_ROLE}),
        scopes=ALL_SCOPES,
        dataplanes=DataplaneBinding(all=True, ids=frozenset()),
        token_kind="master_key",
    )


class TokenVerifier:
    """Resolves a bearer credential of any supported kind to a Principal."""

    def __init__(
        self, master_key: str, sessions: SessionTokenCodec, oidc: OidcTokenVerifier | None, policy: RolePolicy
    ) -> None:
        self._master_key = master_key
        self._sessions = sessions
        self._oidc = oidc
        self._policy = policy

    def verify(self, credential: str) -> VerifiedToken:
        if secrets.compare_digest(credential, self._master_key):
            return master_key_principal()
        if not _looks_like_jwt(credential):
            return Rejected("credential is neither the master key nor a JWT")
        if self._oidc is not None and _issuer_of(credential) == self._oidc.issuer:
            return self._oidc.verify(credential)
        return self._sessions.verify(credential)


def _looks_like_jwt(credential: str) -> bool:
    return credential.count(".") == 2


def _issuer_of(credential: str) -> str | None:
    try:
        unverified: Final = jwt.decode(credential, options={"verify_signature": False})
    except jwt.PyJWTError:
        return None
    return _as_optional_string(unverified.get("iss"))


def build_token_verifier(
    settings: ControlPlaneSettings, policy: RolePolicy, keys: JwksKeyLookup | None = None
) -> TokenVerifier:
    master_key: Final = settings.master_key.get_secret_value()
    sessions: Final = SessionTokenCodec(
        master_key=master_key,
        issuer=settings.session_issuer,
        audience=settings.session_audience,
        ttl=timedelta(seconds=settings.session_ttl_seconds),
        policy=policy,
    )
    oidc_settings: Final = settings.oidc
    if oidc_settings is None:
        return TokenVerifier(master_key, sessions, None, policy)
    jwks_url: Final = oidc_settings.jwks_url or f"{oidc_settings.issuer.rstrip('/')}/.well-known/jwks.json"
    lookup: Final = keys or PyJwkClientLookup(jwks_url)
    return TokenVerifier(master_key, sessions, OidcTokenVerifier(oidc_settings, lookup, policy), policy)
