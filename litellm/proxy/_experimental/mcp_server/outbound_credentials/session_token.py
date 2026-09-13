"""Identity-only session tokens for the gateway-level (aggregate ``/mcp``) DCR front door.

A DCR client that signs in through LiteLLM SSO holds ONE bearer that carries ONLY a
litellm identity; unlike the :mod:`.envelope` bridge bearer it seals no upstream
credential, because the custody model vaults every upstream token server-side in
``LiteLLM_MCPUserCredentials`` and egress resolves them by user at call time. The token
is therefore a stable REFERENCE, not an authorization: admission reloads the live user
record and policy on every request, so deactivating the user (or their team) kills
outstanding sessions immediately without a revocation store.

Wire shape: ``llm_session_`` (access) / ``llm_srefresh_`` (refresh) + a JWT signed with
the injected key material: HS256 under the default master-key-derived secret (the same
signing approach as :mod:`.envelope`), or RS256 under an operator-provided RSA private
key (:class:`AsymmetricSessionKeys`) so downstream validators hold only the public half.
Claims are ``iss``/``iat``/``exp``
plus ``jti`` (per-mint uniqueness, so two tokens minted in the same second never
collide and a future revocation list has a stable handle), ``kind``, ``user_id``, and
``client_id``; ``client_id`` binds the refresh token
to the DCR client it was issued to (RFC 6749 section 6) and is carried on the access
token for parity and audit. There is no encrypted payload: nothing in a session token
is secret beyond the signature, and reprs never print the signed value because minted
tokens are ``SecretStr``.

This module is pure and unwired: it imports nothing from endpoint or edge code, reads
no proxy globals, and takes all key material and the clock as explicit parameters.
Failures are values: :func:`open_session_token` and :func:`open_session_refresh_token`
are total over hostile, attacker-controlled input and return a
``SessionTokenOpenError`` variant rather than raising. PyJWT's ``iat``/``nbf``/``exp``
validators are disabled for the same reasons documented in :mod:`.envelope` (they
raise on hostile claim types and compare against the wall clock instead of the
injected ``now``); the strict pydantic claims model is the sole, total type gate.
"""

from __future__ import annotations

import secrets
from collections import Counter
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Final, Literal, TypeAlias

import jwt
from cryptography.exceptions import UnsupportedAlgorithm
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError, field_validator, model_validator

SESSION_TOKEN_PREFIX: Final = "llm_session_"
"""Marker prefix on every serialized session ACCESS token so the admission edge can cheaply
tell a gateway session from a litellm key, JWT, or bridge envelope before doing any
cryptography. Distinct from the ``llm_env_``/``llm_refresh_`` envelope prefixes."""

SESSION_REFRESH_PREFIX: Final = "llm_srefresh_"
"""Marker prefix on every serialized session REFRESH token. A distinct prefix keeps the two
credentials routable without crypto and, together with the signed ``kind`` claim, stops one
from being presented where the other is expected: the refresh token is only ever presented
back to the token endpoint, never at the MCP edge."""

SESSION_ISSUER: Final = "litellm-mcp-gateway"
"""``iss`` claim stamped into every session token and required back on open. Distinct from
the envelope issuer so a token of one family can never validate in the other even under a
hypothetical shared signing key."""

SESSION_TTL_SECONDS: Final = 3600
"""Session ACCESS token lifetime (1h), matching the BYOK session bearer window: a
client-held credential never outlives a bounded window, and each refresh re-validates
the live user before re-minting."""

SESSION_REFRESH_TTL_SECONDS: Final = 1209600
"""Session REFRESH token lifetime (14 days), matching the refresh-envelope bound. Each
renewal re-validates the sealed user against the live record (deactivation gates it) and
rotates the refresh token, so the practical bound is idle time, not a fixed session."""

MAX_SESSION_TOKEN_BYTES: Final = 4096
"""Size cap on the serialized token (prefix + JWT, in bytes) and on any candidate accepted
by the openers. Session claims are small; the only variable-length field is ``client_id``
(a sealed DCR client record), and 4096 leaves ample headroom under common 8-16KB header
limits while bounding hostile input before JWT parsing."""

_SESSION_JWT_ALGORITHM: Final = "HS256"

_SESSION_RSA_ALGORITHM: Final = "RS256"

_MIN_RSA_KEY_BITS: Final = 2048
"""RFC 7518 section 3.3: RS256 requires a key of at least 2048 bits."""

SessionTokenKind = Literal["session", "session_refresh"]
"""Which credential a session token is. Stamped into the signed claims and required to match
on open, so a signature-valid token of one kind cannot be replayed as the other even if its
wire prefix is swapped (the prefix is not part of the signed payload; this claim is)."""

SessionAudience = Literal["proxy_api"]
"""The non-MCP audience a session REFRESH token can be minted for. ``None`` (the default and
the only value ever on an MCP wire) means the aggregate MCP gateway; ``"proxy_api"`` means the
refresh grant re-mints the proxy-API CLI credential instead of an MCP session pair. The audience
is read only from the signed claims, never from the request, so a token of one audience can
never be redeemed as the other."""


class SessionPrincipal(BaseModel):
    """The litellm user a session token identifies and the DCR client it was issued to.

    ``user_id`` is the SSO-established litellm user subject, never a credential: admission
    reloads the live user record by it, so current role, team, and revocation state are
    enforced at use time rather than frozen at mint time. ``client_id`` is the (stateless,
    gateway-sealed) DCR client identifier the token was issued to; the token endpoint
    requires it to match on the refresh grant.

    ``resource_server_id`` is the single MCP server this session was authorized for when
    the client requested a per-server RFC 8707 resource at authorize time, or ``None`` for
    the aggregate scope. It is a RESTRICTION carried for admission to intersect against
    the live grant resolution, never a grant by itself; the refresh grant re-mints from
    this principal so the restriction survives rotation.
    """

    model_config = ConfigDict(frozen=True)
    user_id: str = Field(min_length=1)
    client_id: str = Field(min_length=1)
    resource_server_id: str | None = None
    audience: SessionAudience | None = None
    team_id: str | None = None


class SessionKeys(BaseModel):
    """Injected key material: the HS256 signing key.

    ``signing_key`` must be at least 32 bytes: HS256's HMAC-SHA256 has a 256-bit security
    level, RFC 7518 requires a key of at least that size, and a shorter key makes PyJWT
    emit ``InsecureKeyLengthWarning``.
    """

    model_config = ConfigDict(frozen=True)
    signing_key: SecretStr = Field(min_length=32)


class SessionRotatedPublicKey(BaseModel):
    """The public half of a retired signing key, kept verifiable under its ``kid`` during a
    rotation window so tokens minted before the rotation stay valid until they expire."""

    model_config = ConfigDict(frozen=True)
    kid: str = Field(min_length=1)
    public_key_pem: str = Field(min_length=1)

    @field_validator("public_key_pem")
    @classmethod
    def _pem_is_an_rsa_public_key(cls, value: str) -> str:
        try:
            loaded: Final = serialization.load_pem_public_key(value.encode())
        except (ValueError, TypeError, UnsupportedAlgorithm) as exc:
            raise ValueError(f"public_key_pem is not a loadable PEM public key: {exc}") from exc
        if not isinstance(loaded, rsa.RSAPublicKey):
            raise ValueError("public_key_pem must be an RSA public key in PEM format")  # noqa: TRY004  # pydantic validators must raise ValueError
        if loaded.key_size < _MIN_RSA_KEY_BITS:
            raise ValueError(f"public_key_pem must be an RSA key of at least {_MIN_RSA_KEY_BITS} bits")
        return value


class AsymmetricSessionKeys(BaseModel):
    """Injected RS256 key material: the issuer-held RSA private key and the stable ``kid``
    stamped into every minted token's JOSE header, plus the public halves of previously
    rotated keys that verification still accepts while their tokens age out. Downstream
    validators never need the private key: :func:`session_public_key_pem` yields the
    public half to distribute."""

    model_config = ConfigDict(frozen=True)
    private_key_pem: SecretStr
    kid: str = Field(min_length=1)
    previous_public_keys: tuple[SessionRotatedPublicKey, ...] = ()

    @field_validator("private_key_pem")
    @classmethod
    def _pem_is_a_strong_rsa_private_key(cls, value: SecretStr) -> SecretStr:
        try:
            loaded: Final = serialization.load_pem_private_key(value.get_secret_value().encode(), password=None)
        except (ValueError, TypeError, UnsupportedAlgorithm) as exc:
            raise ValueError(f"private_key_pem is not a loadable unencrypted PEM private key: {exc}") from exc
        if not isinstance(loaded, rsa.RSAPrivateKey):
            raise ValueError("private_key_pem must be an unencrypted RSA private key in PEM format")  # noqa: TRY004  # pydantic validators must raise ValueError
        if loaded.key_size < _MIN_RSA_KEY_BITS:
            raise ValueError(f"private_key_pem must be an RSA key of at least {_MIN_RSA_KEY_BITS} bits")
        return value

    @model_validator(mode="after")
    def _kids_are_unique(self) -> AsymmetricSessionKeys:
        kids: Final = (self.kid, *(previous.kid for previous in self.previous_public_keys))
        duplicates: Final = tuple(kid for kid, count in Counter(kids).items() if count > 1)
        if duplicates:
            raise ValueError(
                f"every kid must be unique across the current and previous keys; duplicated: {', '.join(duplicates)}"
            )
        return self


SessionSigningKeys: TypeAlias = SessionKeys | AsymmetricSessionKeys
"""Every key material shape the mints and openers accept: the default master-key-derived
HS256 secret, or operator-configured RS256 RSA keys."""


@lru_cache(maxsize=8)
def _public_key_pem_from_private(private_key_pem: str) -> str:
    loaded: Final = serialization.load_pem_private_key(private_key_pem.encode(), password=None)
    return (
        loaded.public_key()
        .public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
        .decode()
    )


def session_public_key_pem(keys: AsymmetricSessionKeys) -> str:
    """The PEM public half of the current RS256 signing key: the only material a downstream
    validator (an external gateway verifying ``kid``-matched tokens) ever needs."""
    return _public_key_pem_from_private(keys.private_key_pem.get_secret_value())


class MintedSessionToken(BaseModel):
    """A minted session token: the client-held bearer value and when it expires."""

    model_config = ConfigDict(frozen=True)
    token: SecretStr
    expires_at: datetime


class OpenedSessionToken(BaseModel):
    """A validated session token of either kind: the principal it was minted for, the
    ``jti`` so the token endpoint can enforce single-use rotation on a refresh token, and
    the signed ``kind``/``iat``/``exp`` so an introspection response can report the
    token's metadata without re-decoding."""

    model_config = ConfigDict(frozen=True)
    principal: SessionPrincipal
    jti: str
    kind: SessionTokenKind
    iat: int
    exp: int


class SessionTokenTooLarge(BaseModel):
    """The serialized token exceeded ``MAX_SESSION_TOKEN_BYTES``; carries sizes only. Only
    reachable through an oversized ``client_id``, which registration should have bounded."""

    model_config = ConfigDict(frozen=True)
    tag: Literal["session_token_too_large"] = "session_token_too_large"
    size_bytes: int
    max_bytes: int


SessionTokenMintError: TypeAlias = SessionTokenTooLarge


class NotASessionToken(BaseModel):
    """The candidate does not carry the expected session prefix."""

    model_config = ConfigDict(frozen=True)
    tag: Literal["not_a_session_token"] = "not_a_session_token"


class SessionBadSignature(BaseModel):
    """The JWT signature does not verify under the provided signing key."""

    model_config = ConfigDict(frozen=True)
    tag: Literal["session_bad_signature"] = "session_bad_signature"


class SessionExpired(BaseModel):
    """The token's ``exp`` is not in the future relative to the provided ``now``."""

    model_config = ConfigDict(frozen=True)
    tag: Literal["session_expired"] = "session_expired"


class SessionMalformed(BaseModel):
    """The token is not a well-formed session token: undecodable JWT, wrong issuer, wrong
    ``kind``, or missing/mistyped/extra claims."""

    model_config = ConfigDict(frozen=True)
    tag: Literal["session_malformed"] = "session_malformed"


SessionTokenOpenError: TypeAlias = NotASessionToken | SessionBadSignature | SessionExpired | SessionMalformed


class _SessionClaims(BaseModel):
    """Decoded-claims boundary that pins the exact shape the mints emit.

    ``user_id``/``client_id`` mirror the ``min_length`` constraints of
    :class:`SessionPrincipal` so any claim set that validates here also constructs a
    principal, keeping the openers raise-free: a correctly signed JWT with an empty
    identity claim fails here and maps to ``SessionMalformed``. ``strict`` rejects coerced
    types (``exp: "123"``) and ``extra="forbid"`` rejects any claim the gateway never
    mints; PyJWT's own registered-claim validators are disabled at decode (see module
    docstring), so this model is the sole, total type gate for every claim.
    """

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    iss: str
    iat: int
    exp: int
    jti: str = Field(min_length=1)
    kind: SessionTokenKind
    user_id: str = Field(min_length=1)
    client_id: str = Field(min_length=1)
    resource_server_id: str | None = None
    audience: SessionAudience | None = None
    team_id: str | None = None


def is_session_token(candidate: str) -> bool:
    """Cheap prefix check for a session ACCESS token so the admission edge can route gateway
    sessions vs keys, JWTs, and envelopes without crypto."""
    return candidate.startswith(SESSION_TOKEN_PREFIX)


def is_session_refresh_token(candidate: str) -> bool:
    """Cheap prefix check for a session REFRESH token so the token endpoint can route a
    refresh grant without crypto."""
    return candidate.startswith(SESSION_REFRESH_PREFIX)


def mint_session_token(
    principal: SessionPrincipal,
    keys: SessionSigningKeys,
    now: datetime,
) -> MintedSessionToken | SessionTokenMintError:
    """Mint the short-lived session ACCESS token for ``principal``.

    ``exp`` is ``SESSION_TTL_SECONDS`` from ``now``. Returns ``SessionTokenTooLarge`` when
    the serialized token exceeds ``MAX_SESSION_TOKEN_BYTES``.
    """
    return _mint(
        kind="session",
        prefix=SESSION_TOKEN_PREFIX,
        principal=principal,
        expires_at=now + timedelta(seconds=SESSION_TTL_SECONDS),
        keys=keys,
        now=now,
    )


def mint_session_refresh_token(
    principal: SessionPrincipal,
    keys: SessionSigningKeys,
    now: datetime,
) -> MintedSessionToken | SessionTokenMintError:
    """Mint the long-lived session REFRESH token for ``principal``.

    ``exp`` is ``SESSION_REFRESH_TTL_SECONDS`` from ``now``. Minting a distinct
    ``kind="session_refresh"`` claim is what keeps a refresh token from ever opening as an
    access credential at the MCP edge.
    """
    return _mint(
        kind="session_refresh",
        prefix=SESSION_REFRESH_PREFIX,
        principal=principal,
        expires_at=now + timedelta(seconds=SESSION_REFRESH_TTL_SECONDS),
        keys=keys,
        now=now,
    )


def open_session_token(
    candidate: str,
    keys: SessionSigningKeys,
    now: datetime,
) -> OpenedSessionToken | SessionTokenOpenError:
    """Validate a session ACCESS ``candidate`` and recover the principal.

    Never raises for bad input: every invalid, expired, tampered, or wrong-kind candidate
    maps to a distinct ``SessionTokenOpenError`` variant.
    """
    return _open(candidate, prefix=SESSION_TOKEN_PREFIX, expected_kind="session", keys=keys, now=now)


def open_session_refresh_token(
    candidate: str,
    keys: SessionSigningKeys,
    now: datetime,
) -> OpenedSessionToken | SessionTokenOpenError:
    """Validate a session REFRESH ``candidate`` and recover the principal.

    Total over hostile input exactly like :func:`open_session_token`. The
    ``kind="session_refresh"`` claim is required, so an access token re-prefixed as a
    refresh one is rejected as ``SessionMalformed``.
    """
    return _open(candidate, prefix=SESSION_REFRESH_PREFIX, expected_kind="session_refresh", keys=keys, now=now)


def _mint(
    kind: SessionTokenKind,
    prefix: str,
    principal: SessionPrincipal,
    expires_at: datetime,
    keys: SessionSigningKeys,
    now: datetime,
) -> MintedSessionToken | SessionTokenTooLarge:
    """Sign the claims for either token kind and enforce the size cap. Shared by both mints
    so the JWT shape, issuer, and size guard cannot drift between access and refresh."""
    claims: Final = _SessionClaims(
        iss=SESSION_ISSUER,
        iat=int(now.timestamp()),
        exp=int(expires_at.timestamp()),
        jti=secrets.token_urlsafe(16),
        kind=kind,
        user_id=principal.user_id,
        client_id=principal.client_id,
        resource_server_id=principal.resource_server_id,
        audience=principal.audience,
        team_id=principal.team_id,
    )
    token: Final = prefix + _sign_claims(claims, keys)
    size_bytes: Final = len(token.encode("utf-8"))
    if size_bytes > MAX_SESSION_TOKEN_BYTES:
        return SessionTokenTooLarge(size_bytes=size_bytes, max_bytes=MAX_SESSION_TOKEN_BYTES)
    return MintedSessionToken(token=SecretStr(token), expires_at=expires_at)


def _sign_claims(claims: _SessionClaims, keys: SessionSigningKeys) -> str:
    """Sign the claim set under whichever key material was injected: RS256 with the ``kid``
    in the JOSE header (so a validator can pick the right public key), or the default
    HS256 secret with no header extras (byte-compatible with every pre-RS256 token)."""
    payload: Final = claims.model_dump(exclude_none=True)
    if isinstance(keys, AsymmetricSessionKeys):
        return jwt.encode(
            payload,
            keys.private_key_pem.get_secret_value(),
            algorithm=_SESSION_RSA_ALGORITHM,
            headers={"kid": keys.kid},
        )
    return jwt.encode(payload, keys.signing_key.get_secret_value(), algorithm=_SESSION_JWT_ALGORITHM)


def _open(
    candidate: str,
    prefix: str,
    expected_kind: SessionTokenKind,
    keys: SessionSigningKeys,
    now: datetime,
) -> OpenedSessionToken | SessionTokenOpenError:
    """Prefix-route, size-bound, signature-verify, kind-check, and expiry-check an
    attacker-controlled candidate, shared by both openers so the security gate is identical
    for access and refresh. Returns the opened token or a distinct error; never raises."""
    if not candidate.startswith(prefix):
        return NotASessionToken()
    # UTF-8 byte length is never below character length, so a character count already over
    # the cap rejects an oversize candidate in O(1) without encoding it; the exact byte
    # check then runs only on candidates already bounded to the cap in characters.
    if len(candidate) > MAX_SESSION_TOKEN_BYTES:
        return SessionMalformed()
    if len(candidate.encode("utf-8", "surrogatepass")) > MAX_SESSION_TOKEN_BYTES:
        return SessionMalformed()
    claims: Final = _decode_claims(candidate.removeprefix(prefix), keys)
    if not isinstance(claims, _SessionClaims):
        return claims
    if claims.kind != expected_kind:
        return SessionMalformed()
    if now.timestamp() >= claims.exp:
        return SessionExpired()
    return OpenedSessionToken(
        principal=SessionPrincipal(
            user_id=claims.user_id,
            client_id=claims.client_id,
            resource_server_id=claims.resource_server_id,
            audience=claims.audience,
            team_id=claims.team_id,
        ),
        jti=claims.jti,
        kind=claims.kind,
        iat=claims.iat,
        exp=claims.exp,
    )


class _VerificationMaterial(BaseModel):
    model_config = ConfigDict(frozen=True)
    key: SecretStr
    algorithm: Literal["HS256", "RS256"]


def _verification_material(
    compact: str,
    keys: SessionSigningKeys,
) -> _VerificationMaterial | SessionBadSignature | SessionMalformed:
    """Pick the single key and algorithm the candidate is allowed to verify under.

    HS256 mode has exactly one secret. RS256 mode routes by the JOSE header ``kid``: the
    current key's derived public half, or a retired key's stored public half during a
    rotation window. An unknown or missing ``kid`` is ``SessionBadSignature`` (a foreign
    key), and an undecodable header is ``SessionMalformed``. The algorithm is pinned per
    key shape, never read from the header, so an HS256 token can never be verified
    against a public key or vice versa.
    """
    if isinstance(keys, SessionKeys):
        return _VerificationMaterial(key=keys.signing_key, algorithm=_SESSION_JWT_ALGORITHM)
    try:
        header: Final = jwt.get_unverified_header(compact)
    except jwt.InvalidTokenError:
        return SessionMalformed()
    kid: Final = header.get("kid")
    if kid == keys.kid:
        return _VerificationMaterial(key=SecretStr(session_public_key_pem(keys)), algorithm=_SESSION_RSA_ALGORITHM)
    for previous in keys.previous_public_keys:
        if previous.kid == kid:
            return _VerificationMaterial(key=SecretStr(previous.public_key_pem), algorithm=_SESSION_RSA_ALGORITHM)
    return SessionBadSignature()


def _decode_claims(
    compact: str,
    keys: SessionSigningKeys,
) -> _SessionClaims | SessionBadSignature | SessionMalformed:
    """Verify the signature and shape of an attacker-controlled compact JWT.

    ``compact`` is fully hostile and bounded to ``MAX_SESSION_TOKEN_BYTES`` by the caller.
    The accepted algorithm is pinned by :func:`_verification_material` from the injected
    key shape, so ``alg`` confusion (``none``, or HS256 signed with a public key as the
    secret) fails before or at signature verification. PyJWT's ``iat``/``nbf``/``exp``
    validators are disabled: they raise on hostile claim
    types and, for ``iat``/``nbf``, compare against the wall clock rather than the injected
    ``now`` (``exp`` is checked by the caller against ``now``). Apart from a signature
    mismatch, every decode failure is ``SessionMalformed``: a non-UTF-8 candidate surfaces
    as ``UnicodeEncodeError`` (a ``ValueError``), a non-string registered claim as a
    ``TypeError`` from PyJWT's claim validators, and a wrong issuer or structurally invalid
    token as an ``InvalidTokenError``. ``_SessionClaims`` is the total type gate.
    """
    material: Final = _verification_material(compact, keys)
    if not isinstance(material, _VerificationMaterial):
        return material
    try:
        payload: Final = jwt.decode(
            compact,
            material.key.get_secret_value(),
            algorithms=[material.algorithm],
            issuer=SESSION_ISSUER,
            options={
                "verify_exp": False,
                "verify_iat": False,
                "verify_nbf": False,
                "require": ["iss", "iat", "exp"],
            },
        )
    except jwt.InvalidSignatureError:
        return SessionBadSignature()
    except (jwt.InvalidTokenError, ValueError, TypeError):
        return SessionMalformed()
    try:
        return _SessionClaims.model_validate(payload)
    except ValidationError:
        return SessionMalformed()
