"""Identity-only session tokens for the gateway-level (aggregate ``/mcp``) DCR front door.

A DCR client that signs in through LiteLLM SSO holds ONE bearer that carries ONLY a
litellm identity; unlike the :mod:`.envelope` bridge bearer it seals no upstream
credential, because the custody model vaults every upstream token server-side in
``LiteLLM_MCPUserCredentials`` and egress resolves them by user at call time. The token
is therefore a stable REFERENCE, not an authorization: admission reloads the live user
record and policy on every request, so deactivating the user (or their team) kills
outstanding sessions immediately without a revocation store.

Wire shape: ``llm_session_`` (access) / ``llm_srefresh_`` (refresh) + an HS256 JWT,
the same signing approach as :mod:`.envelope`. Claims are ``iss``/``iat``/``exp``
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
from datetime import datetime, timedelta
from typing import Literal, TypeAlias

import jwt
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError

SESSION_TOKEN_PREFIX = "llm_session_"
"""Marker prefix on every serialized session ACCESS token so the admission edge can cheaply
tell a gateway session from a litellm key, JWT, or bridge envelope before doing any
cryptography. Distinct from the ``llm_env_``/``llm_refresh_`` envelope prefixes."""

SESSION_REFRESH_PREFIX = "llm_srefresh_"
"""Marker prefix on every serialized session REFRESH token. A distinct prefix keeps the two
credentials routable without crypto and, together with the signed ``kind`` claim, stops one
from being presented where the other is expected: the refresh token is only ever presented
back to the token endpoint, never at the MCP edge."""

SESSION_ISSUER = "litellm-mcp-gateway"
"""``iss`` claim stamped into every session token and required back on open. Distinct from
the envelope issuer so a token of one family can never validate in the other even under a
hypothetical shared signing key."""

SESSION_TTL_SECONDS = 3600
"""Session ACCESS token lifetime (1h), matching the access-envelope and BYOK session bearer
windows: a client-held credential never outlives a bounded window, and each refresh
re-validates the live user before re-minting."""

SESSION_REFRESH_TTL_SECONDS = 1209600
"""Session REFRESH token lifetime (14 days), matching the refresh-envelope bound. Each
renewal re-validates the sealed user against the live record (deactivation gates it) and
rotates the refresh token, so the practical bound is idle time, not a fixed session."""

MAX_SESSION_TOKEN_BYTES = 4096
"""Size cap on the serialized token (prefix + JWT, in bytes) and on any candidate accepted
by the openers. Session claims are small; the only variable-length field is ``client_id``
(a sealed DCR client record), and 4096 leaves ample headroom under common 8-16KB header
limits while bounding hostile input before JWT parsing."""

_SESSION_JWT_ALGORITHM = "HS256"

SessionTokenKind = Literal["session", "session_refresh"]
"""Which credential a session token is. Stamped into the signed claims and required to match
on open, so a signature-valid token of one kind cannot be replayed as the other even if its
wire prefix is swapped (the prefix is not part of the signed payload; this claim is)."""


class SessionPrincipal(BaseModel):
    """The litellm user a session token identifies and the DCR client it was issued to.

    ``user_id`` is the SSO-established litellm user subject, never a credential: admission
    reloads the live user record by it, so current role, team, and revocation state are
    enforced at use time rather than frozen at mint time. ``client_id`` is the (stateless,
    gateway-sealed) DCR client identifier the token was issued to; the token endpoint
    requires it to match on the refresh grant.
    """

    model_config = ConfigDict(frozen=True)
    user_id: str = Field(min_length=1)
    client_id: str = Field(min_length=1)


class SessionKeys(BaseModel):
    """Injected key material: the HS256 signing key.

    ``signing_key`` must be at least 32 bytes: HS256's HMAC-SHA256 has a 256-bit security
    level, RFC 7518 requires a key of at least that size, and a shorter key makes PyJWT
    emit ``InsecureKeyLengthWarning``.
    """

    model_config = ConfigDict(frozen=True)
    signing_key: SecretStr = Field(min_length=32)


class MintedSessionToken(BaseModel):
    """A minted session token: the client-held bearer value and when it expires."""

    model_config = ConfigDict(frozen=True)
    token: SecretStr
    expires_at: datetime


class OpenedSessionToken(BaseModel):
    """A validated session token of either kind: the principal it was minted for, plus the
    ``jti`` so the token endpoint can enforce single-use rotation on a refresh token."""

    model_config = ConfigDict(frozen=True)
    principal: SessionPrincipal
    jti: str


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
    keys: SessionKeys,
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
    keys: SessionKeys,
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
    keys: SessionKeys,
    now: datetime,
) -> OpenedSessionToken | SessionTokenOpenError:
    """Validate a session ACCESS ``candidate`` and recover the principal.

    Never raises for bad input: every invalid, expired, tampered, or wrong-kind candidate
    maps to a distinct ``SessionTokenOpenError`` variant.
    """
    return _open(candidate, prefix=SESSION_TOKEN_PREFIX, expected_kind="session", keys=keys, now=now)


def open_session_refresh_token(
    candidate: str,
    keys: SessionKeys,
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
    keys: SessionKeys,
    now: datetime,
) -> MintedSessionToken | SessionTokenTooLarge:
    """Sign the claims for either token kind and enforce the size cap. Shared by both mints
    so the JWT shape, issuer, and size guard cannot drift between access and refresh."""
    claims = _SessionClaims(
        iss=SESSION_ISSUER,
        iat=int(now.timestamp()),
        exp=int(expires_at.timestamp()),
        jti=secrets.token_urlsafe(16),
        kind=kind,
        user_id=principal.user_id,
        client_id=principal.client_id,
    )
    token = prefix + jwt.encode(
        claims.model_dump(), keys.signing_key.get_secret_value(), algorithm=_SESSION_JWT_ALGORITHM
    )
    size_bytes = len(token.encode("utf-8"))
    if size_bytes > MAX_SESSION_TOKEN_BYTES:
        return SessionTokenTooLarge(size_bytes=size_bytes, max_bytes=MAX_SESSION_TOKEN_BYTES)
    return MintedSessionToken(token=SecretStr(token), expires_at=expires_at)


def _open(
    candidate: str,
    prefix: str,
    expected_kind: SessionTokenKind,
    keys: SessionKeys,
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
    claims = _decode_claims(candidate.removeprefix(prefix), keys.signing_key)
    if not isinstance(claims, _SessionClaims):
        return claims
    if claims.kind != expected_kind:
        return SessionMalformed()
    if now.timestamp() >= claims.exp:
        return SessionExpired()
    return OpenedSessionToken(
        principal=SessionPrincipal(user_id=claims.user_id, client_id=claims.client_id), jti=claims.jti
    )


def _decode_claims(
    compact: str,
    signing_key: SecretStr,
) -> _SessionClaims | SessionBadSignature | SessionMalformed:
    """Verify the HS256 signature and shape of an attacker-controlled compact JWT.

    ``compact`` is fully hostile and bounded to ``MAX_SESSION_TOKEN_BYTES`` by the caller.
    PyJWT's ``iat``/``nbf``/``exp`` validators are disabled: they raise on hostile claim
    types and, for ``iat``/``nbf``, compare against the wall clock rather than the injected
    ``now`` (``exp`` is checked by the caller against ``now``). Apart from a signature
    mismatch, every decode failure is ``SessionMalformed``: a non-UTF-8 candidate surfaces
    as ``UnicodeEncodeError`` (a ``ValueError``), a non-string registered claim as a
    ``TypeError`` from PyJWT's claim validators, and a wrong issuer or structurally invalid
    token as an ``InvalidTokenError``. ``_SessionClaims`` is the total type gate.
    """
    try:
        payload = jwt.decode(
            compact,
            signing_key.get_secret_value(),
            algorithms=[_SESSION_JWT_ALGORITHM],
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
