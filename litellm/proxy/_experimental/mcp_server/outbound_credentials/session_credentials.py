"""Producer and consumer helpers for the gateway-level DCR session token.

The aggregate ``/mcp`` front door (``mcp_gateway_dcr``) issues the identity-only session
tokens defined in :mod:`.session_token`. The gateway token endpoint mints them (producer)
after SSO sign-in, and at the MCP admission edge the gateway derives the session signing
key from the proxy ``master_key``, opens the bearer, and admits the request under the
recovered litellm user (consumer), reloading the live user record and policy before
anything runs. This module is the pure surface for both sides; the token-endpoint and
admission wiring live in their respective call sites.

The signing key is derived with the same memory-hard scrypt construction as
:func:`~.bridge_credentials.envelope_keys_from_master_key` but under a distinct domain
label, so session tokens and bridge envelopes never share key material: a token of one
family is unverifiable in the other by key separation, on top of the distinct issuers,
prefixes, and claim shapes.
"""

import hashlib
from datetime import datetime
from functools import lru_cache
from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, SecretStr

from litellm.proxy._experimental.mcp_server.outbound_credentials.session_token import (
    OpenedSessionToken,
    SessionExpired,
    SessionKeys,
    SessionPrincipal,
    is_session_refresh_token,
    is_session_token,
    open_session_refresh_token,
    open_session_token,
)

_SESSION_SIGNING_KEY_DOMAIN = b"litellm-mcp-gateway:session-signing:"

# scrypt work factors (RFC 7914), identical to the envelope KDF: memory-hard so a captured
# session token is not a cheap offline oracle for the master key.
_SCRYPT_N = 2**15
_SCRYPT_R = 8
_SCRYPT_P = 1
_SCRYPT_MAXMEM = 128 * _SCRYPT_N * _SCRYPT_R * _SCRYPT_P * 2
_DERIVED_KEY_BYTES = 32


@lru_cache(maxsize=8)
def session_keys_from_master_key(master_key: str) -> SessionKeys:
    """Derive the session signing key from the proxy master key.

    A memory-hard scrypt KDF (RFC 7914) over a session-specific domain-label salt yields a
    256-bit subkey from the one secret, so the producer (mint) and consumer (open) agree on
    the key without persisting any. The domain label differs from both envelope labels in
    :mod:`.bridge_credentials`, so compromise or misuse of one token family never crosses
    into the other. The result is cached (the master key is fixed for a process); rotating
    ``master_key`` invalidates every outstanding session, which is the intended behavior
    for a signing-key change.
    """
    signing = hashlib.scrypt(
        master_key.encode(),
        salt=_SESSION_SIGNING_KEY_DOMAIN,
        n=_SCRYPT_N,
        r=_SCRYPT_R,
        p=_SCRYPT_P,
        maxmem=_SCRYPT_MAXMEM,
        dklen=_DERIVED_KEY_BYTES,
    ).hex()
    return SessionKeys(signing_key=SecretStr(signing))


class NotSessionBearer(BaseModel):
    """The bearer is not session-shaped; admission continues on its normal path."""

    model_config = ConfigDict(frozen=True)
    tag: Literal["not_session_bearer"] = "not_session_bearer"


class SessionBearerAdmitted(BaseModel):
    """A valid session access token: the principal to admit under after a live reload."""

    model_config = ConfigDict(frozen=True)
    tag: Literal["admitted"] = "admitted"
    principal: SessionPrincipal


class SessionBearerInvalid(BaseModel):
    """The bearer is session-shaped but must not admit (expired, tampered, wrong key, or a
    refresh token presented at the tool-call edge); admission fails closed with the
    ``invalid_token`` challenge rather than falling through to another arm. ``expired``
    distinguishes a routine expiry (debug-log worthy) from a tampered or foreign token."""

    model_config = ConfigDict(frozen=True)
    tag: Literal["invalid"] = "invalid"
    expired: bool = False


SessionBearerResult: TypeAlias = NotSessionBearer | SessionBearerAdmitted | SessionBearerInvalid


def _strip_bearer(value: str) -> str:
    parts = value.split(None, 1)
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1]
    return value


def is_session_bearer_shaped(authorization_value: str) -> bool:
    """Cheap, keyless test that an ``Authorization`` value carries a session token of either
    kind (optional ``Bearer`` scheme stripped). The admission edge engages the session arm
    for an access token (to admit) and for a refresh token (to reject it explicitly, since
    a refresh credential is never usable at the tool-call edge); anything else falls
    through to normal admission."""
    candidate = _strip_bearer(authorization_value)
    return is_session_token(candidate) or is_session_refresh_token(candidate)


def resolve_session_bearer(
    authorization_value: str,
    keys: SessionKeys,
    now: datetime,
) -> SessionBearerResult:
    """Classify an ``Authorization`` value presented at the aggregate MCP edge.

    Strips an optional ``Bearer`` scheme, then returns ``NotSessionBearer`` for a
    non-session bearer (normal admission continues), ``SessionBearerAdmitted`` with the
    recovered principal for a valid access token, and ``SessionBearerInvalid`` for a
    session-shaped bearer that must not admit. Never raises: total over hostile input via
    :func:`~.session_token.open_session_token`.

    A refresh token is ``SessionBearerInvalid`` here: it is a valid gateway credential but
    only ever presented back to the token endpoint, so admission must fail it closed rather
    than let it fall through to another arm.
    """
    candidate = _strip_bearer(authorization_value)
    if is_session_refresh_token(candidate):
        return SessionBearerInvalid()
    if not is_session_token(candidate):
        return NotSessionBearer()
    opened = open_session_token(candidate, keys, now)
    if isinstance(opened, OpenedSessionToken):
        return SessionBearerAdmitted(principal=opened.principal)
    return SessionBearerInvalid(expired=isinstance(opened, SessionExpired))


class SessionRefreshOpened(BaseModel):
    """A valid session refresh token presented to the token endpoint: the principal to
    re-validate and renew under."""

    model_config = ConfigDict(frozen=True)
    tag: Literal["opened"] = "opened"
    principal: SessionPrincipal


class SessionRefreshInvalid(BaseModel):
    """The presented refresh grant is not a valid session refresh token for this client
    (not refresh-shaped, will not open, or bound to a different ``client_id``); the token
    endpoint fails the refresh closed."""

    model_config = ConfigDict(frozen=True)
    tag: Literal["invalid"] = "invalid"


SessionRefreshResult: TypeAlias = SessionRefreshOpened | SessionRefreshInvalid


def open_session_refresh_bearer(
    refresh_value: str,
    keys: SessionKeys,
    now: datetime,
    expected_client_id: str,
) -> SessionRefreshResult:
    """Open a session refresh token presented on a ``refresh_token`` grant.

    The token-endpoint mirror of :func:`resolve_session_bearer`: strips an optional
    ``Bearer`` scheme, then returns ``SessionRefreshOpened`` with the recovered principal,
    or ``SessionRefreshInvalid`` for anything that is not a valid session refresh token
    issued to ``expected_client_id``. Never raises. The client binding (RFC 6749 section 6)
    stops a refresh token stolen from one DCR client from being renewed through another;
    ``client_id`` is not a secret (the caller presents it), so a plain equality check is
    sufficient and, unlike ``hmac.compare_digest`` on ``str``, does not raise on non-ASCII.
    """
    candidate = _strip_bearer(refresh_value)
    if not is_session_refresh_token(candidate):
        return SessionRefreshInvalid()
    opened = open_session_refresh_token(candidate, keys, now)
    if not isinstance(opened, OpenedSessionToken):
        return SessionRefreshInvalid()
    if opened.principal.client_id != expected_client_id:
        return SessionRefreshInvalid()
    return SessionRefreshOpened(principal=opened.principal)
