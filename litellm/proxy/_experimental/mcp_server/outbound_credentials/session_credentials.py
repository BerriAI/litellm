"""Producer and consumer helpers for the gateway-level DCR session token.

The aggregate ``/mcp`` front door (``mcp_gateway_dcr``) issues the identity-only session
tokens defined in :mod:`.session_token`. The gateway token endpoint mints them (producer)
after SSO sign-in, and at the MCP admission edge the gateway derives the session signing
key from the proxy ``master_key``, opens the bearer, and admits the request under the
recovered litellm user (consumer), reloading the live user record and policy before
anything runs. This module is the pure surface for both sides; the token-endpoint and
admission wiring live in their respective call sites.

The signing key is derived with the same HKDF-SHA256 construction as
:func:`~.bridge_credentials.envelope_keys_from_master_key` but under a distinct domain
label, so session tokens and bridge envelopes never share key material: a token of one
family is unverifiable in the other by key separation, on top of the distinct issuers,
prefixes, and claim shapes.
"""

import os
from collections.abc import Callable
from datetime import datetime
from functools import lru_cache
from typing import Final, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError

from litellm.proxy._experimental.mcp_server.outbound_credentials.key_derivation import (
    hkdf_sha256,
    legacy_kdf_grace_enabled,
    legacy_scrypt,
)
from litellm.proxy._experimental.mcp_server.outbound_credentials.session_token import (
    AsymmetricSessionKeys,
    OpenedSessionToken,
    SessionExpired,
    SessionKeys,
    SessionPrincipal,
    SessionRotatedPublicKey,
    SessionSigningKeys,
    SessionTokenOpenError,
    is_session_refresh_token,
    is_session_token,
    open_session_refresh_token,
    open_session_token,
)

_SESSION_SIGNING_KEY_DOMAIN: Final = b"litellm-mcp-gateway:session-signing:"


@lru_cache(maxsize=8)
def session_keys_from_master_key(master_key: str) -> SessionKeys:
    """Derive the session signing key from the proxy master key.

    HKDF-SHA256 (RFC 5869) over the master key with a session-specific domain label as
    ``info`` yields a 256-bit subkey from the one secret, so the producer (mint) and
    consumer (open) agree on the key without persisting any. The label differs from both
    envelope labels in :mod:`.bridge_credentials`, so compromise or misuse of one token
    family never crosses into the other. The result is cached (the master key is fixed for
    a process); rotating ``master_key`` invalidates every outstanding session, which is
    the intended behavior for a signing-key change.
    """
    return SessionKeys(signing_key=SecretStr(hkdf_sha256(master_key, _SESSION_SIGNING_KEY_DOMAIN)))


@lru_cache(maxsize=8)
def _legacy_session_keys(master_key: str) -> SessionKeys | None:
    """The pre-rotation scrypt signing key, kept so session tokens minted before the KDF
    rotation can still be opened during the opt-in grace window. ``None`` when scrypt is
    unavailable."""
    signing: Final = legacy_scrypt(master_key, _SESSION_SIGNING_KEY_DOMAIN)
    if signing is None:
        return None
    return SessionKeys(signing_key=SecretStr(signing))


def legacy_session_keys_from_master_key(
    master_key: str,
    active: SessionSigningKeys,
    environ: Callable[[str], str | None] = os.environ.get,
) -> SessionKeys | None:
    """The legacy session key when the operator enabled the KDF grace window, else ``None``.

    The fallback only applies to the default master-key HS256 path: a configured
    ``mcp_session_token_signing`` RS256 key was never scrypt-derived, so an
    :class:`AsymmetricSessionKeys` ``active`` gets no legacy key even with grace on.
    """
    if not isinstance(active, SessionKeys) or not legacy_kdf_grace_enabled(environ):
        return None
    return _legacy_session_keys(master_key)


class SessionSigningPreviousKey(BaseModel):
    """One retired key in ``mcp_session_token_signing.previous_public_keys``: its ``kid``
    and the PEM public half (inline or an ``os.environ/`` reference)."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    kid: str = Field(min_length=1)
    public_key: str = Field(min_length=1)


class MCPSessionTokenSigningSettings(BaseModel):
    """The ``general_settings.mcp_session_token_signing`` block: opt-in asymmetric signing
    for the gateway session tokens. Absent, the gateway keeps the backward-compatible
    HS256 key derived from ``master_key``. ``private_key`` and each ``public_key`` accept
    a PEM string inline or an ``os.environ/<NAME>`` (or secret manager) reference."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    algorithm: Literal["RS256"]
    kid: str = Field(min_length=1)
    private_key: str = Field(min_length=1)
    previous_public_keys: tuple[SessionSigningPreviousKey, ...] = ()


class SessionSigningConfigError(BaseModel):
    """``mcp_session_token_signing`` is present but unusable (bad shape, unresolvable
    secret reference, or a key that is not a loadable RSA PEM); the caller fails closed
    with a server error instead of silently falling back to HS256."""

    model_config = ConfigDict(frozen=True)
    tag: Literal["session_signing_config_error"] = "session_signing_config_error"
    detail: str


def _resolve_key_material(value: str) -> str | None:
    if not value.startswith("os.environ/"):
        return value
    from litellm.secret_managers.main import get_secret_str  # noqa: PLC0415  # heavy import kept off the pure path

    return get_secret_str(value)


def resolve_session_signing_keys(
    master_key: str,
    raw_settings: object | None,
) -> SessionSigningKeys | SessionSigningConfigError:
    """Turn the operator's ``mcp_session_token_signing`` setting into signing key material.

    ``None`` (the setting absent) keeps the backward-compatible HS256 key derived from
    ``master_key``. A present setting must fully validate into RS256 material; any defect
    is a ``SessionSigningConfigError`` value so token issuance and admission fail closed
    rather than minting under a key the operator did not intend.
    """
    if raw_settings is None:
        return session_keys_from_master_key(master_key)
    try:
        settings: Final = MCPSessionTokenSigningSettings.model_validate(raw_settings)
    except ValidationError as exc:
        return SessionSigningConfigError(detail=f"mcp_session_token_signing is malformed: {exc}")
    private_pem: Final = _resolve_key_material(settings.private_key)
    if private_pem is None:
        return SessionSigningConfigError(detail="mcp_session_token_signing.private_key reference did not resolve")
    resolved_previous: Final = tuple(
        (previous.kid, _resolve_key_material(previous.public_key)) for previous in settings.previous_public_keys
    )
    unresolved: Final = tuple(kid for kid, pem in resolved_previous if pem is None)
    if unresolved:
        return SessionSigningConfigError(
            detail=f"mcp_session_token_signing.previous_public_keys reference did not resolve for kid(s): {', '.join(unresolved)}"
        )
    try:
        return AsymmetricSessionKeys(
            private_key_pem=SecretStr(private_pem),
            kid=settings.kid,
            previous_public_keys=tuple(
                SessionRotatedPublicKey(kid=kid, public_key_pem=pem)
                for kid, pem in resolved_previous
                if pem is not None
            ),
        )
    except ValidationError as exc:
        return SessionSigningConfigError(
            detail=f"mcp_session_token_signing keys are not usable RSA PEM material: {exc}"
        )


def active_session_signing_keys(master_key: str) -> SessionSigningKeys | SessionSigningConfigError:
    """Wiring helper for the token endpoint and the admission edge: resolve the signing
    keys from the live ``general_settings.mcp_session_token_signing`` block, or derive the
    default HS256 key from ``master_key`` when the block is absent."""
    from litellm.proxy.proxy_server import general_settings  # noqa: PLC0415  # circular import at module load

    return resolve_session_signing_keys(master_key, general_settings.get("mcp_session_token_signing"))


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
    parts: Final = value.split(None, 1)
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1]
    return value


def is_session_bearer_shaped(authorization_value: str) -> bool:
    """Cheap, keyless test that an ``Authorization`` value carries a session token of either
    kind (optional ``Bearer`` scheme stripped). The admission edge engages the session arm
    for an access token (to admit) and for a refresh token (to reject it explicitly, since
    a refresh credential is never usable at the tool-call edge); anything else falls
    through to normal admission."""
    candidate: Final = _strip_bearer(authorization_value)
    return is_session_token(candidate) or is_session_refresh_token(candidate)


def resolve_session_bearer(
    authorization_value: str,
    keys: SessionSigningKeys,
    now: datetime,
    legacy_keys: SessionKeys | None = None,
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
    candidate: Final = _strip_bearer(authorization_value)
    if is_session_refresh_token(candidate):
        return SessionBearerInvalid()
    if not is_session_token(candidate):
        return NotSessionBearer()
    opened: Final = open_session_credential_with_legacy(open_session_token, candidate, keys, legacy_keys, now)
    if isinstance(opened, OpenedSessionToken):
        return SessionBearerAdmitted(principal=opened.principal)
    return SessionBearerInvalid(expired=isinstance(opened, SessionExpired))


class SessionRefreshOpened(BaseModel):
    """A valid session refresh token presented to the token endpoint: the principal to
    re-validate and renew under."""

    model_config = ConfigDict(frozen=True)
    tag: Literal["opened"] = "opened"
    principal: SessionPrincipal
    jti: str


class SessionRefreshInvalid(BaseModel):
    """The presented refresh grant is not a valid session refresh token for this client
    (not refresh-shaped, will not open, or bound to a different ``client_id``); the token
    endpoint fails the refresh closed."""

    model_config = ConfigDict(frozen=True)
    tag: Literal["invalid"] = "invalid"


SessionRefreshResult: TypeAlias = SessionRefreshOpened | SessionRefreshInvalid


def open_session_credential_with_legacy(
    open_token: Callable[[str, SessionSigningKeys, datetime], OpenedSessionToken | SessionTokenOpenError],
    token: str,
    keys: SessionSigningKeys,
    legacy_keys: SessionKeys | None,
    now: datetime,
) -> OpenedSessionToken | SessionTokenOpenError:
    """Open a session credential under the active keys, then under the legacy keys when
    grace supplied them. An :class:`OpenedSessionToken` or :class:`SessionExpired` under
    the active keys is final: expiry must still report as expiry, so only a signature or
    format failure falls through to the legacy derivation."""
    primary: Final = open_token(token, keys, now)
    if isinstance(primary, (OpenedSessionToken, SessionExpired)) or legacy_keys is None:
        return primary
    return open_token(token, legacy_keys, now)


def open_session_refresh_bearer(
    refresh_value: str,
    keys: SessionSigningKeys,
    now: datetime,
    expected_client_id: str,
    legacy_keys: SessionKeys | None = None,
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
    candidate: Final = _strip_bearer(refresh_value)
    if not is_session_refresh_token(candidate):
        return SessionRefreshInvalid()
    opened: Final = open_session_credential_with_legacy(open_session_refresh_token, candidate, keys, legacy_keys, now)
    if not isinstance(opened, OpenedSessionToken):
        return SessionRefreshInvalid()
    if opened.principal.client_id != expected_client_id:
        return SessionRefreshInvalid()
    return SessionRefreshOpened(principal=opened.principal, jti=opened.jti)
