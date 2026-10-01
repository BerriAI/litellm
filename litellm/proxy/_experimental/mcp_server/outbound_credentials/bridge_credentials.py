"""Producer and consumer helpers for the DCR-bridge ``oauth_delegate`` envelope.

A DCR-bridge ``oauth_delegate`` client presents ONE bearer that is a litellm-signed
envelope (see :mod:`.envelope`) carrying both a litellm identity and the upstream OAuth
token. The gateway token endpoint mints it (producer) at OAuth issuance, and at the MCP
admission edge the gateway derives the envelope keys from the proxy ``master_key``, opens
it, admits the request under the recovered identity, and forwards the inner upstream token
to the upstream MCP server (consumer). This module is the pure surface for both sides; the
token-endpoint and admission wiring live in their respective call sites.
"""

import os
from collections.abc import Callable
from datetime import datetime
from functools import lru_cache
from typing import Final, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, SecretStr

from litellm.proxy._experimental.mcp_server.outbound_credentials.envelope import (
    EnvelopeIdentity,
    EnvelopeKeys,
    EnvelopeMintError,
    EnvelopeOpenError,
    OpenedEnvelope,
    OpenedRefreshEnvelope,
    RefreshCredential,
    SealedEnvelope,
    UpstreamTokenGrant,
    is_envelope,
    is_refresh_envelope,
    mint_envelope,
    mint_refresh_envelope,
    open_envelope,
    open_refresh_envelope,
)
from litellm.proxy._experimental.mcp_server.outbound_credentials.key_derivation import (
    hkdf_sha256,
    legacy_kdf_grace_enabled,
    legacy_scrypt,
)

_SIGNING_KEY_DOMAIN: Final = b"litellm-mcp-bridge:envelope-signing:"
_ENCRYPTION_KEY_DOMAIN: Final = b"litellm-mcp-bridge:envelope-encryption:"


@lru_cache(maxsize=8)
def envelope_keys_from_master_key(master_key: str) -> EnvelopeKeys:
    """Derive the envelope signing and encryption keys from the proxy master key.

    HKDF-SHA256 (RFC 5869) over the master key, once per domain label as ``info``, yields
    two independent 256-bit subkeys from the one secret, so the producer (mint) and consumer
    (open) agree on keys without persisting any. No memory-hard guessing cost is needed:
    the master key is a high-entropy secret, so a captured envelope is not a cheap offline
    oracle for it either way. The result is cached (the master key is fixed for a process),
    so the KDF runs once per key and adds nothing to the per-request admission path. The
    derivation is deterministic; rotating ``master_key`` invalidates every outstanding
    envelope, which is the intended behavior for a signing-key change.
    """
    return EnvelopeKeys(
        signing_key=SecretStr(hkdf_sha256(master_key, _SIGNING_KEY_DOMAIN)),
        encryption_key=SecretStr(hkdf_sha256(master_key, _ENCRYPTION_KEY_DOMAIN)),
    )


@lru_cache(maxsize=8)
def _legacy_envelope_keys(master_key: str) -> EnvelopeKeys | None:
    """The pre-rotation scrypt subkeys, kept so envelopes minted before the KDF rotation
    can still be opened during the opt-in grace window. ``None`` when scrypt is unavailable."""
    signing: Final = legacy_scrypt(master_key, _SIGNING_KEY_DOMAIN)
    encryption: Final = legacy_scrypt(master_key, _ENCRYPTION_KEY_DOMAIN)
    if signing is None or encryption is None:
        return None
    return EnvelopeKeys(signing_key=SecretStr(signing), encryption_key=SecretStr(encryption))


def legacy_envelope_keys_from_master_key(
    master_key: str, environ: Callable[[str], str | None] = os.environ.get
) -> EnvelopeKeys | None:
    """The legacy envelope keys when the operator enabled the KDF grace window, else ``None``."""
    if not legacy_kdf_grace_enabled(environ):
        return None
    return _legacy_envelope_keys(master_key)


def build_bridge_token_response(
    identity: EnvelopeIdentity,
    grant: UpstreamTokenGrant,
    keys: EnvelopeKeys,
    now: datetime,
) -> SealedEnvelope | EnvelopeMintError:
    """Seal ``grant`` for ``identity`` into the client-held bearer the token endpoint returns.

    The producer mirror of :func:`resolve_bridge_envelope`: a thin, pure wrapper over
    :func:`mint_envelope` that returns the sealed envelope, or the mint error as a value
    for the caller to map onto an OAuth error response.
    """
    return mint_envelope(identity, grant, keys, now)


def build_bridge_refresh_token_response(
    identity: EnvelopeIdentity,
    refresh: RefreshCredential,
    keys: EnvelopeKeys,
    now: datetime,
) -> SealedEnvelope | EnvelopeMintError:
    """Seal ``refresh`` for ``identity`` into the long-lived refresh envelope the token endpoint returns
    alongside the access envelope, so the client can renew without re-authenticating. A thin, pure
    wrapper over :func:`mint_refresh_envelope`; returns the mint error as a value for the caller to map.
    """
    return mint_refresh_envelope(identity, refresh, keys, now)


class BridgeRefreshOpened(BaseModel):
    """A valid refresh envelope presented to the token endpoint: the identity to re-validate and renew
    under, and the upstream refresh grant to exchange."""

    model_config = ConfigDict(frozen=True)
    tag: Literal["opened"] = "opened"
    identity: EnvelopeIdentity
    refresh: RefreshCredential


class BridgeRefreshInvalid(BaseModel):
    """The presented refresh grant is not a valid refresh envelope for this server (not refresh-shaped,
    will not open, or minted for a different server); the token endpoint fails the refresh closed."""

    model_config = ConfigDict(frozen=True)
    tag: Literal["invalid"] = "invalid"


BridgeRefreshResult: TypeAlias = BridgeRefreshOpened | BridgeRefreshInvalid


def _open_envelope_with_fallback(
    candidate: str,
    keys: EnvelopeKeys,
    legacy_keys: EnvelopeKeys | None,
    now: datetime,
) -> OpenedEnvelope | EnvelopeOpenError:
    """Open an access envelope under the active keys, then under the legacy keys when
    grace supplied them."""
    primary: Final = open_envelope(candidate, keys, now)
    if isinstance(primary, OpenedEnvelope) or legacy_keys is None:
        return primary
    return open_envelope(candidate, legacy_keys, now)


def _open_refresh_envelope_with_fallback(
    candidate: str,
    keys: EnvelopeKeys,
    legacy_keys: EnvelopeKeys | None,
    now: datetime,
) -> OpenedRefreshEnvelope | EnvelopeOpenError:
    """Open a refresh envelope under the active keys, then under the legacy keys when
    grace supplied them."""
    primary: Final = open_refresh_envelope(candidate, keys, now)
    if isinstance(primary, OpenedRefreshEnvelope) or legacy_keys is None:
        return primary
    return open_refresh_envelope(candidate, legacy_keys, now)


def open_bridge_refresh_envelope(
    refresh_value: str,
    keys: EnvelopeKeys,
    now: datetime,
    expected_server_id: str,
    legacy_keys: EnvelopeKeys | None = None,
) -> BridgeRefreshResult:
    """Open a refresh envelope a bridge ``oauth_delegate`` client presented on a refresh_token grant.

    The token-endpoint mirror of :func:`resolve_bridge_envelope`: strips an optional ``Bearer`` scheme,
    then returns ``BridgeRefreshOpened`` with the recovered identity and upstream refresh grant, or
    ``BridgeRefreshInvalid`` for anything that is not a valid refresh envelope for this server. Never
    raises; total over hostile input via :func:`open_refresh_envelope`. ``expected_server_id`` binds the
    envelope to the server the request targets, so a refresh envelope minted for one server cannot renew
    against another. A raw upstream refresh token (not envelope-shaped) is ``BridgeRefreshInvalid``: this
    mode never hands the client a bare upstream refresh token, so it must never accept one.
    """
    candidate: Final = _strip_bearer(refresh_value)
    if not is_refresh_envelope(candidate):
        return BridgeRefreshInvalid()
    opened: Final = _open_refresh_envelope_with_fallback(candidate, keys, legacy_keys, now)
    if not isinstance(opened, OpenedRefreshEnvelope):
        return BridgeRefreshInvalid()
    if opened.identity.server_id != expected_server_id:
        return BridgeRefreshInvalid()
    return BridgeRefreshOpened(identity=opened.identity, refresh=opened.refresh)


class NotBridgeEnvelope(BaseModel):
    """The bearer is not an envelope; admission continues on its normal path."""

    model_config = ConfigDict(frozen=True)
    tag: Literal["not_bridge_envelope"] = "not_bridge_envelope"


class BridgeEnvelopeAdmitted(BaseModel):
    """A valid envelope: the identity to admit under and the full upstream ``Authorization``
    value (``token_type access_token``) to forward to the upstream MCP server."""

    model_config = ConfigDict(frozen=True)
    tag: Literal["admitted"] = "admitted"
    identity: EnvelopeIdentity
    upstream_authorization: SecretStr


class BridgeEnvelopeInvalid(BaseModel):
    """The bearer is envelope-shaped but did not open (expired, tampered, wrong key);
    admission must fail closed rather than fall through to normal validation."""

    model_config = ConfigDict(frozen=True)
    tag: Literal["invalid"] = "invalid"


BridgeEnvelopeResult: TypeAlias = NotBridgeEnvelope | BridgeEnvelopeAdmitted | BridgeEnvelopeInvalid


def _strip_bearer(value: str) -> str:
    parts: Final = value.split(None, 1)
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1]
    return value


def is_bridge_envelope_shaped(authorization_value: str) -> bool:
    """Cheap, keyless test that an ``Authorization`` value carries an envelope of either kind (optional
    ``Bearer`` scheme stripped). The admission edge engages the bridge arm for an access envelope (to
    admit) and for a refresh envelope (to reject it explicitly, since a refresh credential is never
    usable at the tool-call edge); a plain upstream bearer falls through to normal oauth2 admission."""
    candidate: Final = _strip_bearer(authorization_value)
    return is_envelope(candidate) or is_refresh_envelope(candidate)


def resolve_bridge_envelope(
    authorization_value: str,
    keys: EnvelopeKeys,
    now: datetime,
    expected_server_id: str,
    legacy_keys: EnvelopeKeys | None = None,
) -> BridgeEnvelopeResult:
    """Classify an ``Authorization`` value presented to a bridge ``oauth_delegate`` server.

    Strips an optional ``Bearer`` scheme, then returns ``NotBridgeEnvelope`` for a
    non-envelope bearer (normal admission continues), ``BridgeEnvelopeAdmitted`` with the
    recovered identity and the upstream ``Authorization`` value to forward for a valid
    envelope, and ``BridgeEnvelopeInvalid`` for an envelope-shaped bearer that will not
    open. Never raises: it is total over hostile input via :func:`open_envelope`.

    A refresh envelope is ``BridgeEnvelopeInvalid`` here: it is a valid gateway credential but only ever
    presented back to the token endpoint, never usable to authenticate a tool call, so admission must
    fail it closed rather than let it fall through to another arm.

    ``expected_server_id`` is the ``server_id`` of the MCP server the request targets; an
    opened envelope whose sealed ``server_id`` does not match is rejected as
    ``BridgeEnvelopeInvalid``. Binding here (rather than leaving it to the caller) prevents
    replaying an envelope minted for one server against another, which would forward the
    first server's upstream credential across a server boundary. ``server_id`` is not a
    secret (the caller targets that server), so a plain equality check is sufficient and,
    unlike ``hmac.compare_digest`` on ``str``, does not raise on a non-ASCII server_id.
    """
    candidate: Final = _strip_bearer(authorization_value)
    if is_refresh_envelope(candidate):
        return BridgeEnvelopeInvalid()
    if not is_envelope(candidate):
        return NotBridgeEnvelope()
    opened: Final = _open_envelope_with_fallback(candidate, keys, legacy_keys, now)
    if not isinstance(opened, OpenedEnvelope):
        return BridgeEnvelopeInvalid()
    if opened.identity.server_id != expected_server_id:
        return BridgeEnvelopeInvalid()
    grant: Final = opened.grant
    authorization_scheme: Final = "Bearer" if grant.token_type.lower() == "bearer" else grant.token_type
    upstream_authorization: Final = f"{authorization_scheme} {grant.access_token.get_secret_value()}"
    return BridgeEnvelopeAdmitted(identity=opened.identity, upstream_authorization=SecretStr(upstream_authorization))
