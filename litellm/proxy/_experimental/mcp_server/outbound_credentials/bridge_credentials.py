"""Consumer-side helpers for the DCR-bridge ``oauth_delegate`` envelope.

A DCR-bridge ``oauth_delegate`` client presents ONE bearer that is a litellm-signed
envelope (see :mod:`.envelope`) carrying both a litellm identity and the upstream OAuth
token. At the MCP admission edge the gateway derives the envelope keys from the proxy
``master_key``, opens the envelope, admits the request under the recovered identity, and
forwards the inner upstream token to the upstream MCP server. This module is the pure
consumer surface; the admission wiring lives in ``user_api_key_auth_mcp.py`` and the
producer (mint) side lands with the token-endpoint flow.
"""

import hashlib
from datetime import datetime
from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, SecretStr

from litellm.proxy._experimental.mcp_server.outbound_credentials.envelope import (
    EnvelopeIdentity,
    EnvelopeKeys,
    OpenedEnvelope,
    is_envelope,
    open_envelope,
)

_SIGNING_KEY_DOMAIN = "litellm-mcp-bridge:envelope-signing:"
_ENCRYPTION_KEY_DOMAIN = "litellm-mcp-bridge:envelope-encryption:"
_BEARER_PREFIX = "bearer "


def envelope_keys_from_master_key(master_key: str) -> EnvelopeKeys:
    """Derive the envelope signing and encryption keys from the proxy master key.

    Domain-separated SHA-256 yields two distinct 64-char (256-bit) hex keys from the one
    secret, so the producer (mint) and consumer (open) agree on keys without persisting
    any, and even a short master key still produces a >= 32-byte signing key (HS256's
    requirement). The derivation is deterministic; rotating ``master_key`` invalidates
    every outstanding envelope, which is the intended behavior for a signing-key change.
    """
    signing = hashlib.sha256((_SIGNING_KEY_DOMAIN + master_key).encode()).hexdigest()
    encryption = hashlib.sha256((_ENCRYPTION_KEY_DOMAIN + master_key).encode()).hexdigest()
    return EnvelopeKeys(signing_key=SecretStr(signing), encryption_key=SecretStr(encryption))


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
    if value[: len(_BEARER_PREFIX)].lower() == _BEARER_PREFIX:
        return value[len(_BEARER_PREFIX) :]
    return value


def resolve_bridge_envelope(authorization_value: str, keys: EnvelopeKeys, now: datetime) -> BridgeEnvelopeResult:
    """Classify an ``Authorization`` value presented to a bridge ``oauth_delegate`` server.

    Strips an optional ``Bearer`` scheme, then returns ``NotBridgeEnvelope`` for a
    non-envelope bearer (normal admission continues), ``BridgeEnvelopeAdmitted`` with the
    recovered identity and the upstream ``Authorization`` value to forward for a valid
    envelope, and ``BridgeEnvelopeInvalid`` for an envelope-shaped bearer that will not
    open. Never raises: it is total over hostile input via :func:`open_envelope`.
    """
    candidate = _strip_bearer(authorization_value)
    if not is_envelope(candidate):
        return NotBridgeEnvelope()
    opened = open_envelope(candidate, keys, now)
    if not isinstance(opened, OpenedEnvelope):
        return BridgeEnvelopeInvalid()
    grant = opened.grant
    upstream_authorization = f"{grant.token_type} {grant.access_token.get_secret_value()}"
    return BridgeEnvelopeAdmitted(identity=opened.identity, upstream_authorization=SecretStr(upstream_authorization))
