"""Client-held sealed envelope for the oauth_delegate DCR bridge.

A DCR-bridge client holds ONE bearer that must carry BOTH a litellm identity and the
upstream OAuth grant, with zero server-side storage. The gateway token endpoint mints a
litellm-signed envelope (:func:`mint_envelope`); the MCP edge validates it, recovers the
identity claims and the inner upstream grant (:func:`open_envelope`), and forwards the
inner access token upstream. This module is pure and unwired: it imports nothing from
endpoint or edge code, reads no proxy globals, and takes all key material and the clock
as explicit parameters.

Wire shape: ``llm_env_`` + an HS256 JWT (same signing approach as the BYOK session
bearer in ``byok_oauth_endpoints.py``). Registered claims are ``iss``/``iat``/``exp``;
custom claims are ``server_id``, ``key_hash``, and ``grant``, where ``grant`` is the
upstream token grant serialized to JSON, encrypted with the repo's symmetric
encryption helpers (``encrypt_value``/``decrypt_value`` from
``encrypt_decrypt_utils`` — the same family ``encrypt_value_helper`` applies to
persisted DCR credentials), and base64url-encoded, so the inner token never appears
in plaintext anywhere in the envelope.

Failures are values: :func:`open_envelope` returns one of the frozen
``EnvelopeOpenError`` variants (discriminated on ``tag``) for invalid, expired,
tampered, or undecryptable input, and :func:`mint_envelope` returns
``EnvelopeTooLarge`` for oversized grants. Error values carry tags and sizes only,
never token material.

The pydantic input models reject programmer errors at construction (e.g. a
non-positive ``expires_in`` or an empty required field). :func:`open_envelope` is
additionally total over hostile, attacker-controlled input: it never raises, only
returns an ``EnvelopeOpenError``. :func:`mint_envelope` operates on a
gateway-supplied grant (an upstream IdP's UTF-8 JSON token response), so it does not
defend against non-UTF-8 field content that cannot survive JSON parsing; its only
value-typed failure is ``EnvelopeTooLarge``.
"""

from __future__ import annotations

import base64
from datetime import datetime, timedelta
from typing import Literal, TypeAlias

import jwt
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError

from litellm.proxy.common_utils.encrypt_decrypt_utils import decrypt_value, encrypt_value

ENVELOPE_PREFIX = "llm_env_"
"""Marker prefix on every serialized envelope so the edge can cheaply tell an envelope
from a raw upstream token before doing any cryptography."""

ENVELOPE_ISSUER = "litellm-mcp-bridge"
"""``iss`` claim stamped into every envelope and required back on open."""

MAX_ENVELOPE_TTL_SECONDS = 3600
"""Hard ceiling on envelope lifetime. ``exp`` is ``min(upstream expires_in, this cap)``
(the cap alone when the upstream omits ``expires_in``), matching the 1h lifetime of the
BYOK session bearer this module's signing approach is borrowed from: a client-held
credential should never outlive a bounded window even when the upstream token does."""

MAX_ENVELOPE_BYTES = 12288
"""Size cap on the final serialized envelope (prefix + JWT, in bytes). Upstream JWTs
commonly run 2-4KB; base64 plus encryption overhead roughly doubles that inside the
envelope, and common proxy/server header limits sit around 16KB total. 12288 leaves
comfortable headroom for a large upstream token while keeping the envelope safely
transmittable as a single Authorization header. Oversized grants are rejected with a
typed error, never truncated."""

_ENVELOPE_JWT_ALGORITHM = "HS256"


class EnvelopeIdentity(BaseModel):
    """The litellm identity the envelope binds the inner grant to.

    ``key_hash`` is the hashed litellm key that authorized the mint, never a raw
    credential (and the edge rejects a bare hash presented as a bearer). Admission
    reloads the live key record by it, so the key's current team/org/object-permission
    restrictions and its revocation state are enforced at use time rather than frozen at
    mint time. ``server_id`` binds the envelope to one MCP server so it cannot be replayed
    across a server boundary.
    """

    model_config = ConfigDict(frozen=True)
    server_id: str = Field(min_length=1)
    key_hash: str = Field(min_length=1)


class UpstreamTokenGrant(BaseModel):
    """The upstream OAuth token response fields sealed inside the envelope.

    ``expires_in`` must be positive when present; a non-positive value is a programmer
    error rejected at construction. Token fields are ``SecretStr`` so reprs never leak
    them.
    """

    model_config = ConfigDict(frozen=True)
    access_token: SecretStr = Field(min_length=1)
    token_type: str = Field(min_length=1)
    refresh_token: SecretStr | None = None
    scope: str | None = None
    expires_in: int | None = Field(default=None, gt=0)


class EnvelopeKeys(BaseModel):
    """Injected key material: the HS256 signing key and the symmetric encryption key.

    ``signing_key`` must be at least 32 bytes: HS256's HMAC-SHA256 has a 256-bit
    security level, RFC 7518 requires a key of at least that size, and a shorter key
    makes PyJWT emit ``InsecureKeyLengthWarning``.
    """

    model_config = ConfigDict(frozen=True)
    signing_key: SecretStr = Field(min_length=32)
    encryption_key: SecretStr = Field(min_length=1)


class SealedEnvelope(BaseModel):
    """A minted envelope: the client-held bearer value and when it expires."""

    model_config = ConfigDict(frozen=True)
    token: SecretStr
    expires_at: datetime


class OpenedEnvelope(BaseModel):
    """A validated envelope: the identity it was minted for and the recovered grant."""

    model_config = ConfigDict(frozen=True)
    identity: EnvelopeIdentity
    grant: UpstreamTokenGrant


class EnvelopeTooLarge(BaseModel):
    """The serialized envelope exceeded ``MAX_ENVELOPE_BYTES``; carries sizes only."""

    model_config = ConfigDict(frozen=True)
    tag: Literal["envelope_too_large"] = "envelope_too_large"
    size_bytes: int
    max_bytes: int


EnvelopeMintError: TypeAlias = EnvelopeTooLarge


class NotAnEnvelope(BaseModel):
    """The candidate does not carry the envelope prefix."""

    model_config = ConfigDict(frozen=True)
    tag: Literal["not_an_envelope"] = "not_an_envelope"


class BadSignature(BaseModel):
    """The JWT signature does not verify under the provided signing key."""

    model_config = ConfigDict(frozen=True)
    tag: Literal["bad_signature"] = "bad_signature"


class Expired(BaseModel):
    """The envelope's ``exp`` is not in the future relative to the provided ``now``."""

    model_config = ConfigDict(frozen=True)
    tag: Literal["expired"] = "expired"


class MalformedPayload(BaseModel):
    """The token is not a well-formed envelope: undecodable JWT, wrong issuer, missing
    or mistyped claims, or a decrypted grant that fails validation."""

    model_config = ConfigDict(frozen=True)
    tag: Literal["malformed_payload"] = "malformed_payload"


class DecryptFailed(BaseModel):
    """The signed ``grant`` blob could not be decrypted under the provided key."""

    model_config = ConfigDict(frozen=True)
    tag: Literal["decrypt_failed"] = "decrypt_failed"


EnvelopeOpenError: TypeAlias = NotAnEnvelope | BadSignature | Expired | MalformedPayload | DecryptFailed


class _EnvelopeClaims(BaseModel):
    """Decoded-claims boundary that pins the exact shape :func:`mint_envelope` emits.

    ``server_id``/``key_hash`` mirror the ``min_length`` constraints of
    :class:`EnvelopeIdentity` so any claim set that validates here also constructs an
    identity, keeping :func:`open_envelope` raise-free: a correctly signed JWT with an
    empty identity claim fails here and maps to ``MalformedPayload``.

    ``strict`` rejects coerced types (``exp: "123"``, ``exp: 123.0``) rather than opening
    on them, and ``extra="forbid"`` rejects any claim the gateway never mints (a hostile
    ``nbf``/``aud``/... rides along on a re-signed token). Since PyJWT's own ``iat``/
    ``nbf``/``exp`` validators are disabled at decode (they raise on hostile claim types
    and, for ``iat``/``nbf``, compare against the wall clock rather than the injected
    ``now``), this model is the sole, total type gate for every registered claim.
    """

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    iss: str
    iat: int
    exp: int
    server_id: str = Field(min_length=1)
    key_hash: str = Field(min_length=1)
    grant: str = Field(min_length=1)


class _GrantWire(BaseModel):
    model_config = ConfigDict(frozen=True)
    access_token: str
    token_type: str
    refresh_token: str | None = None
    scope: str | None = None
    expires_in: int | None = None


def is_envelope(candidate: str) -> bool:
    """Cheap prefix check so the edge can route envelopes vs raw tokens without crypto."""
    return candidate.startswith(ENVELOPE_PREFIX)


def mint_envelope(
    identity: EnvelopeIdentity,
    grant: UpstreamTokenGrant,
    keys: EnvelopeKeys,
    now: datetime,
) -> SealedEnvelope | EnvelopeMintError:
    """Seal ``grant`` for ``identity`` into a client-held envelope.

    ``exp`` is ``min(grant.expires_in, MAX_ENVELOPE_TTL_SECONDS)`` seconds from ``now``
    (the cap alone when ``expires_in`` is absent). Returns ``EnvelopeTooLarge`` when the
    serialized envelope exceeds ``MAX_ENVELOPE_BYTES``.
    """
    expires_at = now + timedelta(seconds=_envelope_ttl_seconds(grant.expires_in))
    claims = _EnvelopeClaims(
        iss=ENVELOPE_ISSUER,
        iat=int(now.timestamp()),
        exp=int(expires_at.timestamp()),
        server_id=identity.server_id,
        key_hash=identity.key_hash,
        grant=_encrypt_grant_blob(_grant_plaintext(grant), keys.encryption_key),
    )
    token = ENVELOPE_PREFIX + jwt.encode(
        claims.model_dump(),
        keys.signing_key.get_secret_value(),
        algorithm=_ENVELOPE_JWT_ALGORITHM,
    )
    size_bytes = len(token.encode("utf-8"))
    if size_bytes > MAX_ENVELOPE_BYTES:
        return EnvelopeTooLarge(size_bytes=size_bytes, max_bytes=MAX_ENVELOPE_BYTES)
    return SealedEnvelope(token=SecretStr(token), expires_at=expires_at)


def open_envelope(
    candidate: str,
    keys: EnvelopeKeys,
    now: datetime,
) -> OpenedEnvelope | EnvelopeOpenError:
    """Validate ``candidate`` and recover the identity and inner grant.

    Never raises for bad input: every invalid, expired, tampered, or undecryptable
    candidate maps to a distinct ``EnvelopeOpenError`` variant. The recovered
    ``grant.expires_in`` is the value the upstream reported at mint time and is not
    re-derived, so it is stale by up to the envelope's lifetime; callers that need a
    live remaining lifetime should use ``now`` against the upstream, not this field.
    """
    if not is_envelope(candidate):
        return NotAnEnvelope()
    # UTF-8 byte length is never below character length, so a character count already over the
    # cap rejects an oversize candidate in O(1) without encoding it; the exact byte check then
    # runs only on candidates already bounded to <= MAX_ENVELOPE_BYTES characters.
    if len(candidate) > MAX_ENVELOPE_BYTES:
        return MalformedPayload()
    if len(candidate.encode("utf-8", "surrogatepass")) > MAX_ENVELOPE_BYTES:
        return MalformedPayload()
    claims = _decode_claims(candidate.removeprefix(ENVELOPE_PREFIX), keys.signing_key)
    if not isinstance(claims, _EnvelopeClaims):
        return claims
    if now.timestamp() >= claims.exp:
        return Expired()
    grant = _decrypt_grant(claims.grant, keys.encryption_key)
    if not isinstance(grant, UpstreamTokenGrant):
        return grant
    return OpenedEnvelope(
        identity=EnvelopeIdentity(server_id=claims.server_id, key_hash=claims.key_hash),
        grant=grant,
    )


def _envelope_ttl_seconds(upstream_expires_in: int | None) -> int:
    if upstream_expires_in is None:
        return MAX_ENVELOPE_TTL_SECONDS
    return min(upstream_expires_in, MAX_ENVELOPE_TTL_SECONDS)


def _grant_plaintext(grant: UpstreamTokenGrant) -> str:
    wire = _GrantWire(
        access_token=grant.access_token.get_secret_value(),
        token_type=grant.token_type,
        refresh_token=None if grant.refresh_token is None else grant.refresh_token.get_secret_value(),
        scope=grant.scope,
        expires_in=grant.expires_in,
    )
    return wire.model_dump_json(exclude_none=True)


def _decode_claims(
    compact: str,
    signing_key: SecretStr,
) -> _EnvelopeClaims | BadSignature | MalformedPayload:
    """Verify the HS256 signature and shape of an attacker-controlled compact JWT.

    ``compact`` is fully hostile and bounded to ``MAX_ENVELOPE_BYTES`` by the caller.
    PyJWT's ``iat``/``nbf``/``exp`` validators are disabled: they raise on hostile claim
    types and, for ``iat``/``nbf``, compare against the wall clock rather than the
    injected ``now`` (``exp`` is checked by the caller against ``now``). Apart from a
    signature mismatch (``BadSignature``), every decode failure is ``MalformedPayload``:
    a non-UTF-8 candidate surfaces as ``UnicodeEncodeError`` (a ``ValueError``), a
    non-string registered claim such as ``iss`` as a ``TypeError`` from PyJWT's claim
    validators, and a wrong issuer or structurally invalid token as an
    ``InvalidTokenError``. ``_EnvelopeClaims`` is the total type gate for the payload.
    """
    try:
        payload = jwt.decode(
            compact,
            signing_key.get_secret_value(),
            algorithms=[_ENVELOPE_JWT_ALGORITHM],
            issuer=ENVELOPE_ISSUER,
            options={
                "verify_exp": False,
                "verify_iat": False,
                "verify_nbf": False,
                "require": ["iss", "iat", "exp"],
            },
        )
    except jwt.InvalidSignatureError:
        return BadSignature()
    except (jwt.InvalidTokenError, ValueError, TypeError):
        return MalformedPayload()
    try:
        return _EnvelopeClaims.model_validate(payload)
    except ValidationError:
        return MalformedPayload()


def _encrypt_grant_blob(plaintext: str, encryption_key: SecretStr) -> str:
    ciphertext = bytes(encrypt_value(value=plaintext, signing_key=encryption_key.get_secret_value()))
    return base64.urlsafe_b64encode(ciphertext).decode("ascii")


def _decrypt_grant(
    blob: str,
    encryption_key: SecretStr,
) -> UpstreamTokenGrant | DecryptFailed | MalformedPayload:
    from nacl.exceptions import CryptoError

    try:
        plaintext = decrypt_value(
            value=base64.urlsafe_b64decode(blob),
            signing_key=encryption_key.get_secret_value(),
        )
    except (CryptoError, ValueError):
        return DecryptFailed()
    try:
        return UpstreamTokenGrant.model_validate_json(plaintext)
    except ValidationError:
        return MalformedPayload()
