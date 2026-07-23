"""The three sealed artifacts of the aggregate gateway DCR front door.

A public OAuth client (Claude Desktop, Claude Code, MCP Inspector) that discovers the gateway as
its authorization server registers, is sent through LiteLLM SSO, authorizes its upstream servers on
the connect grid, and finally swaps an authorization code for the identity-only session bearer in
:mod:`.session_token`. Three values have to survive a browser round trip in between:

``llm_dcrc_`` (client)
    The registered client itself. RFC 7591 registration is public by design, so the ``client_id``
    is not a secret; sealing the client's ``redirect_uris`` INTO the identifier is what makes it
    unforgeable, and it means registration needs no table, no migration, and no lifecycle. Never
    expires: an OAuth client registration is long-lived, and the controls that bound it are PKCE,
    the consent screen, and the redirect-URI binding, not a clock.

``llm_gflow_`` (connect flow)
    The pending authorization request, parked while the user signs in and authorizes servers on the
    grid. Rides in an HttpOnly cookie, so the browser carries it across the interlude with no
    server-side session store and it works across replicas.

``llm_gcode_`` (authorization code)
    The RFC 6749 authorization code, bound to the client, the redirect URI, and the PKCE challenge,
    and short-lived because it is redeemed by the client within one redirect.

All three are HS256 JWTs built the same way as :mod:`.session_token`: strict, ``extra="forbid"``
claims as the sole total type gate, PyJWT's own time validators disabled (they raise on hostile
claim types and read the wall clock instead of the injected ``now``), a size cap applied before any
parsing, and a ``jti`` so a single-use guard has a stable handle. Signing key material is derived
from the proxy ``master_key`` under a domain label distinct from the session and bridge labels, so
no artifact of one family can ever validate as another.

The deliberately-not-used alternative is the older ``encrypt_value_helper`` family that seals the
bridge authorization code and OAuth state: it has no TTL, no claims, no kind separation, and no
domain-separated KDF, and its own callers document that codes minted before an expiry check was
added stay valid indefinitely.

This module is pure: it reads no proxy globals, imports no endpoint code, and takes key material
and the clock as parameters. Every opener is total over hostile input and returns a typed failure
rather than raising.
"""

from __future__ import annotations

import hashlib
import secrets
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Literal, TypeAlias, TypeVar

import jwt
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError

CLIENT_PREFIX = "llm_dcrc_"
CONNECT_FLOW_PREFIX = "llm_gflow_"
AUTHORIZATION_CODE_PREFIX = "llm_gcode_"

GATEWAY_DCR_ISSUER = "litellm-mcp-gateway-dcr"
"""``iss`` on every artifact here, distinct from the session and bridge issuers."""

CONNECT_FLOW_TTL_SECONDS = 1800
"""How long the user has to sign in and authorize servers on the grid before the pending
authorization request goes stale (30 min). Long enough to complete several upstream OAuth
round trips, short enough that an abandoned flow cookie stops being redeemable."""

AUTHORIZATION_CODE_TTL_SECONDS = 120
"""RFC 6749 section 4.1.2 recommends a maximum authorization code lifetime of 10 minutes; the code
is redeemed by the client immediately on redirect, so 2 minutes is ample and narrows the replay
window before the single-use guard is even consulted."""

MAX_ARTIFACT_BYTES = 8192
"""Cap on any serialized artifact and on any candidate accepted by the openers, applied before
parsing. The only unbounded input is a client's ``redirect_uris`` list, which registration bounds
separately; 8192 keeps a sealed client_id usable inside common header and URL limits."""

MAX_REDIRECT_URIS = 8
"""How many redirect URIs one registration may seal in. RFC 7591 sets no limit, but each one
inflates every ``client_id`` (and therefore every session token that carries it), so this bounds
the artifact rather than the client's ambitions."""

_ALGORITHM = "HS256"
_SIGNING_KEY_DOMAIN = b"litellm-mcp-gateway:dcr-artifact-signing:"
_SCRYPT_N = 2**15
_SCRYPT_R = 8
_SCRYPT_P = 1
_SCRYPT_MAXMEM = 128 * _SCRYPT_N * _SCRYPT_R * _SCRYPT_P * 2
_DERIVED_KEY_BYTES = 32


class GatewayDCRKeys(BaseModel):
    """Injected key material: the HS256 signing key (at least 32 bytes, per RFC 7518)."""

    model_config = ConfigDict(frozen=True)
    signing_key: SecretStr = Field(min_length=32)


@lru_cache(maxsize=8)
def gateway_dcr_keys_from_master_key(master_key: str) -> GatewayDCRKeys:
    """Derive the artifact signing key from the proxy master key.

    Memory-hard scrypt (RFC 7914) over a domain label that differs from the session and bridge
    labels, so the three families are cryptographically separated on top of their distinct issuers,
    prefixes, and claim shapes. Rotating ``master_key`` invalidates every outstanding registration,
    flow, and code, which is the intended behavior for a signing-key change.
    """
    signing = hashlib.scrypt(
        master_key.encode(),
        salt=_SIGNING_KEY_DOMAIN,
        n=_SCRYPT_N,
        r=_SCRYPT_R,
        p=_SCRYPT_P,
        maxmem=_SCRYPT_MAXMEM,
        dklen=_DERIVED_KEY_BYTES,
    ).hex()
    return GatewayDCRKeys(signing_key=SecretStr(signing))


class RegisteredClient(BaseModel):
    """A public DCR client: the redirect URIs it registered and its self-reported name."""

    model_config = ConfigDict(frozen=True)
    redirect_uris: tuple[str, ...] = Field(min_length=1, max_length=MAX_REDIRECT_URIS)
    client_name: str | None = None


class ConnectFlow(BaseModel):
    """An authorization request parked across sign-in and the connect grid.

    ``user_id`` is stamped once the SSO session is resolved, so completing the flow can require
    that the browser finishing it is still the user who started it.
    """

    model_config = ConfigDict(frozen=True)
    client_id: str = Field(min_length=1)
    redirect_uri: str = Field(min_length=1)
    code_challenge: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    state: str = ""


class AuthorizationCode(BaseModel):
    """An issued authorization code, bound to everything the token endpoint must re-check.

    ``redirect_uri`` and ``client_id`` are bound per RFC 6749 section 4.1.3, and ``code_challenge``
    per RFC 7636 section 4.6, so a code intercepted in a redirect is useless without the verifier.
    """

    model_config = ConfigDict(frozen=True)
    client_id: str = Field(min_length=1)
    redirect_uri: str = Field(min_length=1)
    code_challenge: str = Field(min_length=1)
    user_id: str = Field(min_length=1)


class OpenedConnectFlow(BaseModel):
    """A validated connect flow and the ``jti`` its single-use guard claims."""

    model_config = ConfigDict(frozen=True)
    flow: ConnectFlow
    jti: str = Field(min_length=1)


class OpenedAuthorizationCode(BaseModel):
    """A validated authorization code and the ``jti`` its single-use guard claims."""

    model_config = ConfigDict(frozen=True)
    code: AuthorizationCode
    jti: str = Field(min_length=1)


class _ClientClaims(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    iss: str
    iat: int
    jti: str = Field(min_length=1)
    kind: Literal["client"]
    # list, not tuple: these claims are a JSON wire boundary and ``strict`` mode rejects a decoded
    # JSON array for a tuple-typed field, so a sealed client would never reopen.
    redirect_uris: list[str] = Field(min_length=1, max_length=MAX_REDIRECT_URIS)
    client_name: str | None = None


class _ExpiringClaims(BaseModel):
    """Base for the two artifacts that expire, so ``exp`` is a type-level fact and the shared
    opener enforces it in one place instead of each caller remembering to."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    iss: str
    iat: int
    exp: int
    jti: str = Field(min_length=1)


class _ConnectFlowClaims(_ExpiringClaims):
    kind: Literal["connect_flow"]
    client_id: str = Field(min_length=1)
    redirect_uri: str = Field(min_length=1)
    code_challenge: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    state: str


class _AuthorizationCodeClaims(_ExpiringClaims):
    kind: Literal["authorization_code"]
    client_id: str = Field(min_length=1)
    redirect_uri: str = Field(min_length=1)
    code_challenge: str = Field(min_length=1)
    user_id: str = Field(min_length=1)


class ArtifactTooLarge(BaseModel):
    """The serialized artifact exceeded ``MAX_ARTIFACT_BYTES``; carries sizes only."""

    model_config = ConfigDict(frozen=True)
    tag: Literal["artifact_too_large"] = "artifact_too_large"
    size_bytes: int
    max_bytes: int


class NotThisArtifact(BaseModel):
    """The candidate does not carry this artifact's prefix."""

    model_config = ConfigDict(frozen=True)
    tag: Literal["not_this_artifact"] = "not_this_artifact"


class ArtifactBadSignature(BaseModel):
    """The signature does not verify under the provided key."""

    model_config = ConfigDict(frozen=True)
    tag: Literal["artifact_bad_signature"] = "artifact_bad_signature"


class ArtifactExpired(BaseModel):
    """``exp`` is not in the future relative to the provided ``now``."""

    model_config = ConfigDict(frozen=True)
    tag: Literal["artifact_expired"] = "artifact_expired"


class ArtifactMalformed(BaseModel):
    """Undecodable, wrong issuer, wrong kind, or missing/mistyped/extra claims."""

    model_config = ConfigDict(frozen=True)
    tag: Literal["artifact_malformed"] = "artifact_malformed"


ArtifactOpenError: TypeAlias = NotThisArtifact | ArtifactBadSignature | ArtifactExpired | ArtifactMalformed

_ClaimsT = TypeVar("_ClaimsT", _ClientClaims, _ConnectFlowClaims, _AuthorizationCodeClaims)


def is_gateway_dcr_client_id(candidate: str) -> bool:
    """Cheap, keyless test that a ``client_id`` is a gateway DCR registration.

    This is what lets the shared root ``/authorize`` route a gateway client to the aggregate flow
    before falling back to the legacy single-server resolution. It asks IDENTITY only (is this
    ours), never usability, so an otherwise invalid client is still recognized here and fails with
    the aggregate flow's own error rather than silently falling through to a different flow.
    """
    return candidate.startswith(CLIENT_PREFIX)


def is_gateway_authorization_code(candidate: str) -> bool:
    """Cheap, keyless test that a ``code`` was issued by the aggregate flow."""
    return candidate.startswith(AUTHORIZATION_CODE_PREFIX)


def seal_client(client: RegisteredClient, keys: GatewayDCRKeys, now: datetime) -> str | ArtifactTooLarge:
    """Seal a registration into its own ``client_id``. No expiry (see module docstring)."""
    return _sign(
        _ClientClaims(
            iss=GATEWAY_DCR_ISSUER,
            iat=int(now.timestamp()),
            jti=secrets.token_urlsafe(16),
            kind="client",
            redirect_uris=list(client.redirect_uris),
            client_name=client.client_name,
        ),
        CLIENT_PREFIX,
        keys,
    )


def open_client(candidate: str, keys: GatewayDCRKeys) -> RegisteredClient | ArtifactOpenError:
    """Recover a registration from a ``client_id``. Total over hostile input."""
    claims = _open(candidate, CLIENT_PREFIX, _ClientClaims, keys, now=None)
    if not isinstance(claims, _ClientClaims):
        return claims
    return RegisteredClient(redirect_uris=tuple(claims.redirect_uris), client_name=claims.client_name)


def seal_connect_flow(flow: ConnectFlow, keys: GatewayDCRKeys, now: datetime) -> str | ArtifactTooLarge:
    """Seal the pending authorization request for the connect-grid interlude."""
    return _sign(
        _ConnectFlowClaims(
            iss=GATEWAY_DCR_ISSUER,
            iat=int(now.timestamp()),
            exp=int((now + timedelta(seconds=CONNECT_FLOW_TTL_SECONDS)).timestamp()),
            jti=secrets.token_urlsafe(16),
            kind="connect_flow",
            client_id=flow.client_id,
            redirect_uri=flow.redirect_uri,
            code_challenge=flow.code_challenge,
            user_id=flow.user_id,
            state=flow.state,
        ),
        CONNECT_FLOW_PREFIX,
        keys,
    )


def open_connect_flow(candidate: str, keys: GatewayDCRKeys, now: datetime) -> OpenedConnectFlow | ArtifactOpenError:
    """Recover a parked authorization request. Total over hostile input."""
    claims = _open(candidate, CONNECT_FLOW_PREFIX, _ConnectFlowClaims, keys, now)
    if not isinstance(claims, _ConnectFlowClaims):
        return claims
    return OpenedConnectFlow(
        flow=ConnectFlow(
            client_id=claims.client_id,
            redirect_uri=claims.redirect_uri,
            code_challenge=claims.code_challenge,
            user_id=claims.user_id,
            state=claims.state,
        ),
        jti=claims.jti,
    )


def seal_authorization_code(code: AuthorizationCode, keys: GatewayDCRKeys, now: datetime) -> str | ArtifactTooLarge:
    """Seal the authorization code handed back to the client on the final redirect."""
    return _sign(
        _AuthorizationCodeClaims(
            iss=GATEWAY_DCR_ISSUER,
            iat=int(now.timestamp()),
            exp=int((now + timedelta(seconds=AUTHORIZATION_CODE_TTL_SECONDS)).timestamp()),
            jti=secrets.token_urlsafe(16),
            kind="authorization_code",
            client_id=code.client_id,
            redirect_uri=code.redirect_uri,
            code_challenge=code.code_challenge,
            user_id=code.user_id,
        ),
        AUTHORIZATION_CODE_PREFIX,
        keys,
    )


def open_authorization_code(
    candidate: str, keys: GatewayDCRKeys, now: datetime
) -> OpenedAuthorizationCode | ArtifactOpenError:
    """Recover an authorization code at the token endpoint. Total over hostile input."""
    claims = _open(candidate, AUTHORIZATION_CODE_PREFIX, _AuthorizationCodeClaims, keys, now)
    if not isinstance(claims, _AuthorizationCodeClaims):
        return claims
    return OpenedAuthorizationCode(
        code=AuthorizationCode(
            client_id=claims.client_id,
            redirect_uri=claims.redirect_uri,
            code_challenge=claims.code_challenge,
            user_id=claims.user_id,
        ),
        jti=claims.jti,
    )


def _sign(claims: BaseModel, prefix: str, keys: GatewayDCRKeys) -> str | ArtifactTooLarge:
    """Sign one artifact's claims and enforce the size cap.

    Shared by all three so the JWT shape, issuer, and size guard cannot drift between them.
    """
    sealed = prefix + jwt.encode(claims.model_dump(), keys.signing_key.get_secret_value(), algorithm=_ALGORITHM)
    size_bytes = len(sealed.encode("utf-8"))
    if size_bytes > MAX_ARTIFACT_BYTES:
        return ArtifactTooLarge(size_bytes=size_bytes, max_bytes=MAX_ARTIFACT_BYTES)
    return sealed


def _open(
    candidate: str,
    prefix: str,
    claims_model: type[_ClaimsT],
    keys: GatewayDCRKeys,
    now: datetime | None,
) -> _ClaimsT | ArtifactOpenError:
    """Prefix-route, size-bound, verify and expiry-check an attacker-controlled value.

    The ``kind`` claim is enforced by each claims model's ``Literal``, so re-prefixing one artifact
    as another fails validation here rather than needing a separate comparison. ``now`` is ``None``
    for the client artifact, which carries no ``exp`` by design. Never raises.
    """
    if not candidate.startswith(prefix):
        return NotThisArtifact()
    # A character count already over the cap rejects in O(1), since UTF-8 byte length is never
    # below character length; the exact byte check then runs only on already-bounded candidates.
    if len(candidate) > MAX_ARTIFACT_BYTES:
        return ArtifactMalformed()
    if len(candidate.encode("utf-8", "surrogatepass")) > MAX_ARTIFACT_BYTES:
        return ArtifactMalformed()
    required = ["iss", "iat"] if now is None else ["iss", "iat", "exp"]
    try:
        payload = jwt.decode(
            candidate.removeprefix(prefix),
            keys.signing_key.get_secret_value(),
            algorithms=[_ALGORITHM],
            issuer=GATEWAY_DCR_ISSUER,
            options={"verify_exp": False, "verify_iat": False, "verify_nbf": False, "require": required},
        )
    except jwt.InvalidSignatureError:
        return ArtifactBadSignature()
    except (jwt.InvalidTokenError, ValueError, TypeError):
        return ArtifactMalformed()
    try:
        claims = claims_model.model_validate(payload)
    except ValidationError:
        return ArtifactMalformed()
    if isinstance(claims, _ExpiringClaims) and now is not None and now.timestamp() >= claims.exp:
        return ArtifactExpired()
    return claims
