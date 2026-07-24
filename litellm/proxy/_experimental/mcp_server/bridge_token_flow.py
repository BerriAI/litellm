"""Bridge token flow: litellm identity resolution and the DCR-bridge oauth_delegate mint/refresh pipeline."""

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Literal, Optional

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import SecretStr
from typing_extensions import assert_never

from litellm._logging import verbose_logger
from litellm.proxy._experimental.mcp_server.oauth_utils import TOKEN_NO_CACHE_HEADERS
from litellm.types.mcp_server.mcp_server_manager import MCPServer

if TYPE_CHECKING:
    from litellm.proxy._experimental.mcp_server.discoverable_endpoints import _BridgeAuthorizationCode
    from litellm.proxy._experimental.mcp_server.outbound_credentials.envelope import (
        EnvelopeIdentity,
        EnvelopeKeys,
        RefreshCredential,
        UpstreamTokenGrant,
    )
    from litellm.proxy._types import UserAPIKeyAuth


def _litellm_key_from_request(request: Request) -> Optional[str]:
    """Return the LiteLLM API key presented on the request, or ``None``.

    Accepts the key from ``x-litellm-api-key`` (what MCP clients such as Claude Desktop/Code
    send) as well as ``Authorization``; either may carry a bare token or ``Bearer <token>``.
    ``x-litellm-api-key`` wins when both are present, since ``Authorization`` may instead carry
    an OAuth/upstream bearer.
    """
    for header_value in (
        request.headers.get("x-litellm-api-key"),
        request.headers.get("Authorization") or request.headers.get("authorization"),
    ):
        if not header_value:
            continue
        value = header_value.strip()
        if value.lower().startswith("bearer "):
            value = value[7:].strip()
        if value:
            return value
    return None


def _key_is_active(key_obj: "UserAPIKeyAuth") -> bool:
    """``True`` when the presented key is neither blocked nor past its expiry.

    The OAuth token endpoint is unauthenticated, so the presented key is validated here before it is
    trusted; a revoked or expired key must not mint a bridge envelope or write a stored credential.
    ``get_key_object`` resolves a row without these checks (the main ``user_api_key_auth`` pipeline
    enforces them downstream, which this endpoint bypasses), so they are applied here. Deleted keys
    are already rejected upstream, where ``get_key_object`` raises on a row that no longer exists.

    This is an active-state gate only; it deliberately does not require a ``user_id``. A valid
    team-scoped or service-account key has no ``user_id`` yet is a legitimate credential, so gating
    on ``user_id`` presence would wrongly reject it. Callers that need the user (the per-user token
    store) derive it separately via :func:`_active_key_user_id`.

    Total by design: ``expires`` is typed ``str | datetime``, and an unparseable string would make
    ``datetime.fromisoformat`` raise. Since the callers run this outside their key-resolution
    ``try``, an uncaught parse error would surface as a 500 instead of the endpoint's fail-closed
    behavior, so a malformed expiry is treated as inactive (return ``False``) rather than raising.
    """
    if key_obj.blocked is True:
        return False
    expires = key_obj.expires
    if expires is not None:
        if isinstance(expires, datetime):
            expiry = expires
        else:
            try:
                expiry = datetime.fromisoformat(expires)
            except (ValueError, TypeError):
                return False
        if expiry.tzinfo is None or expiry.tzinfo.utcoffset(expiry) is None:
            expiry = expiry.replace(tzinfo=timezone.utc)
        if expiry < datetime.now(timezone.utc):
            return False
    return True


def _active_key_user_id(key_obj: "UserAPIKeyAuth") -> str | None:
    """The active key's ``user_id``, or ``None`` when the key is blocked/expired or simply has no
    ``user_id`` (a team-scoped or service-account key). Used only by the per-user token store, which
    needs a user to key the stored credential; the bridge mint uses the key hash and does not."""
    return key_obj.user_id if _key_is_active(key_obj) else None


@dataclass(frozen=True, slots=True)
class _ResolvedKey:
    """An active litellm key resolved from the token request: its hash (the value ``get_key_object``
    and the cache/DB layer key the record by) and the live record."""

    key_hash: str
    key: "UserAPIKeyAuth"


_KeyResolutionFailure = Literal["no_active_key", "unavailable", "unresolvable"]
"""Why a token request yielded no active litellm key, kept distinct so a caller statuses each truthfully
instead of blaming the client for a gateway problem:
- ``no_active_key``: none was presented, or the presented key is unknown / blocked / expired (the
  caller's request is at fault)
- ``unavailable``: the auth database was transiently unreachable while resolving (retryable)
- ``unresolvable``: the gateway cannot resolve identity right now (no DB connection, or an unexpected
  error) -- a gateway fault, not the caller's
The classification mirrors admission's ``_reload_admitted_key`` so the mint (ingress) and admission
(egress) never disagree on the status of the same outage."""


async def _resolve_active_litellm_key(request: Request) -> "_ResolvedKey | _KeyResolutionFailure":
    """Resolve the presented litellm key to an active key record, or say precisely why not.

    Single resolution path the OAuth token endpoint reuses, resolving authoritatively via
    ``get_key_object`` (cache first, then DB). The failure is a value, not a bare ``None``, so a caller
    can tell "the client sent no usable credential" (a request error) apart from "the gateway could not
    check" (an infrastructure error) and status each truthfully; collapsing both to ``None`` is what let
    a DB outage read as a 400. A resolved key is still gated by ``_key_is_active``, so a blocked or
    expired key is ``no_active_key`` while a valid team-scoped or service-account key (no ``user_id``)
    resolves. Classification mirrors admission's ``_reload_admitted_key``: no DB connection is a gateway
    fault, a ``ProxyException`` / ``HTTPException`` from ``get_key_object`` is an unknown or invalid key,
    a database-service-unavailable error is a retryable outage, and anything else is an unexpected
    gateway fault."""
    token = _litellm_key_from_request(request)
    if not token:
        return "no_active_key"
    from litellm.proxy._types import hash_token  # noqa: PLC0415  # inline import avoids a module-load circular import

    return await _reload_active_key_by_hash(hash_token(token))


async def _reload_active_key_by_hash(key_hash: str) -> "_ResolvedKey | _KeyResolutionFailure":
    """Reload the live key record for ``key_hash`` (cache first, then DB) and gate it on active state,
    returning the resolved key or a precise failure. Shared by the token request's presented-key
    resolution (:func:`_resolve_active_litellm_key`, which hashes the presented key) and the refresh
    path (which already holds the hash sealed in the refresh envelope), so both re-validate identity
    through one active-key gate and one failure classification. Classification mirrors admission's
    ``_reload_admitted_key``: no DB connection is a gateway fault, a ``ProxyException`` / ``HTTPException``
    from ``get_key_object`` is an unknown or invalid key, a database-service-unavailable error is a
    retryable outage, and anything else is an unexpected gateway fault. A blocked or expired key is
    ``no_active_key``, so a revoked key can neither mint nor refresh a bridge envelope."""
    from litellm.proxy._types import (
        ProxyException,  # noqa: PLC0415  # inline import avoids a module-load circular import
    )
    from litellm.proxy.auth.auth_checks import (  # noqa: PLC0415  # inline import avoids a module-load circular import
        get_key_object,
    )
    from litellm.proxy.db.exception_handler import (  # noqa: PLC0415  # inline import avoids a module-load circular import
        PrismaDBExceptionHandler,
    )
    from litellm.proxy.proxy_server import (  # noqa: PLC0415  # inline import avoids a module-load circular import
        prisma_client,
        user_api_key_cache,
    )

    if prisma_client is None:
        return "unresolvable"
    try:
        key_obj = await get_key_object(
            hashed_token=key_hash,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
        )
    except (ProxyException, HTTPException):
        return "no_active_key"
    except Exception as exc:  # noqa: BLE001  # classify: a DB outage is retryable, anything else is an opaque gateway fault
        if PrismaDBExceptionHandler.is_database_service_unavailable_error(exc):
            return "unavailable"
        verbose_logger.debug(
            "_reload_active_key_by_hash: unexpected key-resolution error (%s)",
            type(exc).__name__,
        )
        return "unresolvable"
    if not _key_is_active(key_obj):
        return "no_active_key"
    return _ResolvedKey(key_hash=key_hash, key=key_obj)


async def _reload_active_user_by_id(user_id: str) -> "_KeyResolutionFailure | None":
    """Re-validate a live litellm user by id, returning ``None`` when the user is active or a precise
    failure otherwise. The interactive DCR client authenticates via SSO, so its refresh envelope seals a
    user subject; renewing it must re-check the user is still live (present and not SCIM-deactivated) so a
    deactivated user cannot keep refreshing, mirroring how admission re-validates the same user subject on
    the egress side. No DB connection is a gateway fault (``unresolvable``) and a
    database-service-unavailable error is a retryable outage (``unavailable``). Everything else fails
    closed as ``no_active_key`` (the caller maps it to invalid_grant): a ``ProxyException`` /
    ``HTTPException``, a SCIM-deactivated user, and, unlike the key path, a missing user. ``get_user_object``
    catches every DB failure and re-raises a bare ``ValueError`` (a deleted user and a real outage look
    identical, the original error surviving only as ``__context__``), so the outage check walks the cause
    chain, and a missing user falls through to ``no_active_key`` rather than an opaque gateway fault."""
    from litellm.proxy._types import (
        ProxyException,  # noqa: PLC0415  # inline import avoids a module-load circular import
    )
    from litellm.proxy.auth.auth_checks import (  # noqa: PLC0415  # inline import avoids a module-load circular import
        get_user_object,
    )
    from litellm.proxy.db.exception_handler import (  # noqa: PLC0415  # inline import avoids a module-load circular import
        PrismaDBExceptionHandler,
    )
    from litellm.proxy.proxy_server import (  # noqa: PLC0415  # inline import avoids a module-load circular import
        prisma_client,
        user_api_key_cache,
    )

    if prisma_client is None:
        return "unresolvable"
    try:
        user_object = await get_user_object(
            user_id=user_id,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            user_id_upsert=False,
        )
    except (ProxyException, HTTPException):
        return "no_active_key"
    except Exception as exc:  # noqa: BLE001  # a DB outage is retryable; a missing user (get_user_object's wrapped ValueError) or any other resolution failure fails closed as no_active_key, never a 500
        if PrismaDBExceptionHandler.is_database_service_unavailable_error_in_chain(exc):
            return "unavailable"
        verbose_logger.debug("_reload_active_user_by_id: user-resolution error (%s)", type(exc).__name__)
        return "no_active_key"
    if user_object is None:
        return "no_active_key"
    if isinstance(user_object.metadata, dict) and user_object.metadata.get("scim_active") is False:
        return "no_active_key"
    return None


async def _key_owner_scim_deactivated(key: "UserAPIKeyAuth") -> bool:
    """True only when the key's owning user was explicitly SCIM-deactivated, so a refresh revokes an
    offboarded owner's key exactly as admission does via ``_reject_if_admitted_owner_scim_deactivated``.
    A key with no owner, a missing owner record, or a failed lookup fails OPEN (returns ``False``),
    matching admission and the standard builder: a key may outlive its owner record, and a transient DB
    blip must not revoke a live key. Only an explicit ``scim_active`` of ``False`` gates renewal."""
    if key.user_id is None:
        return False
    from litellm.proxy.auth.auth_checks import (  # noqa: PLC0415  # inline import avoids a module-load circular import
        get_user_object,
    )
    from litellm.proxy.proxy_server import (  # noqa: PLC0415  # inline import avoids a module-load circular import
        prisma_client,
        user_api_key_cache,
    )

    if prisma_client is None:
        return False
    try:
        owner = await get_user_object(
            user_id=key.user_id,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            user_id_upsert=False,
        )
    except Exception as exc:  # noqa: BLE001  # fail open: a missing owner (get_user_object's wrapped ValueError) or a DB blip must not revoke a live key
        verbose_logger.debug("refresh: key-owner SCIM lookup failed, not revoking (%s)", type(exc).__name__)
        return False
    return owner is not None and isinstance(owner.metadata, dict) and owner.metadata.get("scim_active") is False


async def _revalidate_active_subject(identity: "EnvelopeIdentity") -> "_KeyResolutionFailure | None":
    """Re-validate that the subject sealed in a refresh envelope is still live, dispatching on its type:
    a key_hash reloads the virtual key, a user_id reloads the user. Returns ``None`` when the subject is
    active or a precise failure otherwise, so revocation gates renewal for either identity source the same
    way admission gates the egress: a blocked or expired key, a SCIM-deactivated key owner (mirroring
    admission's owner check, so an offboarded user cannot keep renewing a still-active key), and a
    deactivated or deleted user all fail closed to ``no_active_key``."""
    match identity.subject_type:
        case "key_hash":
            reloaded = await _reload_active_key_by_hash(identity.subject)
            if not isinstance(reloaded, _ResolvedKey):
                return reloaded
            if await _key_owner_scim_deactivated(reloaded.key):
                return "no_active_key"
            return None
        case "user_id":
            return await _reload_active_user_by_id(identity.subject)
        case _:
            assert_never(identity.subject_type)


async def _extract_user_id_from_request(request: Request) -> str | None:
    """The litellm ``user_id`` for the token request, so a per-user token is stored under the same
    identity the egress later reads it by. Storage is best-effort, so every non-resolved outcome
    (including a transient DB outage) collapses to ``None`` here and the caller simply skips the store;
    the bridge mint, which must status those outcomes differently, consumes
    :func:`_resolve_active_litellm_key` directly."""
    resolved = await _resolve_active_litellm_key(request)
    if not isinstance(resolved, _ResolvedKey):
        return None
    return _active_key_user_id(resolved.key)


_UpstreamGrantRejection = Literal["no_access_token", "expired_lifetime"]
"""Why an upstream token response cannot back a bridge envelope:
- ``no_access_token``: the response carries no usable ``access_token``
- ``expired_lifetime``: the response reports a parseable, non-positive ``expires_in``, i.e. an upstream
  token that is already dead, so sealing it would forward a bearer the edge cannot use
An absent or unparseable ``expires_in`` is NOT a rejection; the lifetime is merely unknown and the
envelope caps it, the by-design behaviour for an upstream that omits the field."""


def _classify_upstream_lifetime(raw_expires_in: object) -> "int | Literal['unspecified', 'expired']":
    """Classify an upstream ``expires_in`` into a positive number of seconds, ``"unspecified"`` (absent
    or unparseable, so the envelope caps it), or ``"expired"`` (a non-positive value the upstream reports
    as already elapsed). Telling "we do not know the lifetime" apart from "the upstream says it is
    already dead" is what stops an explicitly-expired token from silently receiving the envelope's 1h
    cap. The expired decision is made on the parsed numeric value, not on ``int(...)`` of it, so a
    positive sub-second lifetime in ``(0, 1)`` is not truncated to ``0`` and misread as elapsed; the
    envelope works in whole seconds, so such a lifetime clamps up to its 1s floor. ``bool`` is excluded
    (an ``int`` subclass but never a real lifetime), and the conversions can raise on ``NaN`` /
    ``Infinity`` / oversized input, which reads as unparseable rather than surfacing as a 500."""
    if raw_expires_in is None or isinstance(raw_expires_in, bool) or not isinstance(raw_expires_in, (int, float, str)):
        return "unspecified"
    try:
        numeric = float(raw_expires_in)
        seconds = int(numeric)
    except (ValueError, TypeError, OverflowError):
        return "unspecified"
    if numeric <= 0:
        return "expired"
    return max(1, seconds)


def _bridge_grant_from_token_response(token_response: object) -> "UpstreamTokenGrant | _UpstreamGrantRejection":
    """Validate an upstream OAuth token response into a typed grant, or say why it cannot back an
    envelope. Each field is isinstance-checked so nothing untyped from ``response.json()`` reaches the
    grant. ``expires_in`` is read three ways (see :func:`_classify_upstream_lifetime`): an unknown
    lifetime leaves the grant ``expires_in`` ``None`` for the envelope to cap, a positive value is
    honoured, and an explicit already-elapsed value is a rejection rather than a silent fall-through to
    the cap."""
    from litellm.proxy._experimental.mcp_server.outbound_credentials.envelope import (  # noqa: PLC0415  # inline import avoids a module-load circular import
        UpstreamTokenGrant,
    )

    if not isinstance(token_response, dict):
        return "no_access_token"
    access = token_response.get("access_token")
    if not isinstance(access, str) or not access:
        return "no_access_token"
    lifetime = _classify_upstream_lifetime(token_response.get("expires_in"))
    if lifetime == "expired":
        return "expired_lifetime"
    token_type = token_response.get("token_type")
    scope = token_response.get("scope")
    return UpstreamTokenGrant(
        access_token=SecretStr(access),
        token_type=token_type if isinstance(token_type, str) and token_type else "Bearer",
        # The upstream refresh_token is deliberately NOT sealed: the edge never consumes it (it forwards
        # only token_type + access_token), so it would be dead weight embedding a long-lived upstream
        # credential in the client-held bearer, and it enlarges the envelope. Refresh support is a
        # follow-up (a dedicated refresh-envelope); the client re-runs authorization_code at the cap.
        refresh_token=None,
        scope=scope if isinstance(scope, str) and scope else None,
        expires_in=lifetime if isinstance(lifetime, int) else None,
    )


# ---------------------------------------------------------------------------
# DCR-bridge oauth_delegate mint: a three-phase pipeline whose failures are values.
#
#   prepare  (before the upstream exchange) -> validate every precondition and resolve identity+keys
#   exchange (the single-use upstream code is consumed here, in exchange_token_with_server)
#   finish   (after the exchange)           -> seal the upstream grant into the client-held envelope
#
# Every precondition lives in ``prepare``, which runs BEFORE the exchange, so no failure can burn the
# single-use code or rotate a refresh token, for either grant type -- that whole class of bug is gone
# by construction rather than guarded case by case. Failures are values mapped to an OAuth-shaped
# response in one place (``_bridge_mint_error_response``), so status codes and the RFC 6749 §5.2 body
# shape are uniform. Adding a failure mode is a new literal plus a match arm the type checker forces.
# ---------------------------------------------------------------------------

_BridgeMintError = Literal[
    "no_identity",
    "invalid_refresh",
    "identity_unavailable",
    "identity_unresolvable",
    "not_configured",
    "no_upstream_token",
    "upstream_token_expired",
    "too_large",
]


@dataclass(frozen=True, slots=True)
class _BridgeMintReady:
    """Everything the seal needs, resolved once before the exchange: the identity to bind the envelope
    to and the master-key-derived envelope keys. The identity is a key_hash subject for the scripted
    two-header client (resolved from the litellm key it presents) or a user_id subject for the
    interactive SSO client (the user recovered from the gateway authorization code), so one phase-3 seal
    serves both. Resolving identity here means ``_finish_bridge_mint`` has no preconditions left to
    fail."""

    identity: "EnvelopeIdentity"
    keys: "EnvelopeKeys"


def _bridge_mint_error_response(error: _BridgeMintError) -> JSONResponse:
    """Map a bridge-mint failure value to its token-endpoint response: one place, RFC 6749 §5.2 shape
    (top-level ``error``, no-store headers) for every case, with a status truthful about where the
    failure is. The caller's request is 400, a transient gateway outage is 503, a gateway
    misconfiguration is 500, and an upstream problem is 502. The identity-resolution statuses match how
    admission statuses the same conditions on the egress side, so mint and admit never disagree under
    one outage."""
    match error:
        case "no_identity":
            status, code, desc = (
                400,
                "invalid_request",
                "this server issues a gateway-bound credential; complete the interactive sign-in, or "
                "send a litellm credential (x-litellm-api-key or Authorization) on the token request",
            )
        case "invalid_refresh":
            status, code, desc = (
                400,
                "invalid_grant",
                "the refresh credential is not a valid, live refresh envelope for this server; "
                "re-run authorization_code to obtain a new one",
            )
        case "identity_unavailable":
            status, code, desc = (
                503,
                "temporarily_unavailable",
                "the authentication database is temporarily unreachable; retry shortly",
            )
        case "identity_unresolvable":
            status, code, desc = (
                500,
                "server_error",
                "the gateway could not resolve the litellm identity for this request",
            )
        case "not_configured":
            status, code, desc = (
                500,
                "server_error",
                "the gateway is not configured to mint a gateway-bound credential (master_key is not set)",
            )
        case "no_upstream_token":
            status, code, desc = (
                502,
                "server_error",
                "the upstream token response has no usable access_token",
            )
        case "upstream_token_expired":
            status, code, desc = (
                502,
                "server_error",
                "the upstream token response reports an already-expired lifetime",
            )
        case "too_large":
            status, code, desc = (
                502,
                "server_error",
                "the upstream token is too large to seal into a gateway-bound credential",
            )
        case _:
            assert_never(error)
    return JSONResponse(
        status_code=status, content={"error": code, "error_description": desc}, headers=TOKEN_NO_CACHE_HEADERS
    )


def _key_resolution_failure_to_mint_error(failure: _KeyResolutionFailure) -> _BridgeMintError:
    """Lift an identity-resolution failure into the mint taxonomy, preserving origin so the status stays
    truthful: the caller's missing credential is 400, a transient DB outage is 503, and a gateway that
    cannot resolve identity is 500."""
    match failure:
        case "no_active_key":
            return "no_identity"
        case "unavailable":
            return "identity_unavailable"
        case "unresolvable":
            return "identity_unresolvable"
        case _:
            assert_never(failure)


def _upstream_rejection_to_mint_error(rejection: _UpstreamGrantRejection) -> _BridgeMintError:
    """Lift an upstream-response rejection into the mint taxonomy; both are upstream faults (502)."""
    match rejection:
        case "no_access_token":
            return "no_upstream_token"
        case "expired_lifetime":
            return "upstream_token_expired"
        case _:
            assert_never(rejection)


async def _prepare_bridge_mint(
    request: Request,
    mcp_server: MCPServer,
    bridge_identity: "_BridgeAuthorizationCode | None" = None,
) -> "_BridgeMintReady | _BridgeMintError":
    """Phase 1 for the authorization_code grant, BEFORE the upstream exchange: confirm the gateway can
    mint (master_key set), resolve the litellm identity, and derive the envelope keys. Returns a ready
    context or a precise failure value. Running before the exchange is what makes every failure here fail
    closed without consuming the single-use code.

    Two identity sources, one envelope. The interactive DCR client authenticates via SSO at the bridged
    authorize, so its identity arrives as ``bridge_identity`` (the user recovered from the gateway
    authorization code) and mints a user subject. The scripted two-header client presents a litellm key
    on the token request instead, so its identity is the active key's hash and mints a key_hash subject.
    A missing or invalid presented key keeps its resolution origin so the mapper statuses it truthfully;
    neither source present is ``no_identity``. The refresh_token grant has its own phase-1
    (:func:`_prepare_bridge_refresh`), which recovers identity from the presented refresh envelope."""
    from litellm.proxy._experimental.mcp_server.outbound_credentials.bridge_credentials import (  # noqa: PLC0415  # inline import avoids a module-load circular import
        envelope_keys_from_master_key,
    )
    from litellm.proxy._experimental.mcp_server.outbound_credentials.envelope import (  # noqa: PLC0415  # inline import avoids a module-load circular import
        key_hash_identity,
        user_identity,
    )
    from litellm.proxy.proxy_server import (  # noqa: PLC0415  # inline import avoids a module-load circular import
        master_key,
    )

    if not master_key:
        return "not_configured"
    keys = envelope_keys_from_master_key(master_key)
    if bridge_identity is not None:
        identity = user_identity(server_id=mcp_server.server_id, user_id=bridge_identity.litellm_user_id)
        return _BridgeMintReady(identity=identity, keys=keys)
    resolved = await _resolve_active_litellm_key(request)
    if not isinstance(resolved, _ResolvedKey):
        return _key_resolution_failure_to_mint_error(resolved)
    identity = key_hash_identity(server_id=mcp_server.server_id, key_hash=resolved.key_hash)
    return _BridgeMintReady(identity=identity, keys=keys)


@dataclass(frozen=True, slots=True)
class _BridgeRefreshReady:
    """A validated refresh request: the identity+keys to mint the renewed pair under, the upstream refresh
    token (unwrapped from the client's refresh envelope) to exchange with the upstream IdP, and the scope
    sealed alongside it at mint. The upstream refresh token is a ``SecretStr`` like every other credential
    in this layer, so a repr or a traceback that captures this value never exposes the raw upstream refresh
    token in plaintext. ``upstream_scope`` carries the originally-granted scope so the renewal re-requests
    it when the client (a DCR/MCP client that typically omits scope on refresh) sends none, keeping the
    renewed token's scope stable against an upstream that would otherwise narrow or drop it."""

    ready: "_BridgeMintReady"
    upstream_refresh_token: SecretStr
    upstream_scope: str | None = None


def _refresh_key_failure_to_mint_error(failure: _KeyResolutionFailure) -> _BridgeMintError:
    """Lift an identity-resolution failure on the refresh path into the mint taxonomy. Unlike the mint
    path, a resolved-but-inactive (or unknown) key is ``invalid_grant`` rather than ``invalid_request``:
    the client did present an identity (sealed in the refresh envelope), but it is no longer live, so the
    refresh is invalid and the client must re-authenticate. A transient outage is still 503 and a gateway
    fault still 500, matching the mint path and admission."""
    match failure:
        case "no_active_key":
            return "invalid_refresh"
        case "unavailable":
            return "identity_unavailable"
        case "unresolvable":
            return "identity_unresolvable"
        case _:
            assert_never(failure)


async def _prepare_bridge_refresh(
    mcp_server: MCPServer, refresh_value: str | None
) -> "_BridgeRefreshReady | _BridgeMintError":
    """Phase 1 for the refresh_token grant, BEFORE the upstream exchange: open the client's refresh
    envelope, re-validate the sealed litellm identity so a revoked key cannot keep refreshing, and
    recover the upstream refresh token to exchange. Identity comes entirely from the sealed envelope, not
    the HTTP request, so the request object is not needed here. The client presents a refresh envelope,
    never a raw upstream refresh token, so a missing value, a non-envelope, an unopenable envelope, or one
    minted for another server is ``invalid_grant``. Running before the exchange means a rejected refresh
    never consumes or rotates the upstream refresh token."""
    from litellm.proxy._experimental.mcp_server.outbound_credentials.bridge_credentials import (  # noqa: PLC0415  # inline import avoids a module-load circular import
        BridgeRefreshOpened,
        envelope_keys_from_master_key,
        open_bridge_refresh_envelope,
    )
    from litellm.proxy.proxy_server import (  # noqa: PLC0415  # inline import avoids a module-load circular import
        master_key,
    )

    if not master_key:
        return "not_configured"
    if not refresh_value:
        return "invalid_refresh"
    keys = envelope_keys_from_master_key(master_key)
    opened = open_bridge_refresh_envelope(refresh_value, keys, datetime.now(timezone.utc), mcp_server.server_id)
    if not isinstance(opened, BridgeRefreshOpened):
        return "invalid_refresh"
    failure = await _revalidate_active_subject(opened.identity)
    if failure is not None:
        return _refresh_key_failure_to_mint_error(failure)
    return _BridgeRefreshReady(
        ready=_BridgeMintReady(identity=opened.identity, keys=keys),
        upstream_refresh_token=opened.refresh.refresh_token,
        upstream_scope=opened.refresh.scope,
    )


def _finish_bridge_mint(
    ready: "_BridgeMintReady", mcp_server: MCPServer, token_response: object, now: datetime
) -> "JSONResponse | _BridgeMintError":
    """Phase 3, AFTER the upstream exchange: seal the upstream grant into the client-held access envelope
    using the pre-resolved identity and keys, and, when the upstream returned a refresh token, seal a
    long-lived refresh envelope alongside it so the client can renew without re-authenticating. Shared by
    the authorization_code and refresh_token paths, so a renewal that the upstream rotates re-issues a
    fresh refresh envelope. The only hard failures here are properties of the upstream access token (no
    usable token, an already-expired lifetime, or a token too large to seal); a refresh token that cannot
    be sealed degrades to an access-only response rather than failing the whole exchange."""
    from litellm.proxy._experimental.mcp_server.outbound_credentials.bridge_credentials import (  # noqa: PLC0415  # inline import avoids a module-load circular import
        build_bridge_token_response,
    )
    from litellm.proxy._experimental.mcp_server.outbound_credentials.envelope import (  # noqa: PLC0415  # inline import avoids a module-load circular import
        SealedEnvelope,
        UpstreamTokenGrant,
    )

    grant = _bridge_grant_from_token_response(token_response)
    if not isinstance(grant, UpstreamTokenGrant):
        return _upstream_rejection_to_mint_error(grant)
    sealed = build_bridge_token_response(ready.identity, grant, ready.keys, now)
    if not isinstance(sealed, SealedEnvelope):
        return "too_large"
    # Report expires_in from the JWT's own second-truncated exp, rounding the elapsed portion up, so the
    # client is never told the bearer lives past the point admission (which uses that exp) rejects it.
    expires_in = max(0, int(sealed.expires_at.timestamp()) - math.ceil(now.timestamp()))
    refresh_envelope = _mint_refresh_envelope_value(ready.identity, token_response, ready.keys, now, mcp_server)
    body = {
        "access_token": sealed.token.get_secret_value(),
        "token_type": "Bearer",
        "expires_in": expires_in,
        # A refresh envelope rides along only when the upstream returned a refresh token to seal; when it
        # rotates on renewal, the client receives the new one and the old envelope's upstream token dies.
        **({"refresh_token": refresh_envelope} if refresh_envelope is not None else {}),
    }
    return JSONResponse(body, headers=TOKEN_NO_CACHE_HEADERS)


def _upstream_refresh_credential(token_response: object) -> "RefreshCredential | None":
    """Extract the upstream refresh grant from a token response, or ``None`` when there is none to seal.
    Each field is isinstance-checked so nothing untyped reaches the refresh envelope; ``refresh_expires_in``
    (the refresh token's own lifetime, when the upstream reports it) is classified like ``expires_in`` and
    bounds the refresh envelope's TTL. An upstream that reports the refresh token itself as already elapsed
    (``refresh_expires_in`` non-positive) yields ``None`` rather than a refresh envelope: sealing a dead
    token would hand the client a full-TTL-capped envelope the IdP will reject, so the exchange degrades to
    an access-only response (the client re-authenticates at access expiry), mirroring how
    :func:`_bridge_grant_from_token_response` refuses an already-elapsed access token instead of capping it."""
    from litellm.proxy._experimental.mcp_server.outbound_credentials.envelope import (  # noqa: PLC0415  # inline import avoids a module-load circular import
        RefreshCredential,
    )

    if not isinstance(token_response, dict):
        return None
    refresh = token_response.get("refresh_token")
    if not isinstance(refresh, str) or not refresh:
        return None
    lifetime = _classify_upstream_lifetime(token_response.get("refresh_expires_in"))
    if lifetime == "expired":
        return None
    scope = token_response.get("scope")
    return RefreshCredential(
        refresh_token=SecretStr(refresh),
        scope=scope if isinstance(scope, str) and scope else None,
        expires_in=lifetime if isinstance(lifetime, int) else None,
    )


def _mint_refresh_envelope_value(
    identity: "EnvelopeIdentity", token_response: object, keys: "EnvelopeKeys", now: datetime, mcp_server: MCPServer
) -> str | None:
    """Seal the upstream refresh grant (if any) into a refresh envelope and return its bearer string, or
    ``None`` when the upstream returned no refresh token or the refresh token is too large to seal. A
    too-large refresh token degrades to an access-only response (logged) rather than failing an exchange
    that already succeeded upstream: the client simply re-authenticates when the access envelope expires."""
    from litellm.proxy._experimental.mcp_server.outbound_credentials.bridge_credentials import (  # noqa: PLC0415  # inline import avoids a module-load circular import
        build_bridge_refresh_token_response,
    )
    from litellm.proxy._experimental.mcp_server.outbound_credentials.envelope import (  # noqa: PLC0415  # inline import avoids a module-load circular import
        SealedEnvelope,
    )

    refresh_credential = _upstream_refresh_credential(token_response)
    if refresh_credential is None:
        return None
    sealed = build_bridge_refresh_token_response(identity, refresh_credential, keys, now)
    if isinstance(sealed, SealedEnvelope):
        return sealed.token.get_secret_value()
    verbose_logger.warning(
        "bridge mint: the upstream refresh token is too large to seal into a refresh envelope for "
        "server=%s; issuing an access-only response, so the client re-authenticates at access expiry",
        mcp_server.server_id,
    )
    return None
