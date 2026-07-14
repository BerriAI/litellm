import asyncio
import html as _html
import json
import math
import secrets
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional, Tuple
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import httpx
from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError
from typing_extensions import assert_never

from litellm._logging import verbose_logger
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.proxy._experimental.mcp_server.auth.token_endpoint_auth import (
    TokenEndpointAuthConfigError,
    build_token_endpoint_client_auth,
)
from litellm.proxy._experimental.mcp_server.faults import (
    CallerRejected,
    CredentialSource,
    UpstreamProtocolFault,
    classify_upstream_dcr_rejection,
    classify_upstream_token_rejection,
    dcr_fault_detail,
    render_token_fault,
)
from litellm.proxy._experimental.mcp_server.gateway_dcr_flow import (
    aggregate_authorize,
    aggregate_token,
    complete_connect_flow,
    is_gateway_dcr_client_id,
    register_aggregate_client,
    relative_request_url,
)
from litellm.proxy._experimental.mcp_server.oauth_utils import (
    TOKEN_NO_CACHE_HEADERS,
    get_request_base_url,
    validate_trusted_redirect_uri,
)
from litellm.proxy.auth.ip_address_utils import IPAddressUtils
from litellm.proxy.common_utils.encrypt_decrypt_utils import (
    decrypt_value_helper,
    encrypt_value_helper,
)
from litellm.proxy.common_utils.http_parsing_utils import _read_request_body
from litellm.proxy.utils import get_server_root_path
from litellm.types.mcp import MCPAuth, MCPCredentials
from litellm.types.mcp_server.mcp_server_manager import MCPServer

if TYPE_CHECKING:
    from litellm.proxy._experimental.mcp_server.outbound_credentials.envelope import (
        EnvelopeIdentity,
        EnvelopeKeys,
        RefreshCredential,
        UpstreamTokenGrant,
    )
    from litellm.proxy._types import LiteLLM_MCPServerTable, UserAPIKeyAuth

# TTL cache for upstream OAuth metadata fetched from pass-through MCP servers.
# Keeps us from hammering the upstream IdP on each discovery request.
# Keyed by (server_id, resource_url) → (expires_at_epoch, payload).
# A payload of ``None`` is a negative-result entry that prevents repeated
# upstream fetches when the IdP consistently has no metadata to serve.
_OAUTH_METADATA_CACHE: Dict[Tuple[str, str], Tuple[float, Optional[dict]]] = {}
_OAUTH_METADATA_CACHE_TTL_SECONDS = 300
_OAUTH_METADATA_NEGATIVE_CACHE_TTL_SECONDS = 60
_OAUTH_METADATA_CACHE_MAX_SIZE = 128
# Per-(server_id, resource_url) async locks so concurrent discovery requests
# coalesce onto a single upstream fetch instead of issuing N parallel calls.
_OAUTH_METADATA_FETCH_LOCKS: Dict[Tuple[str, str], asyncio.Lock] = {}

router = APIRouter(
    tags=["mcp"],
)


def _prune_oauth_metadata_cache(now: Optional[float] = None) -> None:
    now = now if now is not None else time.time()
    expired_cache_keys = [
        cache_key for cache_key, (expires_at, _payload) in _OAUTH_METADATA_CACHE.items() if expires_at <= now
    ]
    for cache_key in expired_cache_keys:
        _OAUTH_METADATA_CACHE.pop(cache_key, None)

    if len(_OAUTH_METADATA_CACHE) > _OAUTH_METADATA_CACHE_MAX_SIZE:
        overflow = len(_OAUTH_METADATA_CACHE) - _OAUTH_METADATA_CACHE_MAX_SIZE
        cache_keys_by_expiry = sorted(
            _OAUTH_METADATA_CACHE,
            key=lambda cache_key: _OAUTH_METADATA_CACHE[cache_key][0],
        )
        for cache_key in cache_keys_by_expiry[:overflow]:
            _OAUTH_METADATA_CACHE.pop(cache_key, None)

    # Drop locks whose cache entry has been evicted and that aren't currently
    # held; held locks stay so in-flight callers continue to coalesce.
    for cache_key in list(_OAUTH_METADATA_FETCH_LOCKS):
        if cache_key in _OAUTH_METADATA_CACHE:
            continue
        lock = _OAUTH_METADATA_FETCH_LOCKS.get(cache_key)
        if lock is None or lock.locked():
            continue
        _OAUTH_METADATA_FETCH_LOCKS.pop(cache_key, None)


def encode_state_with_base_url(
    base_url: str,
    original_state: str,
    code_challenge: Optional[str] = None,
    code_challenge_method: Optional[str] = None,
    client_redirect_uri: Optional[str] = None,
    litellm_user_id: str | None = None,
    mcp_server_id: str | None = None,
) -> str:
    """
    Encode the base_url, original state, and PKCE parameters using encryption.

    Args:
        base_url: The base URL to encode
        original_state: The original state parameter
        code_challenge: PKCE code challenge from client
        code_challenge_method: PKCE code challenge method from client
        client_redirect_uri: Original redirect_uri from client
        litellm_user_id: The SSO-authenticated litellm user captured at the bridge authorize
            (interactive dcr_bridge oauth_delegate only); the callback seals it into the gateway
            authorization code so the token mint can bind the envelope to this user
        mcp_server_id: The bridge server the interactive flow targets, sealed alongside
            litellm_user_id so the gateway code cannot be replayed against another server

    Returns:
        An encrypted string that encodes all values
    """
    state_data = {
        "base_url": base_url,
        "original_state": original_state,
        "code_challenge": code_challenge,
        "code_challenge_method": code_challenge_method,
        "client_redirect_uri": client_redirect_uri,
        "litellm_user_id": litellm_user_id,
        "mcp_server_id": mcp_server_id,
    }
    state_json = json.dumps(state_data, sort_keys=True)
    encrypted_state = encrypt_value_helper(state_json)
    return encrypted_state


def decode_state_hash(encrypted_state: str) -> dict:
    """
    Decode an encrypted state to retrieve all OAuth session data.

    Args:
        encrypted_state: The encrypted string to decode

    Returns:
        A dict containing base_url, original_state, and optional PKCE parameters

    Raises:
        Exception: If decryption fails or data is malformed
    """
    decrypted_json = decrypt_value_helper(encrypted_state, "oauth_state")
    if decrypted_json is None:
        raise ValueError("Failed to decrypt state parameter")

    state_data = json.loads(decrypted_json)
    return state_data


_BRIDGE_AUTH_CODE_PREFIX = "llm_bcode_"


class _BridgeAuthorizationCode(BaseModel):
    """The identity and upstream code the gateway seals into the authorization code it hands a DCR
    client for an interactive dcr_bridge oauth_delegate sign-in, recovered at the token endpoint."""

    model_config = ConfigDict(frozen=True)
    upstream_code: str = Field(min_length=1)
    litellm_user_id: str = Field(min_length=1)
    mcp_server_id: str = Field(min_length=1)


def is_bridge_authorization_code(code: str) -> bool:
    """Cheap prefix check that ``code`` is a gateway-sealed bridge authorization code rather than a
    raw upstream code, so the token endpoint can route without decrypting."""
    return code.startswith(_BRIDGE_AUTH_CODE_PREFIX)


def seal_bridge_authorization_code(upstream_code: str, litellm_user_id: str, mcp_server_id: str) -> str:
    """Seal the upstream authorization code and the SSO-captured litellm user into a gateway
    authorization code. The DCR client only echoes this opaque value back at the token endpoint; the
    gateway decrypts it there to recover the user (to bind the envelope) and the upstream code (to
    exchange with the upstream), so a litellm identity captured in the browser at authorize survives
    to the back-channel token call with nothing stored server-side. Encrypted with the repo's
    authenticated symmetric helper (the same family the OAuth state uses), so the client can neither
    read nor forge it."""
    payload = json.dumps(
        {"upstream_code": upstream_code, "litellm_user_id": litellm_user_id, "mcp_server_id": mcp_server_id},
        sort_keys=True,
    )
    return _BRIDGE_AUTH_CODE_PREFIX + encrypt_value_helper(payload)


def open_bridge_authorization_code(code: str) -> _BridgeAuthorizationCode | None:
    """Recover the sealed identity and upstream code, or ``None`` when ``code`` is not a gateway
    bridge code or does not decrypt / validate. Total over hostile input: a raw upstream code (the
    scripted two-header path) returns ``None`` and the caller falls through to the existing
    behavior."""
    if not is_bridge_authorization_code(code):
        return None
    decrypted = decrypt_value_helper(
        code[len(_BRIDGE_AUTH_CODE_PREFIX) :], "bridge_authorization_code", return_original_value=False
    )
    if not isinstance(decrypted, str):
        return None
    try:
        return _BridgeAuthorizationCode.model_validate_json(decrypted)
    except ValidationError:
        return None


def _session_cookie_user_id(request: Request) -> str | None:
    """The signed-in litellm user for a browser request, or ``None``. Thin wrapper so the
    aggregate DCR flow's verbs receive the identity as a plain value instead of parsing
    cookies themselves."""
    from litellm.proxy._experimental.mcp_server.byok_oauth_endpoints import (  # noqa: PLC0415  # circular import at module load
        _user_id_from_session_cookie,
    )

    return _user_id_from_session_cookie(request)


def _redirect_to_litellm_login(request: Request) -> RedirectResponse:
    """Send an unauthenticated browser through litellm login before the interactive bridge authorize
    can capture its identity. The bridge oauth_delegate flow seals the SSO user into the gateway code,
    so a session is required; without one there is nothing to bind. A same-origin relative
    ``return_to`` (honored by the SSO callback) brings the browser straight back to this authorize
    request after login instead of stranding it on the dashboard."""
    base_url = get_request_base_url(request)
    return RedirectResponse(f"{base_url}/sso/key/generate?{urlencode({'return_to': relative_request_url(request)})}")


# LIT-4197: some upstream authorization servers reject an over-long ``state``
# (the encrypted OAuth session blob routinely exceeds their limit). The upstream
# only needs an opaque value it echoes back on ``/callback``, so we forward a
# short random handle and keep the encrypted session in a per-flow HttpOnly
# cookie bound to that handle. The browser carries the cookie across the
# upstream round trip, so the flow stays correct with no server-side session
# store (works across proxy replicas, unlike an in-process map).
_OAUTH_STATE_COOKIE_PREFIX = "mcp_oauth_state_"
_OAUTH_STATE_COOKIE_TTL_SECONDS = 600
_OAUTH_STATE_HANDLE_BYTES = 32


def _oauth_state_cookie_name(relay_state: str) -> str:
    return f"{_OAUTH_STATE_COOKIE_PREFIX}{relay_state}"


def _oauth_state_cookie_path_and_secure(request: Request) -> tuple[str, bool]:
    parsed = urlparse(get_request_base_url(request))
    return parsed.path or "/", parsed.scheme == "https"


def _set_oauth_state_cookie(
    response: Response,
    request: Request,
    relay_state: str,
    encoded_state: str,
) -> None:
    path, secure = _oauth_state_cookie_path_and_secure(request)
    response.set_cookie(
        key=_oauth_state_cookie_name(relay_state),
        value=encoded_state,
        max_age=_OAUTH_STATE_COOKIE_TTL_SECONDS,
        path=path,
        secure=secure,
        httponly=True,
        samesite="lax",
    )


def _resolve_encoded_oauth_state(request: Request, state: str) -> str:
    """Return the encrypted OAuth session for a ``/callback`` request.

    New flows carry it in a per-flow cookie keyed by the short handle we
    forwarded upstream (the IdP echoes that handle back as ``state``). Flows
    started before this change - or in flight across a deploy - carry the
    encrypted blob directly in ``state``, so fall back to it when the cookie
    is absent.
    """
    cookie_value = request.cookies.get(_oauth_state_cookie_name(state))
    return cookie_value if cookie_value else state


def _clear_oauth_state_cookie(response: Response, request: Request, state: str) -> None:
    cookie_name = _oauth_state_cookie_name(state)
    if cookie_name not in request.cookies:
        return
    path, secure = _oauth_state_cookie_path_and_secure(request)
    response.delete_cookie(
        key=cookie_name,
        path=path,
        secure=secure,
        httponly=True,
        samesite="lax",
    )


def _get_validated_client_redirect_uri(request: Request, state_data: Dict[str, Any]) -> str:
    """Return a trusted (same-origin, loopback, or ops-allowlisted)
    client redirect URI from OAuth state.
    """
    redirect_uri = state_data.get("client_redirect_uri") or state_data.get("base_url")
    if not redirect_uri or not isinstance(redirect_uri, str):
        raise HTTPException(status_code=400, detail="Invalid redirect URI")
    validate_trusted_redirect_uri(request, redirect_uri)
    return redirect_uri


def _append_query_params(url: str, params: Dict[str, str]) -> str:
    parsed = urlparse(url)
    query_params = parse_qsl(parsed.query, keep_blank_values=True)
    query_params.extend(params.items())
    return urlunparse(parsed._replace(query=urlencode(query_params)))


def _resolve_oauth2_server_for_root_endpoints(
    client_ip: Optional[str] = None,
) -> Optional[MCPServer]:
    """
    Resolve the MCP server for root-level OAuth endpoints (no server name in path).

    When the MCP SDK hits root-level endpoints like /register, /authorize, /token
    without a server name prefix, we try to find the right server automatically.
    Returns the server if exactly one OAuth2 server is configured, else None.
    """
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
        global_mcp_server_manager,
    )

    registry = global_mcp_server_manager.get_filtered_registry(client_ip=client_ip)
    oauth2_servers = [s for s in registry.values() if s.auth_type == MCPAuth.oauth2]
    if len(oauth2_servers) == 1:
        return oauth2_servers[0]
    return None


def _normalize_for_token_comparison(value: Any) -> str:
    """Stringify ``value`` for token-rule comparison.

    Booleans are lower-cased so Python's ``True`` / ``False`` line up with
    JSON-style ``"true"`` / ``"false"`` rules from admin config.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _validate_token_response(
    token_response: Dict[str, Any],
    validation_rules: Dict[str, Any],
    server_id: str,
) -> None:
    """Raise HTTPException 403 if any validation rule doesn't match the token response.

    Supports dot-notation for nested fields (e.g. ``"team.enterprise_id"`` checks
    ``token_response["team"]["enterprise_id"]``).  Top-level keys are tried first,
    then dot-split traversal.  All comparisons are string-coerced so that numeric
    values in the response (e.g. ``"org_id": 12345``) match string rules
    (``"org_id": "12345"``).  Booleans are normalised to JSON-style ``"true"`` /
    ``"false"`` so admin rules written as ``{"verified": "true"}`` match upstream
    responses of ``{"verified": true}``.
    """
    for key, expected in validation_rules.items():
        actual: Any = token_response.get(key)
        # Try dot-notation traversal when top-level lookup returns None
        if actual is None and "." in key:
            obj: Any = token_response
            for part in key.split("."):
                if isinstance(obj, dict):
                    obj = obj.get(part)
                else:
                    obj = None
                    break
            actual = obj
        # Treat absent fields as a distinct failure from a mismatched value
        if actual is None:
            raise HTTPException(
                status_code=403,
                detail={
                    "error": "token_validation_failed",
                    "server_id": server_id,
                    "field": key,
                    "message": (f"OAuth token rejected: required field '{key}' is absent"),
                },
            )
        if _normalize_for_token_comparison(actual) != _normalize_for_token_comparison(expected):
            raise HTTPException(
                status_code=403,
                detail={
                    "error": "token_validation_failed",
                    "server_id": server_id,
                    "field": key,
                    "message": (f"OAuth token rejected: '{key}' = '{actual}', expected '{expected}'"),
                },
            )


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


async def _store_per_user_token_server_side(
    server: MCPServer,
    user_id: str,
    token_response: Dict[str, Any],
) -> None:
    """Persist the OAuth token server-side and warm the Redis cache.

    Called from the token endpoint after a successful code exchange or refresh.
    Errors are logged but NOT re-raised — the token is always returned to the
    client even when server-side storage fails.
    """
    from litellm.proxy._experimental.mcp_server.oauth2_token_cache import (  # noqa: PLC0415
        _compute_per_user_token_ttl,
        mcp_per_user_token_cache,
    )
    from litellm.proxy.utils import get_prisma_client_or_throw  # noqa: PLC0415

    access_token: Optional[str] = token_response.get("access_token")
    if not access_token:
        return

    raw_expires = token_response.get("expires_in")
    try:
        expires_in: Optional[int] = int(raw_expires) if raw_expires is not None else None
    except (TypeError, ValueError):
        expires_in = None

    refresh_token: Optional[str] = token_response.get("refresh_token") or None
    raw_scope = token_response.get("scope")
    scopes: Optional[list] = raw_scope.split() if isinstance(raw_scope, str) and raw_scope else None

    try:
        prisma_client = get_prisma_client_or_throw("Database not connected. Cannot store per-user OAuth token.")
        from litellm.proxy._experimental.mcp_server.db import (  # noqa: PLC0415
            store_user_oauth_credential,
        )

        await store_user_oauth_credential(
            prisma_client=prisma_client,
            user_id=user_id,
            server_id=server.server_id,
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=expires_in,
            scopes=scopes,
        )
        verbose_logger.info(
            "_store_per_user_token_server_side: stored token for user=%s server=%s",
            user_id,
            server.server_id,
        )
    except Exception as exc:
        verbose_logger.warning(
            "_store_per_user_token_server_side: DB storage failed for user=%s server=%s: %s",
            user_id,
            server.server_id,
            exc,
        )
        return  # Don't warm Redis if DB write failed

    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (  # noqa: PLC0415
        global_mcp_server_manager,
    )

    await global_mcp_server_manager.invalidate_user_oauth_token_cache(user_id, server.server_id)

    # Warm the Redis cache so the first subsequent MCP call is a cache hit
    ttl = _compute_per_user_token_ttl(server, expires_in)
    await mcp_per_user_token_cache.set(
        user_id=user_id,
        server_id=server.server_id,
        access_token=access_token,
        ttl=ttl,
    )


def _raise_if_not_oauth2(mcp_server: MCPServer) -> None:
    """Reject a server without upstream OAuth from the gateway's authorize/token/register flow.

    The client-forwarded token modes (``true_passthrough`` / ``oauth_delegate``) are allowed
    through: the caller owns the upstream token, and this relayed flow is how a browser obtains
    one against the upstream IdP (the admin UI's browser-only Authorize uses it). The minted
    token is upstream-audienced and held by the caller; the gateway persists nothing for these
    modes (``_persist_dcr_client_registration`` skips them unconditionally, so even the admin
    Authorize path with ``persist_credentials`` enabled writes nothing to the server row).
    """
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (  # noqa: PLC0415  # circular import with mcp_server_manager at module load
        _UPSTREAM_OAUTH_DISCOVERY_AUTH_TYPES,
    )

    if mcp_server.auth_type in _UPSTREAM_OAUTH_DISCOVERY_AUTH_TYPES:
        return
    raise HTTPException(
        status_code=400,
        detail={
            "error": "server_not_oauth2",
            "message": (
                f"MCP server '{mcp_server.server_name or mcp_server.name}' does not use OAuth "
                f"(auth_type={mcp_server.auth_type}). This server does not support the authorization-code "
                "flow; it has no client_id, authorize, token, or registration endpoint. "
                "Access is controlled by the server's configured auth_type and access groups"
            ),
        },
    )


def _raise_unless_oauth2_discovery_server(
    mcp_server: Optional[MCPServer],
    mcp_server_name: Optional[str],
    description: str,
) -> None:
    """404 a NAMED discovery request unless it resolves to an oauth2 or DCR-bridge server.

    A named server that is unknown (or hidden from the caller) and one that exists
    but is non-oauth2 both return the same 404, so the well-known discovery paths
    cannot be used to enumerate non-OAuth server names. Root discovery (no name) is
    unaffected, and pass-through servers are resolved by the caller before this runs.
    DCR-bridge servers are admitted because they serve the gateway's own authorization
    server metadata (the register, authorize, and token relays).
    """
    if mcp_server_name is None:
        return
    if mcp_server is not None and mcp_server.auth_type == MCPAuth.oauth2:
        return
    if mcp_server is not None and mcp_server.is_dcr_bridge:
        return
    raise HTTPException(
        status_code=404,
        detail=f"MCP server '{mcp_server_name}' is {description}",
    )


def _dcr_bridge_relays_client_registration(mcp_server: MCPServer) -> bool:
    """True when a DCR-bridge server relays client registration to the upstream authorization
    server instead of short-circuiting to an admin-configured OAuth client. In the relay arm the
    upstream holds each client's own registration, so the authorize and token relays pass the
    client's ``client_id`` and ``redirect_uri`` through verbatim and the authorization code
    returns directly to the client's redirect URI without transiting the gateway. Gateway-side
    redirect trust and the ``/callback`` state relay therefore only apply to the short-circuit
    arm, where the upstream only knows the gateway's own callback."""
    return mcp_server.is_dcr_bridge and bool(mcp_server.registration_url) and not mcp_server.client_id


def _require_s256_pkce(
    code_challenge: Optional[str],
    code_challenge_method: Optional[str],
) -> Tuple[str, str]:
    """DCR-bridge servers serve unauthenticated public OAuth clients, so the PKCE downgrade
    paths (no challenge, or a non-S256 method; RFC 7636 defaults a missing method to ``plain``)
    are rejected at the gateway instead of relying on upstream enforcement. Returns the
    validated pair so callers get non-optional values."""
    if code_challenge and code_challenge_method == "S256":
        return code_challenge, code_challenge_method
    raise HTTPException(
        status_code=400,
        detail=(
            "This server requires PKCE: send code_challenge with "
            "code_challenge_method=S256 on the authorization request"
        ),
    )


def _redirect_to_upstream_authorize(
    *,
    mcp_server: MCPServer,
    client_id: str,
    redirect_uri: str,
    state: str,
    code_challenge: str,
    code_challenge_method: str,
    response_type: Optional[str],
    scope: Optional[str],
) -> RedirectResponse:
    """The bridge relay arm's authorize redirect: every client-supplied parameter passes through
    to the upstream authorize endpoint verbatim, no relay state cookie is set, and the upstream
    enforces its own registered redirect binding for the client."""
    scope_value = scope or (" ".join(mcp_server.scopes) if mcp_server.scopes else None)
    passthrough_params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "state": state,
        "response_type": response_type or "code",
        "code_challenge": code_challenge,
        "code_challenge_method": code_challenge_method,
        **({"scope": scope_value} if scope_value else {}),
    }
    parsed_auth_url = urlparse(mcp_server.authorization_url or "")
    merged_params = {**dict(parse_qsl(parsed_auth_url.query)), **passthrough_params}
    return RedirectResponse(urlunparse(parsed_auth_url._replace(query=urlencode(merged_params))))


async def authorize_with_server(
    request: Request,
    mcp_server: MCPServer,
    client_id: str,
    redirect_uri: str,
    state: str = "",
    code_challenge: Optional[str] = None,
    code_challenge_method: Optional[str] = None,
    response_type: Optional[str] = None,
    scope: Optional[str] = None,
):
    _raise_if_not_oauth2(mcp_server)
    if mcp_server.authorization_url is None:
        raise HTTPException(status_code=400, detail="MCP server authorization url is not set")

    if mcp_server.is_dcr_bridge:
        # Enforce S256 PKCE on both bridge arms. The relay arm forwards the validated,
        # now-non-optional pair to the upstream authorize; the short-circuit arm keeps
        # calling this for its enforcement side effect, then falls through to the gateway
        # /callback flow below, which reads the original code_challenge names.
        bridge_challenge, bridge_method = _require_s256_pkce(code_challenge, code_challenge_method)
        if _dcr_bridge_relays_client_registration(mcp_server):
            return _redirect_to_upstream_authorize(
                mcp_server=mcp_server,
                client_id=client_id,
                redirect_uri=redirect_uri,
                state=state,
                code_challenge=bridge_challenge,
                code_challenge_method=bridge_method,
                response_type=response_type,
                scope=scope,
            )

    # Trusted redirect_uri: same-origin, loopback, or ops-allowlisted.
    # The URI is encrypted into the OAuth state and decoded on
    # /callback to redirect the user back; a non-trusted URI would be
    # an open-redirect + code-theft primitive (VERIA-57 root cause B).
    validate_trusted_redirect_uri(request, redirect_uri)
    parsed = urlparse(redirect_uri)
    base_url = urlunparse(parsed._replace(query=""))
    request_base_url = get_request_base_url(request)

    # Interactive dcr_bridge oauth_delegate sign-in: this arm runs the gateway /callback and /token in
    # the loop, so the gateway can capture the litellm user here (from the browser's UI session) and
    # carry it to the back-channel token mint. Seal the SSO user and the target server into the state;
    # the callback reads them back to mint the gateway authorization code. A DCR client cannot present a
    # litellm key, so the browser session is the only identity source; without one there is nothing to
    # bind, so send the user through login first. Every other oauth2 server keeps the identity-less state.
    litellm_user_id: str | None = None
    if mcp_server.is_dcr_bridge and mcp_server.is_oauth_delegate:
        from litellm.proxy._experimental.mcp_server.byok_oauth_endpoints import (  # noqa: PLC0415  # inline import avoids a module-load circular import
            _user_id_from_session_cookie,
        )

        litellm_user_id = _user_id_from_session_cookie(request)
        if litellm_user_id is None:
            return _redirect_to_litellm_login(request)

    encoded_state = encode_state_with_base_url(
        base_url=base_url,
        original_state=state,
        code_challenge=code_challenge,
        code_challenge_method=code_challenge_method,
        client_redirect_uri=redirect_uri,
        litellm_user_id=litellm_user_id,
        mcp_server_id=mcp_server.server_id if litellm_user_id else None,
    )
    relay_state = secrets.token_urlsafe(_OAUTH_STATE_HANDLE_BYTES)

    params = {
        "client_id": mcp_server.client_id if mcp_server.client_id else client_id,
        "redirect_uri": f"{request_base_url}/callback",
        "state": relay_state,
        "response_type": response_type or "code",
    }
    if scope:
        params["scope"] = scope
    elif mcp_server.scopes:
        params["scope"] = " ".join(mcp_server.scopes)

    if code_challenge:
        params["code_challenge"] = code_challenge
    if code_challenge_method:
        params["code_challenge_method"] = code_challenge_method

    parsed_auth_url = urlparse(mcp_server.authorization_url)
    existing_params = dict(parse_qsl(parsed_auth_url.query))
    existing_params.update(params)
    final_url = urlunparse(parsed_auth_url._replace(query=urlencode(existing_params)))
    response = RedirectResponse(final_url)
    _set_oauth_state_cookie(response, request, relay_state, encoded_state)
    return response


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
    bridge_identity: _BridgeAuthorizationCode | None = None,
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


def _token_credential_source(mcp_server: MCPServer) -> CredentialSource:
    """Mirrors the resolved-client rule in :func:`exchange_token_with_server`: when the server has a
    stored client_id the gateway presents its own credentials upstream, so a credential rejection is
    the operator's fault, not the caller's."""
    return "gateway_stored" if mcp_server.client_id else "caller_supplied"


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


async def exchange_token_with_server(
    request: Request,
    mcp_server: MCPServer,
    grant_type: str,
    code: Optional[str],
    redirect_uri: Optional[str],
    client_id: str,
    client_secret: Optional[str],
    code_verifier: Optional[str],
    refresh_token: Optional[str] = None,
    scope: Optional[str] = None,
):
    _raise_if_not_oauth2(mcp_server)
    if grant_type not in ("authorization_code", "refresh_token"):
        raise HTTPException(status_code=400, detail="Unsupported grant_type")

    if mcp_server.token_url is None:
        raise HTTPException(status_code=400, detail="MCP server token url is not set")

    # The id and secret must come from the same source. When the server-side client_id wins,
    # falling back to the caller's secret pairs the persisted client with a foreign secret; the
    # register short-circuit hands clients a placeholder secret ("dummy"), so a re-auth against a
    # persisted public PKCE client (no stored secret) would send that placeholder and the IdP 401s.
    resolved_client_id = mcp_server.client_id if mcp_server.client_id else client_id
    resolved_client_secret = mcp_server.client_secret if mcp_server.client_id else client_secret
    try:
        client_auth = build_token_endpoint_client_auth(
            auth_method=mcp_server.token_endpoint_auth_method,
            client_id=resolved_client_id,
            client_secret=resolved_client_secret,
        )
    except TokenEndpointAuthConfigError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    bridge_identity: _BridgeAuthorizationCode | None = None
    bridge_mint_ready: _BridgeMintReady | None = None
    bridge_upstream_refresh: SecretStr | None = None
    bridge_upstream_scope: str | None = None
    refresh_request_scope: str | None = None
    is_bridge = mcp_server.is_oauth_delegate and mcp_server.is_dcr_bridge

    if grant_type == "refresh_token":
        # Phase 1 for a bridge refresh: open the client's refresh envelope, re-validate the sealed
        # identity, and unwrap the real upstream refresh token BEFORE building token_data, so the exchange
        # sends the upstream token and never the envelope. A failure returns without touching the upstream.
        if is_bridge:
            prepared_refresh = await _prepare_bridge_refresh(mcp_server, refresh_token)
            if not isinstance(prepared_refresh, _BridgeRefreshReady):
                return _bridge_mint_error_response(prepared_refresh)
            bridge_mint_ready = prepared_refresh.ready
            bridge_upstream_refresh = prepared_refresh.upstream_refresh_token
            bridge_upstream_scope = prepared_refresh.upstream_scope
        # A bridge server sends the unwrapped upstream refresh token recovered from the client's refresh
        # envelope above; every other server sends the client's own refresh token verbatim.
        upstream_refresh_token = (
            bridge_upstream_refresh.get_secret_value() if bridge_upstream_refresh is not None else refresh_token
        )
        if not upstream_refresh_token:
            raise HTTPException(
                status_code=400,
                detail="refresh_token is required for refresh_token grant",
            )
        token_data: dict = {
            "grant_type": "refresh_token",
            "refresh_token": upstream_refresh_token,
            **client_auth.body,
        }
        refresh_request_scope = scope or bridge_upstream_scope
        if refresh_request_scope:
            token_data["scope"] = refresh_request_scope
    else:
        if not code:
            raise HTTPException(
                status_code=400,
                detail="code is required for authorization_code grant",
            )
        # Interactive dcr_bridge oauth_delegate: the client presents the gateway authorization code the
        # callback sealed. Recover the SSO user and the real upstream code from it; the upstream exchange
        # below uses the upstream code, and the mint binds the envelope to the recovered user. Bind the
        # sealed server to this request so a code minted for one bridge server cannot be spent at another.
        # A raw upstream code (scripted path) opens to None and the code is used as-is.
        bridge_identity = open_bridge_authorization_code(code)
        if bridge_identity is not None:
            if bridge_identity.mcp_server_id != mcp_server.server_id:
                raise HTTPException(
                    status_code=400,
                    detail="Authorization code was issued for a different MCP server",
                )
            code = bridge_identity.upstream_code
        bridge_token_relay = _dcr_bridge_relays_client_registration(mcp_server)
        if bridge_token_relay and not redirect_uri:
            raise HTTPException(
                status_code=400,
                detail=(
                    "redirect_uri is required for the authorization_code grant on this server; "
                    "send the same redirect_uri used on the authorization request"
                ),
            )
        proxy_base_url = get_request_base_url(request)
        resolved_redirect_uri = redirect_uri if bridge_token_relay else f"{proxy_base_url}/callback"
        token_data = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": resolved_redirect_uri,
            **client_auth.body,
        }
        if code_verifier:
            token_data["code_verifier"] = code_verifier
        # Phase 1 for a bridge authorization_code mint: resolve identity (the SSO user recovered above, or
        # the presented litellm key) and the envelope keys BEFORE the exchange consumes the single-use code.
        if is_bridge:
            prepared = await _prepare_bridge_mint(request, mcp_server, bridge_identity)
            if not isinstance(prepared, _BridgeMintReady):
                return _bridge_mint_error_response(prepared)
            bridge_mint_ready = prepared
    async_client = get_async_httpx_client(llm_provider=httpxSpecialProvider.Oauth2Check)
    try:
        response = await async_client.post(
            mcp_server.token_url,
            headers={"Accept": "application/json", **client_auth.headers},
            data=token_data,
        )
        if response is not None:
            response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        fault = classify_upstream_token_rejection(
            exc.response,
            credential_source=_token_credential_source(mcp_server),
            log_context=mcp_server.server_id,
        )
        upstream_rejected_bridge_refresh = (
            is_bridge
            and grant_type == "refresh_token"
            and isinstance(fault, CallerRejected)
            and fault.code == "invalid_grant"
        )
        if upstream_rejected_bridge_refresh:
            verbose_logger.info(
                "bridge refresh: the upstream rejected the sealed refresh token for server=%s with "
                "invalid_grant (revoked or expired at the IdP); returning invalid_grant so the client "
                "re-runs authorization_code rather than an opaque upstream error",
                mcp_server.server_id,
            )
            return _bridge_mint_error_response("invalid_refresh")
        return render_token_fault(fault)
    if response is None:
        raise HTTPException(
            status_code=502,
            detail="MCP upstream token endpoint returned no response",
        )
    token_response = response.json()

    # Validate token response against server-configured rules before any storage.
    # This rejects tokens from wrong Slack workspaces, Atlassian orgs, etc.
    if mcp_server.token_validation and isinstance(mcp_server.token_validation, dict):
        _validate_token_response(
            token_response=token_response,
            validation_rules=mcp_server.token_validation,
            server_id=mcp_server.server_id,
        )

    # Store server-side when the server is configured for per-user OAuth and
    # the calling client has provided a valid LiteLLM identity.
    # Errors are non-fatal: the token is still returned to the client.
    if mcp_server.needs_user_oauth_token:
        user_id = await _extract_user_id_from_request(request)
        if user_id:
            try:
                await _store_per_user_token_server_side(
                    server=mcp_server,
                    user_id=user_id,
                    token_response=token_response,
                )
            except Exception as exc:
                verbose_logger.warning(
                    "exchange_token_with_server: server-side storage failed for user=%s server=%s: %s",
                    user_id,
                    mcp_server.server_id,
                    exc,
                )
        else:
            verbose_logger.warning(
                "exchange_token_with_server: could not resolve a LiteLLM user_id for the request, "
                "so the per-user token for server=%s was NOT stored. The authorization_code egress "
                "requires the stored token, so the client will be challenged with 401 on reconnect. "
                "Ensure the request carries a valid LiteLLM key (x-litellm-api-key or Authorization), "
                "or store it via POST /mcp/server/{id}/oauth-user-credential.",
                mcp_server.server_id,
            )

    # A DCR-bridge oauth_delegate server hands the client a gateway-bound envelope (identity plus the
    # upstream token) instead of the raw upstream token, so the one bearer both admits the caller and
    # forwards the upstream credential. Only this mode mints; every other server returns the raw token.
    if bridge_mint_ready is not None:
        if refresh_request_scope and isinstance(token_response, dict) and not token_response.get("scope"):
            token_response = {**token_response, "scope": refresh_request_scope}
        # Phase 3: seal the upstream grant into the client-held envelope; failures map through the same
        # OAuth-shaped response as the phase-1 preconditions.
        minted = _finish_bridge_mint(bridge_mint_ready, mcp_server, token_response, datetime.now(timezone.utc))
        return minted if isinstance(minted, JSONResponse) else _bridge_mint_error_response(minted)

    raw_access_token = token_response.get("access_token") if isinstance(token_response, dict) else None
    if not isinstance(raw_access_token, str) or not raw_access_token:
        return render_token_fault(UpstreamProtocolFault(note="the upstream token response has no usable access_token"))

    result = {
        "access_token": raw_access_token,
        "token_type": token_response.get("token_type", "Bearer"),
    }

    if token_response.get("expires_in") is not None:
        result["expires_in"] = token_response["expires_in"]
    if token_response.get("refresh_token"):
        result["refresh_token"] = token_response["refresh_token"]
    if token_response.get("scope"):
        result["scope"] = token_response["scope"]

    # RFC 6749 §5.1: token responses must not be cached.
    return JSONResponse(result, headers=TOKEN_NO_CACHE_HEADERS)


class _DcrClientRegistration(BaseModel):
    """RFC 7591 dynamic client registration response, narrowed to the fields the gateway
    must persist to authenticate later token-endpoint calls. Extra members are ignored."""

    client_id: str
    client_secret: Optional[str] = None
    token_endpoint_auth_method: Optional[str] = None


class _PersistedDcrCredentials(BaseModel):
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    token_endpoint_auth_method: Optional[str] = None
    redirect_uris: Optional[list[str]] = None


def _redirect_uri_not_registered(credentials: _PersistedDcrCredentials, current_redirect_uri: str) -> bool:
    """Whether a persisted DCR client is positively known NOT to cover the current callback.

    A DCR client is bound to the redirect_uris it was registered with; if the proxy's
    resolved public origin has since changed, every authorize built for it will be
    rejected by the IdP. Clients persisted before ``redirect_uris`` was recorded (and
    admin-configured clients, which never get a recording) return False so they are
    grandfathered rather than re-registered, because re-minting a client_id orphans
    every user's refresh tokens for that server."""
    recorded = credentials.redirect_uris
    if not recorded:
        return False
    return current_redirect_uri not in recorded


def _get_persisted_dcr_credentials(credentials: object) -> Optional[_PersistedDcrCredentials]:
    if not credentials:
        return None
    try:
        return (
            _PersistedDcrCredentials.model_validate_json(credentials)
            if isinstance(credentials, str)
            else _PersistedDcrCredentials.model_validate(credentials)
        )
    except ValidationError:
        return None


def _decrypt_persisted_dcr_credential(value: Optional[str], key: str) -> Optional[str]:
    if value is None:
        return None
    return decrypt_value_helper(
        value=value,
        key=key,
        exception_type="debug",
        return_original_value=True,
    )


def _apply_persisted_dcr_credentials(mcp_server: MCPServer, credentials: _PersistedDcrCredentials) -> bool:
    client_id = _decrypt_persisted_dcr_credential(credentials.client_id, "client_id")
    if not client_id:
        return False
    mcp_server.client_id = client_id
    mcp_server.client_secret = _decrypt_persisted_dcr_credential(credentials.client_secret, "client_secret")
    mcp_server.token_endpoint_auth_method = credentials.token_endpoint_auth_method
    return True


async def _get_persisted_mcp_server_with_dcr_client_id(
    mcp_server: MCPServer,
) -> Optional[tuple["LiteLLM_MCPServerTable", _PersistedDcrCredentials]]:
    from litellm.proxy._experimental.mcp_server.db import get_mcp_server  # noqa: PLC0415
    from litellm.proxy.utils import get_prisma_client_or_throw  # noqa: PLC0415

    try:
        prisma_client = get_prisma_client_or_throw("Database not connected. Cannot read MCP OAuth client registration.")
        persisted_mcp_server = await get_mcp_server(
            prisma_client=prisma_client,
            server_id=mcp_server.server_id,
        )
    except Exception as exc:  # noqa: BLE001
        verbose_logger.debug(
            "register_client_with_server: failed to read persisted DCR client registration for server_id=%s: %s",
            mcp_server.server_id,
            exc,
        )
        return None

    if persisted_mcp_server is None:
        return None

    credentials = _get_persisted_dcr_credentials(persisted_mcp_server.credentials)
    if credentials is None or not credentials.client_id:
        return None

    return persisted_mcp_server, credentials


async def _reuse_persisted_dcr_client_if_available(
    mcp_server: MCPServer, current_redirect_uri: Optional[str] = None
) -> bool:
    persisted = await _get_persisted_mcp_server_with_dcr_client_id(mcp_server)
    if persisted is None:
        return False
    persisted_mcp_server, credentials = persisted
    if current_redirect_uri is not None and _redirect_uri_not_registered(credentials, current_redirect_uri):
        verbose_logger.debug(
            "register_client_with_server: not reusing persisted DCR client for server_id=%s; its registered "
            "redirect_uris=%s do not include the current callback %s. The operator-facing warning for this "
            "re-registration event is emitted once by _persisted_dcr_redirect_uri_is_stale.",
            mcp_server.server_id,
            credentials.redirect_uris,
            current_redirect_uri,
        )
        return False
    if not _apply_persisted_dcr_credentials(mcp_server, credentials):
        return False

    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (  # noqa: PLC0415
        global_mcp_server_manager,
    )

    try:
        await global_mcp_server_manager.update_server(persisted_mcp_server)
    except Exception as exc:  # noqa: BLE001
        verbose_logger.warning(
            "register_client_with_server: failed to refresh persisted DCR client registration for server_id=%s: %s",
            mcp_server.server_id,
            exc,
        )
    return bool(mcp_server.client_id)


async def _persisted_dcr_redirect_uri_is_stale(mcp_server: MCPServer, current_redirect_uri: str) -> bool:
    """Whether the server's persisted DCR client is bound to redirect_uris that no longer
    cover the current proxy callback, meaning authorize is guaranteed to fail IdP-side.

    Consulted when the in-memory server already carries a hydrated client_id, which
    otherwise short-circuits registration before any redirect check can run. Servers
    without a persisted DCR recording (admin-configured client_id, or registered before
    redirect_uris were recorded) are never reported stale."""
    persisted = await _get_persisted_mcp_server_with_dcr_client_id(mcp_server)
    if persisted is None:
        return False
    _, credentials = persisted
    if not _redirect_uri_not_registered(credentials, current_redirect_uri):
        return False
    verbose_logger.warning(
        "register_client_with_server: persisted DCR client for server_id=%s is registered with redirect_uris=%s "
        "which do not include the current callback %s (proxy origin changed); registering a replacement client. "
        "Users previously signed in to this server will need to re-authenticate.",
        mcp_server.server_id,
        credentials.redirect_uris,
        current_redirect_uri,
    )
    return True


DcrRegistrationPersistenceResult = Literal["persisted", "reused", "skipped", "failed"]


async def _persist_dcr_client_registration(
    mcp_server: MCPServer, registration_response: object, current_redirect_uri: str
) -> DcrRegistrationPersistenceResult:
    """Persist the dynamically registered OAuth client (RFC 7591) onto the MCP server row.

    The interactive authorization_code flow mints a ``client_id`` via Dynamic Client
    Registration that discovery cannot re-derive; without persisting it the autonomous
    ``refresh_token`` grant has no client identity, so an expired access token forces a
    full re-authorization instead of a silent refresh. Mirrors the ``encrypt_credentials``
    write that ``client_credentials`` and token exchange already use. Failures are logged,
    never raised: registration still returns to the caller even when persistence fails.

    The client-forwarded token modes (``true_passthrough`` / ``oauth_delegate``) are skipped
    unconditionally: the caller holds the upstream token and the gateway must hold no OAuth
    client identity for these servers. Persisting here would stamp ``oauth2_flow`` and a
    ``client_id`` onto a server whose mode promises the gateway stores nothing, making a
    fresh pass-through server read as gateway-authorized.

    ``redirect_uris`` records what the client is bound to so a later origin change can be
    detected as a positive mismatch and trigger re-registration instead of stranding the
    server on IdP-side redirect_uri rejections. ``client_secret`` and
    ``token_endpoint_auth_method`` are written explicitly (None when absent) because
    ``update_mcp_server`` merges credential blobs: a re-registered public client must not
    inherit the previous client's secret or auth method.
    """
    if mcp_server.is_true_passthrough or mcp_server.is_oauth_delegate:
        return "skipped"

    try:
        registration = _DcrClientRegistration.model_validate(registration_response)
    except ValidationError as exc:
        verbose_logger.warning(
            "register_client_with_server: DCR response has no usable client_id for server_id=%s; "
            "client registration not persisted (%s)",
            mcp_server.server_id,
            exc,
        )
        return "failed"

    if await _reuse_persisted_dcr_client_if_available(mcp_server, current_redirect_uri=current_redirect_uri):
        return "reused"

    credentials: MCPCredentials = {
        "client_id": registration.client_id,
        "client_secret": registration.client_secret,
        "token_endpoint_auth_method": (
            "client_secret_basic" if registration.token_endpoint_auth_method == "client_secret_basic" else None
        ),
        "redirect_uris": [current_redirect_uri],
    }

    from litellm.proxy._experimental.mcp_server.db import update_mcp_server  # noqa: PLC0415
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (  # noqa: PLC0415
        global_mcp_server_manager,
    )
    from litellm.proxy._types import UpdateMCPServerRequest  # noqa: PLC0415
    from litellm.proxy.utils import get_prisma_client_or_throw  # noqa: PLC0415

    try:
        prisma_client = get_prisma_client_or_throw(
            "Database not connected. Cannot persist MCP OAuth client registration."
        )
        updated_row = await update_mcp_server(
            prisma_client=prisma_client,
            data=UpdateMCPServerRequest(
                server_id=mcp_server.server_id,
                credentials=credentials,
                oauth2_flow="authorization_code",
                **({"token_url": mcp_server.token_url} if mcp_server.token_url else {}),
            ),
            touched_by="mcp_oauth_dcr",
        )
        await global_mcp_server_manager.update_server(updated_row)
        return "persisted"
    except Exception as exc:  # noqa: BLE001
        verbose_logger.warning(
            "register_client_with_server: failed to persist DCR client registration for server_id=%s: %s",
            mcp_server.server_id,
            exc,
        )
        return "failed"


async def register_client_with_server(
    request: Request,
    mcp_server: MCPServer,
    client_name: str,
    grant_types: Optional[list],
    response_types: Optional[list],
    token_endpoint_auth_method: Optional[str],
    fallback_client_id: Optional[str] = None,
    persist_credentials: bool = False,
    client_redirect_uris: Optional[list] = None,
):
    _raise_if_not_oauth2(mcp_server)
    request_base_url = get_request_base_url(request)
    current_redirect_uri = f"{request_base_url}/callback"
    dummy_return = {
        "client_id": fallback_client_id or mcp_server.server_name,
        "client_secret": "dummy",
        "redirect_uris": [current_redirect_uri],
    }

    if mcp_server.client_id and not (
        persist_credentials
        and mcp_server.registration_url
        and await _persisted_dcr_redirect_uri_is_stale(mcp_server, current_redirect_uri)
    ):
        return dummy_return

    if await _reuse_persisted_dcr_client_if_available(
        mcp_server,
        current_redirect_uri=current_redirect_uri if persist_credentials else None,
    ):
        return dummy_return

    if mcp_server.authorization_url is None:
        raise HTTPException(status_code=400, detail="MCP server authorization url is not set")

    if mcp_server.registration_url is None:
        return dummy_return

    bridge_relay = _dcr_bridge_relays_client_registration(mcp_server)
    if bridge_relay and not client_redirect_uris:
        raise HTTPException(
            status_code=400,
            detail="redirect_uris is required to register a client with this server",
        )

    register_data = {
        "client_name": client_name,
        "redirect_uris": client_redirect_uris if bridge_relay else [current_redirect_uri],
        "grant_types": grant_types or (["authorization_code", "refresh_token"] if bridge_relay else []),
        "response_types": response_types or (["code"] if bridge_relay else []),
        "token_endpoint_auth_method": token_endpoint_auth_method or ("none" if bridge_relay else ""),
    }
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    async_client = get_async_httpx_client(llm_provider=httpxSpecialProvider.Oauth2Register)
    try:
        response = await async_client.post(
            mcp_server.registration_url,
            headers=headers,
            json=register_data,
        )
        if response is not None:
            response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        status_code, detail = dcr_fault_detail(
            classify_upstream_dcr_rejection(exc.response, log_context=mcp_server.server_id)
        )
        raise HTTPException(status_code=status_code, detail=detail) from exc
    if response is None:
        raise HTTPException(
            status_code=502,
            detail="MCP upstream registration endpoint returned no response",
        )

    token_response = response.json()

    if persist_credentials and not bridge_relay:
        persistence_result = await _persist_dcr_client_registration(mcp_server, token_response, current_redirect_uri)
        if persistence_result == "reused":
            return dummy_return

    return JSONResponse(token_response)


@router.get("/{mcp_server_name}/authorize")
@router.get("/authorize")
async def authorize(
    request: Request,
    redirect_uri: str,
    client_id: Optional[str] = None,
    state: str = "",
    mcp_server_name: Optional[str] = None,
    code_challenge: Optional[str] = None,
    code_challenge_method: Optional[str] = None,
    response_type: Optional[str] = None,
    scope: Optional[str] = None,
):
    # Redirect to real OAuth provider with PKCE support
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
        global_mcp_server_manager,
    )

    if mcp_server_name is None and client_id and is_gateway_dcr_client_id(client_id) and is_mcp_gateway_dcr_enabled():
        return aggregate_authorize(
            request=request,
            client_id=client_id,
            redirect_uri=redirect_uri,
            state=state,
            code_challenge=code_challenge,
            code_challenge_method=code_challenge_method,
            response_type=response_type,
            session_user_id=_session_cookie_user_id(request),
        )

    lookup_name: Optional[str] = mcp_server_name or client_id
    client_ip = IPAddressUtils.get_mcp_client_ip(request)
    mcp_server = (
        global_mcp_server_manager.get_mcp_server_by_name(lookup_name, client_ip=client_ip) if lookup_name else None
    )
    if mcp_server is None and mcp_server_name is None:
        mcp_server = _resolve_oauth2_server_for_root_endpoints(client_ip=client_ip)
    if mcp_server is None:
        raise HTTPException(status_code=404, detail="MCP server not found")
    _raise_if_not_oauth2(mcp_server)
    # Use server's stored client_id when caller doesn't supply one.
    # Raise a clear error instead of passing an empty string — an empty
    # client_id would silently produce a broken authorization URL.
    resolved_client_id: str = mcp_server.client_id or client_id or ""
    if not resolved_client_id:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "client_id is required but was not supplied and is not "
                "stored on the MCP server record. Provide client_id as a query "
                "parameter or configure it on the server."
            },
        )
    return await authorize_with_server(
        request=request,
        mcp_server=mcp_server,
        client_id=resolved_client_id,
        redirect_uri=redirect_uri,
        state=state,
        code_challenge=code_challenge,
        code_challenge_method=code_challenge_method,
        response_type=response_type,
        scope=scope,
    )


@router.post("/{mcp_server_name}/token")
@router.post("/token")
async def token_endpoint(
    request: Request,
    grant_type: str = Form(...),
    code: str = Form(None),
    redirect_uri: str = Form(None),
    client_id: str = Form(...),
    client_secret: Optional[str] = Form(None),
    code_verifier: str = Form(None),
    refresh_token: Optional[str] = Form(None),
    scope: Optional[str] = Form(None),
    mcp_server_name: Optional[str] = None,
):
    """
    Accept the authorization code from client and exchange it for OAuth token.
    Supports PKCE flow by forwarding code_verifier to upstream provider.

    1. Call the token endpoint with PKCE parameters
    2. Store the user's token in the db - and generate a LiteLLM virtual key
    3. Return the token
    4. Return a virtual key in this response
    """
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
        global_mcp_server_manager,
    )

    if mcp_server_name is None and is_gateway_dcr_client_id(client_id) and is_mcp_gateway_dcr_enabled():
        from litellm.proxy.proxy_server import (  # noqa: PLC0415  # circular import at module load
            master_key,
            user_api_key_cache,
        )

        return await aggregate_token(
            request=request,
            grant_type=grant_type,
            code=code,
            redirect_uri=redirect_uri,
            client_id=client_id,
            code_verifier=code_verifier,
            refresh_token=refresh_token,
            master_key=master_key,
            reload_user=_reload_active_user_by_id,
            cache=user_api_key_cache,
        )

    lookup_name = mcp_server_name or client_id
    client_ip = IPAddressUtils.get_mcp_client_ip(request)
    mcp_server = global_mcp_server_manager.get_mcp_server_by_name(lookup_name, client_ip=client_ip)
    if mcp_server is None and mcp_server_name is None:
        mcp_server = _resolve_oauth2_server_for_root_endpoints(client_ip=client_ip)
    if mcp_server is None:
        raise HTTPException(status_code=404, detail="MCP server not found")
    return await exchange_token_with_server(
        request=request,
        mcp_server=mcp_server,
        grant_type=grant_type,
        code=code,
        redirect_uri=redirect_uri,
        client_id=client_id,
        client_secret=client_secret,
        code_verifier=code_verifier,
        refresh_token=refresh_token,
        scope=scope,
    )


@router.post("/authorize/complete")
async def authorize_complete(request: Request, flow: str = Form(...)):
    """Finish an aggregate connect flow (``mcp_gateway_dcr``): mint the gateway
    authorization code for the signed-in user and redirect back to the DCR client. POST
    plus the per-flow HttpOnly cookie set at /authorize; 404 when the flag is off so the
    route is byte-invisible to existing deployments."""
    if not is_mcp_gateway_dcr_enabled():
        raise HTTPException(status_code=404, detail="Not Found")
    return complete_connect_flow(
        request=request,
        flow_handle=flow,
        session_user_id=_session_cookie_user_id(request),
    )


# Per RFC 6749 §4.1.2.1, an IdP that rejects an OAuth authorization request
# redirects back to the configured redirect URI with ``error`` /
# ``error_description`` / ``error_uri`` query params and no ``code``. The MCP
# loopback flow funnels that response through this /callback endpoint, so
# the endpoint must accept either a successful (``code``+``state``) or an
# error response. Declaring ``code``/``state`` as required would cause
# FastAPI to reject the error response with a 422 before the handler runs,
# which strands the MCP client waiting on the loopback (see LIT-2750).


def _render_oauth_error_html(error: str, description: Optional[str]) -> HTMLResponse:
    """Render an actionable HTML page for an IdP-reported OAuth error.

    Used when we cannot propagate the error back to the registered
    ``redirect_uri`` (state missing or undecryptable). Returned with a 400
    status so the failure is observable to operators while still being a
    human-readable page for the end user.
    """
    safe_error = _html.escape(error or "unknown_error")
    safe_description = _html.escape(description) if description else ""
    description_html = f"<p>{safe_description}</p>" if safe_description else ""
    body = (
        "<html><body>"
        "<h2>Authentication failed</h2>"
        f"<p><strong>Error:</strong> {safe_error}</p>"
        f"{description_html}"
        "<p>You can close this window and try again.</p>"
        "</body></html>"
    )
    return HTMLResponse(body, status_code=400)


@router.get("/callback")
async def callback(
    request: Request,
    code: Optional[str] = None,
    state: Optional[str] = None,
    error: Optional[str] = None,
    error_description: Optional[str] = None,
    error_uri: Optional[str] = None,
):
    """OAuth 2.0 authorization response handler for MCP loopback clients.

    Accepts either:

    - A successful authorization response (``code`` + ``state``), which is
      forwarded back to the validated client ``redirect_uri`` with the
      original (un-wrapped) ``state``.
    - An error response (``error``[+``error_description``/``error_uri``]), per
      RFC 6749 §4.1.2.1. When ``state`` is present and decodes to a trusted
      ``redirect_uri``, the error params are propagated back to the client so
      its OAuth library can surface them. Otherwise we render an HTML error
      page so the user is not left on an opaque 422 / blank screen.
    """
    # 1. IdP-reported error path (e.g. ``?error=access_denied``).
    if error:
        verbose_logger.info(
            "MCP /callback received IdP error: error=%s, error_description=%s",
            error,
            error_description,
        )
        if state:
            encoded_state = _resolve_encoded_oauth_state(request, state)
            try:
                state_data = decode_state_hash(encoded_state)
                original_state = state_data.get("original_state")
                redirect_uri = _get_validated_client_redirect_uri(request, state_data)
            except Exception:
                # Untrusted/invalid client redirect_uri (HTTPException), or an
                # undecryptable state (expired key, tampered): surface the IdP
                # error inline rather than forwarding it to an attacker-controlled
                # URL, and drop the one-time cookie we can no longer consume.
                response = _render_oauth_error_html(error, error_description)
                _clear_oauth_state_cookie(response, request, state)
                return response

            params: Dict[str, str] = {"error": error}
            if error_description:
                params["error_description"] = error_description
            if error_uri:
                params["error_uri"] = error_uri
            if original_state is not None:
                params["state"] = original_state
            complete_returned_url = _append_query_params(redirect_uri, params)
            response = RedirectResponse(url=complete_returned_url, status_code=302)
            _clear_oauth_state_cookie(response, request, state)
            return response

        # No state — nothing to round-trip to. Show the user the error.
        return _render_oauth_error_html(error, error_description)

    # 2. Neither success nor error parameters present — most likely a stray
    #    GET / dropped SSO redirect chain. Surface a 400 instead of 422.
    if not code or not state:
        missing = [name for name, value in (("code", code), ("state", state)) if not value]
        return _render_oauth_error_html(
            "invalid_request",
            f"Missing authorization {' and '.join(repr(m) for m in missing)} parameter(s).",
        )

    # 3. Successful authorization response.
    try:
        encoded_state = _resolve_encoded_oauth_state(request, state)
        state_data = decode_state_hash(encoded_state)
        original_state = state_data["original_state"]

        # Re-validate the client redirect URI at the sink. /authorize
        # rejects untrusted URIs before encoding them into state, but
        # encrypted states minted before that check was added have no
        # expiry and remain valid indefinitely. Validating here blocks
        # the open-redirect + code-theft primitive even for pre-fix
        # states while permitting same-origin / allowlisted clients.
        redirect_uri = _get_validated_client_redirect_uri(request, state_data)

        # Interactive dcr_bridge oauth_delegate: the state carries the litellm user the authorize step
        # captured. Instead of forwarding the raw upstream code (which the client would present at the
        # token endpoint with no way to prove who signed in), seal the user and the upstream code into a
        # gateway authorization code and forward THAT. The token endpoint decrypts it to bind the
        # envelope to this user. Every other flow forwards the raw code unchanged.
        litellm_user_id = state_data.get("litellm_user_id")
        mcp_server_id = state_data.get("mcp_server_id")
        forwarded_code = code
        if isinstance(litellm_user_id, str) and litellm_user_id and isinstance(mcp_server_id, str) and mcp_server_id:
            forwarded_code = seal_bridge_authorization_code(
                upstream_code=code, litellm_user_id=litellm_user_id, mcp_server_id=mcp_server_id
            )

        params = {"code": forwarded_code, "state": original_state}
        complete_returned_url = _append_query_params(redirect_uri, params)
        response = RedirectResponse(url=complete_returned_url, status_code=302)
        _clear_oauth_state_cookie(response, request, state)
        return response

    except HTTPException:
        # Re-raise so a non-loopback base_url surfaces as 400 instead of
        # a generic "authentication incomplete" redirect.
        raise
    except Exception:
        response = HTMLResponse("<html><body>Authentication incomplete. You can close this window.</body></html>")
        _clear_oauth_state_cookie(response, request, state)
        return response


# ------------------------------
# Optional .well-known endpoints for MCP + OAuth discovery
# ------------------------------
"""
    Per SEP-985, the client MUST:
    1. Try resource_metadata from WWW-Authenticate header (if present)
    2. Fall back to path-based well-known URI: /.well-known/oauth-protected-resource/{path}
    (
    If the resource identifier value contains a path or query component, any terminating slash (/)
    following the host component MUST be removed before inserting /.well-known/ and the well-known
    URI path suffix between the host component and the path(include root path) and/or query components.
    https://datatracker.ietf.org/doc/html/rfc9728#section-3.1)
    3. Fall back to root-based well-known URI: /.well-known/oauth-protected-resource

    Dual Pattern Support:
    - Standard MCP pattern: /mcp/{server_name} (recommended, used by mcp-inspector, VSCode Copilot)
    - LiteLLM legacy pattern: /{server_name}/mcp (backward compatibility)

    The resource URL returned matches the pattern used in the discovery request.
"""


async def fetch_upstream_oauth_protected_resource(
    mcp_server: MCPServer,
) -> Optional[dict]:
    """Fetch the upstream MCP server's ``.well-known/oauth-protected-resource``
    metadata for a pass-through server.

    Tries host-only first, then falls back to the RFC 9728 §3.1 path-suffix
    form (e.g. ``https://host/.well-known/oauth-protected-resource/mcp``) to
    cover upstreams that scope metadata per resource path.

    Responses are cached in-process for ~5 minutes keyed on
    ``(server_id, resource_url)`` so we do not hammer the IdP.

    Returns the parsed JSON dict on success, or ``None`` if neither form
    responds with a 2xx JSON payload. Raises on network/connection errors so
    the caller can emit HTTP 502 rather than fabricate a gateway response.
    """
    if not mcp_server.url:
        return None

    upstream = urlparse(mcp_server.url)
    if not upstream.scheme or not upstream.netloc:
        return None

    cache_key = (mcp_server.server_id, mcp_server.url)
    now = time.time()
    _prune_oauth_metadata_cache(now)
    cached = _OAUTH_METADATA_CACHE.get(cache_key)
    if cached is not None and cached[0] > now:
        return cached[1]

    lock = _OAUTH_METADATA_FETCH_LOCKS.setdefault(cache_key, asyncio.Lock())
    async with lock:
        now = time.time()
        cached = _OAUTH_METADATA_CACHE.get(cache_key)
        if cached is not None and cached[0] > now:
            return cached[1]

        host_base = f"{upstream.scheme}://{upstream.netloc}"
        candidates = [f"{host_base}/.well-known/oauth-protected-resource"]
        # RFC 9728 §3.1 path fallback
        if upstream.path and upstream.path not in ("", "/"):
            candidates.append(f"{host_base}/.well-known/oauth-protected-resource{upstream.path.rstrip('/')}")

        async_client = get_async_httpx_client(llm_provider=httpxSpecialProvider.Oauth2Check)

        network_errors: list[Exception] = []
        for candidate in candidates:
            try:
                response = await async_client.get(
                    candidate,
                    headers={"Accept": "application/json"},
                )
            except Exception as exc:
                if is_network_error(exc):
                    network_errors.append(exc)
                else:
                    verbose_logger.warning(
                        "MCP OAuth metadata fetch for %s raised non-transport "
                        "%s: %s — treating as no metadata for this candidate",
                        candidate,
                        type(exc).__name__,
                        exc,
                    )
                continue
            if response.status_code == 200:
                try:
                    payload = response.json()
                except Exception as exc:
                    verbose_logger.warning(
                        "MCP OAuth metadata at %s returned 200 but JSON "
                        "decode failed (%s: %s) — treating as no metadata",
                        candidate,
                        type(exc).__name__,
                        exc,
                    )
                    continue
                if isinstance(payload, dict):
                    now = time.time()
                    _OAUTH_METADATA_CACHE[cache_key] = (
                        now + _OAUTH_METADATA_CACHE_TTL_SECONDS,
                        payload,
                    )
                    _prune_oauth_metadata_cache(now)
                    return payload

        if len(network_errors) == len(candidates):
            raise network_errors[-1]

        # Negative-result caching: when no candidate yielded a usable payload,
        # remember that for a shorter TTL so we don't re-fetch on every
        # subsequent discovery request (and so the per-key lock can be pruned).
        now = time.time()
        _OAUTH_METADATA_CACHE[cache_key] = (
            now + _OAUTH_METADATA_NEGATIVE_CACHE_TTL_SECONDS,
            None,
        )
        _prune_oauth_metadata_cache(now)
        return None


def is_network_error(exc: Exception) -> bool:
    """True for transport-layer failures (connection refused, DNS, TLS, timeout)
    as opposed to HTTP protocol errors (4xx/5xx with a valid response)."""
    return isinstance(exc, httpx.TransportError)


async def _build_oauth_protected_resource_response(
    request: Request,
    mcp_server_name: Optional[str],
    use_standard_pattern: bool,
) -> dict:
    """
    Build OAuth protected resource response with the appropriate URL pattern.

    For pass-through MCP servers, the gateway proxies the upstream's own
    ``oauth-protected-resource`` metadata so standards-compliant MCP clients
    discover the **upstream** IdP instead of the gateway. For ``true_passthrough``
    and ``oauth_delegate`` the metadata is returned verbatim (``resource`` stays
    the upstream): the caller's token is forwarded to and validated by the
    upstream, so its audience must be the upstream — rewriting it to the gateway
    would make a strict IdP (e.g. Entra) refuse to mint it or the upstream reject
    it. Only the legacy ``is_oauth_passthrough`` opt-in rewrites ``resource`` to
    the gateway's own URL so clients present the bearer token back to the gateway.

    Args:
        request: FastAPI Request object
        mcp_server_name: Name of the MCP server
        use_standard_pattern: If True, use /mcp/{server_name} pattern;
                             if False, use /{server_name}/mcp pattern

    Returns:
        OAuth protected resource metadata dict
    """
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
        global_mcp_server_manager,
    )

    request_base_url = get_request_base_url(request)
    client_ip = IPAddressUtils.get_mcp_client_ip(request)

    # When no server name provided, try to resolve the single OAuth2 server
    if mcp_server_name is None:
        resolved = _resolve_oauth2_server_for_root_endpoints(client_ip=client_ip)
        if resolved:
            mcp_server_name = resolved.server_name or resolved.name

    mcp_server: Optional[MCPServer] = None
    if mcp_server_name:
        mcp_server = global_mcp_server_manager.get_mcp_server_by_name(mcp_server_name, client_ip=client_ip)

    # Build resource URL based on the pattern
    if mcp_server_name:
        if use_standard_pattern:
            # Standard MCP pattern: /mcp/{server_name}
            resource_url = f"{request_base_url}/mcp/{mcp_server_name}"
        else:
            # LiteLLM legacy pattern: /{server_name}/mcp
            resource_url = f"{request_base_url}/{mcp_server_name}/mcp"
    else:
        resource_url = f"{request_base_url}/mcp"

    if mcp_server is not None and mcp_server_name and mcp_server.is_dcr_bridge:
        return {
            "authorization_servers": [f"{request_base_url}/{mcp_server_name}"],
            "resource": resource_url,
            "scopes_supported": (mcp_server.scopes if mcp_server.scopes else []),
        }

    # Pass-through branch: proxy the upstream's own metadata so discovery
    # directs the client at the real IdP (Okta, Keycloak, …) instead of us.
    if mcp_server is not None and (
        mcp_server.is_oauth_passthrough or mcp_server.is_oauth_delegate or mcp_server.is_true_passthrough
    ):
        try:
            upstream_metadata = await fetch_upstream_oauth_protected_resource(mcp_server)
        except Exception as exc:
            verbose_logger.warning(
                "Failed to fetch upstream oauth-protected-resource metadata "
                f"for pass-through MCP server {mcp_server.name!r}: {exc}"
            )
            raise HTTPException(
                status_code=502,
                detail=(
                    f"Failed to fetch upstream oauth-protected-resource metadata for MCP server {mcp_server.name!r}"
                ),
            )

        if upstream_metadata is not None:
            if mcp_server.is_true_passthrough or mcp_server.is_oauth_delegate:
                return upstream_metadata
            return {**upstream_metadata, "resource": resource_url}

        # Upstream responded but with non-200 or non-dict payload. For
        # pass-through servers the gateway is NOT the authorization server,
        # so we must not fall through to the default gateway metadata —
        # that would point clients at the wrong IdP.
        verbose_logger.warning(
            f"Upstream oauth-protected-resource metadata unavailable for pass-through MCP server {mcp_server.name!r}"
        )
        raise HTTPException(
            status_code=502,
            detail=(f"Upstream oauth-protected-resource metadata unavailable for MCP server {mcp_server.name!r}"),
        )

    obo_response = _obo_protected_resource_response(mcp_server, resource_url)
    if obo_response is not None:
        return obo_response

    # An OBO server with no configured issuer falls through to the gateway default so discovery still
    # returns metadata; every other non-oauth2 named server 404s to avoid enumeration.
    if mcp_server is None or mcp_server.auth_type != MCPAuth.oauth2_token_exchange:
        _raise_unless_oauth2_discovery_server(mcp_server, mcp_server_name, "not an OAuth-protected resource")

    return {
        "authorization_servers": [
            (f"{request_base_url}/{mcp_server_name}" if mcp_server_name else f"{request_base_url}")
        ],
        "resource": resource_url,
        "scopes_supported": (mcp_server.scopes if mcp_server and mcp_server.scopes else []),
    }


def _obo_protected_resource_response(mcp_server: Optional[MCPServer], resource_url: str) -> Optional[dict]:
    """The OBO (token_exchange) PRM, or None when this server is not OBO / no issuer is configured.

    The client SSOs with the IdP to obtain a subject token, which LiteLLM then exchanges, so discovery
    points at the JWT-auth issuer(s) LiteLLM trusts (the same IdP that issues and validates the
    subject), not the gateway. None falls the caller back to the gateway default so discovery still
    returns metadata; it just can't name the IdP.
    """
    if mcp_server is None or mcp_server.auth_type != MCPAuth.oauth2_token_exchange:
        return None
    issuers = _jwt_auth_issuers()
    if not issuers:
        return None
    return {
        "authorization_servers": issuers,
        "resource": resource_url,
        "scopes_supported": (mcp_server.scopes if mcp_server.scopes else []),
    }


def _jwt_auth_issuers() -> list:
    """The OAuth issuer identifier(s) LiteLLM's JWT auth trusts, for the OBO PRM authorization_servers.

    In token_exchange the IdP that issues the subject JWT is the same one LiteLLM validates it
    against, so OBO discovery points clients at the JWT-auth issuer to obtain a subject token.
    Sourced from ``JWT_ISSUER`` and any configured ``litellm_jwtauth.issuers``.
    """
    import os  # noqa: PLC0415

    from litellm.proxy.proxy_server import general_settings  # noqa: PLC0415

    issuers: list = []
    env_issuer = os.getenv("JWT_ISSUER")
    if env_issuer:
        issuers.append(env_issuer)

    jwtauth = general_settings.get("litellm_jwtauth") if isinstance(general_settings, dict) else None
    raw_issuers = jwtauth.get("issuers") if isinstance(jwtauth, dict) else getattr(jwtauth, "issuers", None)
    for cfg in raw_issuers or []:
        issuer = cfg.get("issuer") if isinstance(cfg, dict) else getattr(cfg, "issuer", None)
        if issuer and issuer not in issuers:
            issuers.append(issuer)
    return issuers


def _build_aggregate_protected_resource_response(request: Request) -> dict:
    """RFC 9728 metadata for the aggregate /mcp resource: the gateway itself is
    the authorization server. No per-server names or scopes leak here; access
    is resolved after sign-in from the authenticated user's grants.

    The advertised authorization server is ``{base}/mcp`` (not the bare
    origin) so RFC 8414 path-insertion resolves its metadata at
    ``/.well-known/oauth-authorization-server/mcp``, a route this module
    owns. The bare-origin well-known is registered first by the BYOK OAuth
    feature and describes the BYOK flow, so it must not be the aggregate
    discovery entry point (same pattern as the per-server documents, which
    advertise ``{base}/{server_name}``)."""
    request_base_url = get_request_base_url(request)
    return {
        "authorization_servers": [f"{request_base_url}/mcp"],
        "resource": f"{request_base_url}/mcp",
        "scopes_supported": [],
    }


def _build_aggregate_authorization_server_response(request: Request) -> dict:
    """RFC 8414 metadata for the gateway as the aggregate authorization server.

    The issuer is ``{base}/mcp`` and must stay equal to the value the
    aggregate protected-resource document advertises: spec clients verify the
    issuer in the metadata matches the one that derived the well-known URL.
    Advertises the root /authorize, /token, and /register endpoints and
    ``token_endpoint_auth_methods_supported: ["none", ...]`` because DCR
    clients (Claude Desktop, MCP Inspector) register as public clients; PKCE
    S256 is mandatory in the gateway's authorize flow."""
    request_base_url = get_request_base_url(request)
    return {
        "issuer": f"{request_base_url}/mcp",
        "authorization_endpoint": f"{request_base_url}/authorize",
        "token_endpoint": f"{request_base_url}/token",
        "registration_endpoint": f"{request_base_url}/register",
        "response_types_supported": ["code"],
        "scopes_supported": [],
        "grant_types_supported": ["authorization_code", "refresh_token"],
        "code_challenge_methods_supported": ["S256"],
        "token_endpoint_auth_methods_supported": ["none", "client_secret_post"],
    }


def _mcp_named_server_exists(request: Request) -> bool:
    """True when a server literally named ``mcp`` is configured and visible to this caller.

    Its per-server authorization-server document is served at
    ``/.well-known/oauth-authorization-server/mcp``, a single segment that collides with the
    aggregate path. When such a server exists the real server wins the route, so that
    deployment keeps its per-server discovery regardless of whether the aggregate front door
    is on."""
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (  # noqa: PLC0415  # circular import with mcp_server_manager at module load
        global_mcp_server_manager,
    )

    client_ip = IPAddressUtils.get_mcp_client_ip(request)
    return global_mcp_server_manager.get_mcp_server_by_name("mcp", client_ip=client_ip) is not None


# RFC 9728 path-appended discovery for the aggregate /mcp endpoint. A client
# pointed at {base}/mcp inserts the well-known segment before the resource
# path, so this exact route must exist for aggregate discovery to work at all.
# Declared before the parameterized well-known routes below: Starlette matches
# in registration order, and /.well-known/oauth-authorization-server/{name}
# would otherwise capture the "/mcp" suffix as a server name.
@router.get(
    f"/.well-known/oauth-protected-resource{'' if get_server_root_path() == '/' else get_server_root_path()}/mcp"
)
async def oauth_protected_resource_aggregate(request: Request):
    """
    OAuth protected resource discovery for the aggregate /mcp endpoint.

    The single-segment ``/mcp`` path does not collide with any per-server PRM pattern
    (those are two-segment: ``/mcp/{server}`` or ``/{server}/mcp``), so this unambiguously
    describes the aggregate resource.
    """
    return _build_aggregate_protected_resource_response(request)


@router.get(
    f"/.well-known/oauth-authorization-server{'' if get_server_root_path() == '/' else get_server_root_path()}/mcp"
)
async def oauth_authorization_server_aggregate(request: Request):
    """
    OAuth authorization server discovery for the aggregate /mcp endpoint, the RFC 8414
    path-inserted form for a client that treats {base}/mcp as its authorization base URL.

    This single-segment path collides with the parameterized ``/{mcp_server_name}`` route
    below, so a server literally named ``mcp`` wins it and keeps its per-server discovery;
    only when no such server exists is the aggregate document served.
    """
    if _mcp_named_server_exists(request):
        return _build_oauth_authorization_server_response(request=request, mcp_server_name="mcp")
    return _build_aggregate_authorization_server_response(request)


# Standard MCP pattern: /.well-known/oauth-protected-resource/mcp/{server_name}
# This is the pattern expected by standard MCP clients (mcp-inspector, VSCode Copilot)
@router.get(
    f"/.well-known/oauth-protected-resource{'' if get_server_root_path() == '/' else get_server_root_path()}/mcp/{{mcp_server_name}}"
)
async def oauth_protected_resource_mcp_standard(request: Request, mcp_server_name: str):
    """
    OAuth protected resource discovery endpoint using standard MCP URL pattern.

    Standard pattern: /mcp/{server_name}
    Discovery path: /.well-known/oauth-protected-resource/mcp/{server_name}

    This endpoint is compliant with MCP specification and works with standard
    MCP clients like mcp-inspector and VSCode Copilot.
    """
    return await _build_oauth_protected_resource_response(
        request=request,
        mcp_server_name=mcp_server_name,
        use_standard_pattern=True,
    )


# LiteLLM legacy pattern: /.well-known/oauth-protected-resource/{server_name}/mcp
# Kept for backward compatibility with existing deployments
@router.get(
    f"/.well-known/oauth-protected-resource{'' if get_server_root_path() == '/' else get_server_root_path()}/{{mcp_server_name}}/mcp"
)
@router.get("/.well-known/oauth-protected-resource")
async def oauth_protected_resource_mcp(request: Request, mcp_server_name: Optional[str] = None):
    """
    OAuth protected resource discovery endpoint using LiteLLM legacy URL pattern.

    Legacy pattern: /{server_name}/mcp
    Discovery path: /.well-known/oauth-protected-resource/{server_name}/mcp

    This endpoint is kept for backward compatibility. New integrations should
    use the standard MCP pattern (/mcp/{server_name}) instead.
    """
    return await _build_oauth_protected_resource_response(
        request=request,
        mcp_server_name=mcp_server_name,
        use_standard_pattern=False,
    )


def _build_oauth_authorization_server_response(
    request: Request,
    mcp_server_name: Optional[str],
) -> dict:
    """Build OAuth authorization server metadata response (gateway-as-AS shape).

    Synchronous because the body only does dict construction and synchronous
    registry lookups; unlike :func:`_build_oauth_protected_resource_response`
    it does not need to await any upstream IO.
    """
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
        global_mcp_server_manager,
    )

    request_base_url = get_request_base_url(request)
    client_ip = IPAddressUtils.get_mcp_client_ip(request)

    # When no server name provided, try to resolve the single OAuth2 server
    if mcp_server_name is None:
        resolved = _resolve_oauth2_server_for_root_endpoints(client_ip=client_ip)
        if resolved:
            mcp_server_name = resolved.server_name or resolved.name

    authorization_endpoint = (
        f"{request_base_url}/{mcp_server_name}/authorize" if mcp_server_name else f"{request_base_url}/authorize"
    )
    token_endpoint = f"{request_base_url}/{mcp_server_name}/token" if mcp_server_name else f"{request_base_url}/token"

    mcp_server: Optional[MCPServer] = None
    if mcp_server_name:
        mcp_server = global_mcp_server_manager.get_mcp_server_by_name(mcp_server_name, client_ip=client_ip)

    _raise_unless_oauth2_discovery_server(mcp_server, mcp_server_name, "not an OAuth authorization server")

    return {
        "issuer": request_base_url,  # point to your proxy
        "authorization_endpoint": authorization_endpoint,
        "token_endpoint": token_endpoint,
        "response_types_supported": ["code"],
        "scopes_supported": (mcp_server.scopes if mcp_server and mcp_server.scopes else []),
        "grant_types_supported": ["authorization_code", "refresh_token"],
        "code_challenge_methods_supported": ["S256"],
        "token_endpoint_auth_methods_supported": ["client_secret_post"],
        # Claude expects a registration endpoint, even if we just fake it
        "registration_endpoint": (
            f"{request_base_url}/{mcp_server_name}/register" if mcp_server_name else f"{request_base_url}/register"
        ),
    }


# Standard MCP pattern: /.well-known/oauth-authorization-server/mcp/{server_name}
@router.get(
    f"/.well-known/oauth-authorization-server{'' if get_server_root_path() == '/' else get_server_root_path()}/mcp/{{mcp_server_name}}"
)
async def oauth_authorization_server_mcp_standard(request: Request, mcp_server_name: str):
    """
    OAuth authorization server discovery endpoint using standard MCP URL pattern.

    Standard pattern: /mcp/{server_name}
    Discovery path: /.well-known/oauth-authorization-server/mcp/{server_name}
    """
    return _build_oauth_authorization_server_response(
        request=request,
        mcp_server_name=mcp_server_name,
    )


# LiteLLM legacy pattern and root endpoint
@router.get(
    f"/.well-known/oauth-authorization-server{'' if get_server_root_path() == '/' else get_server_root_path()}/{{mcp_server_name}}"
)
@router.get("/.well-known/oauth-authorization-server")
async def oauth_authorization_server_mcp(request: Request, mcp_server_name: Optional[str] = None):
    """
    OAuth authorization server discovery endpoint.

    Supports both legacy pattern (/{server_name}) and root endpoint.
    """
    return _build_oauth_authorization_server_response(
        request=request,
        mcp_server_name=mcp_server_name,
    )


# Alias for standard OpenID discovery
@router.get("/.well-known/openid-configuration")
async def openid_configuration(request: Request):
    response = await oauth_authorization_server_mcp(request)

    # If MCPJWTSigner is active, augment the discovery doc with JWKS fields so
    # MCP servers and gateways (e.g. AWS Bedrock AgentCore Gateway) can resolve
    # the signing keys and verify liteLLM-issued tokens.
    try:
        from litellm.proxy.guardrails.guardrail_hooks.mcp_jwt_signer.mcp_jwt_signer import (
            get_mcp_jwt_signer,
        )

        signer = get_mcp_jwt_signer()
        if signer is not None:
            request_base_url = get_request_base_url(request)
            if isinstance(response, dict):
                response = {
                    **response,
                    "jwks_uri": f"{request_base_url}/.well-known/jwks.json",
                    "id_token_signing_alg_values_supported": ["RS256"],
                }
    except ImportError:
        pass

    return response


@router.get("/.well-known/jwks.json")
async def jwks_json(request: Request):
    """
    JSON Web Key Set endpoint.

    Returns the RSA public key used by MCPJWTSigner to sign outbound MCP tokens.
    MCP servers and gateways use this endpoint to verify liteLLM-issued JWTs.

    Returns an empty key set if MCPJWTSigner is not configured.
    """
    try:
        from litellm.proxy.guardrails.guardrail_hooks.mcp_jwt_signer.mcp_jwt_signer import (
            get_mcp_jwt_signer,
        )

        signer = get_mcp_jwt_signer()
        if signer is not None:
            return JSONResponse(
                content=signer.get_jwks(),
                headers={"Cache-Control": f"public, max-age={signer.jwks_max_age}"},
            )
    except ImportError:
        pass

    # No signer active — return empty key set; short cache so activation is picked up quickly.
    return JSONResponse(
        content={"keys": []},
        headers={"Cache-Control": "public, max-age=60"},
    )


# Additional legacy pattern support
@router.get("/.well-known/oauth-authorization-server/{mcp_server_name}/mcp")
async def oauth_authorization_server_legacy(request: Request, mcp_server_name: str):
    """
    OAuth authorization server discovery for legacy /{server_name}/mcp pattern.
    """
    return _build_oauth_authorization_server_response(
        request=request,
        mcp_server_name=mcp_server_name,
    )


@router.post("/{mcp_server_name}/register")
@router.post("/register")
async def register_client(request: Request, mcp_server_name: Optional[str] = None):
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
        global_mcp_server_manager,
    )

    # Get the correct base URL considering X-Forwarded-* headers
    request_base_url = get_request_base_url(request)

    request_data = await _read_request_body(request=request)
    data: dict = {**request_data}

    dummy_return = {
        "client_id": mcp_server_name or "dummy_client",
        "client_secret": "dummy",
        "redirect_uris": [f"{request_base_url}/callback"],
    }
    client_ip = IPAddressUtils.get_mcp_client_ip(request)
    if not mcp_server_name:
        if is_mcp_gateway_dcr_enabled():
            return await register_aggregate_client(request=request, request_body=data)
        resolved = _resolve_oauth2_server_for_root_endpoints(client_ip=client_ip)
        if resolved:
            return await register_client_with_server(
                request=request,
                mcp_server=resolved,
                client_name=data.get("client_name", ""),
                grant_types=data.get("grant_types", []),
                response_types=data.get("response_types", []),
                token_endpoint_auth_method=data.get("token_endpoint_auth_method", ""),
                fallback_client_id=resolved.server_name or resolved.name,
                client_redirect_uris=data.get("redirect_uris"),
            )
        return dummy_return

    mcp_server = global_mcp_server_manager.get_mcp_server_by_name(mcp_server_name, client_ip=client_ip)
    if mcp_server is None:
        return dummy_return
    return await register_client_with_server(
        request=request,
        mcp_server=mcp_server,
        client_name=data.get("client_name", ""),
        grant_types=data.get("grant_types", []),
        response_types=data.get("response_types", []),
        token_endpoint_auth_method=data.get("token_endpoint_auth_method", ""),
        fallback_client_id=mcp_server_name,
        client_redirect_uris=data.get("redirect_uris"),
    )
