import asyncio
import html as _html
import json
import secrets
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional, Tuple
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import httpx
from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from pydantic import BaseModel, ValidationError

from litellm._logging import verbose_logger
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.proxy._experimental.mcp_server.auth.token_endpoint_auth import (
    TokenEndpointAuthConfigError,
    build_token_endpoint_client_auth,
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
) -> str:
    """
    Encode the base_url, original state, and PKCE parameters using encryption.

    Args:
        base_url: The base URL to encode
        original_state: The original state parameter
        code_challenge: PKCE code challenge from client
        code_challenge_method: PKCE code challenge method from client
        client_redirect_uri: Original redirect_uri from client

    Returns:
        An encrypted string that encodes all values
    """
    state_data = {
        "base_url": base_url,
        "original_state": original_state,
        "code_challenge": code_challenge,
        "code_challenge_method": code_challenge_method,
        "client_redirect_uri": client_redirect_uri,
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


def _active_key_user_id(key_obj: "UserAPIKeyAuth") -> Optional[str]:
    """The key's ``user_id``, or ``None`` if the key is blocked or expired.

    The OAuth token endpoint is unauthenticated, so the presented key is validated here before its
    identity is trusted to key a stored credential; a revoked or expired key must not be able to
    write or overwrite the per-user OAuth token. ``get_key_object`` resolves a row without these
    checks (the main ``user_api_key_auth`` pipeline enforces them downstream, which this endpoint
    bypasses), so they are applied here. Deleted keys are already rejected upstream, where
    ``get_key_object`` raises on a row that no longer exists.
    """
    if key_obj.blocked is True:
        return None
    expires = key_obj.expires
    if expires is not None:
        expiry = expires if isinstance(expires, datetime) else datetime.fromisoformat(expires)
        if expiry.tzinfo is None or expiry.tzinfo.utcoffset(expiry) is None:
            expiry = expiry.replace(tzinfo=timezone.utc)
        if expiry < datetime.now(timezone.utc):
            return None
    return key_obj.user_id


async def _extract_user_id_from_request(request: Request) -> Optional[str]:
    """Resolve the LiteLLM ``user_id`` at the OAuth token endpoint so a per-user token is stored
    under the same identity the egress later reads it by (``user_api_key_auth.user_id``).

    Resolves authoritatively via ``get_key_object`` (cache first, then DB) instead of a raw cache
    peek. On a multi-replica gateway the token-exchange request can land on a worker whose in-memory
    cache never saw the key, and a cross-replica Redis hit deserializes to a plain ``dict`` rather
    than a ``UserAPIKeyAuth``; the previous code read only ``Authorization`` and did
    ``getattr(cached, "user_id")`` with no ``model_type`` rehydration and no DB fallback, so it
    silently returned ``None`` and the token was never persisted, which makes the egress 401 on every
    reconnect. The resolved key is validated (``_active_key_user_id``) before its identity is trusted,
    so a blocked or expired key cannot write. Returns ``None`` when no key is present, the key cannot
    be resolved, or it is blocked/expired.
    """
    token = _litellm_key_from_request(request)
    if not token:
        return None
    try:
        from litellm.proxy._types import hash_token  # noqa: PLC0415
        from litellm.proxy.auth.auth_checks import get_key_object  # noqa: PLC0415
        from litellm.proxy.proxy_server import (  # noqa: PLC0415
            prisma_client,
            user_api_key_cache,
        )

        key_obj = await get_key_object(
            hashed_token=hash_token(token),
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
        )
        return _active_key_user_id(key_obj)
    except Exception as exc:
        verbose_logger.debug(
            "_extract_user_id_from_request: could not resolve a LiteLLM user_id for the presented "
            "key (%s); per-user token will not be stored server-side.",
            type(exc).__name__,
        )
        return None


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
    encoded_state = encode_state_with_base_url(
        base_url=base_url,
        original_state=state,
        code_challenge=code_challenge,
        code_challenge_method=code_challenge_method,
        client_redirect_uri=redirect_uri,
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

    if grant_type == "refresh_token":
        if not refresh_token:
            raise HTTPException(
                status_code=400,
                detail="refresh_token is required for refresh_token grant",
            )
        token_data: dict = {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            **client_auth.body,
        }
        if scope:
            token_data["scope"] = scope
    else:
        if not code:
            raise HTTPException(
                status_code=400,
                detail="code is required for authorization_code grant",
            )
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

    async_client = get_async_httpx_client(llm_provider=httpxSpecialProvider.Oauth2Check)
    response = await async_client.post(
        mcp_server.token_url,
        headers={"Accept": "application/json", **client_auth.headers},
        data=token_data,
    )
    if response is None:
        raise HTTPException(
            status_code=502,
            detail="MCP upstream token endpoint returned no response",
        )

    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        if "invalid_target" in exc.response.text:
            verbose_logger.warning(
                "MCP server %s: the upstream authorization server rejected the token request with "
                "invalid_target; it may require RFC 8707 resource indicators, which the gateway "
                "does not send yet (tracked as LIT-4339)",
                mcp_server.server_id,
            )
        raise
    token_response = response.json()
    access_token = token_response["access_token"]

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

    result = {
        "access_token": access_token,
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


_MAX_UPSTREAM_ERROR_CHARS = 500


def _safe_upstream_error_detail(response: httpx.Response) -> str:
    """Bounded plaintext summary of an upstream registration failure for the client.

    RFC 7591 error bodies are small JSON objects (``error`` / ``error_description``); relaying the
    text lets the client read the real reason instead of a bare 500, and the length bound keeps a
    hostile or oversized upstream body from bloating the gateway response."""
    body = response.text
    if not body:
        return response.reason_phrase or "upstream registration failed"
    return body[:_MAX_UPSTREAM_ERROR_CHARS]


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
    response = await async_client.post(
        mcp_server.registration_url,
        headers=headers,
        json=register_data,
    )
    if response is None:
        raise HTTPException(
            status_code=502,
            detail="MCP upstream registration endpoint returned no response",
        )
    if bridge_relay and response.status_code >= 400:
        raise HTTPException(status_code=response.status_code, detail=_safe_upstream_error_detail(response))
    response.raise_for_status()

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

        params = {"code": code, "state": original_state}
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
