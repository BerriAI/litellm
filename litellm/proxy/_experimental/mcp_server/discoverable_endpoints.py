import json
from typing import Any, Dict, Optional
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from litellm._logging import verbose_logger
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
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
from litellm.types.mcp import MCPAuth
from litellm.types.mcp_server.mcp_server_manager import MCPServer

router = APIRouter(
    tags=["mcp"],
)


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


def _get_validated_client_redirect_uri(
    request: Request, state_data: Dict[str, Any]
) -> str:
    """Return a trusted (same-origin or loopback) client redirect URI from OAuth state."""
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
    (``"org_id": "12345"``).
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
                    "message": (
                        f"OAuth token rejected: required field '{key}' is absent"
                    ),
                },
            )
        if str(actual) != str(expected):
            raise HTTPException(
                status_code=403,
                detail={
                    "error": "token_validation_failed",
                    "server_id": server_id,
                    "field": key,
                    "message": (
                        f"OAuth token rejected: '{key}' = '{actual}', "
                        f"expected '{expected}'"
                    ),
                },
            )


async def _extract_user_id_from_request(request: Request) -> Optional[str]:
    """Best-effort extraction of LiteLLM user_id from the request's Authorization header.

    Called at the OAuth token endpoint so that per-user tokens can be stored
    server-side.  Uses a read-only cache lookup to avoid re-running the full
    auth pipeline (which has side effects such as rate-limit increments and
    spend logging).  Returns ``None`` if no cached credential is found.
    """
    auth_header = request.headers.get("Authorization") or request.headers.get(
        "authorization"
    )
    if not auth_header:
        return None
    lower = auth_header.lower()
    if not lower.startswith("bearer "):
        return None
    token = auth_header[7:].strip()
    try:
        from litellm.proxy._types import hash_token  # noqa: PLC0415
        from litellm.proxy.proxy_server import user_api_key_cache  # noqa: PLC0415

        cached = await user_api_key_cache.async_get_cache(hash_token(token))
        return getattr(cached, "user_id", None)
    except Exception:
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
        expires_in: Optional[int] = (
            int(raw_expires) if raw_expires is not None else None
        )
    except (TypeError, ValueError):
        expires_in = None

    refresh_token: Optional[str] = token_response.get("refresh_token") or None
    raw_scope = token_response.get("scope")
    scopes: Optional[list] = (
        raw_scope.split() if isinstance(raw_scope, str) and raw_scope else None
    )

    try:
        prisma_client = get_prisma_client_or_throw(
            "Database not connected. Cannot store per-user OAuth token."
        )
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

    # Warm the Redis cache so the first subsequent MCP call is a cache hit
    ttl = _compute_per_user_token_ttl(server, expires_in)
    await mcp_per_user_token_cache.set(
        user_id=user_id,
        server_id=server.server_id,
        access_token=access_token,
        ttl=ttl,
    )


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
    if mcp_server.auth_type != "oauth2":
        raise HTTPException(status_code=400, detail="MCP server is not OAuth2")
    if mcp_server.authorization_url is None:
        raise HTTPException(
            status_code=400, detail="MCP server authorization url is not set"
        )

    # Loopback OR same-origin redirect_uri. The URI is encrypted into the
    # OAuth state and decoded on /callback to redirect the user back;
    # restricting to trusted origins blocks the open-redirect +
    # code-theft primitive (VERIA-57 root cause B). Loopback supports
    # native MCP clients; same-origin supports the proxy's own UI callback.
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

    params = {
        "client_id": mcp_server.client_id if mcp_server.client_id else client_id,
        "redirect_uri": f"{request_base_url}/callback",
        "state": encoded_state,
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
    return RedirectResponse(final_url)


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
    if grant_type not in ("authorization_code", "refresh_token"):
        raise HTTPException(status_code=400, detail="Unsupported grant_type")

    if mcp_server.token_url is None:
        raise HTTPException(status_code=400, detail="MCP server token url is not set")

    resolved_client_id = mcp_server.client_id if mcp_server.client_id else client_id
    resolved_client_secret = (
        mcp_server.client_secret if mcp_server.client_secret else client_secret
    )

    if grant_type == "refresh_token":
        if not refresh_token:
            raise HTTPException(
                status_code=400,
                detail="refresh_token is required for refresh_token grant",
            )
        token_data: dict = {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": resolved_client_id,
        }
        if resolved_client_secret is not None:
            token_data["client_secret"] = resolved_client_secret
        if scope:
            token_data["scope"] = scope
    else:
        if not code:
            raise HTTPException(
                status_code=400,
                detail="code is required for authorization_code grant",
            )
        proxy_base_url = get_request_base_url(request)
        token_data = {
            "grant_type": "authorization_code",
            "client_id": resolved_client_id,
            "code": code,
            "redirect_uri": f"{proxy_base_url}/callback",
        }
        if resolved_client_secret is not None:
            token_data["client_secret"] = resolved_client_secret
        if code_verifier:
            token_data["code_verifier"] = code_verifier

    async_client = get_async_httpx_client(llm_provider=httpxSpecialProvider.Oauth2Check)
    response = await async_client.post(
        mcp_server.token_url,
        headers={"Accept": "application/json"},
        data=token_data,
    )

    response.raise_for_status()
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
                    "exchange_token_with_server: server-side storage failed "
                    "for user=%s server=%s: %s",
                    user_id,
                    mcp_server.server_id,
                    exc,
                )
        else:
            verbose_logger.debug(
                "exchange_token_with_server: no LiteLLM user_id found in request; "
                "per-user token for server=%s will not be stored server-side. "
                "The client should call POST /mcp/server/{id}/oauth-user-credential "
                "to store it manually.",
                mcp_server.server_id,
            )

    result = {
        "access_token": access_token,
        "token_type": token_response.get("token_type", "Bearer"),
        "expires_in": token_response.get("expires_in", 3600),
    }

    if "refresh_token" in token_response and token_response["refresh_token"]:
        result["refresh_token"] = token_response["refresh_token"]
    if "scope" in token_response and token_response["scope"]:
        result["scope"] = token_response["scope"]

    # RFC 6749 §5.1: token responses must not be cached.
    return JSONResponse(result, headers=TOKEN_NO_CACHE_HEADERS)


async def register_client_with_server(
    request: Request,
    mcp_server: MCPServer,
    client_name: str,
    grant_types: Optional[list],
    response_types: Optional[list],
    token_endpoint_auth_method: Optional[str],
    fallback_client_id: Optional[str] = None,
):
    request_base_url = get_request_base_url(request)
    dummy_return = {
        "client_id": fallback_client_id or mcp_server.server_name,
        "client_secret": "dummy",
        "redirect_uris": [f"{request_base_url}/callback"],
    }

    if mcp_server.client_id and mcp_server.client_secret:
        return dummy_return

    if mcp_server.authorization_url is None:
        raise HTTPException(
            status_code=400, detail="MCP server authorization url is not set"
        )

    if mcp_server.registration_url is None:
        return dummy_return

    register_data = {
        "client_name": client_name,
        "redirect_uris": [f"{request_base_url}/callback"],
        "grant_types": grant_types or [],
        "response_types": response_types or [],
        "token_endpoint_auth_method": token_endpoint_auth_method or "",
    }
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    async_client = get_async_httpx_client(
        llm_provider=httpxSpecialProvider.Oauth2Register
    )
    response = await async_client.post(
        mcp_server.registration_url,
        headers=headers,
        json=register_data,
    )
    response.raise_for_status()

    token_response = response.json()

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
        global_mcp_server_manager.get_mcp_server_by_name(
            lookup_name, client_ip=client_ip
        )
        if lookup_name
        else None
    )
    if mcp_server is None and mcp_server_name is None:
        mcp_server = _resolve_oauth2_server_for_root_endpoints(client_ip=client_ip)
    if mcp_server is None:
        raise HTTPException(status_code=404, detail="MCP server not found")
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
    mcp_server = global_mcp_server_manager.get_mcp_server_by_name(
        lookup_name, client_ip=client_ip
    )
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


@router.get("/callback")
async def callback(request: Request, code: str, state: str):
    try:
        state_data = decode_state_hash(state)
        original_state = state_data["original_state"]

        # Re-validate at the sink. /authorize rejects untrusted
        # redirect_uri before encoding into state, but encrypted states
        # minted before that check was added have no expiry and remain
        # valid indefinitely. Validating here (same-origin OR loopback)
        # blocks the open-redirect + code-theft primitive even for pre-fix
        # states while allowing the UI's same-origin callback to work.
        redirect_uri = _get_validated_client_redirect_uri(request, state_data)

        params = {"code": code, "state": original_state}
        complete_returned_url = _append_query_params(redirect_uri, params)
        return RedirectResponse(url=complete_returned_url, status_code=302)

    except HTTPException:
        # Re-raise so a non-loopback base_url surfaces as 400 instead of
        # a generic "authentication incomplete" redirect.
        raise
    except Exception:
        return HTMLResponse(
            "<html><body>Authentication incomplete. You can close this window.</body></html>"
        )


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


def _build_oauth_protected_resource_response(
    request: Request,
    mcp_server_name: Optional[str],
    use_standard_pattern: bool,
) -> dict:
    """
    Build OAuth protected resource response with the appropriate URL pattern.

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
        mcp_server = global_mcp_server_manager.get_mcp_server_by_name(
            mcp_server_name, client_ip=client_ip
        )

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

    return {
        "authorization_servers": [
            (
                f"{request_base_url}/{mcp_server_name}"
                if mcp_server_name
                else f"{request_base_url}"
            )
        ],
        "resource": resource_url,
        "scopes_supported": (
            mcp_server.scopes if mcp_server and mcp_server.scopes else []
        ),
    }


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
    return _build_oauth_protected_resource_response(
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
async def oauth_protected_resource_mcp(
    request: Request, mcp_server_name: Optional[str] = None
):
    """
    OAuth protected resource discovery endpoint using LiteLLM legacy URL pattern.

    Legacy pattern: /{server_name}/mcp
    Discovery path: /.well-known/oauth-protected-resource/{server_name}/mcp

    This endpoint is kept for backward compatibility. New integrations should
    use the standard MCP pattern (/mcp/{server_name}) instead.
    """
    return _build_oauth_protected_resource_response(
        request=request,
        mcp_server_name=mcp_server_name,
        use_standard_pattern=False,
    )


"""
    https://datatracker.ietf.org/doc/html/rfc8414#section-3.1
    RFC 8414: Path-aware OAuth discovery
    If the issuer identifier value contains a path component, any
    terminating "/" MUST be removed before inserting "/.well-known/" and
    the well-known URI suffix between the host component and the path(include root path)
    component.
"""


def _build_oauth_authorization_server_response(
    request: Request,
    mcp_server_name: Optional[str],
) -> dict:
    """
    Build OAuth authorization server metadata response.

    Args:
        request: FastAPI Request object
        mcp_server_name: Name of the MCP server

    Returns:
        OAuth authorization server metadata dict
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
        f"{request_base_url}/{mcp_server_name}/authorize"
        if mcp_server_name
        else f"{request_base_url}/authorize"
    )
    token_endpoint = (
        f"{request_base_url}/{mcp_server_name}/token"
        if mcp_server_name
        else f"{request_base_url}/token"
    )

    mcp_server: Optional[MCPServer] = None
    if mcp_server_name:
        mcp_server = global_mcp_server_manager.get_mcp_server_by_name(
            mcp_server_name, client_ip=client_ip
        )

    return {
        "issuer": request_base_url,  # point to your proxy
        "authorization_endpoint": authorization_endpoint,
        "token_endpoint": token_endpoint,
        "response_types_supported": ["code"],
        "scopes_supported": (
            mcp_server.scopes if mcp_server and mcp_server.scopes else []
        ),
        "grant_types_supported": ["authorization_code", "refresh_token"],
        "code_challenge_methods_supported": ["S256"],
        "token_endpoint_auth_methods_supported": ["client_secret_post"],
        # Claude expects a registration endpoint, even if we just fake it
        "registration_endpoint": (
            f"{request_base_url}/{mcp_server_name}/register"
            if mcp_server_name
            else f"{request_base_url}/register"
        ),
    }


# Standard MCP pattern: /.well-known/oauth-authorization-server/mcp/{server_name}
@router.get(
    f"/.well-known/oauth-authorization-server{'' if get_server_root_path() == '/' else get_server_root_path()}/mcp/{{mcp_server_name}}"
)
async def oauth_authorization_server_mcp_standard(
    request: Request, mcp_server_name: str
):
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
async def oauth_authorization_server_mcp(
    request: Request, mcp_server_name: Optional[str] = None
):
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
            )
        return dummy_return

    mcp_server = global_mcp_server_manager.get_mcp_server_by_name(
        mcp_server_name, client_ip=client_ip
    )
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
    )
