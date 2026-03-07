"""
OAuth2 authorization code flow endpoints for OpenAPI-backed MCP servers.

These endpoints let users authorize LiteLLM to call an OAuth2-protected API
(e.g. GitHub, Spotify) on their behalf, rather than entering a static key.

Endpoints:
  GET  /v1/mcp/server/{server_id}/oauth2/connect   — initiate the OAuth2 flow
  GET  /v1/mcp/oauth2/callback                     — receive the code from the provider
  GET  /v1/mcp/server/{server_id}/oauth2/status    — check if user is connected
"""

import html as _html_module
import json
import secrets
import time
from typing import Dict, Optional
from urllib.parse import parse_qs, urlencode

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response

from litellm._logging import verbose_proxy_logger
from litellm.proxy._experimental.mcp_server.db import (
    has_user_credential,
    store_user_credential,
)
from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
    get_request_base_url,
)
from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
    global_mcp_server_manager,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

# ---------------------------------------------------------------------------
# In-memory state store for pending OAuth2 flows.
# Each entry: {state: {server_id, user_id, timestamp, expires_at}}
#
# NOTE: This is an in-process dict. In horizontally-scaled deployments
# (multiple uvicorn workers, Kubernetes pods, etc.) the /connect request may
# be handled by one instance while the provider's redirect hits /callback on
# a different instance — that second instance won't find the state and the
# flow will fail.  A follow-up should persist state in a shared store
# (e.g. LiteLLM_MCPUserCredentials with a sentinel user_id, or Redis cache).
# ---------------------------------------------------------------------------
_pending_oauth2_states: Dict[str, dict] = {}

_STATE_TTL_SECONDS = 600  # 10 minutes
_STATES_MAX_SIZE = 1000

router = APIRouter(tags=["mcp"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_callback_base_url(request: "Request") -> str:
    """Return the base URL to use for the OAuth2 redirect_uri.

    Prefers a statically-configured base URL (via the LITELLM_PROXY_BASE_URL
    environment variable) over the request-derived URL to avoid trusting
    potentially-spoofable X-Forwarded-Host headers in the OAuth2 security
    context.  Falls back to the request-derived URL when the env var is unset.
    """
    import os

    static_base = os.environ.get("LITELLM_PROXY_BASE_URL", "").rstrip("/")
    if static_base:
        return static_base
    return get_request_base_url(request)


def _purge_expired_states() -> None:
    now = time.time()
    expired = [k for k, v in _pending_oauth2_states.items() if v["expires_at"] < now]
    for k in expired:
        del _pending_oauth2_states[k]


def _make_state_token() -> str:
    """Return a cryptographically random opaque state token.

    The token is a dict key in `_pending_oauth2_states` — it is never used to
    carry or verify signed data, so a 32-byte (256-bit) random value is
    sufficient and makes the intent explicit.
    """
    return secrets.token_urlsafe(32)


def _build_success_html(server_name: str) -> str:
    e = _html_module.escape
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Connected &mdash; LiteLLM</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         background: #0f172a; min-height: 100vh; display: flex;
         align-items: center; justify-content: center; padding: 24px; }}
  .card {{ background: #fff; border-radius: 16px; padding: 36px 32px;
           width: 440px; max-width: 100%; box-shadow: 0 25px 60px rgba(0,0,0,.35);
           text-align: center; }}
  .check {{ font-size: 48px; margin-bottom: 16px; }}
  h2 {{ color: #16a34a; font-size: 20px; margin-bottom: 12px; }}
  p {{ color: #475569; font-size: 14px; line-height: 1.6; }}
</style>
<script>
  // Auto-close popup if opened as a popup window
  if (window.opener) {{ setTimeout(function() {{ window.close(); }}, 2000); }}
</script>
</head>
<body>
<div class="card">
  <div class="check">&#10003;</div>
  <h2>Connected to {e(server_name)}</h2>
  <p>Authorization successful. You can close this window.</p>
</div>
</body>
</html>"""


def _build_error_html(title: str, message: str) -> str:
    e = _html_module.escape
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{e(title)} &mdash; LiteLLM</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         background: #0f172a; min-height: 100vh; display: flex;
         align-items: center; justify-content: center; padding: 24px; }}
  .card {{ background: #fff; border-radius: 16px; padding: 36px 32px;
           width: 440px; max-width: 100%; box-shadow: 0 25px 60px rgba(0,0,0,.35); }}
  h2 {{ color: #dc2626; font-size: 20px; margin-bottom: 12px; }}
  p {{ color: #475569; font-size: 14px; line-height: 1.6; }}
</style>
<script>
  // Auto-close error popup so the polling loop stops and the spinner clears
  if (window.opener) {{ setTimeout(function() {{ window.close(); }}, 4000); }}
</script>
</head>
<body>
<div class="card">
  <h2>{e(title)}</h2>
  <p>{e(message)}</p>
</div>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/v1/mcp/server/{server_id}/oauth2/connect", include_in_schema=False)
async def openapi_oauth2_connect(
    server_id: str,
    request: Request,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> JSONResponse:
    from litellm.proxy.proxy_server import master_key

    server = global_mcp_server_manager.get_mcp_server_by_id(server_id)
    if server is None:
        raise HTTPException(status_code=404, detail=f"MCP server '{server_id}' not found")

    if not server.authorization_url:
        raise HTTPException(
            status_code=400,
            detail=f"Server '{server_id}' has no authorization_url configured",
        )
    if not server.token_url:
        raise HTTPException(
            status_code=400,
            detail=f"Server '{server_id}' has no token_url configured",
        )
    if not server.client_id:
        raise HTTPException(
            status_code=400,
            detail=f"Server '{server_id}' has no client_id configured",
        )
    if not server.client_secret:
        raise HTTPException(
            status_code=400,
            detail=f"Server '{server_id}' has no client_secret configured",
        )
    if master_key is None:
        raise HTTPException(status_code=500, detail="Master key not configured")

    # Fail early if the DB is unavailable: without it the callback cannot store
    # the token, so sending the user through the provider consent flow is wasted effort.
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=503,
            detail="Database is not configured. Cannot initiate OAuth2 flow.",
        )

    user_id = user_api_key_dict.user_id or user_api_key_dict.api_key or ""
    if not user_id:
        raise HTTPException(status_code=400, detail="Cannot determine user identity from token")

    _purge_expired_states()
    if len(_pending_oauth2_states) >= _STATES_MAX_SIZE:
        raise HTTPException(status_code=503, detail="Too many pending OAuth2 flows")

    timestamp = time.time()
    state = _make_state_token()
    base_url = _get_callback_base_url(request)
    callback_url = f"{base_url}/v1/mcp/oauth2/callback"

    # Store callback_url alongside the state so the /callback handler reuses
    # the exact same value when constructing the token-exchange redirect_uri.
    # Re-deriving it from the /callback request headers can produce a different
    # string (e.g. different proto/host due to reverse-proxy routing), causing
    # redirect_uri_mismatch errors at the provider.
    _pending_oauth2_states[state] = {
        "server_id": server_id,
        "user_id": user_id,
        "timestamp": timestamp,
        "expires_at": timestamp + _STATE_TTL_SECONDS,
        "callback_url": callback_url,
    }

    # NOTE: PKCE (RFC 7636 / OAuth 2.1) is not implemented here because this is
    # a server-side *confidential* client that always presents a client_secret.
    # Confidential clients are significantly less exposed to code-interception
    # attacks than public clients.  PKCE support for public/SPAs is tracked as
    # a follow-up improvement.
    params: dict = {
        "client_id": server.client_id,
        "redirect_uri": callback_url,
        "response_type": "code",
        "state": state,
    }
    if server.scopes:
        params["scope"] = " ".join(server.scopes)

    # Use "&" if the base URL already contains query parameters, otherwise "?"
    sep = "&" if "?" in server.authorization_url else "?"
    authorization_url = f"{server.authorization_url}{sep}{urlencode(params)}"
    server_name = server.server_name or server.name or server_id

    verbose_proxy_logger.debug(
        "openapi_oauth2_connect: user=%s server=%s",
        user_id,
        server_id,
    )

    return JSONResponse(
        {
            "authorization_url": authorization_url,
            "server_id": server_id,
            "server_name": server_name,
        }
    )


@router.get("/v1/mcp/oauth2/callback", include_in_schema=False, response_model=None)
async def openapi_oauth2_callback(
    request: Request,
    code: Optional[str] = Query(default=None),
    state: Optional[str] = Query(default=None),
    error: Optional[str] = Query(default=None),
    error_description: Optional[str] = Query(default=None),
) -> Response:
    from litellm.proxy.proxy_server import prisma_client

    if error:
        msg = error_description or error
        return HTMLResponse(
            content=_build_error_html(
                "Authorization failed",
                f"The provider returned an error: {msg}",
            ),
            status_code=400,
        )

    if not code or not state:
        return HTMLResponse(
            content=_build_error_html(
                "Missing parameters",
                "Expected 'code' and 'state' query parameters.",
            ),
            status_code=400,
        )

    _purge_expired_states()
    state_data = _pending_oauth2_states.get(state)
    if state_data is None:
        return HTMLResponse(
            content=_build_error_html(
                "Invalid state",
                "The OAuth2 state token is invalid or has already been used.",
            ),
            status_code=400,
        )
    if time.time() > state_data["expires_at"]:
        del _pending_oauth2_states[state]
        return HTMLResponse(
            content=_build_error_html(
                "State expired",
                "The OAuth2 state token has expired. Please start the connection flow again.",
            ),
            status_code=400,
        )

    # Consume the state (one-time use)
    del _pending_oauth2_states[state]

    server_id: str = state_data["server_id"]
    user_id: str = state_data["user_id"]

    server = global_mcp_server_manager.get_mcp_server_by_id(server_id)
    if server is None:
        return HTMLResponse(
            content=_build_error_html(
                "Server not found",
                f"MCP server '{server_id}' could not be found.",
            ),
            status_code=404,
        )
    if not server.token_url:
        return HTMLResponse(
            content=_build_error_html(
                "Configuration error",
                f"Server '{server_id}' has no token_url configured.",
            ),
            status_code=500,
        )

    # Reuse the callback_url stored during /connect so the redirect_uri
    # exactly matches what was sent to the provider, regardless of how the
    # two requests are routed through reverse proxies.
    callback_url = state_data.get("callback_url") or (
        f"{_get_callback_base_url(request)}/v1/mcp/oauth2/callback"
    )

    token_request_data = {
        "client_id": server.client_id or "",
        "client_secret": server.client_secret or "",
        "code": code,
        "redirect_uri": callback_url,
        "grant_type": "authorization_code",
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                server.token_url,
                data=token_request_data,
                headers={"Accept": "application/json"},
                timeout=30.0,
            )
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        verbose_proxy_logger.error(
            "openapi_oauth2_callback: token exchange HTTP error user=%s server=%s status=%s",
            user_id,
            server_id,
            exc.response.status_code,
        )
        return HTMLResponse(
            content=_build_error_html(
                "Token exchange failed",
                f"The provider returned HTTP {exc.response.status_code} during token exchange.",
            ),
            status_code=502,
        )
    except Exception as exc:
        verbose_proxy_logger.error(
            "openapi_oauth2_callback: token exchange error user=%s server=%s: %s",
            user_id,
            server_id,
            exc,
        )
        return HTMLResponse(
            content=_build_error_html(
                "Token exchange failed",
                "An unexpected error occurred during token exchange.",
            ),
            status_code=502,
        )

    # Parse response: try JSON first, fall back to URL-encoded form (GitHub can return either)
    # Some providers return HTTP 200 with an error body, so check for error fields explicitly.
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    provider_error: Optional[str] = None
    content_type = response.headers.get("content-type", "")
    if "application/json" in content_type:
        try:
            token_data = response.json()
            if token_data.get("error"):
                err = token_data["error"]
                err_desc = token_data.get("error_description", "")
                provider_error = f"{err}: {err_desc}" if err_desc else err
            else:
                access_token = token_data.get("access_token")
                refresh_token = token_data.get("refresh_token")
        except Exception:
            pass
    if access_token is None and provider_error is None:
        # URL-encoded form fallback (GitHub without Accept: application/json header)
        try:
            form_data = parse_qs(response.text)
            if form_data.get("error"):
                err = form_data["error"][0]
                err_desc_list = form_data.get("error_description", [])
                err_desc = err_desc_list[0] if err_desc_list else ""
                provider_error = f"{err}: {err_desc}" if err_desc else err
            else:
                tokens = form_data.get("access_token", [])
                if tokens:
                    access_token = tokens[0]
                refresh_tokens = form_data.get("refresh_token", [])
                if refresh_tokens:
                    refresh_token = refresh_tokens[0]
        except Exception:
            pass

    if provider_error:
        verbose_proxy_logger.error(
            "openapi_oauth2_callback: provider error user=%s server=%s error=%s",
            user_id,
            server_id,
            provider_error,
        )
        return HTMLResponse(
            content=_build_error_html(
                "Token exchange failed",
                f"The provider returned an error: {provider_error}",
            ),
            status_code=502,
        )

    if not access_token:
        verbose_proxy_logger.error(
            "openapi_oauth2_callback: no access_token in response user=%s server=%s",
            user_id,
            server_id,
        )
        return HTMLResponse(
            content=_build_error_html(
                "Token exchange failed",
                "The provider did not return an access token. Check client credentials and scopes.",
            ),
            status_code=502,
        )

    if prisma_client is None:
        verbose_proxy_logger.warning(
            "openapi_oauth2_callback: prisma_client is None — credential not persisted user=%s server=%s",
            user_id,
            server_id,
        )
        return HTMLResponse(
            content=_build_error_html(
                "Database unavailable",
                "Cannot persist credentials: database is not configured.",
            ),
            status_code=500,
        )

    # Persist the access token (and refresh token if the provider returned one).
    # Stored as a JSON blob so the token retrieval path can surface the refresh
    # token for future renewal without a schema change.
    credential_to_store = (
        json.dumps({"access_token": access_token, "refresh_token": refresh_token})
        if refresh_token
        else access_token
    )

    try:
        await store_user_credential(
            prisma_client=prisma_client,
            user_id=user_id,
            server_id=server_id,
            credential=credential_to_store,
        )
    except Exception as exc:
        verbose_proxy_logger.error(
            "openapi_oauth2_callback: failed to store credential user=%s server=%s: %s",
            user_id,
            server_id,
            exc,
        )
        return HTMLResponse(
            content=_build_error_html(
                "Storage error",
                "Failed to store the access token. Please try again.",
            ),
            status_code=500,
        )

    # Best-effort cache flush; a failure here must NOT mask the successful write above.
    try:
        from litellm.proxy._experimental.mcp_server.server import (
            _invalidate_byok_cred_cache,
        )

        _invalidate_byok_cred_cache(user_id, server_id)
    except Exception as exc:
        verbose_proxy_logger.warning(
            "openapi_oauth2_callback: cache invalidation failed (credential was stored) user=%s server=%s: %s",
            user_id,
            server_id,
            exc,
        )

    verbose_proxy_logger.info(
        "openapi_oauth2_callback: connected user=%s server=%s",
        user_id,
        server_id,
    )
    server_name = server.server_name or server.name or server_id
    return HTMLResponse(content=_build_success_html(server_name), status_code=200)


@router.get("/v1/mcp/server/{server_id}/oauth2/status", include_in_schema=False)
async def openapi_oauth2_status(
    server_id: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> JSONResponse:
    from litellm.proxy.proxy_server import prisma_client

    server = global_mcp_server_manager.get_mcp_server_by_id(server_id)
    if server is None:
        raise HTTPException(status_code=404, detail=f"MCP server '{server_id}' not found")

    user_id = user_api_key_dict.user_id or user_api_key_dict.api_key or ""
    server_name = server.server_name or server.name or server_id

    if prisma_client is None or not user_id:
        return JSONResponse(
            {"connected": False, "server_id": server_id, "server_name": server_name}
        )

    # Check the shared credential cache before issuing a DB query.
    # The frontend polls this endpoint every 2 s; the cache avoids a raw DB
    # hit on each poll.  The callback explicitly invalidates the cache entry
    # via _invalidate_byok_cred_cache, so the next poll after a successful
    # authorization always falls through to the DB and returns connected=True.
    try:
        import time as _time

        from litellm.proxy._experimental.mcp_server.server import (
            _BYOK_CRED_CACHE_TTL,
            _byok_cred_cache,
        )

        cached = _byok_cred_cache.get((user_id, server_id))
        if cached is not None:
            cached_cred, ts = cached
            if _time.monotonic() - ts < _BYOK_CRED_CACHE_TTL:
                return JSONResponse(
                    {
                        "connected": cached_cred is not None,
                        "server_id": server_id,
                        "server_name": server_name,
                    }
                )
    except Exception:
        pass  # If cache import fails, fall through to DB

    try:
        connected = await has_user_credential(
            prisma_client=prisma_client,
            user_id=user_id,
            server_id=server_id,
        )
    except Exception as exc:
        verbose_proxy_logger.error(
            "openapi_oauth2_status: error checking credential user=%s server=%s: %s",
            user_id,
            server_id,
            exc,
        )
        connected = False

    return JSONResponse(
        {"connected": connected, "server_id": server_id, "server_name": server_name}
    )
