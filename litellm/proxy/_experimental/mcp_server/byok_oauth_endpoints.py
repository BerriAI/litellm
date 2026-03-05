"""
BYOK (Bring Your Own Key) OAuth 2.1 Authorization Server endpoints for MCP servers.

When an MCP client connects to a BYOK-enabled server and no stored credential exists,
LiteLLM runs a minimal OAuth 2.1 authorization code flow.  The "authorization page" is
just a form that asks the user for their API key — not a full identity-provider OAuth.

Endpoints implemented here:
  GET  /.well-known/oauth-authorization-server      — OAuth authorization server metadata
  GET  /.well-known/oauth-protected-resource         — OAuth protected resource metadata
  GET  /v1/mcp/oauth/authorize                       — Shows HTML form to collect the API key
  POST /v1/mcp/oauth/authorize                       — Stores temp auth code and redirects
  POST /v1/mcp/oauth/token                           — Exchanges code for a bearer JWT token
"""

import base64
import hashlib
import time
import uuid
from typing import Dict, Optional, cast

import jwt
from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from litellm._logging import verbose_proxy_logger
from litellm.proxy._experimental.mcp_server.db import store_user_credential
from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
    get_request_base_url,
)

# ---------------------------------------------------------------------------
# In-memory store for pending authorization codes.
# Each entry: {code: {api_key, server_id, code_challenge, redirect_uri, user_id, expires_at}}
# ---------------------------------------------------------------------------
_byok_auth_codes: Dict[str, dict] = {}

# Authorization codes expire after 5 minutes.
_AUTH_CODE_TTL_SECONDS = 300

router = APIRouter(tags=["mcp"])


# ---------------------------------------------------------------------------
# PKCE helper
# ---------------------------------------------------------------------------


def _verify_pkce(code_verifier: str, code_challenge: str) -> bool:
    """Return True iff SHA-256(code_verifier) == code_challenge (base64url, no padding)."""
    digest = hashlib.sha256(code_verifier.encode()).digest()
    computed = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
    return computed == code_challenge


# ---------------------------------------------------------------------------
# Cleanup of expired auth codes (called lazily on each request)
# ---------------------------------------------------------------------------


def _purge_expired_codes() -> None:
    now = time.time()
    expired = [k for k, v in _byok_auth_codes.items() if v["expires_at"] < now]
    for k in expired:
        del _byok_auth_codes[k]


# ---------------------------------------------------------------------------
# HTML template for the authorization page
# ---------------------------------------------------------------------------

_AUTHORIZE_HTML = """<!DOCTYPE html>
<html>
<head><title>Connect to {server_name} — LiteLLM</title>
<style>
  body {{ font-family: system-ui; background: #0f172a; display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0; }}
  .card {{ background: #1e293b; border-radius: 12px; padding: 32px; width: 400px; color: white; }}
  h2 {{ margin: 0 0 8px; font-size: 20px; }}
  p {{ color: #94a3b8; margin: 0 0 24px; font-size: 14px; }}
  label {{ font-size: 13px; color: #cbd5e1; display: block; margin-bottom: 6px; }}
  input[type=password] {{ width: 100%; padding: 10px; border-radius: 8px; border: 1px solid #334155; background: #0f172a; color: white; font-size: 14px; box-sizing: border-box; }}
  button {{ width: 100%; margin-top: 20px; padding: 12px; background: #3b82f6; border: none; border-radius: 8px; color: white; font-size: 15px; cursor: pointer; }}
  button:hover {{ background: #2563eb; }}
  .note {{ font-size: 12px; color: #64748b; margin-top: 16px; text-align: center; }}
</style></head>
<body>
  <div class="card">
    <h2>Connect to {server_name}</h2>
    <p>Enter your {server_name} API key to authorize this connection.</p>
    <form method="POST">
      <input type="hidden" name="client_id" value="{client_id}">
      <input type="hidden" name="redirect_uri" value="{redirect_uri}">
      <input type="hidden" name="code_challenge" value="{code_challenge}">
      <input type="hidden" name="code_challenge_method" value="{code_challenge_method}">
      <input type="hidden" name="state" value="{state}">
      <input type="hidden" name="server_id" value="{server_id}">
      <label>{server_name} API Key</label>
      <input type="password" name="api_key" placeholder="Enter your API key" required autofocus>
      <button type="submit">Connect &amp; Authorize</button>
    </form>
    <p class="note">Your key is encrypted at rest and never shared with third parties.</p>
  </div>
</body>
</html>"""


# ---------------------------------------------------------------------------
# OAuth metadata discovery endpoints
# ---------------------------------------------------------------------------


@router.get("/.well-known/oauth-authorization-server", include_in_schema=False)
async def oauth_authorization_server_metadata(request: Request) -> JSONResponse:
    """RFC 8414 Authorization Server Metadata for the BYOK OAuth flow."""
    base_url = get_request_base_url(request)
    return JSONResponse(
        {
            "issuer": base_url,
            "authorization_endpoint": f"{base_url}/v1/mcp/oauth/authorize",
            "token_endpoint": f"{base_url}/v1/mcp/oauth/token",
            "response_types_supported": ["code"],
            "grant_types_supported": ["authorization_code"],
            "code_challenge_methods_supported": ["S256"],
        }
    )


@router.get("/.well-known/oauth-protected-resource", include_in_schema=False)
async def oauth_protected_resource_metadata(request: Request) -> JSONResponse:
    """RFC 9728 Protected Resource Metadata pointing back at this server."""
    base_url = get_request_base_url(request)
    return JSONResponse(
        {
            "resource": base_url,
            "authorization_servers": [base_url],
        }
    )


# ---------------------------------------------------------------------------
# Authorization endpoint — GET (show form) and POST (process form)
# ---------------------------------------------------------------------------


@router.get("/v1/mcp/oauth/authorize", include_in_schema=False)
async def byok_authorize_get(
    request: Request,
    client_id: Optional[str] = None,
    redirect_uri: Optional[str] = None,
    response_type: Optional[str] = None,
    code_challenge: Optional[str] = None,
    code_challenge_method: Optional[str] = None,
    state: Optional[str] = None,
    server_id: Optional[str] = None,
) -> HTMLResponse:
    """
    Show the BYOK API-key entry form.

    The MCP client navigates the user here; the user types their API key and
    clicks "Connect & Authorize", which POSTs back to this same path.
    """
    if response_type != "code":
        raise HTTPException(status_code=400, detail="response_type must be 'code'")
    if not redirect_uri:
        raise HTTPException(status_code=400, detail="redirect_uri is required")
    if not code_challenge:
        raise HTTPException(status_code=400, detail="code_challenge is required")

    # Resolve a human-readable server name.
    server_name = server_id or "MCP Server"
    if server_id:
        try:
            from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
                global_mcp_server_manager,
            )

            registry = global_mcp_server_manager.get_registry()
            if server_id in registry:
                server_name = registry[server_id].server_name or registry[server_id].name
        except Exception:
            pass

    html = _AUTHORIZE_HTML.format(
        server_name=server_name,
        client_id=client_id or "",
        redirect_uri=redirect_uri,
        code_challenge=code_challenge,
        code_challenge_method=code_challenge_method or "S256",
        state=state or "",
        server_id=server_id or "",
    )
    return HTMLResponse(content=html)


@router.post("/v1/mcp/oauth/authorize", include_in_schema=False)
async def byok_authorize_post(
    request: Request,
    client_id: str = Form(default=""),
    redirect_uri: str = Form(...),
    code_challenge: str = Form(...),
    code_challenge_method: str = Form(default="S256"),
    state: str = Form(default=""),
    server_id: str = Form(default=""),
    api_key: str = Form(...),
) -> RedirectResponse:
    """
    Process the BYOK API-key form submission.

    Stores a short-lived authorization code and redirects the client back to
    redirect_uri with ?code=...&state=... query parameters.
    """
    _purge_expired_codes()

    if code_challenge_method != "S256":
        raise HTTPException(
            status_code=400, detail="Only S256 code_challenge_method is supported"
        )

    auth_code = str(uuid.uuid4())
    _byok_auth_codes[auth_code] = {
        "api_key": api_key,
        "server_id": server_id,
        "code_challenge": code_challenge,
        "redirect_uri": redirect_uri,
        "user_id": client_id,  # external client passes LiteLLM user-id as client_id
        "expires_at": time.time() + _AUTH_CODE_TTL_SECONDS,
    }

    separator = "&" if "?" in redirect_uri else "?"
    location = f"{redirect_uri}{separator}code={auth_code}&state={state}"
    return RedirectResponse(url=location, status_code=302)


# ---------------------------------------------------------------------------
# Token endpoint
# ---------------------------------------------------------------------------


@router.post("/v1/mcp/oauth/token", include_in_schema=False)
async def byok_token(
    request: Request,
    grant_type: str = Form(...),
    code: str = Form(...),
    redirect_uri: str = Form(default=""),
    code_verifier: str = Form(...),
    client_id: str = Form(default=""),
) -> JSONResponse:
    """
    Exchange an authorization code for a short-lived BYOK session JWT.

    1. Validates the authorization code and PKCE challenge.
    2. Stores the API key via store_user_credential().
    3. Issues a signed JWT with type="byok_session".
    """
    from litellm.proxy.proxy_server import master_key, prisma_client

    _purge_expired_codes()

    if grant_type != "authorization_code":
        raise HTTPException(status_code=400, detail="unsupported_grant_type")

    record = _byok_auth_codes.get(code)
    if record is None:
        raise HTTPException(status_code=400, detail="invalid_grant")

    if time.time() > record["expires_at"]:
        del _byok_auth_codes[code]
        raise HTTPException(status_code=400, detail="invalid_grant")

    # PKCE verification
    if not _verify_pkce(code_verifier, record["code_challenge"]):
        raise HTTPException(status_code=400, detail="invalid_grant")

    # Consume the code (one-time use)
    del _byok_auth_codes[code]

    server_id: str = record["server_id"]
    api_key_value: str = record["api_key"]
    # Prefer the user_id that was stored when the code was issued; fall back to
    # whatever client_id the token request supplies (they should match).
    user_id: str = record.get("user_id") or client_id

    if not user_id:
        raise HTTPException(
            status_code=400,
            detail="Cannot determine user_id; pass LiteLLM user id as client_id",
        )

    # Persist the BYOK credential
    if prisma_client is not None:
        try:
            await store_user_credential(
                prisma_client=prisma_client,
                user_id=user_id,
                server_id=server_id,
                credential=api_key_value,
            )
        except Exception as exc:
            verbose_proxy_logger.error(
                "byok_token: failed to store user credential for user=%s server=%s: %s",
                user_id,
                server_id,
                exc,
            )
            raise HTTPException(status_code=500, detail="Failed to store credential")
    else:
        verbose_proxy_logger.warning(
            "byok_token: prisma_client is None — credential not persisted"
        )

    if master_key is None:
        raise HTTPException(
            status_code=500, detail="Master key not configured; cannot issue token"
        )

    now = int(time.time())
    payload = {
        "user_id": user_id,
        "server_id": server_id,
        "type": "byok_session",
        "iat": now,
        "exp": now + 3600,
    }
    access_token = jwt.encode(payload, cast(str, master_key), algorithm="HS256")

    return JSONResponse(
        {
            "access_token": access_token,
            "token_type": "bearer",
            "expires_in": 3600,
        }
    )
