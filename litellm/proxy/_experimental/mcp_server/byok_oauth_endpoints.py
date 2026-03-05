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


def _build_authorize_html(
    server_name: str,
    server_initial: str,
    client_id: str,
    redirect_uri: str,
    code_challenge: str,
    code_challenge_method: str,
    state: str,
    server_id: str,
    access_items: list,
    help_url: str,
) -> str:
    """Build the 2-step BYOK OAuth authorization page HTML."""

    # Build access checklist rows
    access_rows = "".join(
        f'<div class="access-item"><span class="check">&#10003;</span>{item}</div>'
        for item in access_items
    )
    access_section = ""
    if access_rows:
        access_section = f"""
        <div class="access-box">
          <div class="access-header">
            <span class="shield">&#9646;</span>
            <span>Requested Access</span>
          </div>
          {access_rows}
        </div>"""

    # Help link for step 2
    help_link_html = ""
    if help_url:
        help_link_html = f'<a class="help-link" href="{help_url}" target="_blank">Where do I find my API key? &#8599;</a>'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Connect {server_name} &mdash; LiteLLM</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #0f172a;
    min-height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 24px;
  }}
  .modal {{
    background: #ffffff;
    border-radius: 20px;
    padding: 36px 32px 32px;
    width: 440px;
    max-width: 100%;
    position: relative;
    box-shadow: 0 25px 60px rgba(0,0,0,0.35);
  }}
  /* Progress dots */
  .dots {{
    display: flex;
    justify-content: center;
    gap: 7px;
    margin-bottom: 28px;
  }}
  .dot {{
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #e2e8f0;
  }}
  .dot.active {{ background: #38bdf8; }}
  /* Close button */
  .close-btn {{
    position: absolute;
    top: 16px; right: 16px;
    background: none; border: none;
    font-size: 16px; color: #94a3b8;
    cursor: pointer; line-height: 1;
    width: 28px; height: 28px;
    border-radius: 6px;
    display: flex; align-items: center; justify-content: center;
  }}
  .close-btn:hover {{ background: #f1f5f9; color: #475569; }}
  /* Logo pair */
  .logos {{
    display: flex; align-items: center; justify-content: center;
    gap: 12px; margin-bottom: 20px;
  }}
  .logo {{
    width: 52px; height: 52px;
    border-radius: 14px;
    display: flex; align-items: center; justify-content: center;
    font-size: 22px; font-weight: 800; color: white;
  }}
  .logo-l {{ background: linear-gradient(135deg, #38bdf8 0%, #0284c7 100%); }}
  .logo-s {{ background: linear-gradient(135deg, #818cf8 0%, #4f46e5 100%); }}
  .logo-arrow {{ color: #cbd5e1; font-size: 20px; font-weight: 300; }}
  /* Headings */
  .step-title {{
    text-align: center;
    font-size: 21px; font-weight: 700;
    color: #0f172a; margin-bottom: 8px;
  }}
  .step-subtitle {{
    text-align: center;
    font-size: 14px; color: #64748b;
    line-height: 1.55; margin-bottom: 22px;
  }}
  /* Info box */
  .info-box {{
    background: #f8fafc;
    border-radius: 12px;
    padding: 14px 16px;
    display: flex; gap: 12px;
    margin-bottom: 14px;
  }}
  .info-icon {{ font-size: 17px; flex-shrink: 0; margin-top: 1px; color: #38bdf8; }}
  .info-box h4 {{ font-size: 13px; font-weight: 600; color: #1e293b; margin-bottom: 4px; }}
  .info-box p {{ font-size: 13px; color: #64748b; line-height: 1.5; }}
  /* Access checklist */
  .access-box {{
    background: #f8fafc;
    border-radius: 12px;
    padding: 14px 16px;
    margin-bottom: 22px;
  }}
  .access-header {{
    display: flex; align-items: center; gap: 8px;
    margin-bottom: 10px;
  }}
  .shield {{ color: #22c55e; font-size: 15px; }}
  .access-header > span:last-child {{
    font-size: 11px; font-weight: 700;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    color: #475569;
  }}
  .access-item {{
    display: flex; align-items: center; gap: 9px;
    font-size: 13.5px; color: #374151;
    padding: 3px 0;
  }}
  .check {{ color: #22c55e; font-weight: 700; font-size: 13px; }}
  /* Primary CTA */
  .btn-primary {{
    width: 100%; padding: 15px;
    background: #0f172a; color: white;
    border: none; border-radius: 12px;
    font-size: 15px; font-weight: 600;
    cursor: pointer; margin-bottom: 10px;
  }}
  .btn-primary:hover {{ background: #1e293b; }}
  .btn-cancel {{
    width: 100%; padding: 8px;
    background: none; border: none;
    font-size: 13.5px; color: #94a3b8;
    cursor: pointer;
  }}
  .btn-cancel:hover {{ color: #64748b; }}
  /* Step 2 nav */
  .step2-nav {{
    display: flex; align-items: center;
    justify-content: space-between;
    margin-bottom: 24px;
  }}
  .back-btn {{
    background: none; border: none;
    font-size: 13.5px; color: #64748b;
    cursor: pointer; display: flex; align-items: center; gap: 4px;
  }}
  .back-btn:hover {{ color: #374151; }}
  /* Key icon */
  .key-icon-wrap {{
    width: 46px; height: 46px;
    background: #e0f2fe;
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 20px; margin-bottom: 14px;
  }}
  /* Form elements */
  .field-label {{
    font-size: 13.5px; font-weight: 600;
    color: #1e293b; display: block;
    margin-bottom: 7px;
  }}
  .key-input {{
    width: 100%; padding: 11px 13px;
    border: 1.5px solid #e2e8f0;
    border-radius: 10px;
    font-size: 14px; color: #0f172a;
    outline: none; transition: border-color 0.15s, box-shadow 0.15s;
  }}
  .key-input:focus {{
    border-color: #38bdf8;
    box-shadow: 0 0 0 3px rgba(56,189,248,0.12);
  }}
  .help-link {{
    display: inline-flex; align-items: center; gap: 4px;
    color: #0ea5e9; font-size: 13px;
    text-decoration: none; margin: 8px 0 16px;
  }}
  .help-link:hover {{ text-decoration: underline; }}
  /* Save toggle card */
  .save-card {{
    border: 1.5px solid #e2e8f0;
    border-radius: 12px;
    padding: 13px 15px;
    margin-bottom: 6px;
  }}
  .save-row {{
    display: flex; align-items: center; gap: 10px;
  }}
  .save-icon {{ font-size: 16px; }}
  .save-label {{
    flex: 1;
    font-size: 14px; font-weight: 500; color: #1e293b;
  }}
  /* Toggle switch */
  .toggle {{ position: relative; width: 44px; height: 24px; flex-shrink: 0; }}
  .toggle input {{ opacity: 0; width: 0; height: 0; }}
  .slider {{
    position: absolute; inset: 0;
    background: #e2e8f0;
    border-radius: 24px; cursor: pointer;
    transition: background 0.18s;
  }}
  .slider::before {{
    content: '';
    position: absolute;
    width: 18px; height: 18px;
    left: 3px; bottom: 3px;
    background: white;
    border-radius: 50%;
    transition: transform 0.18s;
    box-shadow: 0 1px 3px rgba(0,0,0,0.18);
  }}
  input:checked + .slider {{ background: #38bdf8; }}
  input:checked + .slider::before {{ transform: translateX(20px); }}
  /* Duration pills */
  .duration-section {{ margin-top: 14px; }}
  .duration-label {{
    font-size: 12px; font-weight: 600;
    color: #64748b; margin-bottom: 8px;
    text-transform: uppercase; letter-spacing: 0.05em;
  }}
  .pills {{ display: flex; flex-wrap: wrap; gap: 7px; }}
  .pill {{
    padding: 6px 13px;
    border: 1.5px solid #e2e8f0;
    border-radius: 20px;
    font-size: 13px; color: #475569;
    cursor: pointer; background: white;
    transition: all 0.13s;
    user-select: none;
  }}
  .pill:hover {{ border-color: #94a3b8; }}
  .pill.sel {{
    border-color: #38bdf8;
    color: #0284c7;
    background: #e0f2fe;
  }}
  /* Security note */
  .sec-note {{
    background: #f8fafc;
    border-radius: 10px;
    padding: 11px 14px;
    display: flex; gap: 9px; align-items: flex-start;
    margin: 16px 0;
  }}
  .sec-icon {{ font-size: 13px; color: #94a3b8; margin-top: 1px; flex-shrink: 0; }}
  .sec-note p {{ font-size: 12.5px; color: #64748b; line-height: 1.5; }}
  /* Connect button */
  .btn-connect {{
    width: 100%; padding: 15px;
    border: none; border-radius: 12px;
    font-size: 15px; font-weight: 600;
    cursor: pointer;
    background: #bae6fd; color: #0369a1;
    transition: background 0.15s, color 0.15s;
  }}
  .btn-connect.ready {{
    background: #0ea5e9; color: white;
  }}
  .btn-connect.ready:hover {{ background: #0284c7; }}
  /* Step visibility */
  .step {{ display: none; }}
  .step.show {{ display: block; }}
</style>
</head>
<body>
<div class="modal">

  <!-- ── STEP 1: Connect ─────────────────────────────────────── -->
  <div id="s1" class="step show">
    <div class="dots">
      <div class="dot active"></div>
      <div class="dot"></div>
    </div>
    <button class="close-btn" type="button" onclick="doCancel()" title="Close">&times;</button>

    <div class="logos">
      <div class="logo logo-l">L</div>
      <span class="logo-arrow">&#8594;</span>
      <div class="logo logo-s">{server_initial}</div>
    </div>

    <h2 class="step-title">Connect {server_name}</h2>
    <p class="step-subtitle">LiteLLM needs access to {server_name} to complete your request.</p>

    <div class="info-box">
      <span class="info-icon">&#9432;</span>
      <div>
        <h4>How it works</h4>
        <p>LiteLLM acts as a secure bridge. Your requests are routed through our MCP client directly to {server_name}&rsquo;s API.</p>
      </div>
    </div>

    {access_section}

    <button class="btn-primary" type="button" onclick="goStep2()">
      Continue to Authentication &rarr;
    </button>
    <button class="btn-cancel" type="button" onclick="doCancel()">Cancel</button>
  </div>

  <!-- ── STEP 2: Provide API Key ──────────────────────────────── -->
  <div id="s2" class="step">
    <div class="step2-nav">
      <button class="back-btn" type="button" onclick="goStep1()">&#8592; Back</button>
      <div class="dots">
        <div class="dot active"></div>
        <div class="dot active"></div>
      </div>
      <button class="close-btn" style="position:static;" type="button" onclick="doCancel()" title="Close">&times;</button>
    </div>

    <div class="key-icon-wrap">&#128273;</div>
    <h2 class="step-title" style="text-align:left;">Provide API Key</h2>
    <p class="step-subtitle" style="text-align:left;">Enter your {server_name} API key to authorize this connection.</p>

    <form method="POST" id="authForm" onsubmit="prepareSubmit()">
      <input type="hidden" name="client_id"            value="{client_id}">
      <input type="hidden" name="redirect_uri"          value="{redirect_uri}">
      <input type="hidden" name="code_challenge"        value="{code_challenge}">
      <input type="hidden" name="code_challenge_method" value="{code_challenge_method}">
      <input type="hidden" name="state"                 value="{state}">
      <input type="hidden" name="server_id"             value="{server_id}">
      <input type="hidden" name="duration" id="durInput" value="until_revoked">

      <label class="field-label">{server_name} API Key</label>
      <input
        type="password"
        name="api_key"
        id="apiKey"
        class="key-input"
        placeholder="Enter your API key"
        required
        autofocus
        oninput="syncBtn()"
      >

      {help_link_html}

      <div class="save-card">
        <div class="save-row">
          <span class="save-icon">&#128190;</span>
          <span class="save-label">Save key for future use</span>
          <label class="toggle">
            <input type="checkbox" id="saveToggle" onchange="toggleDur()">
            <span class="slider"></span>
          </label>
        </div>
        <div id="durSection" class="duration-section" style="display:none;">
          <div class="duration-label">Duration</div>
          <div class="pills">
            <div class="pill" onclick="selDur('1h',this)">1 hour</div>
            <div class="pill sel" onclick="selDur('24h',this)">24 hours</div>
            <div class="pill" onclick="selDur('7d',this)">7 days</div>
            <div class="pill" onclick="selDur('30d',this)">30 days</div>
            <div class="pill" onclick="selDur('until_revoked',this)">Until I revoke</div>
          </div>
        </div>
      </div>

      <div class="sec-note">
        <span class="sec-icon">&#128274;</span>
        <p>Your key is encrypted at rest and transmitted securely. It is never shared with third parties.</p>
      </div>

      <button type="submit" class="btn-connect" id="connectBtn">
        &#128274; Connect &amp; Authorize
      </button>
    </form>
  </div>

</div>
<script>
  function goStep2() {{
    document.getElementById('s1').classList.remove('show');
    document.getElementById('s2').classList.add('show');
  }}
  function goStep1() {{
    document.getElementById('s2').classList.remove('show');
    document.getElementById('s1').classList.add('show');
  }}
  function doCancel() {{
    if (window.opener) window.close();
    else window.history.back();
  }}
  function toggleDur() {{
    const on = document.getElementById('saveToggle').checked;
    document.getElementById('durSection').style.display = on ? 'block' : 'none';
  }}
  function selDur(val, el) {{
    document.querySelectorAll('.pill').forEach(p => p.classList.remove('sel'));
    el.classList.add('sel');
    document.getElementById('durInput').value = val;
  }}
  function syncBtn() {{
    const btn = document.getElementById('connectBtn');
    if (document.getElementById('apiKey').value.length > 0) {{
      btn.classList.add('ready');
    }} else {{
      btn.classList.remove('ready');
    }}
  }}
  function prepareSubmit() {{
    // nothing extra needed — duration is already in the hidden input
  }}
</script>
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

    # Resolve server metadata (name, description items, help URL).
    server_name = "MCP Server"
    access_items: list = []
    help_url = ""
    if server_id:
        try:
            from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
                global_mcp_server_manager,
            )

            registry = global_mcp_server_manager.get_registry()
            if server_id in registry:
                srv = registry[server_id]
                server_name = srv.server_name or srv.name
                access_items = list(srv.byok_description or [])
                help_url = srv.byok_api_key_help_url or ""
        except Exception:
            pass

    server_initial = (server_name[0].upper()) if server_name else "S"

    html = _build_authorize_html(
        server_name=server_name,
        server_initial=server_initial,
        client_id=client_id or "",
        redirect_uri=redirect_uri,
        code_challenge=code_challenge,
        code_challenge_method=code_challenge_method or "S256",
        state=state or "",
        server_id=server_id or "",
        access_items=access_items,
        help_url=help_url,
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
