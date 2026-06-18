"""
Plugin proxy routes for litellm.

Enables external services to register as plugins and be proxied through
the litellm proxy server.

Config (in litellm config.yaml general_settings):
  plugins:
    - name: my-plugin
      url: "http://localhost:3210"
      display_name: "My Plugin"
      plugin_key: "sk-..."   # optional: plugin's own auth key

Plugin iframe auth:
  The UI calls GET /api/plugins/auth-token to receive the caller's token
  encrypted with the shared LITELLM_SALT_KEY.  The plugin decrypts it with
  the same key — so the raw litellm credential never appears in plaintext
  outside the proxy process, and a postMessage intercept yields only
  useless ciphertext.
"""

import base64
import hashlib
import hmac as _hmac
import json
import os
import time

from cryptography.fernet import Fernet, InvalidToken
from fastapi import APIRouter, Depends, HTTPException, Request, Response

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.llms.custom_httpx.http_handler import get_async_httpx_client
from litellm.types.llms.custom_http import httpxSpecialProvider

router = APIRouter()

# Headers that must never be forwarded TO plugin backends
_REQUEST_STRIP = {
    "host",
    "connection",
    "transfer-encoding",
    "te",
    "trailers",
    "upgrade",
    # Strip litellm credentials — plugin auth uses plugin_key only
    "authorization",
    "x-api-key",
    # Strip session cookies — plugins must not capture litellm JWT cookies
    "cookie",
}

# Headers to strip from plugin RESPONSES before returning to the browser.
# httpx already decompresses and de-chunks the body, so forwarding the wire
# encoding headers causes clients to attempt double-decompression (garbage) or
# incorrect length checks.  set-cookie is removed so plugins cannot overwrite
# litellm session cookies.
_RESPONSE_STRIP = {
    "content-encoding",
    "transfer-encoding",
    "content-length",
    "set-cookie",
}

# In-memory plugin registry — populated from general_settings at startup
_plugin_registry: dict = {}


# ---------------------------------------------------------------------------
# Key derivation — audience-scoped per plugin so compromising one plugin
# cannot be used to forge claims for another.  LITELLM_SALT_KEY is NEVER
# shared with plugins; each plugin only receives a key derived from
# HMAC(LITELLM_SALT_KEY, plugin_name) which reveals nothing about the master.
# ---------------------------------------------------------------------------
def _plugin_fernet(plugin_name: str) -> Fernet:
    """Return a Fernet cipher whose key is scoped to a specific plugin.

    Key material: HMAC-SHA256(LITELLM_SALT_KEY, plugin_name).
    A plugin possessing its own key cannot derive the master salt or
    forge claims intended for a different plugin.
    """
    salt = os.getenv("LITELLM_SALT_KEY", "").encode()
    derived = _hmac.new(salt, plugin_name.encode(), hashlib.sha256).digest()
    return Fernet(base64.urlsafe_b64encode(derived))


_CLAIM_TTL_SECONDS = 30  # identity claims expire after 30 s


def issue_plugin_session_claim(
    plugin_name: str, user_id: str | None, user_role: str | None
) -> str:
    """Issue a short-lived, audience-scoped identity claim for the plugin.

    The claim contains {user_id, user_role, plugin, exp}.  Crucially it
    contains NO litellm bearer token — the plugin can only derive the
    caller's identity, not act as them against the proxy.
    """
    claim = {
        "plugin": plugin_name,
        "user_id": user_id or "",
        "user_role": user_role or "",
        "exp": int(time.time()) + _CLAIM_TTL_SECONDS,
    }
    return _plugin_fernet(plugin_name).encrypt(json.dumps(claim).encode()).decode()


def verify_plugin_session_claim(plugin_name: str, ciphertext: str) -> dict:
    """Verify and decode a plugin session claim.

    Raises ValueError if the HMAC is invalid, the audience is wrong, or
    the claim is expired.  Returns the decoded claim dict on success.
    """
    try:
        raw = _plugin_fernet(plugin_name).decrypt(
            ciphertext.encode(), ttl=_CLAIM_TTL_SECONDS
        )
        claim = json.loads(raw)
    except (InvalidToken, Exception) as exc:
        raise ValueError("Invalid, tampered, or expired plugin session claim") from exc

    if claim.get("plugin") != plugin_name:
        raise ValueError("Plugin claim audience mismatch")
    if int(claim.get("exp", 0)) < int(time.time()):
        raise ValueError("Plugin session claim expired")
    return claim


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def register_plugins_from_config(general_settings: dict) -> None:
    """Replace the plugin registry from general_settings.

    Replaces (not merges) so plugins removed from config are immediately
    unreachable without requiring a process restart.
    """
    _plugin_registry.clear()
    for plugin in general_settings.get("plugins", []):
        name = plugin.get("name")
        if name:
            _plugin_registry[name] = plugin


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@router.get("/api/plugins", tags=["plugins"])
async def list_plugins(
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> list:
    """Return registered plugins for authenticated UI callers.

    plugin_key is only returned to proxy admins — not to regular users —
    so plugin credentials cannot be extracted by non-admin callers.
    """
    is_admin = getattr(user_api_key_dict, "user_role", None) == "proxy_admin"
    result = []
    for name, plugin in _plugin_registry.items():
        entry: dict = {
            "name": name,
            "display_name": plugin.get("display_name", name),
            "url": plugin.get("url", ""),
        }
        if is_admin and plugin.get("plugin_key"):
            entry["plugin_key"] = plugin["plugin_key"]
        result.append(entry)
    return result


@router.get("/api/plugins/auth-token", tags=["plugins"])
async def plugin_auth_token(
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    plugin_name: str = "litellm-platform-plugin",
) -> dict:
    """Issue a short-lived, audience-scoped plugin session claim.

    The claim contains {user_id, user_role, plugin, exp}.  It does NOT
    contain the caller's litellm bearer token — a compromised plugin can
    only learn the caller's identity, not impersonate them against the proxy.

    Encrypted with a key derived from HMAC(LITELLM_SALT_KEY, plugin_name),
    so each plugin holds only its own key and cannot forge claims for others.

    Requires LITELLM_SALT_KEY to be set; returns 503 otherwise.
    """
    if not os.getenv("LITELLM_SALT_KEY"):
        raise HTTPException(
            status_code=503,
            detail="LITELLM_SALT_KEY is not configured; plugin iframe auth unavailable.",
        )
    if plugin_name not in _plugin_registry:
        raise HTTPException(
            status_code=404, detail=f"Plugin '{plugin_name}' is not registered."
        )
    user_id = getattr(user_api_key_dict, "user_id", None)
    user_role = getattr(user_api_key_dict, "user_role", None)
    return {
        "session_claim": issue_plugin_session_claim(plugin_name, user_id, user_role)
    }


@router.api_route(
    "/plugin-proxy/{plugin_name}/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"],
    tags=["plugins"],
    include_in_schema=False,
)
async def plugin_proxy(
    plugin_name: str,
    path: str,
    request: Request,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> Response:
    """Authenticated reverse-proxy to a registered plugin backend.

    Restricted to proxy_admin callers — the shared plugin_key must not be
    usable as a confused-deputy credential by regular users.  Plugin UIs
    talk to the plugin service directly via the iframe; this route is for
    administrative and server-to-server access only.

    The caller's litellm credential is stripped and replaced with the
    plugin's own plugin_key so plugins never receive a live litellm API key.
    """
    if getattr(user_api_key_dict, "user_role", None) != "proxy_admin":
        return Response(
            content="Plugin proxy access requires proxy_admin role.",
            status_code=403,
        )

    plugin = _plugin_registry.get(plugin_name)
    if not plugin:
        return Response(
            content=f"Plugin '{plugin_name}' not registered",
            status_code=404,
        )

    target_url = f"{plugin['url'].rstrip('/')}/{path}"
    query = request.url.query
    if query:
        target_url = f"{target_url}?{query}"

    body = await request.body()

    # Strip caller credentials and hop-by-hop headers from forwarded request
    forward_headers = {
        k: v for k, v in request.headers.items() if k.lower() not in _REQUEST_STRIP
    }

    # Inject plugin's own credential as upstream auth (if configured)
    plugin_key = plugin.get("plugin_key")
    if plugin_key:
        forward_headers["authorization"] = f"Bearer {plugin_key}"

    # Forward caller identity so the plugin can enforce its own access control.
    # The plugin MUST NOT trust these as credentials — they are informational.
    # The plugin_key above is the only authentication mechanism.
    user_id = getattr(user_api_key_dict, "user_id", None)
    user_role = getattr(user_api_key_dict, "user_role", None)
    if user_id:
        forward_headers["x-litellm-user-id"] = str(user_id)
    if user_role:
        forward_headers["x-litellm-user-role"] = str(user_role)

    handler = get_async_httpx_client(
        llm_provider=httpxSpecialProvider.PassThroughEndpoint
    )
    try:
        req = handler.client.build_request(
            method=request.method,
            url=target_url,
            headers=forward_headers,
            content=body,
        )
        # Do not follow redirects — a redirect to an internal URL would allow
        # the plugin to SSRF the proxy into fetching arbitrary internal services.
        resp = await handler.client.send(req, follow_redirects=False)
    except Exception:
        return Response(
            content=f"Cannot connect to plugin '{plugin_name}' at {plugin['url']}",
            status_code=502,
        )

    safe_headers = {
        k: v for k, v in resp.headers.items() if k.lower() not in _RESPONSE_STRIP
    }

    return Response(
        content=resp.content,
        status_code=resp.status_code,
        headers=safe_headers,
    )
