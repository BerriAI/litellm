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
import os

from cryptography.fernet import Fernet, InvalidToken
from fastapi import APIRouter, Depends, HTTPException, Request, Response

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.llms.custom_httpx.http_handler import get_async_httpx_client

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
# Key derivation — deterministic, so plugin and proxy agree without sharing
# raw key material beyond LITELLM_SALT_KEY.
# ---------------------------------------------------------------------------
def _fernet() -> Fernet:
    """Build a Fernet cipher keyed from LITELLM_SALT_KEY.

    SHA-256 stretches the salt into a 32-byte key, which is then
    base64url-encoded as Fernet requires.
    """
    salt = os.getenv("LITELLM_SALT_KEY", "")
    key = base64.urlsafe_b64encode(hashlib.sha256(salt.encode()).digest())
    return Fernet(key)


def encrypt_token(token: str) -> str:
    """Encrypt a token for safe delivery to a plugin iframe."""
    return _fernet().encrypt(token.encode()).decode()


def decrypt_token(ciphertext: str) -> str:
    """Decrypt a token received from the proxy.  Raises ValueError on failure."""
    try:
        return _fernet().decrypt(ciphertext.encode()).decode()
    except (InvalidToken, Exception) as exc:
        raise ValueError("Invalid or tampered plugin auth token") from exc


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
) -> dict:
    """Return an encrypted copy of the caller's token for plugin iframe auth.

    The token is encrypted with a key derived from LITELLM_SALT_KEY.
    The plugin decrypts it with the same shared key — the raw litellm
    credential never appears in plaintext outside the proxy process.

    Requires LITELLM_SALT_KEY to be set; returns 503 otherwise.
    """
    if not os.getenv("LITELLM_SALT_KEY"):
        raise HTTPException(
            status_code=503,
            detail="LITELLM_SALT_KEY is not configured; plugin iframe auth unavailable.",
        )
    token = getattr(user_api_key_dict, "api_key", None)
    if not token:
        raise HTTPException(status_code=401, detail="No token available to encrypt.")
    return {"encrypted_token": encrypt_token(token)}


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

    The caller's litellm credential is stripped and replaced with the
    plugin's own plugin_key so plugins never receive a live litellm API key.
    """
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

    handler = get_async_httpx_client(llm_provider=None)
    try:
        req = handler.client.build_request(
            method=request.method,
            url=target_url,
            headers=forward_headers,
            content=body,
        )
        resp = await handler.client.send(req, follow_redirects=True)
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
