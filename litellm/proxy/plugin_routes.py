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
  The UI calls GET /api/plugins/auth-token to receive a short-lived identity
  claim ({user_id, user_role, plugin, exp}) encrypted with a per-plugin key
  derived as HMAC-SHA256(LITELLM_SALT_KEY, plugin_name).  The claim carries no
  litellm bearer token, so a compromised plugin learns only the caller's
  identity, never their credential.  LITELLM_SALT_KEY itself is never shared
  with plugins — each plugin holds only its own derived key.
"""

import base64
import hashlib
import hmac as _hmac
import json
import os
import time
from collections.abc import Mapping

from cryptography.fernet import Fernet, InvalidToken
from fastapi import APIRouter, Depends, HTTPException, Request, Response

from litellm.proxy._types import PluginConfig, SpecialHeaders, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.llms.custom_httpx.http_handler import get_async_httpx_client
from litellm.types.llms.custom_http import httpxSpecialProvider

router = APIRouter()

# Hop-by-hop headers (RFC 7230) and the litellm session cookie — never forwarded
# to a plugin backend.  Credential headers are added on top per-request from the
# canonical SpecialHeaders set so the plugin only ever authenticates via its own
# injected plugin_key.
_HOP_BY_HOP_STRIP = frozenset(
    {
        "host",
        "connection",
        "transfer-encoding",
        "te",
        "trailers",
        "upgrade",
        "cookie",
    }
)


def _configured_key_header_names() -> frozenset[str]:
    """The lowercased general_settings.litellm_key_header_name, if configured.

    Read live from the proxy module (not import-time) so a custom key header set
    via config is honoured without a restart.  Returns empty when unset.
    """
    try:
        from litellm.proxy import proxy_server
    except Exception:
        return frozenset()
    general_settings = getattr(proxy_server, "general_settings", None)
    if not isinstance(general_settings, dict):
        return frozenset()
    name: object = general_settings.get("litellm_key_header_name")
    return frozenset({name.lower()}) if isinstance(name, str) and name else frozenset()


def _request_strip_headers() -> frozenset[str]:
    """Headers to drop before forwarding a request to a plugin backend.

    Every header user_api_key_auth accepts as a litellm credential is stripped —
    Authorization, x-api-key, API-Key, x-goog-api-key, Ocp-Apim-Subscription-Key,
    x-litellm-api-key, and any configured custom key header — so a plugin can
    never be handed the caller's live litellm key (confused-deputy escalation).
    """
    return _HOP_BY_HOP_STRIP | SpecialHeaders.litellm_credential_header_names() | _configured_key_header_names()


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


def _safe_response_headers(raw: "Mapping[str, str]") -> dict[str, str]:
    """Strip wire-encoding/cookie headers and force proxied responses inert.

    Plugin-controlled bytes are served from the litellm dashboard origin, so a
    compromised plugin could return an HTML/JS document that executes with the
    admin's session against same-origin management APIs.  A sandbox CSP forces
    the response into an opaque origin with scripts disabled, and nosniff stops
    content-type confusion from re-enabling execution.  Both are set last so a
    plugin cannot override them with its own headers.
    """
    return {
        **{k: v for k, v in raw.items() if k.lower() not in _RESPONSE_STRIP},
        "content-security-policy": "sandbox",
        "x-content-type-options": "nosniff",
    }


# In-memory plugin registry — populated from general_settings at startup
_plugin_registry: dict[str, PluginConfig] = {}


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


def issue_plugin_session_claim(plugin_name: str, user_id: str | None, user_role: str | None) -> str:
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
        raw = _plugin_fernet(plugin_name).decrypt(ciphertext.encode(), ttl=_CLAIM_TTL_SECONDS)
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
def register_plugins_from_config(general_settings: dict[str, object]) -> None:
    """Replace the plugin registry from general_settings.

    Replaces (not merges) so plugins removed from config are immediately
    unreachable without requiring a process restart.
    """
    raw = general_settings.get("plugins")
    entries: list[object] = raw if isinstance(raw, list) else []
    new_registry = {p.name: p for p in (PluginConfig.model_validate(entry) for entry in entries)}
    _plugin_registry.clear()
    _plugin_registry.update(new_registry)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@router.get("/api/plugins", tags=["plugins"])
async def list_plugins(
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> list[dict[str, str]]:
    """Return registered plugins for authenticated UI callers.

    plugin_key is never returned — the browser never needs it (the proxy injects
    it server-side from the registry), and exposing it here would leak the
    credential into React state and DevTools.  Admin key management goes through
    the redacted /config/field/info path instead.
    """
    return [
        {
            "name": plugin.name,
            "display_name": plugin.display_name or plugin.name,
            "url": plugin.url,
        }
        for plugin in _plugin_registry.values()
    ]


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
        raise HTTPException(status_code=404, detail=f"Plugin '{plugin_name}' is not registered.")
    user_id = getattr(user_api_key_dict, "user_id", None)
    user_role = getattr(user_api_key_dict, "user_role", None)
    return {"session_claim": issue_plugin_session_claim(plugin_name, user_id, user_role)}


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

    target_url = f"{plugin.url.rstrip('/')}/{path}"
    query = request.url.query
    if query:
        target_url = f"{target_url}?{query}"

    body = await request.body()

    # Strip caller credentials and hop-by-hop headers from forwarded request
    strip = _request_strip_headers()
    forward_headers = {k: v for k, v in request.headers.items() if k.lower() not in strip}

    # Inject plugin's own credential as upstream auth (if configured)
    plugin_key = plugin.plugin_key
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

    handler = get_async_httpx_client(llm_provider=httpxSpecialProvider.PassThroughEndpoint)
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
            content=f"Cannot connect to plugin '{plugin_name}' at {plugin.url}",
            status_code=502,
        )

    return Response(
        content=resp.content,
        status_code=resp.status_code,
        headers=_safe_response_headers(resp.headers),
    )
