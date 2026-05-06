"""Shared helpers for the MCP OAuth authorization endpoints
(BYOK + discoverable / pass-through OAuth proxy)."""

from ipaddress import ip_address
from urllib.parse import urlparse

from fastapi import HTTPException, Request

# RFC 6749 §5.1 / OAuth 2.1 draft-15 §4.1.3: token-endpoint responses
# must not be cached — both success and error bodies may reveal secrets.
TOKEN_NO_CACHE_HEADERS = {"Cache-Control": "no-store", "Pragma": "no-cache"}


def validate_loopback_redirect_uri(redirect_uri: str) -> None:
    """Require a loopback ``redirect_uri`` (OAuth 2.1 §4.1.2.1 + RFC 8252
    §7.3 native-app pattern). MCP clients are native apps that listen on
    a localhost port; rejecting non-loopback URIs prevents a malicious
    client from pointing the callback at its own server to capture the
    authorization code — the credential-theft primitive behind VERIA-57
    and pNr1PHa9.

    Accepts the literal ``localhost`` plus any IP in the loopback ranges
    (IPv4 ``127.0.0.0/8`` and IPv6 ``::1``). A string match on
    ``"127.0.0.1"`` alone would miss ``127.0.0.2`` and the full-form
    IPv6 loopback ``0:0:0:0:0:0:0:1``.
    """
    try:
        parsed = urlparse(redirect_uri)
    except ValueError:
        raise HTTPException(status_code=400, detail="invalid_request")
    if parsed.scheme not in ("http", "https"):
        raise HTTPException(status_code=400, detail="invalid_request")
    # Fragments are not allowed in OAuth redirect URIs (RFC 6749 §3.1.2)
    # — rejecting them prevents a ``http://127.0.0.1/cb#frag?code=...``
    # from silently eating the authorization code.
    if parsed.fragment:
        raise HTTPException(status_code=400, detail="invalid_request")
    host = (parsed.hostname or "").lower()
    if host == "localhost":
        return
    try:
        if ip_address(host).is_loopback:
            return
    except ValueError:
        # Unparseable host (malformed IPv6, etc.) — treat as invalid,
        # don't let it bubble up as a 500.
        pass
    raise HTTPException(status_code=400, detail="invalid_request")


def validate_trusted_redirect_uri(request: Request, redirect_uri: str) -> None:
    """Accept same-origin (proxy's own origin) OR loopback ``redirect_uri``.

    Same-origin is required for the LiteLLM UI's OAuth flow: the UI
    redirects to ``<proxy>/ui/mcp/oauth/callback`` which is not loopback
    but is on the proxy's own trusted HTTPS origin. An attacker cannot
    host content on the proxy's own origin without already owning the
    proxy, so the open-redirect / code-theft primitive that motivated
    :func:`validate_loopback_redirect_uri` does not apply here.

    Loopback continues to be accepted for native MCP clients (per
    OAuth 2.1 §4.1.2.1 + RFC 8252 §7.3).

    Use this in the discoverable OAuth proxy endpoints that serve both
    native clients and the proxy's own UI. BYOK endpoints that only
    support native clients should keep
    :func:`validate_loopback_redirect_uri`.
    """
    # Lazy import to avoid circular dependency with discoverable_endpoints.
    from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
        get_request_base_url,
    )

    try:
        parsed = urlparse(redirect_uri)
    except ValueError:
        raise HTTPException(status_code=400, detail="invalid_request")
    if parsed.scheme not in ("http", "https"):
        raise HTTPException(status_code=400, detail="invalid_request")
    if parsed.fragment:
        raise HTTPException(status_code=400, detail="invalid_request")

    # Same-origin: scheme + netloc (host[:port]) must match the proxy's
    # own base URL at this request (honouring trusted X-Forwarded-*).
    try:
        proxy_base = urlparse(get_request_base_url(request))
        if (
            parsed.netloc
            and parsed.scheme == proxy_base.scheme
            and parsed.netloc.lower() == proxy_base.netloc.lower()
        ):
            return
    except Exception:
        # If we can't determine the proxy's origin, fall through to loopback.
        pass

    host = (parsed.hostname or "").lower()
    if host == "localhost":
        return
    try:
        if ip_address(host).is_loopback:
            return
    except ValueError:
        pass
    raise HTTPException(status_code=400, detail="invalid_request")
