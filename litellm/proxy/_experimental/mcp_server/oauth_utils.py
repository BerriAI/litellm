"""Shared helpers for the MCP OAuth authorization endpoints
(BYOK + discoverable / pass-through OAuth proxy)."""

from ipaddress import ip_address
from urllib.parse import urlparse, urlunparse

from fastapi import HTTPException, Request

from litellm._logging import verbose_logger
from litellm.proxy.auth.ip_address_utils import IPAddressUtils

# RFC 6749 §5.1 / OAuth 2.1 draft-15 §4.1.3: token-endpoint responses
# must not be cached — both success and error bodies may reveal secrets.
TOKEN_NO_CACHE_HEADERS = {"Cache-Control": "no-store", "Pragma": "no-cache"}


def get_request_base_url(request: Request) -> str:
    """
    Get the base URL for the request, considering X-Forwarded-* headers.

    X-Forwarded-Proto / X-Forwarded-Host / X-Forwarded-Port are only honoured
    when the request comes from a configured trusted proxy
    (``use_x_forwarded_for`` enabled AND caller in ``mcp_trusted_proxy_ranges``).
    Otherwise the request's literal ``base_url`` is returned, so an
    untrusted caller cannot poison OAuth-discovery / redirect_uri values
    by injecting headers.

    Args:
        request: FastAPI Request object

    Returns:
        The reconstructed base URL (e.g., "https://proxy.example.com")
    """
    base_url = str(request.base_url).rstrip("/")
    parsed = urlparse(base_url)

    if not IPAddressUtils.is_request_from_trusted_proxy(request):
        return base_url

    x_forwarded_proto = request.headers.get("X-Forwarded-Proto")
    x_forwarded_host = request.headers.get("X-Forwarded-Host")
    x_forwarded_port = request.headers.get("X-Forwarded-Port")

    scheme = x_forwarded_proto if x_forwarded_proto else parsed.scheme

    if x_forwarded_host:
        # X-Forwarded-Host may already include port (e.g., "example.com:8080")
        if ":" in x_forwarded_host and not x_forwarded_host.startswith("["):
            netloc = x_forwarded_host
        elif x_forwarded_port:
            netloc = f"{x_forwarded_host}:{x_forwarded_port}"
        else:
            netloc = x_forwarded_host
    else:
        netloc = parsed.netloc
        if x_forwarded_port and ":" not in netloc:
            netloc = f"{netloc}:{x_forwarded_port}"

    return urlunparse((scheme, netloc, parsed.path, "", "", ""))


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
    except Exception as exc:
        # If we can't determine the proxy's origin, fall through to
        # loopback. Log so the failure is diagnosable in production.
        verbose_logger.warning(
            "validate_trusted_redirect_uri: could not determine proxy origin, "
            "falling back to loopback-only check. error=%s",
            exc,
        )

    host = (parsed.hostname or "").lower()
    if host == "localhost":
        return
    try:
        if ip_address(host).is_loopback:
            return
    except ValueError:
        pass
    raise HTTPException(status_code=400, detail="invalid_request")
