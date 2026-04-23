"""Shared helpers for the MCP OAuth authorization endpoints
(BYOK + discoverable / pass-through OAuth proxy)."""

from ipaddress import ip_address
from urllib.parse import urlparse

from fastapi import HTTPException

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
