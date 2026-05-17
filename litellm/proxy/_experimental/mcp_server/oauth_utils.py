"""Shared helpers for the MCP OAuth authorization endpoints
(BYOK + discoverable / pass-through OAuth proxy)."""

import os
from ipaddress import ip_address
from typing import List, Optional
from urllib.parse import urlparse, urlunparse

from fastapi import HTTPException, Request

from litellm._logging import verbose_logger
from litellm.proxy.auth.ip_address_utils import IPAddressUtils

# RFC 6749 §5.1 / OAuth 2.1 draft-15 §4.1.3: token-endpoint responses
# must not be cached — both success and error bodies may reveal secrets.
TOKEN_NO_CACHE_HEADERS = {"Cache-Control": "no-store", "Pragma": "no-cache"}

# Stripped from netloc before same-origin comparison so
# ``llm.example.com`` matches ``llm.example.com:443`` (load balancers
# routinely set X-Forwarded-Port: 443 even when the client URL has no
# explicit port, which would otherwise break a literal netloc compare).
_DEFAULT_PORTS = {"http": 80, "https": 443}

# Env var for ops to allowlist additional redirect_uri origins beyond
# same-origin + loopback — needed for first-party OAuth clients hosted
# on sister domains (e.g. a web app on app.example.com registering as
# an OAuth client of the MCP proxy on llm.example.com). Comma-separated;
# each entry is ``host`` or ``host:port``; a ``*.`` prefix matches any
# subdomain. HTTPS only.
_TRUSTED_REDIRECT_ORIGINS_ENV = "MCP_TRUSTED_REDIRECT_ORIGINS"


_warned_invalid_proxy_base_url: Optional[str] = None


def _resolve_proxy_base_url_env() -> Optional[str]:
    global _warned_invalid_proxy_base_url
    configured = os.environ.get("PROXY_BASE_URL", "").strip()
    if not configured:
        return None
    parsed = urlparse(configured)
    if parsed.scheme in ("http", "https") and parsed.netloc:
        normalized = urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", "", ""))
        return normalized.rstrip("/")
    if _warned_invalid_proxy_base_url != configured:
        verbose_logger.warning(
            "PROXY_BASE_URL=%r is not a valid http(s) URL (missing scheme "
            "or host) and will be ignored for MCP OAuth origin resolution. "
            "Set it to a full URL like https://litellm.example.com.",
            configured,
        )
        _warned_invalid_proxy_base_url = configured
    return None


def get_request_base_url(request: Request) -> str:
    """
    Get the base URL for the request, considering X-Forwarded-* headers.

    Resolution order: ``PROXY_BASE_URL`` env var, then X-Forwarded-* when
    the caller is a trusted proxy (``use_x_forwarded_for`` enabled AND
    caller in ``mcp_trusted_proxy_ranges``), otherwise the request's
    literal ``base_url``. Untrusted callers cannot poison OAuth-discovery
    / redirect_uri values by injecting headers.
    """
    configured = _resolve_proxy_base_url_env()
    if configured:
        return configured

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


def _strip_default_port(scheme: str, netloc: str) -> str:
    """Return ``netloc`` lowercased with the scheme's default port
    stripped. ``Llm.Example.com:443`` with scheme ``https`` becomes
    ``llm.example.com``. Used so a literal netloc comparison between
    the proxy's origin and the client redirect_uri survives a load-
    balancer that sets ``X-Forwarded-Port: 443``.
    """
    if not netloc:
        return netloc
    lowered = netloc.lower()
    if lowered.startswith("["):
        # IPv6 literal: port (if any) appears after the "]".
        close = lowered.rfind("]")
        if close != -1 and lowered[close + 1 :].startswith(":"):
            try:
                port = int(lowered[close + 2 :])
            except ValueError:
                return lowered
            if _DEFAULT_PORTS.get(scheme) == port:
                return lowered[: close + 1]
        return lowered
    if ":" in lowered:
        host, _, port_str = lowered.rpartition(":")
        try:
            port = int(port_str)
        except ValueError:
            return lowered
        if _DEFAULT_PORTS.get(scheme) == port:
            return host
    return lowered


def _parse_trusted_redirect_origins() -> List[str]:
    """Parse ``MCP_TRUSTED_REDIRECT_ORIGINS`` into normalized entries.
    Empty / unset env var → empty list. Entries are lowercased and any
    scheme / path component the operator included is stripped. Default
    ``:443`` is also stripped from non-wildcard entries so
    ``app.example.com:443`` matches a redirect_netloc whose own ``:443``
    has already been normalized away — the allowlist path is https-only,
    so ``:443`` is the only default port that can legitimately appear.
    """
    raw = os.environ.get(_TRUSTED_REDIRECT_ORIGINS_ENV, "").strip()
    if not raw:
        return []
    entries: List[str] = []
    for token in raw.split(","):
        entry = token.strip().lower()
        if not entry:
            continue
        if "://" in entry:
            entry = entry.split("://", 1)[1]
        entry = entry.split("/", 1)[0]
        if not entry:
            continue
        # Wildcards don't express port constraints; leave them alone.
        if not entry.startswith("*."):
            entry = _strip_default_port("https", entry)
        if entry:
            entries.append(entry)
    return entries


def _matches_trusted_origin_entry(netloc: str, entry: str) -> bool:
    """``entry`` is either ``host[:port]`` (exact match after port
    normalization) or ``*.suffix`` (subdomain wildcard; matches any
    strictly-deeper subdomain of ``suffix`` but not ``suffix`` itself).
    ``netloc`` is the already-port-normalized, lowercased netloc of
    the redirect_uri being validated.
    """
    if entry.startswith("*."):
        suffix = entry[2:]
        if not suffix or suffix.startswith("."):
            return False
        # Strip port from netloc for wildcard host comparison;
        # wildcards don't express port constraints.
        host = netloc.split(":", 1)[0] if ":" in netloc else netloc
        return host != suffix and host.endswith("." + suffix)
    return netloc == entry


def validate_trusted_redirect_uri(request: Request, redirect_uri: str) -> None:
    """Accept ``redirect_uri`` when it is (a) same-origin with the
    proxy's own request origin, (b) loopback, or (c) listed in the
    ``MCP_TRUSTED_REDIRECT_ORIGINS`` ops allowlist.

    Same-origin is VERIA-57's threat-model-safe equivalent of loopback:
    an attacker who can host content on the proxy's own HTTPS origin
    has already compromised the proxy, so the open-redirect + code-
    theft primitive that motivated the loopback-only rule does not
    apply. The same reasoning extends to ops-trusted first-party
    hosts (e.g. an internal web app registering as an OAuth client of
    the proxy on a sister domain).

    Allowlisted non-loopback hosts are accepted only when the
    redirect_uri scheme is ``https`` — an attacker on the network
    cannot elevate to https without controlling the host's TLS key.

    Use this in the discoverable OAuth proxy endpoints that serve both
    native clients and the proxy's UI / cross-origin web clients. The
    BYOK endpoints, which only serve native MCP clients, retain
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
    if not parsed.netloc or parsed.username is not None or parsed.password is not None:
        raise HTTPException(status_code=400, detail="invalid_request")
    # Reject userinfo (``user:pass@host``) outright: OAuth redirect_uris
    # have no legitimate reason to carry credentials, and allowing them
    # opens a host-confusion attack where the netloc *looks* allowlisted
    # (``app.example.com:443@attacker.example``) but the browser navigates
    # to the post-``@`` host and hands the authorization code to the
    # attacker. We compare against ``hostname`` after this, but defense in
    # depth keeps malformed netloc strings from reaching the wildcard
    # splitter.
    if parsed.username is not None or parsed.password is not None:
        raise HTTPException(status_code=400, detail="invalid_request")
    # Reject backslash in netloc: urlparse keeps ``\`` as part of netloc,
    # but browsers normalize ``\`` to ``/`` for http(s) URLs and treat it
    # as the start of the path. An attacker can exploit that split by
    # crafting ``https://attacker.net\app.example.com/cb`` — urlparse sees
    # ``attacker.net\app.example.com`` (matches ``*.example.com``) while
    # the browser navigates to ``attacker.net`` with the auth code.
    if "\\" in parsed.netloc:
        raise HTTPException(status_code=400, detail="invalid_request")

    redirect_netloc = _strip_default_port(parsed.scheme, parsed.netloc)

    # (a) Same-origin. Swallow ``get_request_base_url`` failures so the
    # loopback + allowlist paths remain reachable when the origin can't
    # be determined (e.g. request came from an untrusted proxy and
    # ``get_request_base_url`` raised).
    proxy_base: Optional[str] = None
    try:
        proxy_base = get_request_base_url(request)
    except Exception as exc:
        verbose_logger.warning(
            "validate_trusted_redirect_uri: could not determine proxy origin, "
            "falling back to loopback + allowlist. error=%s",
            exc,
        )
        proxy_base = None
    if proxy_base:
        proxy_parsed = urlparse(proxy_base)
        if (
            parsed.scheme == proxy_parsed.scheme
            and redirect_netloc
            == _strip_default_port(proxy_parsed.scheme, proxy_parsed.netloc)
        ):
            return

    # (b) Loopback — same rule as validate_loopback_redirect_uri.
    host = (parsed.hostname or "").lower()
    if host == "localhost":
        return
    try:
        if ip_address(host).is_loopback:
            return
    except ValueError:
        pass

    # (c) Ops allowlist. https only.
    if parsed.scheme == "https":
        for entry in _parse_trusted_redirect_origins():
            if _matches_trusted_origin_entry(redirect_netloc, entry):
                return

    verbose_logger.warning(
        "MCP OAuth: rejecting redirect_uri %r as invalid_request. "
        "Computed proxy base=%r (PROXY_BASE_URL=%r). "
        "Inbound headers: X-Forwarded-Proto=%r X-Forwarded-Host=%r "
        "X-Forwarded-Port=%r Host=%r. "
        "Trusted-redirect-origins env=%r. "
        "If this should be accepted, either align ingress X-Forwarded-* "
        "with the browser URL, set PROXY_BASE_URL to your public origin, "
        "or add the redirect_uri host to MCP_TRUSTED_REDIRECT_ORIGINS.",
        redirect_uri,
        proxy_base,
        os.environ.get("PROXY_BASE_URL"),
        request.headers.get("X-Forwarded-Proto"),
        request.headers.get("X-Forwarded-Host"),
        request.headers.get("X-Forwarded-Port"),
        request.headers.get("Host"),
        os.environ.get(_TRUSTED_REDIRECT_ORIGINS_ENV),
    )
    raise HTTPException(status_code=400, detail="invalid_request")
