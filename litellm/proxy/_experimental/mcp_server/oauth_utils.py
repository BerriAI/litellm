"""Shared helpers for the MCP OAuth authorization endpoints
(BYOK + discoverable / pass-through OAuth proxy)."""

import os
from ipaddress import ip_address
from typing import Any, Dict, List, NoReturn, Optional
from urllib.parse import ParseResult, urlparse, urlsplit, urlunparse, urlunsplit

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

# Comma-separated private-use URI allowlist for native MCP clients.
# A trailing ``*`` is a prefix match; end the prefix with ``/`` (e.g.
# ``myapp://host/oauth/*``) so ``.../oauth/callback*`` does not also
# match ``.../oauth/callback-2``.
_TRUSTED_NATIVE_REDIRECT_URIS_ENV = "MCP_TRUSTED_NATIVE_REDIRECT_URIS"

# Default allowlist for trusted native redirect URIs.
_DEFAULT_NATIVE_REDIRECT_URIS: List[str] = [
    "cursor://anysphere.cursor-mcp/oauth/callback",
]

_warned_invalid_proxy_base_url: Optional[str] = None


def _oauth_invalid_request(
    error_description: str,
    *,
    hint: Optional[str] = None,
    **extra: Any,
) -> NoReturn:
    """Raise ``invalid_request`` (RFC 6749) with a debuggable description.

    FastAPI serializes ``detail`` as JSON. Callers still see ``error``:
    ``invalid_request``; ``error_description`` and ``hint`` explain what
    failed and how to fix it (e.g. reverse-proxy / PROXY_BASE_URL issues).
    """
    detail: Dict[str, Any] = {
        "error": "invalid_request",
        "error_description": error_description,
    }
    if hint:
        detail["hint"] = hint
    detail.update(extra)
    raise HTTPException(status_code=400, detail=detail)


def _origin_label(scheme: str, netloc: str) -> str:
    """Human-readable origin for error messages (scheme + host[:port])."""
    return f"{scheme}://{netloc}" if netloc else f"{scheme}://"


def _redact_mcp_resource_url(url: Optional[str]) -> Optional[str]:
    """Reduce an MCP server URL to its origin (scheme + host + port) for logging.

    Everything else is dropped: userinfo (``user:pass@``), the query string, the
    fragment, and the path, because hosted MCP servers routinely embed the
    credential in the path (e.g. ``/mcp/s/<token>``) and this value is persisted
    in spend-log metadata that a caller who can invoke the tool can read back.
    Returns None when the URL has no host to identify (nothing safe to log).
    """
    if not isinstance(url, str) or not url:
        return None
    try:
        parts = urlsplit(url)
        hostname = parts.hostname
        port = parts.port
    except ValueError:
        return None
    if not hostname:
        return None
    netloc = f"{hostname}:{port}" if port else hostname
    return urlunsplit((parts.scheme, netloc, "", "", "")) or None


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

    return urlunparse((scheme, _strip_default_port(scheme, netloc), parsed.path, "", "", ""))


def well_known_root_suffix() -> str:
    """The ``SERVER_ROOT_PATH`` segment inserted into a ``.well-known`` path (RFC 8414 / 9728
    path insertion), empty for a root-mounted proxy or an explicit ``/``.

    The discovery route registrations and the 401 challenges that advertise those routes both
    derive their path from this one function, so the ``resource_metadata`` URL a client is told
    to fetch cannot drift from the route that actually serves it.
    """
    root = os.getenv("SERVER_ROOT_PATH", "")
    return "" if root == "/" else root


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
    parsed = _parse_redirect_uri_for_validation(redirect_uri)
    if parsed.scheme not in ("http", "https"):
        _oauth_invalid_request(
            f"redirect_uri scheme {parsed.scheme!r} is not allowed; use http or https.",
        )
    if parsed.fragment:
        _oauth_invalid_request(
            "redirect_uri must not contain a URL fragment (#...).",
        )
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
    _oauth_invalid_request(
        "redirect_uri must use a loopback host (localhost or 127.0.0.0/8).",
        hint="Native MCP clients should register a callback on http://127.0.0.1:<port>/...",
    )


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


def _normalize_native_redirect_uri(
    parsed,
) -> str:
    """Lowercase scheme, netloc, and path for allowlist comparison."""
    return urlunparse(
        (
            (parsed.scheme or "").lower(),
            (parsed.netloc or "").lower(),
            (parsed.path or "").lower(),
            "",
            "",
            "",
        )
    )


def _parse_trusted_native_redirect_uris() -> List[str]:
    """Built-in native MCP callbacks plus ``MCP_TRUSTED_NATIVE_REDIRECT_URIS``."""
    entries: List[str] = [uri.lower() for uri in _DEFAULT_NATIVE_REDIRECT_URIS]
    raw = os.environ.get(_TRUSTED_NATIVE_REDIRECT_URIS_ENV, "").strip()
    if not raw:
        return entries
    for token in raw.split(","):
        entry = token.strip().lower()
        if entry and entry not in entries:
            entries.append(entry)
    return entries


def _native_wildcard_prefix_matches(normalized: str, prefix: str) -> bool:
    """Prefix match for ``entry*`` allowlist rows.

    When the prefix does not end with ``/``, only exact matches or
    deeper path segments (``prefix/...``) are accepted — not siblings
    like ``prefix-2``.
    """
    if not normalized.startswith(prefix):
        return False
    suffix = normalized[len(prefix) :]
    if not suffix:
        return True
    if prefix.endswith("/"):
        return True
    return suffix[0] == "/"


def _matches_trusted_native_redirect_uri(parsed) -> bool:
    """Allowlisted private-use / custom-scheme OAuth callbacks for native MCP clients."""
    if parsed.fragment:
        return False
    # Query strings are not part of registered redirect_uris (RFC 6749 §3.1.2).
    # Rejecting them prevents allowlist bypass via ``.../callback?injected=...``.
    if parsed.query:
        return False
    if not parsed.netloc:
        return False
    if parsed.username is not None or parsed.password is not None:
        return False
    if "\\" in parsed.netloc:
        return False

    normalized = _normalize_native_redirect_uri(parsed)
    for entry in _parse_trusted_native_redirect_uris():
        if entry.endswith("*"):
            if _native_wildcard_prefix_matches(normalized, entry[:-1]):
                return True
        elif normalized == entry:
            return True
    return False


def _parse_redirect_uri_for_validation(redirect_uri: str) -> ParseResult:
    try:
        return urlparse(redirect_uri)
    except ValueError:
        _oauth_invalid_request(
            "redirect_uri is not a valid URL.",
            hint="Use a full absolute URL for redirect_uri (e.g. https://your-host/ui/mcp/oauth/callback).",
        )


def _validate_trusted_http_redirect_shape(parsed: ParseResult) -> bool:
    """Return True when ``parsed`` is an allowlisted native callback (caller may return)."""
    if parsed.scheme not in ("http", "https"):
        if _matches_trusted_native_redirect_uri(parsed):
            return True
        _oauth_invalid_request(
            f"redirect_uri scheme {parsed.scheme!r} is not allowed; use http/https "
            "or a registered native callback (e.g. cursor://).",
            hint="Add the full URI to MCP_TRUSTED_NATIVE_REDIRECT_URIS for custom native clients.",
        )
    if parsed.fragment:
        _oauth_invalid_request(
            "redirect_uri must not contain a URL fragment (#...).",
        )
    if not parsed.netloc:
        _oauth_invalid_request(
            "redirect_uri must include a host (e.g. https://your-host/path).",
        )
    if parsed.username is not None or parsed.password is not None:
        _oauth_invalid_request(
            "redirect_uri must not contain userinfo (user:pass@host).",
        )
    if "\\" in parsed.netloc:
        _oauth_invalid_request(
            "redirect_uri host must not contain backslashes.",
        )
    return False


def _resolve_proxy_base_for_redirect(request: Request) -> Optional[str]:
    try:
        return get_request_base_url(request)
    except Exception as exc:
        verbose_logger.warning(
            "validate_trusted_redirect_uri: could not determine proxy origin, "
            "falling back to loopback + allowlist. error=%s",
            exc,
        )
        return None


def _trusted_redirect_uri_is_allowed(
    parsed: ParseResult,
    redirect_netloc: str,
    proxy_base: Optional[str],
) -> bool:
    if proxy_base:
        proxy_parsed = urlparse(proxy_base)
        if parsed.scheme == proxy_parsed.scheme and redirect_netloc == _strip_default_port(
            proxy_parsed.scheme, proxy_parsed.netloc
        ):
            return True

    host = (parsed.hostname or "").lower()
    if host == "localhost":
        return True
    try:
        if ip_address(host).is_loopback:
            return True
    except ValueError:
        pass

    if parsed.scheme == "https":
        for entry in _parse_trusted_redirect_origins():
            if _matches_trusted_origin_entry(redirect_netloc, entry):
                return True
    return False


def _build_trusted_redirect_rejection_message(
    redirect_uri: str,
    parsed: ParseResult,
    redirect_netloc: str,
    proxy_base: Optional[str],
) -> str:
    """Build a client-facing rejection message.

    Intentionally omits the proxy's resolved scheme / host / port to avoid
    leaking internal network topology (e.g. ``http://litellm-internal:4000``)
    through an unauthenticated endpoint. Full diagnostic detail — including
    the computed proxy base — is logged server-side by the caller.
    """
    redirect_origin = _origin_label(parsed.scheme, redirect_netloc)
    proxy_parsed = urlparse(proxy_base) if proxy_base else None
    proxy_netloc_norm = (
        _strip_default_port(proxy_parsed.scheme, proxy_parsed.netloc) if proxy_parsed and proxy_parsed.netloc else ""
    )

    mismatch_parts: List[str] = []
    if proxy_parsed and proxy_parsed.netloc:
        if parsed.scheme != proxy_parsed.scheme:
            mismatch_parts.append(
                f"scheme: redirect_uri uses {parsed.scheme!r}, but the proxy "
                "resolved a different scheme "
                "(TLS often terminates at ingress — set PROXY_BASE_URL to https://… "
                "or trust X-Forwarded-Proto from your ingress)"
            )
        if redirect_netloc != proxy_netloc_norm:
            mismatch_parts.append(f"host/port: redirect_uri {redirect_netloc!r} does not match the proxy origin")

    if mismatch_parts:
        return f"redirect_uri origin ({redirect_origin}) does not match the proxy origin. " + "; ".join(mismatch_parts)
    return (
        f"redirect_uri ({redirect_uri!r}) is not allowed: not same-origin with "
        f"the proxy origin, not loopback, and not listed in "
        f"{_TRUSTED_REDIRECT_ORIGINS_ENV}."
    )


def _raise_trusted_redirect_uri_rejected(
    request: Request,
    redirect_uri: str,
    parsed: ParseResult,
    redirect_netloc: str,
    proxy_base: Optional[str],
) -> NoReturn:
    description = _build_trusted_redirect_rejection_message(redirect_uri, parsed, redirect_netloc, proxy_base)

    hint = (
        "Align the proxy public URL with the browser URL. Set PROXY_BASE_URL to your "
        "HTTPS origin (e.g. https://litellm.example.com), or enable "
        "general_settings.use_x_forwarded_for with mcp_trusted_proxy_ranges for your "
        "ingress. If the redirect_uri is a legitimate separate-origin OAuth client "
        "(e.g. a web app registering with the proxy from another host via dynamic client "
        f"registration), add its origin to {_TRUSTED_REDIRECT_ORIGINS_ENV}. "
        "Verify: curl https://<host>/.well-known/oauth-authorization-server "
        "| jq .issuer — issuer must match window.location.origin in the UI."
    )

    verbose_logger.warning(
        "MCP OAuth: rejecting redirect_uri %r. %s "
        "Computed proxy base=%r (PROXY_BASE_URL=%r). "
        "Inbound headers: X-Forwarded-Proto=%r X-Forwarded-Host=%r "
        "X-Forwarded-Port=%r Host=%r. "
        "Trusted-redirect-origins env=%r. "
        "Trusted-native-redirect-uris env=%r.",
        redirect_uri,
        description,
        proxy_base,
        os.environ.get("PROXY_BASE_URL"),
        request.headers.get("X-Forwarded-Proto"),
        request.headers.get("X-Forwarded-Host"),
        request.headers.get("X-Forwarded-Port"),
        request.headers.get("Host"),
        os.environ.get(_TRUSTED_REDIRECT_ORIGINS_ENV),
        os.environ.get(_TRUSTED_NATIVE_REDIRECT_URIS_ENV),
    )

    _oauth_invalid_request(
        description,
        hint=hint,
        redirect_uri=redirect_uri,
    )


def validate_trusted_redirect_uri(request: Request, redirect_uri: str) -> None:
    """Accept ``redirect_uri`` when it is (a) same-origin with the
    proxy's own request origin, (b) loopback, (c) listed in the
    ``MCP_TRUSTED_REDIRECT_ORIGINS`` ops allowlist, or (d) a built-in /
    env-configured native MCP client callback (e.g. ``cursor://``).

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
    parsed = _parse_redirect_uri_for_validation(redirect_uri)
    if _validate_trusted_http_redirect_shape(parsed):
        return
    redirect_netloc = _strip_default_port(parsed.scheme, parsed.netloc)
    proxy_base = _resolve_proxy_base_for_redirect(request)
    if _trusted_redirect_uri_is_allowed(parsed, redirect_netloc, proxy_base):
        return
    _raise_trusted_redirect_uri_rejected(request, redirect_uri, parsed, redirect_netloc, proxy_base)
