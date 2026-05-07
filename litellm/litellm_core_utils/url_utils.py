"""
URL validation for user-controlled URLs.

Use validate_url() before fetching any URL that originates from user
input (image_url, file_url, spec_path, etc.) to prevent SSRF attacks.

validate_url() resolves DNS once, validates all IPs, and rewrites the
URL to connect to the validated IP directly — no TOCTOU gap, no DNS
rebinding. Redirects are followed manually with validation at each hop.

Admins can opt out via two ``litellm`` globals (wired from proxy config):

- ``litellm.user_url_validation`` (bool, default True): master switch.
  When False, ``safe_get``/``async_safe_get`` perform a plain fetch with
  no DNS check, no block list, and no rewrite.
- ``litellm.user_url_allowed_hosts`` (List[str], default []): per-host
  allowlist. Entries are ``hostname`` or ``hostname:port`` (IPv6 hosts as
  ``[addr]`` / ``[addr]:port``). Matching hosts skip the blocked-networks
  check but still resolve DNS and still rewrite HTTP to the resolved IP.
"""

import socket
from ipaddress import ip_address, ip_network
from typing import Any, List, Optional, Set, Tuple
from urllib.parse import quote, urlparse, urlunparse

import httpx

import litellm

# Globally-routable IPs that are cloud-internal. Everything else
# non-public is caught by ``not ip.is_global`` (RFC 6890, as implemented by
# Python's ``ipaddress`` module). This list only holds IPs that are
# publicly routable *and* point to cloud-fabric services reachable from
# inside a VM via special in-fabric routing.
_CLOUD_METADATA_EXCEPTIONS = [
    ip_network("168.63.129.16/32"),  # Azure Wire Server
]

_ALLOWED_SCHEMES = ("http", "https")


class SSRFError(ValueError):
    """Raised when a URL targets a blocked network."""

    pass


def encode_url_path_segment(value: Any, *, field_name: str = "path parameter") -> str:
    """Percent-encode one user-controlled URL path segment.

    ``urllib.parse.quote(..., safe="")`` intentionally leaves RFC 3986
    unreserved characters such as ``.`` unescaped, so reject standalone dot
    segments before they can be appended to an upstream URL and normalized by
    the HTTP client.
    """
    if value is None:
        raise ValueError(f"{field_name} is required")

    value_str = str(value)
    if value_str == "":
        raise ValueError(f"{field_name} is required")
    if value_str in {".", ".."}:
        raise ValueError(f"{field_name} cannot be a dot path segment")

    return quote(value_str, safe="")


def encode_url_path_segments(value: Any, *, field_name: str = "path") -> str:
    """Percent-encode a user-controlled URL path made of multiple segments.

    Empty segments are rejected, so leading, trailing, or consecutive slashes
    fail closed instead of being normalized by the HTTP client.
    """
    if value is None:
        raise ValueError(f"{field_name} is required")

    value_str = str(value)
    if value_str == "":
        raise ValueError(f"{field_name} is required")

    encoded_segments = []
    for segment in value_str.split("/"):
        encoded_segments.append(encode_url_path_segment(segment, field_name=field_name))

    return "/".join(encoded_segments)


def _is_blocked_ip(addr: str) -> bool:
    """Return True for any IP not safe to reach from a user-supplied URL.

    Policy: default-deny via ``ip.is_global`` (RFC 6890), plus an explicit
    exception list for globally-routable cloud-fabric IPs that are still
    dangerous from inside a cloud VM (currently just Azure Wire Server).
    Unparseable addresses fail closed.
    """
    try:
        ip = ip_address(addr)
    except ValueError:
        return True  # fail-closed: unparseable addresses are blocked
    if ip.version == 6 and hasattr(ip, "ipv4_mapped") and ip.ipv4_mapped:
        ip = ip.ipv4_mapped
    if not ip.is_global or ip.is_multicast:
        return True
    return any(ip in net for net in _CLOUD_METADATA_EXCEPTIONS)


def _normalize_host(host: str) -> str:
    """Lowercase and strip a trailing dot from a hostname."""
    return host.lower().rstrip(".")


def _default_port_for_scheme(scheme: str) -> int:
    return 443 if scheme == "https" else 80


def _parse_url_destination_allowlist_entry(
    entry: str,
) -> Optional[Tuple[str, Optional[str], Optional[int]]]:
    """Parse an admin allowlist entry into host, optional scheme, optional port.

    Entries may be bare hosts (``api.example.com``), host+port
    (``api.example.com:8443``), or origins (``https://api.example.com``).
    URL paths are intentionally ignored so admins can paste an api_base value.
    """
    entry = entry.strip()
    if not entry:
        return None

    has_scheme = "://" in entry
    parsed = urlparse(entry if has_scheme else f"//{entry}")
    if has_scheme and parsed.scheme not in _ALLOWED_SCHEMES:
        return None
    if parsed.username is not None or parsed.password is not None:
        return None
    if not parsed.hostname:
        return None

    try:
        port = parsed.port
    except ValueError:
        return None

    scheme: Optional[str] = parsed.scheme if has_scheme else None
    if scheme is not None and port is None:
        port = _default_port_for_scheme(scheme)

    return _normalize_host(parsed.hostname), scheme, port


def is_url_destination_allowed_by_host(url: str, allowed_hosts: List[str]) -> bool:
    """Return True when a credential-bearing provider URL is admin-allowlisted.

    This does not fetch, resolve, or rewrite URLs. It only answers whether the
    destination origin is explicitly trusted by configuration. Use ``safe_get``
    for user-controlled content fetches that require SSRF protection.
    """
    parsed = urlparse(url)
    if parsed.scheme not in _ALLOWED_SCHEMES:
        return False
    if parsed.username is not None or parsed.password is not None:
        return False
    if not parsed.hostname:
        return False

    try:
        effective_port = parsed.port or _default_port_for_scheme(parsed.scheme)
    except ValueError:
        return False

    normalized_host = _normalize_host(parsed.hostname)
    configured_entries = (
        [allowed_hosts] if isinstance(allowed_hosts, str) else allowed_hosts
    )
    for entry in configured_entries or []:
        if not isinstance(entry, str):
            continue
        parsed_entry = _parse_url_destination_allowlist_entry(entry)
        if parsed_entry is None:
            continue
        allowed_host, allowed_scheme, allowed_port = parsed_entry
        if allowed_host != normalized_host:
            continue
        if allowed_scheme is not None and allowed_scheme != parsed.scheme:
            continue
        if allowed_port is not None and allowed_port != effective_port:
            continue
        return True
    return False


def _format_host_header(hostname: str, port: int, default_port: int) -> str:
    """Build an RFC 7230 Host header value, bracketing IPv6 literals."""
    bracketed = f"[{hostname}]" if ":" in hostname else hostname
    if port == default_port:
        return bracketed
    return f"{bracketed}:{port}"


def _sockaddr_host(sockaddr: Any) -> str:
    """Return the host element of a ``getaddrinfo`` sockaddr as ``str``.

    ``getaddrinfo`` with ``IPPROTO_TCP`` returns AF_INET / AF_INET6 sockaddrs
    whose first element is always a host string. mypy types it as
    ``str | int`` (since sockaddrs for other families can hold ints), so we
    narrow at the boundary. Fail closed if the stdlib ever returns something
    unexpected — a non-string here would mean we have no IP to check against
    the SSRF blocklist.
    """
    host = sockaddr[0]
    if not isinstance(host, str):
        raise SSRFError(f"getaddrinfo returned non-string host: {host!r}")
    return host


def _is_host_allowlisted(hostname: str, effective_port: int) -> bool:
    """Check whether a host is in the admin-configured allowlist.

    Admin entries may be ``hostname`` (any port) or ``hostname:port``. IPv6
    literals are written bracketed (``[::1]`` / ``[::1]:8080``). Matching
    is case-insensitive on the hostname.
    """
    configured: List[str] = getattr(litellm, "user_url_allowed_hosts", []) or []
    if not configured:
        return False
    normalized_host = _normalize_host(hostname)
    host_repr = f"[{normalized_host}]" if ":" in normalized_host else normalized_host
    candidates: Set[str] = {host_repr, f"{host_repr}:{effective_port}"}
    allowlist: Set[str] = {_normalize_host(entry) for entry in configured if entry}
    return bool(candidates & allowlist)


def validate_url(url: str) -> Tuple[str, str]:
    """
    Validate a user-supplied URL and rewrite it to connect to a validated IP.

    Resolves the hostname, checks all resolved IPs against blocked networks,
    then returns a rewritten URL that points to the validated IP along with
    the original hostname (for use in the Host header).

    This eliminates DNS rebinding because the caller connects to the IP we
    validated, not the hostname that could rebind. Callers should also disable
    follow_redirects to prevent redirect-based SSRF bypasses.

    Args:
        url: The user-supplied URL to validate.

    Returns:
        Tuple of (rewritten_url, host_header).
        The rewritten URL has the hostname replaced with the validated IP.
        The host_header value should be sent as the Host header.

    Raises:
        SSRFError: If the URL scheme is invalid or the hostname resolves
            to a private/internal IP address.
    """
    parsed = urlparse(url)

    if parsed.scheme not in _ALLOWED_SCHEMES:
        raise SSRFError(f"URL scheme '{parsed.scheme}' is not allowed")

    hostname = parsed.hostname
    if not hostname:
        raise SSRFError("URL has no hostname")

    port = parsed.port
    default_port = _default_port_for_scheme(parsed.scheme)
    effective_port = port if port is not None else default_port
    host_header = _format_host_header(hostname, effective_port, default_port)

    is_allowlisted = _is_host_allowlisted(hostname, effective_port)

    # Resolve hostname and validate ALL addresses
    try:
        addrinfo = socket.getaddrinfo(
            hostname, effective_port, proto=socket.IPPROTO_TCP
        )
    except socket.gaierror as e:
        raise SSRFError(f"DNS resolution failed for '{hostname}': {e}")

    if not addrinfo:
        raise SSRFError(f"No addresses found for '{hostname}'")

    if not is_allowlisted:
        for family, type_, proto, canonname, sockaddr in addrinfo:
            resolved_ip = _sockaddr_host(sockaddr)
            if _is_blocked_ip(resolved_ip):
                raise SSRFError(
                    f"URL targets a blocked address ({resolved_ip}). "
                    "If this is a legitimate internal service, add the host "
                    "to `user_url_allowed_hosts` in general_settings."
                )

    # For HTTPS with SSL verification enabled, TLS certificate validation
    # binds the connection to the hostname — DNS rebinding can't redirect
    # to a different server because the cert wouldn't match.
    # When SSL verification is disabled, this defense doesn't apply, so
    # we rewrite to the validated IP like HTTP.
    ssl_verify = getattr(litellm, "ssl_verify", True)
    if parsed.scheme == "https" and ssl_verify is not False:
        return url, host_header

    # For HTTP, rewrite URL to connect to the validated IP directly
    # to prevent DNS rebinding (no TLS to bind the connection).
    validated_ip = _sockaddr_host(addrinfo[0][4])
    is_ipv6 = addrinfo[0][0] == socket.AF_INET6
    ip_host = f"[{validated_ip}]" if is_ipv6 else validated_ip

    if port is not None:
        new_netloc = f"{ip_host}:{port}"
    else:
        new_netloc = ip_host

    rewritten = urlunparse(
        (parsed.scheme, new_netloc, parsed.path, parsed.params, parsed.query, "")
    )

    return rewritten, host_header


def assert_same_origin(candidate_url: str, expected_url: str) -> None:
    """Verify ``candidate_url`` shares scheme, host, and port with ``expected_url``.

    Use when an upstream API returns a URL meant for follow-up requests
    (e.g. an async-job polling URL that will be hit with the operator's
    API key in the headers). The upstream is trusted because the operator
    configured ``api_base``, but the URL it hands back must actually point
    back at the same origin or we'd be blindly forwarding credentials
    wherever the upstream told us to.

    Hostnames are compared case-insensitively. Default ports are made
    explicit (HTTP→80, HTTPS→443) so ``https://api.example.com:443/...``
    and ``https://api.example.com/...`` are treated as the same origin.

    Error messages identify *which* component mismatched but never echo
    the operator's ``expected`` host or the candidate's hostname back to
    the caller — in the SSRF threat model the caller is the attacker,
    and reflecting host info would be a secondary leak of operator
    infrastructure details.
    """
    candidate = urlparse(candidate_url)
    expected = urlparse(expected_url)

    if candidate.scheme not in _ALLOWED_SCHEMES:
        raise SSRFError("URL scheme is not allowed")

    if candidate.scheme != expected.scheme:
        raise SSRFError("Origin mismatch on scheme")

    candidate_host = _normalize_host(candidate.hostname or "")
    expected_host = _normalize_host(expected.hostname or "")
    if not candidate_host or candidate_host != expected_host:
        raise SSRFError("Origin mismatch on host")

    default_port = 443 if candidate.scheme == "https" else 80
    candidate_port = candidate.port if candidate.port is not None else default_port
    expected_port = expected.port if expected.port is not None else default_port
    if candidate_port != expected_port:
        raise SSRFError("Origin mismatch on port")


_MAX_REDIRECTS = 10


def _extract_redirect_url(response: Any, request_url: str) -> str:
    """Extract and resolve the redirect target from a response's Location header."""
    location = response.headers.get("location")
    if not isinstance(location, str) or not location:
        raise SSRFError("Redirect response has no Location header")
    # Resolve relative URLs against the request URL
    return str(httpx.URL(request_url).join(location))


def safe_get(client: Any, url: str, **kwargs: Any) -> Any:
    """
    Fetch a user-supplied URL with SSRF protection on every redirect hop.

    Validates the initial URL and each redirect target before making the
    request. No DNS rebinding (resolve-and-rewrite). No redirect bypass
    (each hop validated). No breaking change for legitimate CDN redirects.

    When ``litellm.user_url_validation`` is False, validation is bypassed
    and this function delegates to ``client.get(url, follow_redirects=True)``.

    Args:
        client: An httpx.Client (sync).
        url: The user-supplied URL.
        **kwargs: Additional kwargs passed to client.get().

    Returns:
        The final httpx.Response.
    """
    if not getattr(litellm, "user_url_validation", True):
        kwargs.setdefault("follow_redirects", True)
        return client.get(url, **kwargs)
    kwargs.pop("follow_redirects", None)
    caller_headers = kwargs.pop("headers", {})
    for _ in range(_MAX_REDIRECTS):
        validated_url, original_host = validate_url(url)
        response = client.get(
            validated_url,
            headers={**caller_headers, "Host": original_host},
            follow_redirects=False,
            **kwargs,
        )
        if not response.is_redirect:
            return response
        # Resolve the next hop against the ORIGINAL (pre-rewrite) URL so
        # relative Location headers keep the original hostname.
        url = _extract_redirect_url(response, url)
    raise SSRFError("Too many redirects")


async def async_safe_get(client: Any, url: str, **kwargs: Any) -> Any:
    """Async version of safe_get."""
    if not getattr(litellm, "user_url_validation", True):
        kwargs.setdefault("follow_redirects", True)
        return await client.get(url, **kwargs)
    kwargs.pop("follow_redirects", None)
    caller_headers = kwargs.pop("headers", {})
    for _ in range(_MAX_REDIRECTS):
        validated_url, original_host = validate_url(url)
        response = await client.get(
            validated_url,
            headers={**caller_headers, "Host": original_host},
            follow_redirects=False,
            **kwargs,
        )
        if not response.is_redirect:
            return response
        # Resolve the next hop against the ORIGINAL (pre-rewrite) URL so
        # relative Location headers keep the original hostname.
        url = _extract_redirect_url(response, url)
    raise SSRFError("Too many redirects")
