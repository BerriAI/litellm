"""
URL validation for user-controlled URLs.

Use validate_url() before fetching any URL that originates from user
input (image_url, file_url, spec_path, etc.) to prevent SSRF attacks.

validate_url() resolves DNS once, validates all IPs, and rewrites the
URL to connect to the validated IP directly — no TOCTOU gap, no DNS
rebinding. Redirects are followed manually with validation at each hop.
"""

import socket
from ipaddress import ip_address, ip_network
from typing import Any, Tuple
from urllib.parse import urlparse, urlunparse

import httpx

import litellm

_BLOCKED_NETWORKS = [
    ip_network("0.0.0.0/8"),
    ip_network("10.0.0.0/8"),
    ip_network("100.64.0.0/10"),
    ip_network("127.0.0.0/8"),
    ip_network("169.254.0.0/16"),
    ip_network("172.16.0.0/12"),
    ip_network("192.0.0.0/24"),
    ip_network("192.168.0.0/16"),
    ip_network("198.18.0.0/15"),
    ip_network("::1/128"),
    ip_network("fc00::/7"),
    ip_network("fe80::/10"),
]

_ALLOWED_SCHEMES = ("http", "https")


class SSRFError(ValueError):
    """Raised when a URL targets a blocked network."""

    pass


def _is_blocked_ip(addr: str) -> bool:
    try:
        ip = ip_address(addr)
    except ValueError:
        return True  # fail-closed: unparseable addresses are blocked
    if ip.version == 6 and hasattr(ip, "ipv4_mapped") and ip.ipv4_mapped:
        ip = ip.ipv4_mapped
    return any(ip in net for net in _BLOCKED_NETWORKS)


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
        Tuple of (rewritten_url, original_hostname).
        The rewritten URL has the hostname replaced with the validated IP.
        The original hostname should be set as the Host header.

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
    default_port = 443 if parsed.scheme == "https" else 80

    # Build the Host header value — include port when non-default
    host_header = (
        hostname if (port is None or port == default_port) else f"{hostname}:{port}"
    )

    # Resolve hostname and validate ALL addresses
    try:
        addrinfo = socket.getaddrinfo(
            hostname, port or default_port, proto=socket.IPPROTO_TCP
        )
    except socket.gaierror as e:
        raise SSRFError(f"DNS resolution failed for '{hostname}': {e}")

    if not addrinfo:
        raise SSRFError(f"No addresses found for '{hostname}'")

    for family, type_, proto, canonname, sockaddr in addrinfo:
        if _is_blocked_ip(sockaddr[0]):
            raise SSRFError(
                f"URL targets a blocked address ({sockaddr[0]}). "
                "If this is a legitimate internal service, use a direct "
                "provider configuration instead of a user-supplied URL."
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
    validated_ip = addrinfo[0][4][0]
    is_ipv6 = addrinfo[0][0] == socket.AF_INET6
    ip_host = f"[{validated_ip}]" if is_ipv6 else validated_ip

    if port:
        new_netloc = f"{ip_host}:{port}"
    else:
        new_netloc = ip_host

    rewritten = urlunparse(
        (parsed.scheme, new_netloc, parsed.path, parsed.params, parsed.query, "")
    )

    return rewritten, host_header


_MAX_REDIRECTS = 10


def _extract_redirect_url(response: Any, request_url: str) -> str:
    """Extract and resolve the redirect target from a response's Location header."""
    location = response.headers.get("location")
    if not location:
        raise SSRFError("Redirect response has no Location header")
    # Resolve relative URLs against the request URL
    return str(httpx.URL(request_url).join(location))


def safe_get(client: Any, url: str, **kwargs: Any) -> Any:
    """
    Fetch a user-supplied URL with SSRF protection on every redirect hop.

    Validates the initial URL and each redirect target before making the
    request. No DNS rebinding (resolve-and-rewrite). No redirect bypass
    (each hop validated). No breaking change for legitimate CDN redirects.

    Args:
        client: An httpx.Client (sync).
        url: The user-supplied URL.
        **kwargs: Additional kwargs passed to client.get().

    Returns:
        The final httpx.Response.
    """
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
        url = _extract_redirect_url(response, validated_url)
    raise SSRFError("Too many redirects")


async def async_safe_get(client: Any, url: str, **kwargs: Any) -> Any:
    """Async version of safe_get."""
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
        url = _extract_redirect_url(response, validated_url)
    raise SSRFError("Too many redirects")
