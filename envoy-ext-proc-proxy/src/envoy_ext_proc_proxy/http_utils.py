"""HTTP utility functions for header processing, error classification, and client session creation."""

import asyncio
import ssl
from collections.abc import Mapping

import aiohttp
import multidict
from aiohttp import ClientTimeout

from envoy_ext_proc_proxy.config import ProxyConfig

# Standard Hop-by-Hop headers defined in RFC 2616 / RFC 7230 / RFC 9110
HOP_BY_HOP_HEADERS: set[str] = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}

# Header used to correlate ext_proc sessions with HTTP proxy requests
REQUEST_ID_HEADER: str = "x-ai-proxy-request-id"

# Headers that must not be propagated per proxy requirements
EXCLUDED_REQUEST_HEADERS: set[str] = {
    "x-forwarded-for",
    "x-forwarded-host",
    "x-forwarded-proto",
    REQUEST_ID_HEADER,
}


def _get_connection_tokens(headers: Mapping[str, str]) -> set[str]:
    """Extract token names specified in the Connection header."""
    tokens: set[str] = set()
    if isinstance(headers, (multidict.CIMultiDictProxy, multidict.CIMultiDict)):
        connection_values = headers.getall("connection", [])
    else:
        connection_values = [v for k, v in headers.items() if k.lower() == "connection"]

    for val in connection_values:
        for token in val.split(","):
            token_clean = token.strip().lower()
            if token_clean:
                tokens.add(token_clean)

    return tokens


def filter_request_headers(
    headers: Mapping[str, str] | multidict.CIMultiDictProxy | multidict.CIMultiDict,
) -> multidict.CIMultiDict:
    """Filter pseudo-headers, hop-by-hop headers, and excluded headers from incoming request headers."""
    filtered = multidict.CIMultiDict()
    connection_tokens = _get_connection_tokens(headers)
    strip_set = HOP_BY_HOP_HEADERS | connection_tokens | EXCLUDED_REQUEST_HEADERS

    for key, value in headers.items():
        key_lower = key.lower()
        if not key_lower.startswith(":") and key_lower not in strip_set:
            filtered.add(key, value)
    return filtered


def filter_response_headers(
    headers: Mapping[str, str] | multidict.CIMultiDictProxy | multidict.CIMultiDict,
) -> multidict.CIMultiDict:
    """Filter pseudo-headers and hop-by-hop headers from upstream response headers."""
    filtered = multidict.CIMultiDict()
    connection_tokens = _get_connection_tokens(headers)
    strip_set = HOP_BY_HOP_HEADERS | connection_tokens

    for key, value in headers.items():
        key_lower = key.lower()
        if not key_lower.startswith(":") and key_lower not in strip_set:
            filtered.add(key, value)
    return filtered


def is_bodyless_response(method: str, status: int) -> bool:
    """Return True if the HTTP response must not contain a body."""
    return method.upper() == "HEAD" or status in (204, 304)


def classify_upstream_error(err: Exception) -> tuple[int, str]:
    """Map upstream exception to HTTP status code and descriptive error message."""
    err_desc = str(err) or repr(err)

    if isinstance(err, (ssl.SSLCertVerificationError, aiohttp.ClientSSLError)):
        return (
            502,
            f"502 Bad Gateway: Upstream TLS certificate verification failed ({err_desc})",
        )
    elif isinstance(err, aiohttp.ClientConnectorError):
        return (
            502,
            f"502 Bad Gateway: Failed to connect to upstream ({err_desc})",
        )
    elif isinstance(err, (asyncio.TimeoutError, TimeoutError)):
        return (
            504,
            "504 Gateway Timeout: Upstream server timed out",
        )
    elif isinstance(err, aiohttp.ClientError):
        return (
            502,
            f"502 Bad Gateway: Upstream client error ({err_desc})",
        )
    else:
        return (
            502,
            f"502 Bad Gateway: Upstream error ({err_desc})",
        )


def create_client_session(
    config: ProxyConfig,
    upstream_ssl_context: ssl.SSLContext | None = None,
) -> aiohttp.ClientSession:
    """Create and configure an aiohttp.ClientSession for upstream proxy requests."""
    if upstream_ssl_context is None:
        upstream_ssl_context = ssl.create_default_context()

    connector = aiohttp.TCPConnector(ssl=upstream_ssl_context)
    timeout = ClientTimeout(total=config.upstream_timeout)
    return aiohttp.ClientSession(
        connector=connector,
        timeout=timeout,
        # Prevent aiohttp from automatically decompressing gzip content.
        # It keeps "Content-Encoding: gzip" header intact, and this confuses
        # some clients, notably LiteLLM itself.
        auto_decompress=False,
    )
