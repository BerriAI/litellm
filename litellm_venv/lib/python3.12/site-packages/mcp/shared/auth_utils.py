"""Utilities for OAuth 2.0 Resource Indicators (RFC 8707)."""

from urllib.parse import urlparse, urlsplit, urlunsplit

from pydantic import AnyUrl, HttpUrl


def resource_url_from_server_url(url: str | HttpUrl | AnyUrl) -> str:
    """Convert server URL to canonical resource URL per RFC 8707.

    RFC 8707 section 2 states that resource URIs "MUST NOT include a fragment component".
    Returns absolute URI with lowercase scheme/host for canonical form.

    Args:
        url: Server URL to convert

    Returns:
        Canonical resource URL string
    """
    # Convert to string if needed
    url_str = str(url)

    # Parse the URL and remove fragment, create canonical form
    parsed = urlsplit(url_str)
    canonical = urlunsplit(
        parsed._replace(
            scheme=parsed.scheme.lower(), netloc=parsed.netloc.lower(), fragment=""
        )
    )

    return canonical


def check_resource_allowed(requested_resource: str, configured_resource: str) -> bool:
    """Check if a requested resource URL matches a configured resource URL.

    A requested resource matches if it has the same scheme, domain, port,
    and its path starts with the configured resource's path. This allows
    hierarchical matching where a token for a parent resource can be used
    for child resources.

    Args:
        requested_resource: The resource URL being requested
        configured_resource: The resource URL that has been configured

    Returns:
        True if the requested resource matches the configured resource
    """
    # Parse both URLs
    requested = urlparse(requested_resource)
    configured = urlparse(configured_resource)

    # Compare scheme, host, and port (origin)
    if (
        requested.scheme.lower() != configured.scheme.lower()
        or requested.netloc.lower() != configured.netloc.lower()
    ):
        return False

    # Handle cases like requested=/foo and configured=/foo/
    requested_path = requested.path
    configured_path = configured.path

    # If requested path is shorter, it cannot be a child
    if len(requested_path) < len(configured_path):
        return False

    # Check if the requested path starts with the configured path
    # Ensure both paths end with / for proper comparison
    # This ensures that paths like "/api123" don't incorrectly match "/api"
    if not requested_path.endswith("/"):
        requested_path += "/"
    if not configured_path.endswith("/"):
        configured_path += "/"

    return requested_path.startswith(configured_path)
