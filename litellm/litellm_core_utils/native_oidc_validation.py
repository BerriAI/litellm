"""
Native OIDC validation primitives.

Shared by the proxy discovery endpoint (which publishes the issuer/client/scopes
trust anchor) and the `lite` CLI (which consumes it). Intentionally depends on
nothing but the standard library so it stays importable from both the proxy
runtime and the thin ``litellm[cli]`` installation.
"""

import ipaddress
from collections.abc import Iterable, Sequence
from urllib.parse import urlsplit

# RFC 6749 appendix A: NQCHAR = %x21 / %x23-5B / %x5D-7E
# i.e. printable ASCII excluding SPACE (%x20), DQUOTE (%x22) and BACKSLASH (%x5C).
# Both scope-token (section 3.3) and the `error` code (section 5.2) are 1*NQCHAR.
NQCHAR_ALLOWED_CHARACTERS = frozenset(chr(code) for code in range(0x21, 0x7F) if code not in (0x22, 0x5C))

PROVIDER_CONFIGURATION_PATH = "/.well-known/openid-configuration"

# The only schemes an issuer or endpoint may use. Plaintext http is accepted here
# and narrowed to numeric loopback by the caller.
URL_SCHEMES = frozenset(("http", "https"))


def has_control_characters(value: str) -> bool:
    """True when ``value`` contains a C0 or C1 control character, or DEL."""
    return any(ord(character) < 32 or 127 <= ord(character) <= 159 for character in value)


def is_numeric_loopback_host(hostname: str) -> bool:
    """True only for numeric loopback literals such as ``127.0.0.1`` or ``::1``.

    Deliberately does not resolve DNS and does not accept ``localhost``: an
    attacker-controlled name must never widen the plaintext-HTTP exception.
    """
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def is_valid_nqchar_string(value: str) -> bool:
    """True when ``value`` is a non-empty RFC 6749 NQCHAR string.

    Control characters, including the ESC that starts an ANSI or OSC escape
    sequence, fall outside NQCHAR, so a value that passes this check is safe to
    interpolate into terminal output.
    """
    return bool(value) and all(character in NQCHAR_ALLOWED_CHARACTERS for character in value)


def is_printable_ascii(value: str) -> bool:
    """True when ``value`` is non-empty and made only of printable ASCII.

    Wider than :func:`is_valid_nqchar_string` because it admits SPACE, DQUOTE
    and BACKSLASH, but still excludes every control character, so the result is
    safe to echo to a terminal.
    """
    return bool(value) and all(0x20 <= ord(character) <= 0x7E for character in value)


def is_valid_scope_token(value: str) -> bool:
    """True when ``value`` is a well-formed RFC 6749 scope-token."""
    return is_valid_nqchar_string(value)


def validate_scope_tokens(scopes: Iterable[str]) -> tuple[str, ...]:
    """Validate each scope independently, rejecting duplicates and preserving order.

    Raises ValueError with a message that never echoes the offending value.
    """
    validated = tuple(scopes)
    for scope in validated:
        if not isinstance(scope, str) or not is_valid_scope_token(scope):
            raise ValueError("must contain only RFC 6749 scope-tokens")
    if len(frozenset(validated)) != len(validated):
        raise ValueError("must not contain duplicate scopes")
    if not validated:
        raise ValueError("must contain at least one scope")
    return validated


def _validate_url_shape(value: str, *, allow_query: bool) -> None:
    """Shared safety checks for issuer identifiers and provider endpoints."""
    if not isinstance(value, str) or not value:
        raise ValueError("must be a non-empty string")
    if value != value.strip():
        raise ValueError("must not have leading or trailing whitespace")
    if has_control_characters(value) or any(character.isspace() for character in value):
        raise ValueError("must not contain whitespace or control characters")

    try:
        parsed = urlsplit(value)
        _ = parsed.port
    except ValueError as error:
        raise ValueError("must contain a valid port") from error

    if parsed.scheme not in URL_SCHEMES:
        raise ValueError("must be an absolute http(s) URL")
    if not parsed.hostname:
        raise ValueError("must contain a host")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("must not contain credentials")
    if parsed.fragment:
        raise ValueError("must not contain a fragment")
    if not allow_query and parsed.query:
        raise ValueError("must not contain a query component")
    if parsed.scheme == "http" and not is_numeric_loopback_host(parsed.hostname):
        raise ValueError("must use HTTPS unless the host is a numeric loopback address")


def validate_issuer(value: str) -> str:
    """Validate an OIDC issuer identifier and return it byte-for-byte unchanged.

    The issuer is a trust anchor compared by exact string equality against the
    provider document, so this must never normalize case, ports, percent
    encoding or trailing slashes.
    """
    _validate_url_shape(value, allow_query=False)
    return value


def validate_endpoint_url(value: str) -> str:
    """Validate a provider endpoint URL and return it unchanged.

    Unlike the issuer, standards-compliant endpoints may legitimately carry a
    query component, and may live on a different host than the issuer.
    """
    _validate_url_shape(value, allow_query=True)
    return value


def derive_provider_configuration_url(issuer: str) -> str:
    """Derive the OpenID Provider Configuration URL from an issuer identifier.

    Per OpenID Connect Discovery 1.0, ``/.well-known/openid-configuration`` is
    appended to the issuer with any single terminating slash removed first. No
    other normalization is applied -- the issuer string itself remains the
    trust anchor.
    """
    return issuer.removesuffix("/") + PROVIDER_CONFIGURATION_PATH


def is_trusted_metadata_origin(base_url: str) -> bool:
    """True when native OIDC bootstrap metadata may be trusted from ``base_url``.

    HTTPS anywhere, or plaintext HTTP only against a numeric loopback address
    for local development. Provider metadata must never be picked up from an
    arbitrary remote plaintext channel.
    """
    try:
        parsed = urlsplit(base_url)
        _ = parsed.port
    except ValueError:
        return False
    if not parsed.hostname:
        return False
    if parsed.scheme == "https":
        return True
    return parsed.scheme == "http" and is_numeric_loopback_host(parsed.hostname)


def format_scopes(scopes: Sequence[str]) -> str:
    """Join already-validated scopes with a single ASCII space."""
    return " ".join(scopes)
