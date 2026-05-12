"""
MCP Server Utilities
"""

from typing import Any, Dict, Iterator, Mapping, Optional, Tuple

import hashlib
import importlib
import os

# Constants
LITELLM_MCP_SERVER_NAME = "litellm-mcp-server"
LITELLM_MCP_SERVER_VERSION = "1.0.0"
LITELLM_MCP_SERVER_DESCRIPTION = "MCP Server for LiteLLM"
MCP_TOOL_PREFIX_SEPARATOR = os.environ.get("MCP_TOOL_PREFIX_SEPARATOR", "-")
MCP_TOOL_PREFIX_FORMAT = "{server_name}{separator}{tool_name}"

# ---------------------------------------------------------------------------
# Short-ID tool prefix (opt-in)
# ---------------------------------------------------------------------------
# When LITELLM_USE_SHORT_MCP_TOOL_PREFIX is truthy the prefix attached to MCP
# tool / prompt / resource / resource-template names switches from the
# (potentially long) human-readable server name to a deterministic three
# character ID derived from the server's ``server_id``.
#
# Why three characters?
#   * The first character is restricted to 52 alphabetic characters
#     ([A-Za-z]) and the remaining two characters use the full base62
#     alphabet ([0-9A-Za-z]).  That guarantees the prefix never starts
#     with a digit so it remains a valid identifier for every model API
#     (some providers historically required a leading alphabetic char).
#   * 52 * 62 * 62 = 199_888 distinct IDs.  The chance of a real local
#     tool name happening to begin with the exact prefix LiteLLM assigned
#     to a given MCP server is negligible in practice.
#   * The IDs are short enough that prefixed tool names stay well under
#     the 60-character upper bound enforced by some model APIs (Anthropic
#     etc.) even for long upstream tool names.
#   * The mapping is deterministic (SHA-256 of ``server_id`` → three
#     characters drawn from the alphabets above), so the prefix is stable
#     across processes, workers and restarts without any persistence
#     layer.  Two servers with different ``server_id`` values can in
#     principle hash to the same three chars; that natural-hash collision
#     IS a routing-correctness issue (the second registrant would otherwise
#     have its tools misrouted to the first), so registration goes through
#     ``MCPServerManager._assign_unique_short_prefix`` which rehashes with
#     a deterministic attempt counter until it finds an unused prefix and
#     caches the result on ``MCPServer.short_prefix``.  A collision is
#     logged at INFO when it happens.
#
# This flag is intentionally opt-in for the first release so customers can
# migrate.  It will become the default in a future release.
SHORT_MCP_TOOL_PREFIX_LENGTH = 3
_BASE62_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
# Subset of _BASE62_ALPHABET used for the *first* character only, to
# guarantee the prefix never starts with a digit.
_BASE52_ALPHA_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


def is_short_mcp_tool_prefix_enabled() -> bool:
    """Return True when the short-ID tool prefix mode is enabled.

    Read at call time (not import time) so tests and runtime config changes
    take effect without reimporting the module.
    """
    raw = os.environ.get("LITELLM_USE_SHORT_MCP_TOOL_PREFIX", "")
    return raw.strip().lower() in ("1", "true", "yes", "on")


def compute_short_server_prefix(server_id: str, attempt: int = 0) -> str:
    """Derive the deterministic three-character prefix for a server.

    Uses SHA-256 of ``f"{server_id}#{attempt}"`` and folds the first eight
    bytes into a fixed-length string whose first character is drawn from
    ``_BASE52_ALPHA_ALPHABET`` (so the prefix never starts with a digit)
    and whose remaining characters are drawn from the full base62
    alphabet.  Pass ``attempt > 0`` to rehash to a different prefix when
    the natural hash collides with a prefix already assigned to another
    server (see ``MCPServerManager._assign_unique_short_prefix``).  An
    empty ``server_id`` raises ``ValueError`` — short prefixes require a
    stable identifier to be deterministic.
    """
    if not server_id:
        raise ValueError("compute_short_server_prefix requires a non-empty server_id")

    seed = server_id if attempt == 0 else f"{server_id}#{attempt}"
    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big")

    # Build chars from least-significant to most-significant; we reverse
    # at the end so the first emitted char comes from the high-order
    # bits of the digest (which is the position we constrain to be
    # alphabetic).
    chars = []
    for position in range(SHORT_MCP_TOOL_PREFIX_LENGTH):
        is_first_char = position == SHORT_MCP_TOOL_PREFIX_LENGTH - 1
        alphabet = _BASE52_ALPHA_ALPHABET if is_first_char else _BASE62_ALPHABET
        value, idx = divmod(value, len(alphabet))
        chars.append(alphabet[idx])
    return "".join(reversed(chars))


def is_mcp_available() -> bool:
    """
    Returns True if the MCP module is available, False otherwise
    """
    try:
        importlib.import_module("mcp")
        return True
    except ImportError:
        return False


def normalize_server_name(server_name: str) -> str:
    """
    Normalize server name by replacing spaces with underscores
    """
    return server_name.replace(" ", "_")


def validate_and_normalize_mcp_server_payload(payload: Any) -> None:
    """
    Validate and normalize MCP server payload fields (server_name and alias).

    This function:
    1. Validates that server_name and alias don't contain the MCP_TOOL_PREFIX_SEPARATOR
    2. Normalizes alias by replacing spaces with underscores
    3. Sets default alias if not provided (using server_name as base)

    Args:
        payload: The payload object containing server_name and alias fields

    Raises:
        HTTPException: If validation fails
    """
    # Server name validation: disallow '-'
    if hasattr(payload, "server_name") and payload.server_name:
        validate_mcp_server_name(payload.server_name, raise_http_exception=True)

    # Alias validation: disallow '-'
    if hasattr(payload, "alias") and payload.alias:
        validate_mcp_server_name(payload.alias, raise_http_exception=True)

    # Alias normalization and defaulting
    alias = getattr(payload, "alias", None)
    server_name = getattr(payload, "server_name", None)

    if not alias and server_name:
        alias = normalize_server_name(server_name)
    elif alias:
        alias = normalize_server_name(alias)

    # Update the payload with normalized alias
    if hasattr(payload, "alias"):
        payload.alias = alias


def add_server_prefix_to_name(name: str, server_name: str) -> str:
    """Add server name prefix to any MCP resource name."""
    formatted_server_name = normalize_server_name(server_name)

    return MCP_TOOL_PREFIX_FORMAT.format(
        server_name=formatted_server_name,
        separator=MCP_TOOL_PREFIX_SEPARATOR,
        tool_name=name,
    )


def get_server_prefix(server: Any) -> str:
    """Return the prefix for a server.

    When the short-prefix mode is enabled (``LITELLM_USE_SHORT_MCP_TOOL_PREFIX``)
    a three-character base62 ID is returned.  We prefer the cached
    ``server.short_prefix`` value when set — that field is populated at
    registration time by ``MCPServerManager._assign_unique_short_prefix``
    and resolves natural-hash collisions deterministically — and only fall
    back to the natural hash for ad-hoc / temp-server objects without a
    cached value.  In default mode the historical behaviour is preserved:
    alias if present, else server_name, else server_id.
    """
    if is_short_mcp_tool_prefix_enabled():
        cached = getattr(server, "short_prefix", None)
        if cached:
            return cached
        server_id = getattr(server, "server_id", None)
        if server_id:
            return compute_short_server_prefix(server_id)

    if hasattr(server, "alias") and server.alias:
        return server.alias
    if hasattr(server, "server_name") and server.server_name:
        return server.server_name
    if hasattr(server, "server_id"):
        return server.server_id
    return ""


def iter_known_server_prefixes(server: Any) -> Iterator[str]:
    """Yield every prefix form that may appear in tool names for ``server``.

    Always includes the *current* prefix returned by ``get_server_prefix``.
    Additionally yields the historical (alias / server_name / server_id) and
    short-ID forms so the routing layer can resolve tool names regardless of
    which prefix mode was active when the client first observed them.
    """
    seen = set()

    def _emit(value: Optional[str]) -> Iterator[str]:
        if value and value not in seen:
            seen.add(value)
            yield value

    yield from _emit(get_server_prefix(server))
    yield from _emit(getattr(server, "short_prefix", None))

    server_id = getattr(server, "server_id", None)
    if server_id:
        try:
            yield from _emit(compute_short_server_prefix(server_id))
        except ValueError:
            pass

    yield from _emit(getattr(server, "alias", None))
    yield from _emit(getattr(server, "server_name", None))
    yield from _emit(server_id)


def split_server_prefix_from_name(prefixed_name: str) -> Tuple[str, str]:
    """Return the unprefixed name plus the server name used as prefix."""
    if MCP_TOOL_PREFIX_SEPARATOR in prefixed_name:
        parts = prefixed_name.split(MCP_TOOL_PREFIX_SEPARATOR, 1)
        if len(parts) == 2:
            return parts[1], parts[0]
    return prefixed_name, ""


def is_tool_name_prefixed(
    tool_name: str,
    known_server_prefixes: Optional[set] = None,
) -> bool:
    """
    Check if tool name has a known MCP server prefix.

    When ``known_server_prefixes`` is provided the function verifies that the
    substring before the first separator is an actual registered server
    prefix.  Without it the check falls back to the legacy heuristic
    (separator present anywhere in the name), which can produce false
    positives for non-MCP tools whose names contain hyphens
    (e.g. ``text-to-speech``, ``code-review``).

    Args:
        tool_name: Tool name to check.
        known_server_prefixes: Optional set of normalised server prefixes
            currently registered in the MCP manager.  Pass this whenever
            the caller has access to the server registry so that the check
            is accurate.

    Returns:
        True if tool name is prefixed, False otherwise.
    """
    if MCP_TOOL_PREFIX_SEPARATOR not in tool_name:
        return False

    if known_server_prefixes is not None:
        candidate_prefix = tool_name.split(MCP_TOOL_PREFIX_SEPARATOR, 1)[0]
        return normalize_server_name(candidate_prefix) in known_server_prefixes

    # Legacy fallback – separator present somewhere in the name.
    return True


def validate_mcp_server_name(
    server_name: str, raise_http_exception: bool = False
) -> None:
    """
    Validate that MCP server name does not contain 'MCP_TOOL_PREFIX_SEPARATOR'.

    Args:
        server_name: The server name to validate
        raise_http_exception: If True, raises HTTPException instead of generic Exception

    Raises:
        Exception or HTTPException: If server name contains 'MCP_TOOL_PREFIX_SEPARATOR'
    """
    if server_name and MCP_TOOL_PREFIX_SEPARATOR in server_name:
        error_message = f"Server name cannot contain '{MCP_TOOL_PREFIX_SEPARATOR}'. Use an alternative character instead Found: {server_name}"
        if raise_http_exception:
            from fastapi import HTTPException
            from starlette import status

            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail={"error": error_message}
            )
        else:
            raise Exception(error_message)


def merge_mcp_headers(
    *,
    extra_headers: Optional[Mapping[str, str]] = None,
    static_headers: Optional[Mapping[str, str]] = None,
) -> Optional[Dict[str, str]]:
    """Merge outbound HTTP headers for MCP calls.

    This is used when calling out to external MCP servers (or OpenAPI-based MCP tools).

    Merge rules:
    - Start with `extra_headers` (typically OAuth2-derived headers)
    - Overlay `static_headers` (user-configured per MCP server)

    If both contain the same key, `static_headers` wins. This matches the existing
    behavior in `MCPServerManager` where `server.static_headers` is applied after
    any caller-provided headers.
    """
    merged: Dict[str, str] = {}

    if extra_headers:
        merged.update({str(k): str(v) for k, v in extra_headers.items()})

    if static_headers:
        merged.update({str(k): str(v) for k, v in static_headers.items()})

    return merged or None
