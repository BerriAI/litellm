"""
MCP Server Utilities
"""

import hashlib
import importlib
import json
import os
import re
import typing
from collections.abc import Iterable, Iterator, Mapping, MutableMapping, MutableSequence
from collections.abc import Set as AbstractSet
from typing import Any, Final, Protocol
from urllib.parse import quote

from litellm.types.mcp_server.mcp_server_manager import MCPServer

if typing.TYPE_CHECKING:
    from fastapi import Request


class _McpServerLike(Protocol):
    @property
    def server_id(self) -> str: ...
    @property
    def server_name(self) -> str | None: ...
    @property
    def alias(self) -> str | None: ...
    @property
    def short_prefix(self) -> str | None: ...


class McpServerPayloadLike(Protocol):
    alias: str | None

    @property
    def server_name(self) -> str | None: ...
    @property
    def tool_name_to_display_name(self) -> Mapping[str, str] | None: ...


# Constants
#
# NOTE: The environment-backed values below are read once, when this module is
# first imported, and cached for the lifetime of the process. Changing the
# corresponding environment variables after import has no effect unless the
# module is reloaded (e.g. ``importlib.reload``). Tests that override these
# variables must reload this module — see
# ``tests/test_litellm/proxy/_experimental/mcp_server/test_mcp_server_identity_env.py``.
LITELLM_MCP_SERVER_NAME: Final = os.environ.get("LITELLM_MCP_SERVER_NAME", "litellm-mcp-server")
LITELLM_MCP_SERVER_VERSION: Final = "1.0.0"
LITELLM_MCP_SERVER_DESCRIPTION: Final = os.environ.get("LITELLM_MCP_SERVER_DESCRIPTION", "MCP Server for LiteLLM")
MCP_TOOL_PREFIX_SEPARATOR: Final = os.environ.get("MCP_TOOL_PREFIX_SEPARATOR", "-")
MCP_TOOL_PREFIX_FORMAT: Final = "{server_name}{separator}{tool_name}"

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
SHORT_MCP_TOOL_PREFIX_LENGTH: Final = 3
_BASE62_ALPHABET: Final = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
# Subset of _BASE62_ALPHABET used for the *first* character only, to
# guarantee the prefix never starts with a digit.
_BASE52_ALPHA_ALPHABET: Final = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


def is_short_mcp_tool_prefix_enabled() -> bool:
    """Return True when the short-ID tool prefix mode is enabled.

    Read at call time (not import time) so tests and runtime config changes
    take effect without reimporting the module.
    """
    raw: Final = os.environ.get("LITELLM_USE_SHORT_MCP_TOOL_PREFIX", "")
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

    seed: Final = server_id if attempt == 0 else f"{server_id}#{attempt}"
    digest: Final = hashlib.sha256(seed.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big")

    # Build chars from least-significant to most-significant; we reverse
    # at the end so the first emitted char comes from the high-order
    # bits of the digest (which is the position we constrain to be
    # alphabetic).
    chars: Final[list[str]] = []
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


_MCP_ALIAS_HEADER_INVALID_RE: Final = re.compile(r"[^a-z0-9_]")


def sanitize_mcp_alias_for_header(alias: str) -> str:
    """
    Sanitize an MCP server alias for x-mcp-{alias}-{header} HTTP headers.

    Must stay in sync with ui/litellm-dashboard/src/utils/mcpHeaderUtils.ts.
    """
    sanitized = _MCP_ALIAS_HEADER_INVALID_RE.sub("_", alias.lower().strip())
    sanitized = re.sub(r"_+", "_", sanitized)
    return sanitized.strip("_")


def lookup_mcp_server_auth_in_headers(
    mcp_server_auth_headers: Mapping[str, str | dict[str, str]],
    *,
    alias: str | None = None,
    server_name: str | None = None,
) -> str | dict[str, str] | None:
    """
    Resolve server-specific auth headers with case-insensitive matching.

    Tries the raw alias/server_name (lowercased) and the header-safe sanitized
    alias so dashboard clients using sanitize_mcp_alias_for_header() still match.
    """
    if not mcp_server_auth_headers:
        return None

    normalized_headers: Final = {k.lower(): v for k, v in mcp_server_auth_headers.items()}

    for identifier in (alias, server_name):
        if not identifier:
            continue
        keys_to_try = [identifier.lower()]
        sanitized = sanitize_mcp_alias_for_header(identifier)
        if sanitized and sanitized not in keys_to_try:
            keys_to_try.append(sanitized)
        for key in keys_to_try:
            if key in normalized_headers:
                return normalized_headers[key]
    return None


MCP_TOOL_ALLOWLIST_ENFORCED_KEY: Final = "tool_allowlist_enforced"


def _parse_mcp_info_dict(mcp_info: object) -> Mapping[str, object] | None:
    if mcp_info is None:
        return None
    if isinstance(mcp_info, dict):
        return mcp_info
    if isinstance(mcp_info, str):
        try:
            parsed: Final[object] = json.loads(mcp_info)
        except (ValueError, TypeError):
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def is_server_tool_allowlist_enforced(mcp_server: object) -> bool:
    mcp_info: Final = _parse_mcp_info_dict(getattr(mcp_server, "mcp_info", None))
    if not mcp_info:
        return False
    return bool(mcp_info.get(MCP_TOOL_ALLOWLIST_ENFORCED_KEY))


def server_applies_tool_allowlist(mcp_server: object) -> bool:
    """Whether server-level allowed_tools whitelist filtering is active."""
    allowed_tools: Final[object] = getattr(mcp_server, "allowed_tools", None) or []
    return is_server_tool_allowlist_enforced(mcp_server) or bool(allowed_tools)


def validate_and_normalize_mcp_server_payload(payload: McpServerPayloadLike) -> None:
    """
    Validate and normalize MCP server payload fields (server_name, alias, and
    tool_name_to_display_name).

    This function:
    1. Validates that server_name and alias don't contain the MCP_TOOL_PREFIX_SEPARATOR
    2. Validates that tool_name_to_display_name values satisfy Bedrock's tool-name pattern
    3. Normalizes alias by replacing spaces with underscores
    4. Sets default alias if not provided (using server_name as base)

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

    # Tool display name validation: must satisfy Bedrock's tool-name pattern
    if hasattr(payload, "tool_name_to_display_name") and payload.tool_name_to_display_name:
        validate_tool_display_names(payload.tool_name_to_display_name)

    # Alias normalization and defaulting
    alias: str | None = getattr(payload, "alias", None)
    server_name: Final[str | None] = getattr(payload, "server_name", None)

    if not alias and server_name:
        alias = normalize_server_name(server_name)
    elif alias:
        alias = normalize_server_name(alias)

    # Update the payload with normalized alias
    if hasattr(payload, "alias"):
        payload.alias = alias


def add_server_prefix_to_name(name: str, server_name: str) -> str:
    """Add server name prefix to any MCP resource name."""
    formatted_server_name: Final = normalize_server_name(server_name)

    return MCP_TOOL_PREFIX_FORMAT.format(
        server_name=formatted_server_name,
        separator=MCP_TOOL_PREFIX_SEPARATOR,
        tool_name=name,
    )


def get_server_prefix(server: object) -> str:
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
        cached: Final[str | None] = getattr(server, "short_prefix", None)
        if cached:
            return cached
        server_id: Final[str | None] = getattr(server, "server_id", None)
        if server_id:
            return compute_short_server_prefix(server_id)

    alias: Final[str | None] = getattr(server, "alias", None)
    if alias:
        return alias
    server_name: Final[str | None] = getattr(server, "server_name", None)
    if server_name:
        return server_name
    if hasattr(server, "server_id"):
        fallback_server_id: Final[str] = getattr(server, "server_id", "")
        return fallback_server_id
    return ""


def iter_known_server_prefixes(server: _McpServerLike) -> Iterator[str]:
    """Yield every prefix form that may appear in tool names for ``server``.

    Always includes the *current* prefix returned by ``get_server_prefix``.
    Additionally yields the historical (alias / server_name / server_id) and
    short-ID forms so the routing layer can resolve tool names regardless of
    which prefix mode was active when the client first observed them.
    """
    seen: Final = set()

    def _emit(value: str | None) -> Iterator[str]:
        if value and value not in seen:
            seen.add(value)
            yield value

    yield from _emit(get_server_prefix(server))
    yield from _emit(getattr(server, "short_prefix", None))

    server_id: Final[str | None] = getattr(server, "server_id", None)
    if server_id:
        try:
            yield from _emit(compute_short_server_prefix(server_id))
        except ValueError:
            pass

    yield from _emit(getattr(server, "alias", None))
    yield from _emit(getattr(server, "server_name", None))
    yield from _emit(server_id)


def iter_known_tool_name_spellings(tool_name: str, server: MCPServer) -> Iterator[str]:
    """Yield every name that denotes the bare ``tool_name`` on ``server``: the bare name,
    then its wire spelling under each prefix ``iter_known_server_prefixes`` accepts.
    ``get_server_prefix`` covers only the currently published one, and that moves with the
    alias and with ``LITELLM_USE_SHORT_MCP_TOOL_PREFIX``.
    """
    yield tool_name
    for prefix in iter_known_server_prefixes(server):
        yield add_server_prefix_to_name(tool_name, prefix)


def openapi_tool_name(operation_id: str) -> str:
    """Return the tool name ``_register_openapi_tools`` registers ``operation_id`` under.

    The single transform between a spec's operationId and the name the gateway serves.
    Policy recovers the link by replaying this exact function, which is what keeps it from
    deciding for a tool it does not name: two operationIds that register as two tools
    necessarily normalize to two names here, because this is the map that registered them.
    """
    return operation_id.replace(" ", "_").lower()


def match_known_tool_name(tool_name: str, server: MCPServer, names: Iterable[str]) -> str | None:
    """Return the entry of ``names`` that denotes ``tool_name`` on ``server``, else ``None``.

    The single question every tool-name-keyed site asks: the allow list, the deny list,
    ``allowed_params`` and the discovery filter, so discovery hides exactly what dispatch
    refuses. It spans every spelling routing accepts and no more, because a tool's identity
    is the exact name routing dispatches; anything looser lets one policy decide two tools.

    On an OpenAPI server the configured entry holds the spec's operationId while routing
    holds :func:`openapi_tool_name` of it, so both sides go through that map first. Doing it
    with the registering function rather than a lookalike is the whole safety argument: a
    coarser one collapses operationIds that registration keeps apart.

    Callers read the returned entry rather than testing a container's values, which is what
    stops an explicitly empty ``allowed_params`` list from reading as "nothing configured".
    """
    normalize: Final = openapi_tool_name if getattr(server, "spec_path", None) else str
    spellings: Final = {normalize(spelling) for spelling in iter_known_tool_name_spellings(tool_name, server)}
    return next((name for name in names if normalize(name) in spellings), None)


def split_server_prefix_from_name(prefixed_name: str) -> tuple[str, str]:
    """Return the unprefixed name plus the server name used as prefix.

    Cuts at the FIRST separator, so the two halves are only trustworthy as a
    pair: they reassemble into ``prefixed_name`` exactly, which is what makes
    this safe for routing. Reading one half on its own is a guess about where the
    boundary fell, and that guess is wrong whenever the prefix itself contains
    the separator. Callers that compare a half against configuration must use
    :func:`match_known_server_prefix` or :func:`strip_known_server_prefix`.
    """
    if MCP_TOOL_PREFIX_SEPARATOR in prefixed_name:
        parts: Final = prefixed_name.split(MCP_TOOL_PREFIX_SEPARATOR, 1)
        if len(parts) == 2:
            return parts[1], parts[0]
    return prefixed_name, ""


def match_known_server_prefix(name: str, known_prefixes: Iterable[str]) -> tuple[str, str] | None:
    """Return ``(matched_prefix, bare_name)`` when ``name`` carries a known prefix.

    Candidates are normalized and tried LONGEST first, so a prefix that itself
    contains :data:`MCP_TOOL_PREFIX_SEPARATOR` (the UUID ``server_id`` used when
    a server has no alias, or a legacy hyphenated alias) wins over a shorter
    prefix that is merely its leading segment. Returns ``None`` when no candidate
    matches, i.e. ``name`` carries none of these prefixes.
    """
    candidates: Final = sorted(
        {normalize_server_name(prefix) for prefix in known_prefixes if prefix},
        key=len,
        reverse=True,
    )
    for prefix in candidates:
        separator_suffixed = prefix + MCP_TOOL_PREFIX_SEPARATOR
        if name.startswith(separator_suffixed):
            return prefix, name[len(separator_suffixed) :]
    return None


def strip_known_server_prefix(name: str, server: _McpServerLike | None) -> str:
    """Strip ``server``'s registered prefix from a prefixed tool/resource name.

    Unlike :func:`split_server_prefix_from_name`, which guesses the boundary at
    the first separator, this removes exactly ``{known_prefix}{separator}`` for
    one of the server's actual registered prefixes. It therefore stays correct
    when a prefix itself contains the separator (e.g. the UUID ``server_id``
    used as the fallback prefix when a server has no alias, or a legacy
    hyphenated alias), where the first-separator split would cut inside the
    prefix and never match the stored bare tool name.

    Returns ``name`` unchanged when ``server`` is known but none of its prefixes
    match (the name is already unprefixed). Falls back to the legacy split only
    when ``server`` is ``None``.
    """
    if server is None:
        return split_server_prefix_from_name(name)[0]
    matched: Final = match_known_server_prefix(name, iter_known_server_prefixes(server))
    return name if matched is None else matched[1]


def is_tool_name_prefixed(
    tool_name: str,
    known_server_prefixes: AbstractSet[str] | None = None,
) -> bool:
    """
    Check if tool name has a known MCP server prefix.

    When ``known_server_prefixes`` is provided the function verifies that the
    name actually starts with one of those prefixes followed by the separator,
    matching the longest candidate first so a prefix containing the separator
    still resolves.  Without it the check falls back to the legacy heuristic
    (separator present anywhere in the name), which can produce false
    positives for non-MCP tools whose names contain hyphens
    (e.g. ``text-to-speech``, ``code-review``).

    Args:
        tool_name: Tool name to check.
        known_server_prefixes: Optional set of normalized server prefixes
            currently registered in the MCP manager.  Pass this whenever
            the caller has access to the server registry so that the check
            is accurate.

    Returns:
        True if tool name is prefixed, False otherwise.
    """
    if MCP_TOOL_PREFIX_SEPARATOR not in tool_name:
        return False

    if known_server_prefixes is not None:
        return match_known_server_prefix(tool_name, known_server_prefixes) is not None

    # Legacy fallback – separator present somewhere in the name.
    return True


def validate_mcp_server_name(server_name: str, raise_http_exception: bool = False) -> None:
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

            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail={"error": error_message})
        else:
            raise Exception(error_message)


def extract_mcp_tool_result_error_message(result: object) -> str | None:
    """The first text content of an ``isError=True`` tool result, or ``None``
    when the result is not an error.

    Accepts both ``mcp.types.CallToolResult`` objects and their dict
    equivalents, duck-typed so the ``mcp`` package is not required.
    """
    is_error: Final[object] = result.get("isError") if isinstance(result, Mapping) else getattr(result, "isError", None)
    if is_error is not True:
        return None
    content: Final[object] = result.get("content") if isinstance(result, Mapping) else getattr(result, "content", None)
    if isinstance(content, (list, tuple)):
        for item in content:
            text: object = item.get("text") if isinstance(item, Mapping) else getattr(item, "text", None)
            if isinstance(text, str) and text:
                return text
    return "MCP tool call returned isError=true"


def mcp_tool_result_content_list(result: object) -> MutableSequence[object] | None:  # mutable-ok: see below
    """The mutable content list of an MCP tool result, or ``None`` when it has none.

    Deliberately mutable: a guardrail masking the result rewrites entries in place,
    because the logging payload captured before the guardrail runs references this
    same list, so handing back a copy would leave the unmasked text in the spend log
    and the OTel span.

    Accepts both ``mcp.types.CallToolResult`` objects and their dict
    equivalents, duck-typed so the ``mcp`` package is not required.
    """
    content: Final[object] = result.get("content") if isinstance(result, Mapping) else getattr(result, "content", None)
    if isinstance(content, MutableSequence):
        return content
    return None


def mcp_content_item_text(item: object) -> str | None:
    """The ``text`` of a rewritable MCP content item, or ``None``.

    Only mappings and Pydantic-style models report a text, because those are the
    only shapes ``with_mcp_content_item_text`` can rewrite; a caller therefore
    never reads text it would be unable to write back (e.g. masked by a
    guardrail). Non-text content (images, embedded resources) has no ``text``
    and is reported as ``None``.
    """
    text: object
    if isinstance(item, Mapping):
        text = item.get("text")
    elif callable(getattr(item, "model_copy", None)):
        text = getattr(item, "text", None)
    else:
        return None
    return text if isinstance(text, str) else None


def with_mcp_content_item_text(item: object, text: str) -> object:
    """A copy of an MCP content item carrying ``text`` instead of its own.

    Only meaningful for items ``mcp_content_item_text`` returned a text for; any
    other item is returned unchanged.
    """
    if isinstance(item, Mapping):
        return {**item, "text": text}
    model_copy: Final = getattr(item, "model_copy", None)
    if callable(model_copy):
        return model_copy(update={"text": text})
    return item


TOOL_DISPLAY_NAME_PATTERN: Final = re.compile(r"^[a-zA-Z0-9_-]+$")


def validate_tool_display_names(tool_name_to_display_name: Mapping[str, str] | None) -> None:
    """
    Validate tool display name overrides against Bedrock's tool-name constraint.

    A display name replaces the tool name sent to the LLM provider, so it must
    satisfy the strictest provider requirement in use (Bedrock's
    ``[a-zA-Z0-9_-]+``); a name with spaces or other characters saves
    successfully but fails every subsequent Bedrock tool call.

    Raises:
        HTTPException: If any display name fails the pattern.
    """
    if not tool_name_to_display_name:
        return

    for original_name, display_name in tool_name_to_display_name.items():
        if display_name and not TOOL_DISPLAY_NAME_PATTERN.match(display_name):
            from fastapi import HTTPException
            from starlette import status

            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": (
                        f"Invalid display name '{display_name}' for tool '{original_name}'. "
                        "Display names may only contain letters, digits, underscores, and "
                        "hyphens (no spaces or other special characters), since they replace "
                        "the tool name sent to the LLM provider."
                    )
                },
            )


class MCPMissingUserEnvVarsError(Exception):
    """Raised when an MCP request can't be built because the calling user has
    not supplied one or more required per-user environment variables.

    The error message is user-facing and includes a URL the user can visit
    to fill them in.
    """

    def __init__(
        self,
        *,
        server_id: str,
        server_name: str | None,
        missing: list[str],
        setup_url: str,
    ) -> None:
        self.server_id = server_id
        self.server_name = server_name
        self.missing = missing
        self.setup_url = setup_url
        label: Final = server_name or server_id
        bullet_list: Final = "\n".join(f"- {name}" for name in missing)
        message: Final = (
            f'Cannot connect to MCP server "{label}".\n\n'
            f"Your administrator configured this server to require per-user "
            f"variables, but you haven't set the following yet:\n"
            f"{bullet_list}\n\n"
            f"Set your credentials here:\n"
            f"{setup_url}"
        )
        super().__init__(message)


# Pattern for ``${NAME}`` substitution. Matches the standard env-var
# identifier rules — letters, digits, underscores, can't start with a digit.
_ENV_VAR_PATTERN: Final = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def parse_admin_env_vars(
    env_vars: Iterable[Any] | None,
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    """Split admin-configured env var entries into globals and per-user specs.

    Accepts the raw value of ``MCPServer.env_vars`` (list of dicts or Pydantic
    models). Returns:

    - ``global_values``: ``{name: value}`` for entries with ``scope=="global"``.
    - ``user_specs``: list of ``{name, description}`` for entries with
      ``scope=="user"`` — these are the names the user must fill in.

    Unknown / malformed entries are skipped silently.
    """
    global_values: Final[dict[str, str]] = {}
    user_specs: Final[list[dict[str, Any]]] = []
    if not env_vars:
        return global_values, user_specs
    for raw in env_vars:
        if raw is None:
            continue
        if hasattr(raw, "model_dump"):
            entry: Mapping[str, object] = raw.model_dump()
        elif isinstance(raw, dict):
            entry = raw
        else:
            continue
        name = entry.get("name")
        if not isinstance(name, str) or not name:
            continue
        scope = entry.get("scope") or "global"
        if scope == "user":
            user_specs.append({"name": name, "description": entry.get("description")})
        else:
            value = entry.get("value")
            global_values[name] = "" if value is None else str(value)
    return global_values, user_specs


def find_env_var_references(value: str) -> set[str]:
    """Return the set of ``${NAME}`` identifiers referenced inside ``value``."""
    if not value:
        return set()
    return set(_ENV_VAR_PATTERN.findall(value))


def collect_env_var_references(*, strings: Iterable[str]) -> set[str]:
    """Union of every ``${NAME}`` reference across a collection of strings."""
    refs: set[str] = set()
    for s in strings:
        if isinstance(s, str):
            refs |= find_env_var_references(s)
    return refs


def interpolate_env_vars(value: str, variables: Mapping[str, str]) -> str:
    """Replace ``${NAME}`` references in ``value`` with the matching mapping
    entry. Unknown names are left untouched so callers can detect them via
    ``find_env_var_references`` on the result if needed.
    """
    if not value:
        return value

    def _sub(match: "re.Match[str]") -> str:
        name: Final = match.group(1)
        if name in variables:
            return variables[name]
        return match.group(0)

    return _ENV_VAR_PATTERN.sub(_sub, value)


def interpolate_headers(headers: Mapping[str, str], variables: Mapping[str, str]) -> dict[str, str]:
    """Return a copy of ``headers`` with every value passed through ``interpolate_env_vars``."""
    return {k: interpolate_env_vars(v, variables) for k, v in headers.items()}


def build_env_var_setup_url(server_id: str) -> str:
    """The frontend URL where a user can fill in their per-user env vars."""
    base: Final = os.environ.get("PROXY_BASE_URL", "").rstrip("/")
    path: Final = f"/ui/?page=mcp-servers&fill_env_vars={quote(server_id, safe='')}"
    return f"{base}{path}" if base else path


def merge_mcp_headers(
    *,
    extra_headers: Mapping[str, str] | None = None,
    static_headers: Mapping[str, str] | None = None,
) -> dict[str, str] | None:
    """Merge outbound HTTP headers for MCP calls.

    This is used when calling out to external MCP servers (or OpenAPI-based MCP tools).

    Merge rules:
    - Start with `extra_headers` (typically OAuth2-derived headers)
    - Overlay `static_headers` (user-configured per MCP server)

    If both contain the same key, `static_headers` wins. This matches the existing
    behavior in `MCPServerManager` where `server.static_headers` is applied after
    any caller-provided headers.
    """
    merged: Final[dict[str, str]] = {}

    if extra_headers:
        merged.update({str(k): str(v) for k, v in extra_headers.items()})

    if static_headers:
        merged.update({str(k): str(v) for k, v in static_headers.items()})

    return merged or None


# Local rather than litellm.constants: this module deliberately imports no litellm
# package, so pulling one in for a single integer would drag in litellm/__init__.
MAX_STRUCTURED_CONTENT_SCAN_DEPTH: Final = 100


JSONLeafPath = tuple[str | int, ...]


def _flatten_leaf_groups(
    groups: Iterable[tuple[tuple[JSONLeafPath, str], ...] | None],
) -> tuple[tuple[JSONLeafPath, str], ...] | None:
    """Concatenate child leaf groups, propagating the too-deep sentinel."""
    materialized: Final = tuple(groups)
    if any(group is None for group in materialized):
        return None
    return tuple(leaf for group in materialized if group is not None for leaf in group)


def json_string_leaves(value: object, path: JSONLeafPath = ()) -> tuple[tuple[JSONLeafPath, str], ...] | None:
    """Depth-first, deterministically ordered string leaves of a JSON value.

    Returns ``None`` when the value is nested past ``MAX_STRUCTURED_CONTENT_SCAN_DEPTH``,
    so the caller blocks rather than letting deeper values through unscanned; an
    empty tuple means there was simply nothing to scan. A sentinel rather than an
    exception because this module is reloaded by tests (see the note above the
    environment-backed constants), which would give a custom exception class a new
    identity and let it escape a caller's ``except``.
    """
    if len(path) > MAX_STRUCTURED_CONTENT_SCAN_DEPTH:
        return None
    if isinstance(value, str):
        return ((path, value),)
    if isinstance(value, dict):
        return _flatten_leaf_groups(json_string_leaves(item, (*path, key)) for key, item in value.items())
    if isinstance(value, list):
        return _flatten_leaf_groups(json_string_leaves(item, (*path, index)) for index, item in enumerate(value))
    return ()


def with_json_string_leaves(
    value: object,
    replacements: Mapping[JSONLeafPath, str],
    path: JSONLeafPath = (),
) -> object:
    """Rebuild a JSON value with the guardrail's rewritten string leaves."""
    if isinstance(value, str):
        return replacements.get(path, value)
    if isinstance(value, dict):
        return {key: with_json_string_leaves(item, replacements, (*path, key)) for key, item in value.items()}
    if isinstance(value, list):
        return [with_json_string_leaves(item, replacements, (*path, index)) for index, item in enumerate(value)]
    return value


def json_unrewritable_labels(value: object, path_depth: int = 0) -> tuple[str, ...] | None:
    """Strings in a JSON value that carry meaning but cannot be rewritten.

    Dictionary keys and non-string scalars: masking either would change the
    payload's contract rather than redact a value, so a caller scans these and
    blocks on a match instead of rewriting, matching what the content filter
    already does for MCP tool call arguments. ``None`` means the value is nested
    past the scan depth, same contract as ``json_string_leaves``.
    """
    if path_depth > MAX_STRUCTURED_CONTENT_SCAN_DEPTH:
        return None
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return ()
    if isinstance(value, (int, float)):
        return (str(value),)
    if isinstance(value, dict):
        own: Final = tuple(key for key in value if isinstance(key, str))
        nested = tuple(json_unrewritable_labels(item, path_depth + 1) for item in value.values())
        if any(group is None for group in nested):
            return None
        return own + tuple(label for group in nested if group is not None for label in group)
    if isinstance(value, list):
        nested = tuple(json_unrewritable_labels(item, path_depth + 1) for item in value)
        if any(group is None for group in nested):
            return None
        return tuple(label for group in nested if group is not None for label in group)
    return ()


def mcp_tool_result_structured_content(result: object) -> object:
    """The ``structuredContent`` of an MCP tool result, or ``None`` when it has none."""
    if isinstance(result, Mapping):
        return result.get("structuredContent")
    return getattr(result, "structuredContent", None)


def set_mcp_tool_result_structured_content(result: object, value: object) -> bool:
    """Replace ``structuredContent`` in place; ``False`` when the shape does not carry it.

    In place for the same reason the content list is: the logging payload captured
    before the guardrail ran references this object, so a copy would leave the
    unmasked value in the spend log and the OTel span.
    """
    if isinstance(result, MutableMapping):
        result["structuredContent"] = value
        return True
    if not hasattr(result, "structuredContent"):
        return False
    try:
        setattr(result, "structuredContent", value)  # attribute name is fixed by the MCP result shape
        return True
    except (AttributeError, TypeError, ValueError):
        return False


_HOP_BY_HOP_HEADERS: Final = frozenset(
    {
        "content-length",
        "transfer-encoding",
        "connection",
        "keep-alive",
        "upgrade",
        "te",
        "trailer",
    }
)

_SYNTHETIC_REQUEST_EXCLUDED_HEADERS: Final = _HOP_BY_HOP_HEADERS | frozenset(
    {"content-type", "host", "x-forwarded-for"}
)

_SYNTHETIC_REQUEST_SERVER: Final = ("127.0.0.1", 4000)

_MCP_SERVER_AUTH_HEADER_PREFIX: Final = "x-mcp-"


def _custom_litellm_key_header_name() -> str | None:
    """``general_settings.litellm_key_header_name``, the deployment's custom header name for
    the proxy virtual key, so it is stripped from observability copies like the standard ones."""
    try:
        from litellm.proxy.proxy_server import general_settings
    except ImportError:
        return None
    return general_settings.get("litellm_key_header_name") if general_settings else None


def _mcp_client_side_auth_header_name() -> str:
    """The header name the client passes the upstream MCP credential in, falling back to the
    default when ``general_settings`` is unavailable (the SDK, outside a running proxy)."""
    from .auth.user_api_key_auth_mcp import MCPRequestHandler

    try:
        return MCPRequestHandler.get_mcp_client_side_auth_header_name()
    except ImportError:
        return MCPRequestHandler.LITELLM_MCP_AUTH_HEADER_NAME


def _identity_header_names() -> frozenset[str]:
    """Lowercased header names the deployment reads the caller's identity out of. A name here
    is a claim about who the caller is rather than a secret, and ``get_user_from_headers``
    resolves it off the request this module reconstructs, so dropping one would lose end user
    attribution on the MCP paths that leave ``end_user_id`` unset at connect time.

    ``user_header_mappings`` is accepted as a bare mapping as well as a list of them, matching
    ``get_internal_user_header_from_mapping`` and ``get_customer_user_header_from_mapping``.
    Iterating the bare form without normalizing yields its keys, which would silently exempt
    nothing."""
    try:
        from litellm.proxy.proxy_server import general_settings
    except ImportError:
        return frozenset()
    if not general_settings:
        return frozenset()
    user_header: Final = general_settings.get("user_header_name")
    configured: Final = general_settings.get("user_header_mappings")
    mappings: Final = configured if isinstance(configured, list) else (configured,) if configured else ()
    mapped: Final = (mapping.get("header_name") for mapping in mappings if isinstance(mapping, Mapping))
    return frozenset(name.lower() for name in (user_header, *mapped) if isinstance(name, str) and name)


def _forwarded_upstream_header_names() -> frozenset[str]:
    """Lowercased header names that a configured MCP server forwards upstream through its
    ``extra_headers`` allowlist. The names are chosen by the admin, so no prefix rule can
    recognize them, and a caller supplied value under one of them is an upstream credential.

    ``authorization`` is left out because ``clean_headers`` already strips it, and claiming it
    here would change which header ``authenticated_with_header`` resolves to on the oauth
    passthrough config, which lists it in ``extra_headers`` by design. Identity headers are
    left out for the same reason: naming one in ``extra_headers`` forwards the caller's
    identity upstream, it does not turn that identity into a secret."""
    try:
        from .mcp_server_manager import global_mcp_server_manager
    except ImportError:
        return frozenset()
    exempt: Final = _identity_header_names() | frozenset({"authorization"})
    return frozenset(
        name.lower()
        for server in global_mcp_server_manager.get_registry().values()
        for name in (server.extra_headers or ())
        if name.lower() not in exempt
    )


def _upstream_credential_headers(header_names: Iterable[str]) -> frozenset[str]:
    """Lowercased names of the headers in ``header_names`` that carry an upstream MCP
    credential rather than request context: the configured client side auth header, any
    header name a configured server forwards upstream via ``extra_headers``, and the
    per-server ``x-mcp-{alias}-{header}`` family. ``clean_headers`` only knows the
    credential headers of the chat completions path, so these are dropped on top of it.
    """
    from .auth.user_api_key_auth_mcp import MCPRequestHandler

    non_credential: Final = frozenset(
        {
            MCPRequestHandler.LITELLM_MCP_SERVERS_HEADER_NAME.lower(),
            MCPRequestHandler.LITELLM_MCP_ACCESS_GROUPS_HEADER_NAME.lower(),
        }
    )
    client_side_auth: Final = _mcp_client_side_auth_header_name().lower()
    forwarded_upstream: Final = _forwarded_upstream_header_names()
    return frozenset(
        name
        for name in (raw_name.lower() for raw_name in header_names)
        if name == client_side_auth
        or name in forwarded_upstream
        or (name.startswith(_MCP_SERVER_AUTH_HEADER_PREFIX) and name not in non_credential)
    )


def build_synthetic_mcp_request(
    *,
    path: str,
    raw_headers: Mapping[str, str] | None = None,
    client_ip: str | None = None,
) -> "Request":
    """A synthetic FastAPI ``Request`` carrying the MCP connection's HTTP headers.

    The MCP protocol transports do not hand a per-call ``Request`` to the tool
    handlers, so one is reconstructed from the connection's ``raw_headers``. That
    lets ``add_litellm_data_to_request`` derive ``metadata.headers``,
    ``proxy_server_request``, header-based tags, guardrails and trace correlation
    exactly as on the chat completions path. Hop-by-hop headers describe the
    original HTTP framing rather than the logical request, so they are dropped, and
    ``x-forwarded-for`` comes from the resolved ``client_ip`` to avoid spoofing. ``host`` is
    dropped for the same reason: it is what ``Request.url`` is built from, so forwarding it
    would let a caller choose the URL every logging callback records. Upstream
    MCP credentials and the deployment's proxy key header, including a custom
    ``litellm_key_header_name``, are dropped so they cannot reach a callback or a guardrail
    through the derived metadata even when a caller omits ``general_settings``.
    """
    from fastapi import Request

    custom_key_header: Final = _custom_litellm_key_header_name()
    excluded: Final = (
        _SYNTHETIC_REQUEST_EXCLUDED_HEADERS
        | _upstream_credential_headers(raw_headers.keys() if raw_headers else ())
        | (frozenset({custom_key_header.lower()}) if custom_key_header else frozenset())
    )
    forwarded: Final = tuple(
        (
            name.lower().encode("latin-1", errors="replace"),
            value.encode("utf-8", errors="replace"),
        )
        for name, value in (raw_headers.items() if raw_headers else ())
        if name.lower() not in excluded
    )
    xff: Final = ((b"x-forwarded-for", client_ip.encode("utf-8")),) if client_ip else ()
    return Request(
        scope={
            "type": "http",
            "method": "POST",
            "path": path,
            "scheme": "http",
            "server": _SYNTHETIC_REQUEST_SERVER,
            "query_string": b"",
            "root_path": "",
            "headers": ((b"content-type", b"application/json"), *forwarded, *xff),
            **({"client": (client_ip, 0)} if client_ip else {}),
        }
    )


def logging_safe_mcp_headers(raw_headers: Mapping[str, str] | None) -> Mapping[str, str]:
    """The MCP request's client headers, sanitized the way the chat completions path
    sanitizes them before they reach a logging callback or a guardrail: proxy key
    headers stripped, including the custom key header name the deployment configured,
    upstream MCP credentials dropped, and credential-bearing values masked.

    Client-controlled behaviour flags (``litellm-disable-message-redaction``) are dropped
    too: these headers are read back out of the metadata to change proxy behaviour, so
    leaving one in place would let any MCP client turn off the redaction an admin
    configured. This path carries no key or team object to authorize an opt-out with, so
    it always strips them. ``host`` goes too, so that a caller cannot name the deployment in
    the guardrail payload and the spend row the way it could once name the request URL."""
    from starlette.datastructures import Headers

    from litellm.proxy.litellm_pre_call_utils import (
        UNTRUSTED_REQUEST_HEADER_CONTROL_FIELDS,
        clean_headers,
        redact_credential_headers,
    )

    excluded: Final = (
        _upstream_credential_headers(raw_headers.keys() if raw_headers else ())
        | UNTRUSTED_REQUEST_HEADER_CONTROL_FIELDS
        | frozenset({"host"})
    )
    cleaned: Final = clean_headers(
        Headers(raw_headers),
        litellm_key_header_name=_custom_litellm_key_header_name(),
    )
    return redact_credential_headers({name: value for name, value in cleaned.items() if name.lower() not in excluded})
