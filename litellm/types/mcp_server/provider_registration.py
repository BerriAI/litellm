import json
from collections.abc import Mapping
from typing import Final
from urllib.parse import urlsplit, urlunsplit

from pydantic import AnyHttpUrl, TypeAdapter

_STR: Final = TypeAdapter(str)
_TOOLS: Final = TypeAdapter(list[str])
_MAP: Final = TypeAdapter(dict[str, object])
_URL: Final = TypeAdapter(AnyHttpUrl)
_CONNECTION_FIELDS: Final = (
    "transport",
    "auth_type",
    "credentials",
    "spec_path",
    "extra_headers",
    "static_headers",
    "env_vars",
    "command",
    "args",
    "env",
    "issuer",
    "authorization_url",
    "token_url",
    "registration_url",
    "oauth2_flow",
    "token_exchange_endpoint",
    "audience",
    "subject_token_type",
    "token_exchange_profile",
    "delegate_auth_to_upstream",
    "oauth_passthrough",
    "dcr_bridge",
    "per_server_oauth_discovery",
    "is_byok",
    "byok_description",
    "byok_api_key_help_url",
    "timeout",
    "max_concurrent_requests",
    "rpm",
    "pinned_tools",
)
_FLAG_FIELDS: Final = frozenset(
    {"delegate_auth_to_upstream", "oauth_passthrough", "dcr_bridge", "per_server_oauth_discovery", "is_byok"}
)
_EMPTY_FIELDS: Final = frozenset(
    {"credentials", "extra_headers", "static_headers", "env_vars", "args", "env", "byok_description", "pinned_tools"}
)
_JSON_FIELDS: Final = frozenset({"credentials", "static_headers", "env_vars", "env", "pinned_tools"})


def canonical_provider_mcp_url(value: object) -> str:
    if value is None:
        raise ValueError("Provider MCP url is required")
    url: Final = _STR.validate_python(value)
    if any(char.isspace() or ord(char) < 32 for char in url) or "\\" in url:
        raise ValueError("Provider MCP url must not contain whitespace, control characters, or backslashes")
    _URL.validate_python(url)
    parsed: Final = urlsplit(url)
    if parsed.scheme.lower() != "https" or not parsed.hostname or parsed.username is not None or "#" in url:
        raise ValueError("Provider MCP url must use HTTPS without userinfo or a fragment")
    host: Final = parsed.hostname.lower().encode("idna").decode("ascii")
    hostname: Final = f"[{host}]" if ":" in host else host
    port: Final = parsed.port
    authority: Final = f"{hostname}:{port}" if port is not None and port != 443 else hostname
    return urlunsplit(("https", authority, parsed.path, parsed.query, ""))


def _decoded_json(value: object) -> object:
    return json.loads(value) if isinstance(value, str) else value


def _active_connection_field(field: str, raw: object) -> bool:
    value: Final = _decoded_json(raw) if field in _JSON_FIELDS else raw
    return not (
        value is None or (field in _FLAG_FIELDS and value is False) or (field in _EMPTY_FIELDS and value in ({}, []))
    )


def normalize_provider_registration(values: Mapping[str, object]) -> dict[str, object]:
    unsupported: Final = next(
        (field for field in _CONNECTION_FIELDS if _active_connection_field(field, values.get(field))), None
    )
    if unsupported is not None:
        raise ValueError(f"Provider-executed MCP registrations do not accept {unsupported}")
    if values.get("available_on_public_internet", True) is not True:
        raise ValueError("Provider-executed MCP registrations do not support gateway network restrictions")
    if values.get("allowed_tools") is None:
        raise ValueError("Provider registrations require an explicit allowed_tools list")
    tools: Final = _TOOLS.validate_python(values.get("allowed_tools"))
    if any(not tool.strip() or any(char in tool for char in "*?[]") for tool in tools):
        raise ValueError("allowed_tools must contain nonblank exact tool names without wildcards")
    info: Final = _MAP.validate_python(_decoded_json(values.get("mcp_info")) or {})
    if "tool_allowlist_enforced" in info and info["tool_allowlist_enforced"] is not True:
        raise ValueError("Provider registrations require mcp_info.tool_allowlist_enforced=true")
    if info.get("protocol_version", "auto") != "auto":
        raise ValueError("Provider registrations do not configure the upstream MCP protocol")
    descriptions: Final = _MAP.validate_python(_decoded_json(values.get("tool_name_to_description")) or {})
    if not descriptions.keys() <= set(tools):
        raise ValueError("Tool descriptions must reference allowed_tools")
    return {
        **values,
        "url": canonical_provider_mcp_url(values.get("url")),
        "allowed_tools": list(dict.fromkeys(tools)),
        "mcp_info": {**info, "tool_allowlist_enforced": True},
        "transport": None,
        "auth_type": None,
        "credentials": None,
    }
