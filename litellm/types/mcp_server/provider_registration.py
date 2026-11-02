import json
from collections.abc import Collection, Mapping
from datetime import datetime
from typing import Final, Literal
from urllib.parse import urlsplit, urlunsplit

from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field, TypeAdapter, field_validator, model_validator
from typing_extensions import Self

from litellm.types.mcp import MCPExecutionMode

_STR: Final = TypeAdapter(str)
_MAP: Final = TypeAdapter(dict[str, object])
_URL: Final = TypeAdapter(AnyHttpUrl)


def _is_neutral_registration_value(value: object) -> bool:
    if isinstance(value, str):
        return value.strip() in ("", "null", "{}", "[]")
    return value is None or value is False or value in ({}, [])


def active_registration_extras(values: Mapping[str, object], known_fields: Collection[str]) -> tuple[str, ...]:
    return tuple(
        field
        for field, value in values.items()
        if field not in known_fields and not _is_neutral_registration_value(value)
    )


class ProviderMCPRegistration(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    execution_mode: Literal[MCPExecutionMode.provider] = MCPExecutionMode.provider
    server_id: str | None = None
    server_name: str | None = None
    alias: str | None = None
    description: str | None = None
    url: str
    allowed_tools: list[str]
    mcp_info: dict[str, object] = Field(default_factory=dict, validate_default=True)
    mcp_access_groups: list[str] = Field(default_factory=list)
    tool_name_to_display_name: dict[str, str] | None = None
    tool_name_to_description: dict[str, str] | None = None
    instructions: str | None = None
    source_url: str | None = None
    allow_all_keys: bool = False
    available_on_public_internet: Literal[True] = True
    protocol_version: Literal["auto"] = "auto"
    approval_status: str | None = None
    submitted_by: str | None = None
    submitted_at: datetime | None = None

    @model_validator(mode="before")
    @classmethod
    def discard_neutral_extras(cls, values: object) -> object:
        if not isinstance(values, Mapping):
            return values
        fields: Final = _MAP.validate_python(values)
        return {
            field: value
            for field, value in fields.items()
            if field in cls.model_fields or not _is_neutral_registration_value(value)
        }

    @field_validator("url", mode="before")
    @classmethod
    def normalize_url(cls, value: object) -> str:
        return canonical_provider_mcp_url(value)

    @field_validator("allowed_tools")
    @classmethod
    def normalize_allowed_tools(cls, tools: list[str]) -> list[str]:
        if any(not tool.strip() or any(char in tool for char in "*?[]") for tool in tools):
            raise ValueError("allowed_tools must contain nonblank exact tool names without wildcards")
        return list(dict.fromkeys(tools))

    @field_validator("mcp_info", mode="before")
    @classmethod
    def normalize_metadata(cls, value: object) -> dict[str, object]:
        info: Final = _MAP.validate_python(_decoded_json(value) or {})
        if "tool_allowlist_enforced" in info and info["tool_allowlist_enforced"] is not True:
            raise ValueError("Provider registrations require mcp_info.tool_allowlist_enforced=true")
        if info.get("protocol_version", "auto") != "auto":
            raise ValueError("Provider registrations do not configure the upstream MCP protocol")
        return {**info, "tool_allowlist_enforced": True}

    @field_validator("tool_name_to_description", "tool_name_to_display_name", mode="before")
    @classmethod
    def decode_stored_metadata(cls, value: object) -> object:
        return _decoded_json(value)

    @model_validator(mode="after")
    def validate_tool_descriptions(self) -> Self:
        if not (self.tool_name_to_description or {}).keys() <= set(self.allowed_tools):
            raise ValueError("Tool descriptions must reference allowed_tools")
        return self


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


def normalize_provider_registration(values: Mapping[str, object]) -> dict[str, object]:
    registration: Final = ProviderMCPRegistration.model_validate(values)
    return {
        **registration.model_dump(exclude_unset=True),
        "mcp_info": registration.mcp_info,
        "transport": None,
        "auth_type": None,
        "credentials": None,
    }
