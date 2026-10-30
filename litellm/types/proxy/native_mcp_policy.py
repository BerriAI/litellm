from collections.abc import Mapping
from typing import Annotated, Final, Literal, TypeAlias
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, StringConstraints, field_validator

from litellm.litellm_core_utils.core_helpers import as_str_mapping

MCPName: TypeAlias = Annotated[str, StringConstraints(strict=True, min_length=1, pattern=r"^[^\s*]+$")]
MCPToolScope: TypeAlias = tuple[MCPName, ...] | Literal["*"]


class NativeMCPServerGrant(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: MCPName
    url: str
    tools: MCPToolScope

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        parsed: Final = urlsplit(value)
        if (
            value != value.strip()
            or parsed.scheme not in ("https", "http")
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.fragment
        ):
            raise ValueError("MCP URL must be an absolute HTTP(S) URL without credentials or a fragment")
        _ = parsed.port
        return value


class NativeMCPPolicy(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    mcp_servers: tuple[NativeMCPServerGrant, ...] = ()


def normalize_native_mcp_policy_input(value: object) -> object:
    data: Final = as_str_mapping(value)
    if data is None:
        return value
    metadata: Final = as_str_mapping(data.get("metadata")) or {}
    if "native_mcp_policy" not in data and "native_mcp_policy" not in metadata:
        return value
    top_level: Final = data.get("native_mcp_policy", metadata.get("native_mcp_policy"))
    policy: Final = NativeMCPPolicy.model_validate(top_level) if top_level is not None else None
    if "native_mcp_policy" in data and "native_mcp_policy" in metadata:
        nested: Final = metadata["native_mcp_policy"]
        nested_policy: Final = NativeMCPPolicy.model_validate(nested) if nested is not None else None
        if policy != nested_policy:
            raise ValueError("native_mcp_policy conflicts with metadata.native_mcp_policy")
    persisted: Final[Mapping[str, object] | None] = policy.model_dump(mode="json") if policy is not None else None
    return {
        **data,
        **({"native_mcp_policy": policy} if "native_mcp_policy" in data else {}),
        **({"metadata": {**metadata, "native_mcp_policy": persisted}} if "native_mcp_policy" in metadata else {}),
    }
