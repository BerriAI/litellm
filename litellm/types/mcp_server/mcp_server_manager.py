from typing import TYPE_CHECKING, Optional

from pydantic import BaseModel, ConfigDict
from typing_extensions import TypedDict

from litellm.proxy._types import MCPAuthType, MCPSpecVersionType, MCPTransportType


class MCPInfo(TypedDict, total=False):
    server_name: str
    description: Optional[str]
    logo_url: Optional[str]


class MCPServer(BaseModel):
    server_id: str
    name: str
    url: str
    # TODO: alter the types to be the Literal explicit
    transport: MCPTransportType
    spec_version: MCPSpecVersionType
    auth_type: Optional[MCPAuthType] = None
    mcp_info: Optional[MCPInfo] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)
