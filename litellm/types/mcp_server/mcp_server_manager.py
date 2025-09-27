from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict
from typing_extensions import TypedDict

from litellm.proxy._types import MCPAuthType, MCPTransportType
from litellm.types.mcp import MCPServerCostInfo


# MCPInfo now allows arbitrary additional fields for custom metadata
MCPInfo = Dict[str, Any]


class MCPServer(BaseModel):
    server_id: str
    name: str
    alias: Optional[str] = None
    server_name: Optional[str] = None
    url: Optional[str] = None
    transport: MCPTransportType
    auth_type: Optional[MCPAuthType] = None
    authentication_token: Optional[str] = None
    mcp_info: Optional[MCPInfo] = None
    # Stdio-specific fields
    command: Optional[str] = None
    args: Optional[List[str]] = None
    env: Optional[Dict[str, str]] = None
    access_groups: Optional[List[str]] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)
