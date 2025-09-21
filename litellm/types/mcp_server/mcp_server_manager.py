from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict
from typing_extensions import TypedDict

from litellm.proxy._types import MCPAuthType, MCPTransportType
from litellm.types.mcp import MCPServerCostInfo


class MCPInfo(TypedDict, total=False):
    server_name: str
    description: Optional[str]
    logo_url: Optional[str]
    mcp_server_cost_info: Optional[MCPServerCostInfo]


class MCPOAuthInfo(BaseModel):
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    scopes: Optional[List[str]] = None
    authorization_url: Optional[str] = None
    token_url: Optional[str] = None


class MCPServer(BaseModel):
    server_id: str
    name: str
    alias: Optional[str] = None
    server_name: Optional[str] = None
    url: Optional[str] = None
    transport: MCPTransportType
    auth_type: Optional[MCPAuthType] = None
    authentication_token: Optional[str] = None
    oauth_info: Optional[MCPOAuthInfo] = None
    mcp_info: Optional[MCPInfo] = None
    # Stdio-specific fields
    command: Optional[str] = None
    args: Optional[List[str]] = None
    env: Optional[Dict[str, str]] = None
    access_groups: Optional[List[str]] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)
