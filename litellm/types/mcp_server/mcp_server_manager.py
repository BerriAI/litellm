from typing import TYPE_CHECKING, Optional

from pydantic import BaseModel, ConfigDict
from typing_extensions import TypedDict


class MCPInfo(TypedDict, total=False):
    server_name: str
    description: Optional[str]
    logo_url: Optional[str]


class MCPServer(BaseModel):
    name: str
    url: str
    # TODO: alter the types to be the Literal explicit
    transport: str
    spec_version: str
    auth_type: Optional[str] = None
    mcp_info: Optional[MCPInfo] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)
