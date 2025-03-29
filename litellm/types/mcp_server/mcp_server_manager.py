from typing import Any, Dict, List, Optional

from mcp import ClientSession
from mcp.types import Tool as MCPTool
from pydantic import BaseModel, ConfigDict
from typing_extensions import TypedDict


class MCPInfo(TypedDict, total=False):
    server_name: str
    logo_url: Optional[str]


class MCPSSEServer(BaseModel):
    name: str
    url: str
    client_session: Optional[ClientSession] = None
    mcp_info: Optional[MCPInfo] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)


class ListMCPToolsRestAPIResponseObject(MCPTool):
    """
    Object returned by the /tools/list REST API route.
    """

    mcp_info: Optional[MCPInfo] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)
