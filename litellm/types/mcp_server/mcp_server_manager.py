from typing import Any, Dict, List, Optional

from mcp import ClientSession
from mcp.types import Tool as MCPTool
from pydantic import BaseModel, ConfigDict


class MCPSSEServer(BaseModel):
    name: str
    url: str
    client_session: Optional[ClientSession] = None
    mcp_info: Optional[Dict[str, Any]] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)


class ListMCPToolsRestAPIResponseObject(BaseModel):
    """
    Object returned by the /tools/list REST API route.
    """

    tools: List[MCPTool]
    mcp_info: Optional[Dict[str, Any]] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)
