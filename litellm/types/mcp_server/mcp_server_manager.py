from typing import TYPE_CHECKING, Optional

from pydantic import BaseModel, ConfigDict
from typing_extensions import TypedDict

if TYPE_CHECKING:
    from mcp import ClientSession
    from mcp.types import Tool as MCPTool
else:
    # Provide fallback types for runtime incase `mcp` is not installed
    ClientSession = None
    MCPTool = object


class MCPInfo(TypedDict, total=False):
    server_name: str
    logo_url: Optional[str]


class MCPSSEServer(BaseModel):
    name: str
    url: str
    mcp_info: Optional[MCPInfo] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)


class ListMCPToolsRestAPIResponseObject(MCPTool):
    """
    Object returned by the /tools/list REST API route.
    """

    mcp_info: Optional[MCPInfo] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)
