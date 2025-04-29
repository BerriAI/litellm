from typing import TYPE_CHECKING, Optional

from pydantic import BaseModel, ConfigDict
from typing_extensions import TypedDict


class MCPInfo(TypedDict, total=False):
    server_name: str
    logo_url: Optional[str]


class MCPSSEServer(BaseModel):
    name: str
    url: str
    mcp_info: Optional[MCPInfo] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)
