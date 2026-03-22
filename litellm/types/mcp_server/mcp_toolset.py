from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel
from typing_extensions import TypedDict


class MCPToolsetTool(TypedDict):
    server_id: str
    tool_name: str


class MCPToolset(BaseModel):
    toolset_id: str
    toolset_name: str
    description: Optional[str] = None
    tools: List[MCPToolsetTool] = []
    created_at: Optional[datetime] = None
    created_by: Optional[str] = None
    updated_at: Optional[datetime] = None
    updated_by: Optional[str] = None


class NewMCPToolsetRequest(BaseModel):
    toolset_name: str
    description: Optional[str] = None
    tools: List[MCPToolsetTool] = []


class UpdateMCPToolsetRequest(BaseModel):
    toolset_id: str
    toolset_name: Optional[str] = None
    description: Optional[str] = None
    tools: Optional[List[MCPToolsetTool]] = None
