from datetime import datetime

from pydantic import BaseModel
from typing_extensions import TypedDict


class MCPToolsetTool(TypedDict):
    server_id: str
    tool_name: str


class MCPToolset(BaseModel):
    toolset_id: str
    toolset_name: str
    description: str | None = None
    tools: list[MCPToolsetTool] = []
    created_at: datetime | None = None
    created_by: str | None = None
    updated_at: datetime | None = None
    updated_by: str | None = None


class NewMCPToolsetRequest(BaseModel):
    toolset_name: str
    description: str | None = None
    tools: list[MCPToolsetTool] = []


class UpdateMCPToolsetRequest(BaseModel):
    toolset_id: str
    toolset_name: str | None = None
    description: str | None = None
    tools: list[MCPToolsetTool] | None = None
