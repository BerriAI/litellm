from datetime import datetime

from pydantic import Field
from typing_extensions import TypedDict

from litellm.types.llms.base import LiteLLMBaseModel


class MCPToolsetTool(TypedDict):
    server_id: str
    tool_name: str


class MCPToolset(LiteLLMBaseModel):
    toolset_id: str
    toolset_name: str
    description: str | None = None
    tools: list[MCPToolsetTool] = Field(default=[])
    created_at: datetime | None = None
    created_by: str | None = None
    updated_at: datetime | None = None
    updated_by: str | None = None


class NewMCPToolsetRequest(LiteLLMBaseModel):
    toolset_name: str
    description: str | None = None
    tools: list[MCPToolsetTool] = Field(default=[])


class UpdateMCPToolsetRequest(LiteLLMBaseModel):
    toolset_id: str
    toolset_name: str | None = None
    description: str | None = None
    tools: list[MCPToolsetTool] | None = None
