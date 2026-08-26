from collections.abc import Callable
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict


class MCPTool(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(arbitrary_types_allowed=True)
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable


class ToolSchema(BaseModel):
    name: str
    description: str
    inputSchema: dict[str, Any]


class ListToolsResponse(BaseModel):
    tools: list[ToolSchema]
    nextCursor: str | None = None
    _meta: dict[str, Any] | None = None


class CallToolRequest(BaseModel):
    method: str = "tools/call"
    params: dict[str, Any]


class ContentItem(BaseModel):
    type: str
    text: str | None = None
