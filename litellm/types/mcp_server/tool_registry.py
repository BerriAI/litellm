from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel


class MCPTool(BaseModel):
    name: str
    description: str
    input_schema: Dict[str, Any]
    handler: Callable

    class Config:
        arbitrary_types_allowed = True


class ToolSchema(BaseModel):
    name: str
    description: str
    inputSchema: Dict[str, Any]


class ListToolsResponse(BaseModel):
    tools: List[ToolSchema]
    nextCursor: Optional[str] = None
    _meta: Optional[Dict[str, Any]] = None


class CallToolRequest(BaseModel):
    method: str = "tools/call"
    params: Dict[str, Any]


class ContentItem(BaseModel):
    type: str
    text: Optional[str] = None
