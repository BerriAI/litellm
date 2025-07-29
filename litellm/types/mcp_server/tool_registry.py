from typing import Any, Callable, ClassVar, Dict, List, Optional

from pydantic import BaseModel, ConfigDict


class MCPTool(BaseModel):

    model_config: ClassVar[ConfigDict] = ConfigDict(arbitrary_types_allowed=True)
    name: str
    description: str
    input_schema: Dict[str, Any]
    handler: Callable


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
