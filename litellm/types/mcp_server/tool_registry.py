from collections.abc import Callable
from typing import Any, ClassVar

from pydantic import ConfigDict, Field

from litellm.types.llms.base import LiteLLMBaseModel


class MCPTool(LiteLLMBaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(arbitrary_types_allowed=True)
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable
    server_id: str | None = Field(default=None, frozen=True)


class ToolSchema(LiteLLMBaseModel):
    name: str
    description: str
    inputSchema: dict[str, Any]


class ListToolsResponse(LiteLLMBaseModel):
    tools: list[ToolSchema]
    nextCursor: str | None = None
    _meta: dict[str, Any] | None = None


class CallToolRequest(LiteLLMBaseModel):
    method: str = "tools/call"
    params: dict[str, Any]


class ContentItem(LiteLLMBaseModel):
    type: str
    text: str | None = None
