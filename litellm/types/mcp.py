import enum
from typing import Any, Dict, Literal, Optional

from mcp.types import EmbeddedResource as MCPEmbeddedResource
from mcp.types import ImageContent as MCPImageContent
from mcp.types import TextContent as MCPTextContent
from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import TypedDict


class MCPTransport(str, enum.Enum):
    sse = "sse"
    http = "http"


class MCPSpecVersion(str, enum.Enum):
    nov_2024 = "2024-11-05"
    mar_2025 = "2025-03-26"

class MCPAuth(str, enum.Enum):
    none = "none"
    api_key = "api_key"
    bearer_token = "bearer_token"
    basic = "basic"


# MCP Literals
MCPTransportType = Literal[MCPTransport.sse, MCPTransport.http]
MCPSpecVersionType = Literal[MCPSpecVersion.nov_2024, MCPSpecVersion.mar_2025]
MCPAuthType = Optional[
    Literal[MCPAuth.none, MCPAuth.api_key, MCPAuth.bearer_token, MCPAuth.basic]
]



class MCPServerCostInfo(TypedDict, total=False):
    default_cost_per_query: Optional[float]
    """
    Default cost per query for the MCP server tool call
    """

    tool_name_to_cost_per_query: Optional[Dict[str, float]]
    """
    Granular, set a custom cost for each tool in the MCP server
    """


########################################################################################


class LiteLLM_MCPTextContent(MCPTextContent):
    type: Literal["text"]
    text: str
    _hidden_params: Dict[str, Any] = Field(default_factory=dict, exclude=True)


class LiteLLM_MCPImageContent(MCPImageContent):
    type: Literal["image"]
    data: str
    mimeType: str
    _hidden_params: Dict[str, Any] = Field(default_factory=dict, exclude=True)

class LiteLLM_MCPEmbeddedResource(MCPEmbeddedResource):
    type: Literal["resource"]
    data: str
    mimeType: str
    _hidden_params: Dict[str, Any] = Field(default_factory=dict, exclude=True)
