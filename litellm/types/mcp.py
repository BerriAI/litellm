import enum
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel
from typing_extensions import TypedDict

from litellm.types.llms.base import HiddenParams

if TYPE_CHECKING:
    from mcp.types import EmbeddedResource as MCPEmbeddedResource
    from mcp.types import ImageContent as MCPImageContent
    from mcp.types import TextContent as MCPTextContent
else:
    MCPEmbeddedResource = Any
    MCPImageContent = Any
    MCPTextContent = Any


class MCPTransport(str, enum.Enum):
    sse = "sse"
    http = "http"
    stdio = "stdio"


class MCPSpecVersion(str, enum.Enum):
    nov_2024 = "2024-11-05"
    mar_2025 = "2025-03-26"
    jun_2025 = "2025-06-18"


class MCPAuth(str, enum.Enum):
    none = "none"
    api_key = "api_key"
    bearer_token = "bearer_token"
    basic = "basic"
    authorization = "authorization"
    oauth2 = "oauth2"


# MCP Literals
MCPTransportType = Literal[MCPTransport.sse, MCPTransport.http, MCPTransport.stdio]
MCPSpecVersionType = Literal[
    MCPSpecVersion.nov_2024, MCPSpecVersion.mar_2025, MCPSpecVersion.jun_2025
]
MCPAuthType = Optional[
    Literal[
        MCPAuth.none,
        MCPAuth.api_key,
        MCPAuth.bearer_token,
        MCPAuth.basic,
        MCPAuth.authorization,
        MCPAuth.oauth2,
    ]
]


class MCPPublicServer(BaseModel):
    """
    Safe params for public MCP servers
    """

    server_id: str
    name: str
    alias: Optional[str] = None
    server_name: Optional[str] = None
    url: Optional[str] = None
    transport: MCPTransportType
    spec_path: Optional[str] = None
    auth_type: Optional[MCPAuthType] = None
    mcp_info: Optional[Dict[str, Any]] = None


class MCPCredentials(TypedDict, total=False):
    auth_value: Optional[str]
    """
    Authentication value
    """

    client_id: Optional[str]
    """
    OAuth 2.0 client identifier used when auth_type is oauth2
    """

    client_secret: Optional[str]
    """
    OAuth 2.0 client secret used when auth_type is oauth2
    """

    scopes: Optional[List[str]]
    """
    OAuth 2.0 scopes to request when exchanging the client credentials
    """


class MCPServerCostInfo(TypedDict, total=False):
    default_cost_per_query: Optional[float]
    """
    Default cost per query for the MCP server tool call
    """

    tool_name_to_cost_per_query: Optional[Dict[str, float]]
    """
    Granular, set a custom cost for each tool in the MCP server
    """


class MCPStdioConfig(TypedDict, total=False):
    command: str
    """
    Command to run the MCP server (e.g., 'npx', 'python', 'node')
    """

    args: List[str]
    """
    Arguments to pass to the command
    """

    env: Optional[Dict[str, str]]
    """
    Environment variables to set when running the command
    """


class MCPPreCallRequestObject(BaseModel):
    """
    Pydantic object used for MCP pre_call_hook request validation and modification
    """

    tool_name: str
    arguments: Dict[str, Any]
    server_name: Optional[str] = None
    user_api_key_auth: Optional[Dict[str, Any]] = None
    hidden_params: HiddenParams = HiddenParams()


class MCPPreCallResponseObject(BaseModel):
    """
    Pydantic object used for MCP pre_call_hook response
    """

    should_proceed: bool = True
    modified_arguments: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    hidden_params: HiddenParams = HiddenParams()


class MCPDuringCallRequestObject(BaseModel):
    """
    Pydantic object used for MCP during_call_hook request
    """

    tool_name: str
    arguments: Dict[str, Any]
    server_name: Optional[str] = None
    start_time: Optional[float] = None
    hidden_params: HiddenParams = HiddenParams()


class MCPDuringCallResponseObject(BaseModel):
    """
    Pydantic object used for MCP during_call_hook response
    """

    should_continue: bool = True
    error_message: Optional[str] = None
    hidden_params: HiddenParams = HiddenParams()


class MCPPostCallResponseObject(BaseModel):
    """
    Pydantic object used for MCP post_call_hook response
    """

    mcp_tool_call_response: List[
        Union[MCPTextContent, MCPImageContent, MCPEmbeddedResource]
    ]
    hidden_params: HiddenParams
