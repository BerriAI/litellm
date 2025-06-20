import enum
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict
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
