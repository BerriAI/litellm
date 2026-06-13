"""types — the single vocabulary import point for the gateway.

Everything else imports the FP spine, the SDK wire types, and the gateway
error/result vocabulary from here, so there is exactly one place to look for
the project's nouns.
"""

from __future__ import annotations

from expression import (
    Error,
    Nothing,
    Ok,
    Option,
    Result,
    Some,
    case,
    pipe,
    tag,
    tagged_union,
)
from mcp.types import CallToolResult, ListToolsResult, TextContent, Tool
from pydantic import BaseModel, ConfigDict

from litellm.proxy.gateway.mcp.foundation.errors import GatewayError
from litellm.proxy.gateway.mcp.foundation.result import GatewayResult


class Subject(BaseModel):
    model_config = ConfigDict(frozen=True)

    subject_id: str
    tenant: str


__all__ = [
    "BaseModel",
    "CallToolResult",
    "ConfigDict",
    "Error",
    "GatewayError",
    "GatewayResult",
    "ListToolsResult",
    "Nothing",
    "Ok",
    "Option",
    "Result",
    "Some",
    "Subject",
    "TextContent",
    "Tool",
    "case",
    "pipe",
    "tag",
    "tagged_union",
]
