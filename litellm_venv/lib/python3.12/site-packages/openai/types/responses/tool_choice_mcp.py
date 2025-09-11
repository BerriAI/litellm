# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import Optional
from typing_extensions import Literal

from ..._models import BaseModel

__all__ = ["ToolChoiceMcp"]


class ToolChoiceMcp(BaseModel):
    server_label: str
    """The label of the MCP server to use."""

    type: Literal["mcp"]
    """For MCP tools, the type is always `mcp`."""

    name: Optional[str] = None
    """The name of the tool to call on the server."""
