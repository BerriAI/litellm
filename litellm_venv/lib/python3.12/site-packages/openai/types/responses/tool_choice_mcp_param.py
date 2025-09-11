# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing import Optional
from typing_extensions import Literal, Required, TypedDict

__all__ = ["ToolChoiceMcpParam"]


class ToolChoiceMcpParam(TypedDict, total=False):
    server_label: Required[str]
    """The label of the MCP server to use."""

    type: Required[Literal["mcp"]]
    """For MCP tools, the type is always `mcp`."""

    name: Optional[str]
    """The name of the tool to call on the server."""
