# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing_extensions import Literal

from ..._models import BaseModel

__all__ = ["ResponseMcpCallFailedEvent"]


class ResponseMcpCallFailedEvent(BaseModel):
    item_id: str
    """The ID of the MCP tool call item that failed."""

    output_index: int
    """The index of the output item that failed."""

    sequence_number: int
    """The sequence number of this event."""

    type: Literal["response.mcp_call.failed"]
    """The type of the event. Always 'response.mcp_call.failed'."""
