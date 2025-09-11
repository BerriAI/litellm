# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing_extensions import Literal

from ..._models import BaseModel

__all__ = ["ResponseMcpCallArgumentsDeltaEvent"]


class ResponseMcpCallArgumentsDeltaEvent(BaseModel):
    delta: str
    """
    A JSON string containing the partial update to the arguments for the MCP tool
    call.
    """

    item_id: str
    """The unique identifier of the MCP tool call item being processed."""

    output_index: int
    """The index of the output item in the response's output array."""

    sequence_number: int
    """The sequence number of this event."""

    type: Literal["response.mcp_call_arguments.delta"]
    """The type of the event. Always 'response.mcp_call_arguments.delta'."""
