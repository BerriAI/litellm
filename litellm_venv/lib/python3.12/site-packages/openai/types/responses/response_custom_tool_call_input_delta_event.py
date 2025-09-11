# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing_extensions import Literal

from ..._models import BaseModel

__all__ = ["ResponseCustomToolCallInputDeltaEvent"]


class ResponseCustomToolCallInputDeltaEvent(BaseModel):
    delta: str
    """The incremental input data (delta) for the custom tool call."""

    item_id: str
    """Unique identifier for the API item associated with this event."""

    output_index: int
    """The index of the output this delta applies to."""

    sequence_number: int
    """The sequence number of this event."""

    type: Literal["response.custom_tool_call_input.delta"]
    """The event type identifier."""
