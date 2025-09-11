# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing_extensions import Literal

from ..._models import BaseModel

__all__ = ["ResponseReasoningTextDeltaEvent"]


class ResponseReasoningTextDeltaEvent(BaseModel):
    content_index: int
    """The index of the reasoning content part this delta is associated with."""

    delta: str
    """The text delta that was added to the reasoning content."""

    item_id: str
    """The ID of the item this reasoning text delta is associated with."""

    output_index: int
    """The index of the output item this reasoning text delta is associated with."""

    sequence_number: int
    """The sequence number of this event."""

    type: Literal["response.reasoning_text.delta"]
    """The type of the event. Always `response.reasoning_text.delta`."""
