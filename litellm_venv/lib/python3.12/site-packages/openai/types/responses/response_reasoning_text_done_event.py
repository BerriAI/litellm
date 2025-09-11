# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing_extensions import Literal

from ..._models import BaseModel

__all__ = ["ResponseReasoningTextDoneEvent"]


class ResponseReasoningTextDoneEvent(BaseModel):
    content_index: int
    """The index of the reasoning content part."""

    item_id: str
    """The ID of the item this reasoning text is associated with."""

    output_index: int
    """The index of the output item this reasoning text is associated with."""

    sequence_number: int
    """The sequence number of this event."""

    text: str
    """The full text of the completed reasoning content."""

    type: Literal["response.reasoning_text.done"]
    """The type of the event. Always `response.reasoning_text.done`."""
