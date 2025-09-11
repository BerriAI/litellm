# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing_extensions import Literal

from ..._models import BaseModel

__all__ = ["ResponseRefusalDoneEvent"]


class ResponseRefusalDoneEvent(BaseModel):
    content_index: int
    """The index of the content part that the refusal text is finalized."""

    item_id: str
    """The ID of the output item that the refusal text is finalized."""

    output_index: int
    """The index of the output item that the refusal text is finalized."""

    refusal: str
    """The refusal text that is finalized."""

    sequence_number: int
    """The sequence number of this event."""

    type: Literal["response.refusal.done"]
    """The type of the event. Always `response.refusal.done`."""
