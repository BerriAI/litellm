# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing_extensions import Literal

from ..._models import BaseModel

__all__ = ["ResponseRefusalDeltaEvent"]


class ResponseRefusalDeltaEvent(BaseModel):
    content_index: int
    """The index of the content part that the refusal text is added to."""

    delta: str
    """The refusal text that is added."""

    item_id: str
    """The ID of the output item that the refusal text is added to."""

    output_index: int
    """The index of the output item that the refusal text is added to."""

    sequence_number: int
    """The sequence number of this event."""

    type: Literal["response.refusal.delta"]
    """The type of the event. Always `response.refusal.delta`."""
