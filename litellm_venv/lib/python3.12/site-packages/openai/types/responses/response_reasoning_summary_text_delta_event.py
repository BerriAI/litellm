# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing_extensions import Literal

from ..._models import BaseModel

__all__ = ["ResponseReasoningSummaryTextDeltaEvent"]


class ResponseReasoningSummaryTextDeltaEvent(BaseModel):
    delta: str
    """The text delta that was added to the summary."""

    item_id: str
    """The ID of the item this summary text delta is associated with."""

    output_index: int
    """The index of the output item this summary text delta is associated with."""

    sequence_number: int
    """The sequence number of this event."""

    summary_index: int
    """The index of the summary part within the reasoning summary."""

    type: Literal["response.reasoning_summary_text.delta"]
    """The type of the event. Always `response.reasoning_summary_text.delta`."""
