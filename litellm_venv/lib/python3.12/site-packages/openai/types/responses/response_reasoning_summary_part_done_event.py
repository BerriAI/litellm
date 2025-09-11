# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing_extensions import Literal

from ..._models import BaseModel

__all__ = ["ResponseReasoningSummaryPartDoneEvent", "Part"]


class Part(BaseModel):
    text: str
    """The text of the summary part."""

    type: Literal["summary_text"]
    """The type of the summary part. Always `summary_text`."""


class ResponseReasoningSummaryPartDoneEvent(BaseModel):
    item_id: str
    """The ID of the item this summary part is associated with."""

    output_index: int
    """The index of the output item this summary part is associated with."""

    part: Part
    """The completed summary part."""

    sequence_number: int
    """The sequence number of this event."""

    summary_index: int
    """The index of the summary part within the reasoning summary."""

    type: Literal["response.reasoning_summary_part.done"]
    """The type of the event. Always `response.reasoning_summary_part.done`."""
