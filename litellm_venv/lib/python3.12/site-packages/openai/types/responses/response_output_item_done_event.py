# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing_extensions import Literal

from ..._models import BaseModel
from .response_output_item import ResponseOutputItem

__all__ = ["ResponseOutputItemDoneEvent"]


class ResponseOutputItemDoneEvent(BaseModel):
    item: ResponseOutputItem
    """The output item that was marked done."""

    output_index: int
    """The index of the output item that was marked done."""

    sequence_number: int
    """The sequence number of this event."""

    type: Literal["response.output_item.done"]
    """The type of the event. Always `response.output_item.done`."""
