# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing_extensions import Literal

from ..._models import BaseModel

__all__ = ["ResponseWebSearchCallInProgressEvent"]


class ResponseWebSearchCallInProgressEvent(BaseModel):
    item_id: str
    """Unique ID for the output item associated with the web search call."""

    output_index: int
    """The index of the output item that the web search call is associated with."""

    sequence_number: int
    """The sequence number of the web search call being processed."""

    type: Literal["response.web_search_call.in_progress"]
    """The type of the event. Always `response.web_search_call.in_progress`."""
