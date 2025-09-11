# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing_extensions import Literal

from ...._models import BaseModel

__all__ = ["ResponseFunctionCallArgumentsDeltaEvent"]


class ResponseFunctionCallArgumentsDeltaEvent(BaseModel):
    call_id: str
    """The ID of the function call."""

    delta: str
    """The arguments delta as a JSON string."""

    event_id: str
    """The unique ID of the server event."""

    item_id: str
    """The ID of the function call item."""

    output_index: int
    """The index of the output item in the response."""

    response_id: str
    """The ID of the response."""

    type: Literal["response.function_call_arguments.delta"]
    """The event type, must be `response.function_call_arguments.delta`."""
