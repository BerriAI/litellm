# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing_extensions import Literal

from ..._models import BaseModel

__all__ = ["ResponseFunctionCallArgumentsDeltaEvent"]


class ResponseFunctionCallArgumentsDeltaEvent(BaseModel):
    delta: str
    """The function-call arguments delta that is added."""

    item_id: str
    """The ID of the output item that the function-call arguments delta is added to."""

    output_index: int
    """
    The index of the output item that the function-call arguments delta is added to.
    """

    sequence_number: int
    """The sequence number of this event."""

    type: Literal["response.function_call_arguments.delta"]
    """The type of the event. Always `response.function_call_arguments.delta`."""
