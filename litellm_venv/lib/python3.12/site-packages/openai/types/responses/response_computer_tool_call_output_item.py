# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import List, Optional
from typing_extensions import Literal

from ..._models import BaseModel
from .response_computer_tool_call_output_screenshot import (
    ResponseComputerToolCallOutputScreenshot,
)

__all__ = ["ResponseComputerToolCallOutputItem", "AcknowledgedSafetyCheck"]


class AcknowledgedSafetyCheck(BaseModel):
    id: str
    """The ID of the pending safety check."""

    code: str
    """The type of the pending safety check."""

    message: str
    """Details about the pending safety check."""


class ResponseComputerToolCallOutputItem(BaseModel):
    id: str
    """The unique ID of the computer call tool output."""

    call_id: str
    """The ID of the computer tool call that produced the output."""

    output: ResponseComputerToolCallOutputScreenshot
    """A computer screenshot image used with the computer use tool."""

    type: Literal["computer_call_output"]
    """The type of the computer tool call output. Always `computer_call_output`."""

    acknowledged_safety_checks: Optional[List[AcknowledgedSafetyCheck]] = None
    """
    The safety checks reported by the API that have been acknowledged by the
    developer.
    """

    status: Optional[Literal["in_progress", "completed", "incomplete"]] = None
    """The status of the message input.

    One of `in_progress`, `completed`, or `incomplete`. Populated when input items
    are returned via API.
    """
