# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import Optional
from typing_extensions import Literal

from ....._models import BaseModel

__all__ = ["RunStepDeltaMessageDelta", "MessageCreation"]


class MessageCreation(BaseModel):
    message_id: Optional[str] = None
    """The ID of the message that was created by this run step."""


class RunStepDeltaMessageDelta(BaseModel):
    type: Literal["message_creation"]
    """Always `message_creation`."""

    message_creation: Optional[MessageCreation] = None
