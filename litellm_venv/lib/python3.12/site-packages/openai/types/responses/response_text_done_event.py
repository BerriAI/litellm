# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import List, Optional
from typing_extensions import Literal

from ..._models import BaseModel

__all__ = ["ResponseTextDoneEvent", "Logprob", "LogprobTopLogprob"]


class LogprobTopLogprob(BaseModel):
    token: Optional[str] = None
    """A possible text token."""

    logprob: Optional[float] = None
    """The log probability of this token."""


class Logprob(BaseModel):
    token: str
    """A possible text token."""

    logprob: float
    """The log probability of this token."""

    top_logprobs: Optional[List[LogprobTopLogprob]] = None
    """The log probability of the top 20 most likely tokens."""


class ResponseTextDoneEvent(BaseModel):
    content_index: int
    """The index of the content part that the text content is finalized."""

    item_id: str
    """The ID of the output item that the text content is finalized."""

    logprobs: List[Logprob]
    """The log probabilities of the tokens in the delta."""

    output_index: int
    """The index of the output item that the text content is finalized."""

    sequence_number: int
    """The sequence number for this event."""

    text: str
    """The text content that is finalized."""

    type: Literal["response.output_text.done"]
    """The type of the event. Always `response.output_text.done`."""
