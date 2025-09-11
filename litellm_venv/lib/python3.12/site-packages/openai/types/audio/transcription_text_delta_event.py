# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import List, Optional
from typing_extensions import Literal

from ..._models import BaseModel

__all__ = ["TranscriptionTextDeltaEvent", "Logprob"]


class Logprob(BaseModel):
    token: Optional[str] = None
    """The token that was used to generate the log probability."""

    bytes: Optional[List[int]] = None
    """The bytes that were used to generate the log probability."""

    logprob: Optional[float] = None
    """The log probability of the token."""


class TranscriptionTextDeltaEvent(BaseModel):
    delta: str
    """The text delta that was additionally transcribed."""

    type: Literal["transcript.text.delta"]
    """The type of the event. Always `transcript.text.delta`."""

    logprobs: Optional[List[Logprob]] = None
    """The log probabilities of the delta.

    Only included if you
    [create a transcription](https://platform.openai.com/docs/api-reference/audio/create-transcription)
    with the `include[]` parameter set to `logprobs`.
    """
