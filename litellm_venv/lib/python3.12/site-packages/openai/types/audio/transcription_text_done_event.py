# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import List, Optional
from typing_extensions import Literal

from ..._models import BaseModel

__all__ = ["TranscriptionTextDoneEvent", "Logprob", "Usage", "UsageInputTokenDetails"]


class Logprob(BaseModel):
    token: Optional[str] = None
    """The token that was used to generate the log probability."""

    bytes: Optional[List[int]] = None
    """The bytes that were used to generate the log probability."""

    logprob: Optional[float] = None
    """The log probability of the token."""


class UsageInputTokenDetails(BaseModel):
    audio_tokens: Optional[int] = None
    """Number of audio tokens billed for this request."""

    text_tokens: Optional[int] = None
    """Number of text tokens billed for this request."""


class Usage(BaseModel):
    input_tokens: int
    """Number of input tokens billed for this request."""

    output_tokens: int
    """Number of output tokens generated."""

    total_tokens: int
    """Total number of tokens used (input + output)."""

    type: Literal["tokens"]
    """The type of the usage object. Always `tokens` for this variant."""

    input_token_details: Optional[UsageInputTokenDetails] = None
    """Details about the input tokens billed for this request."""


class TranscriptionTextDoneEvent(BaseModel):
    text: str
    """The text that was transcribed."""

    type: Literal["transcript.text.done"]
    """The type of the event. Always `transcript.text.done`."""

    logprobs: Optional[List[Logprob]] = None
    """The log probabilities of the individual tokens in the transcription.

    Only included if you
    [create a transcription](https://platform.openai.com/docs/api-reference/audio/create-transcription)
    with the `include[]` parameter set to `logprobs`.
    """

    usage: Optional[Usage] = None
    """Usage statistics for models billed by token usage."""
