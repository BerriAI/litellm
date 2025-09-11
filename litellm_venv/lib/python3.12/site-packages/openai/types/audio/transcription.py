# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import List, Union, Optional
from typing_extensions import Literal, Annotated, TypeAlias

from ..._utils import PropertyInfo
from ..._models import BaseModel

__all__ = [
    "Transcription",
    "Logprob",
    "Usage",
    "UsageTokens",
    "UsageTokensInputTokenDetails",
    "UsageDuration",
]


class Logprob(BaseModel):
    token: Optional[str] = None
    """The token in the transcription."""

    bytes: Optional[List[float]] = None
    """The bytes of the token."""

    logprob: Optional[float] = None
    """The log probability of the token."""


class UsageTokensInputTokenDetails(BaseModel):
    audio_tokens: Optional[int] = None
    """Number of audio tokens billed for this request."""

    text_tokens: Optional[int] = None
    """Number of text tokens billed for this request."""


class UsageTokens(BaseModel):
    input_tokens: int
    """Number of input tokens billed for this request."""

    output_tokens: int
    """Number of output tokens generated."""

    total_tokens: int
    """Total number of tokens used (input + output)."""

    type: Literal["tokens"]
    """The type of the usage object. Always `tokens` for this variant."""

    input_token_details: Optional[UsageTokensInputTokenDetails] = None
    """Details about the input tokens billed for this request."""


class UsageDuration(BaseModel):
    seconds: float
    """Duration of the input audio in seconds."""

    type: Literal["duration"]
    """The type of the usage object. Always `duration` for this variant."""


Usage: TypeAlias = Annotated[
    Union[UsageTokens, UsageDuration], PropertyInfo(discriminator="type")
]


class Transcription(BaseModel):
    text: str
    """The transcribed text."""

    logprobs: Optional[List[Logprob]] = None
    """The log probabilities of the tokens in the transcription.

    Only returned with the models `gpt-4o-transcribe` and `gpt-4o-mini-transcribe`
    if `logprobs` is added to the `include` array.
    """

    usage: Optional[Usage] = None
    """Token usage statistics for the request."""
