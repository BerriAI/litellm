# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import Optional

from ...._models import BaseModel

__all__ = ["RealtimeResponseUsage", "InputTokenDetails", "OutputTokenDetails"]


class InputTokenDetails(BaseModel):
    audio_tokens: Optional[int] = None
    """The number of audio tokens used in the Response."""

    cached_tokens: Optional[int] = None
    """The number of cached tokens used in the Response."""

    text_tokens: Optional[int] = None
    """The number of text tokens used in the Response."""


class OutputTokenDetails(BaseModel):
    audio_tokens: Optional[int] = None
    """The number of audio tokens used in the Response."""

    text_tokens: Optional[int] = None
    """The number of text tokens used in the Response."""


class RealtimeResponseUsage(BaseModel):
    input_token_details: Optional[InputTokenDetails] = None
    """Details about the input tokens used in the Response."""

    input_tokens: Optional[int] = None
    """
    The number of input tokens used in the Response, including text and audio
    tokens.
    """

    output_token_details: Optional[OutputTokenDetails] = None
    """Details about the output tokens used in the Response."""

    output_tokens: Optional[int] = None
    """
    The number of output tokens sent in the Response, including text and audio
    tokens.
    """

    total_tokens: Optional[int] = None
    """
    The total number of tokens in the Response including input and output text and
    audio tokens.
    """
