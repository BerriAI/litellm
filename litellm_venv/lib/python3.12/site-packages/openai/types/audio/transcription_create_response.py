# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import Union
from typing_extensions import TypeAlias

from .transcription import Transcription
from .transcription_verbose import TranscriptionVerbose

__all__ = ["TranscriptionCreateResponse"]

TranscriptionCreateResponse: TypeAlias = Union[Transcription, TranscriptionVerbose]
