# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing_extensions import Literal, Required, TypedDict

__all__ = ["InputAudioBufferAppendEventParam"]


class InputAudioBufferAppendEventParam(TypedDict, total=False):
    audio: Required[str]
    """Base64-encoded audio bytes.

    This must be in the format specified by the `input_audio_format` field in the
    session configuration.
    """

    type: Required[Literal["input_audio_buffer.append"]]
    """The event type, must be `input_audio_buffer.append`."""

    event_id: str
    """Optional client-generated ID used to identify this event."""
