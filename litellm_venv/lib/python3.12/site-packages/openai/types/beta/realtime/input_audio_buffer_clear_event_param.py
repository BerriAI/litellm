# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing_extensions import Literal, Required, TypedDict

__all__ = ["InputAudioBufferClearEventParam"]


class InputAudioBufferClearEventParam(TypedDict, total=False):
    type: Required[Literal["input_audio_buffer.clear"]]
    """The event type, must be `input_audio_buffer.clear`."""

    event_id: str
    """Optional client-generated ID used to identify this event."""
