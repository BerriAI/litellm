# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing_extensions import Literal, Required, TypedDict

__all__ = ["InputAudioBufferCommitEventParam"]


class InputAudioBufferCommitEventParam(TypedDict, total=False):
    type: Required[Literal["input_audio_buffer.commit"]]
    """The event type, must be `input_audio_buffer.commit`."""

    event_id: str
    """Optional client-generated ID used to identify this event."""
