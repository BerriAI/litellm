# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing_extensions import Literal

from ...._models import BaseModel
from .transcription_session import TranscriptionSession

__all__ = ["TranscriptionSessionUpdatedEvent"]


class TranscriptionSessionUpdatedEvent(BaseModel):
    event_id: str
    """The unique ID of the server event."""

    session: TranscriptionSession
    """A new Realtime transcription session configuration.

    When a session is created on the server via REST API, the session object also
    contains an ephemeral key. Default TTL for keys is 10 minutes. This property is
    not present when a session is updated via the WebSocket API.
    """

    type: Literal["transcription_session.updated"]
    """The event type, must be `transcription_session.updated`."""
