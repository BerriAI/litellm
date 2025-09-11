# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing_extensions import Literal

from .session import Session
from ...._models import BaseModel

__all__ = ["SessionCreatedEvent"]


class SessionCreatedEvent(BaseModel):
    event_id: str
    """The unique ID of the server event."""

    session: Session
    """Realtime session object configuration."""

    type: Literal["session.created"]
    """The event type, must be `session.created`."""
