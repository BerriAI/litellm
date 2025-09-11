# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing_extensions import Literal

from .session import Session
from ...._models import BaseModel

__all__ = ["SessionUpdatedEvent"]


class SessionUpdatedEvent(BaseModel):
    event_id: str
    """The unique ID of the server event."""

    session: Session
    """Realtime session object configuration."""

    type: Literal["session.updated"]
    """The event type, must be `session.updated`."""
