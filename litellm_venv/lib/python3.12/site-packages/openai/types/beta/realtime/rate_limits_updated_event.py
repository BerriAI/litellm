# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import List, Optional
from typing_extensions import Literal

from ...._models import BaseModel

__all__ = ["RateLimitsUpdatedEvent", "RateLimit"]


class RateLimit(BaseModel):
    limit: Optional[int] = None
    """The maximum allowed value for the rate limit."""

    name: Optional[Literal["requests", "tokens"]] = None
    """The name of the rate limit (`requests`, `tokens`)."""

    remaining: Optional[int] = None
    """The remaining value before the limit is reached."""

    reset_seconds: Optional[float] = None
    """Seconds until the rate limit resets."""


class RateLimitsUpdatedEvent(BaseModel):
    event_id: str
    """The unique ID of the server event."""

    rate_limits: List[RateLimit]
    """List of rate limit information."""

    type: Literal["rate_limits.updated"]
    """The event type, must be `rate_limits.updated`."""
