# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from datetime import datetime

from ..._models import BaseModel

__all__ = ["BetaContainer"]


class BetaContainer(BaseModel):
    id: str
    """Identifier for the container used in this request"""

    expires_at: datetime
    """The time at which the container will expire."""
