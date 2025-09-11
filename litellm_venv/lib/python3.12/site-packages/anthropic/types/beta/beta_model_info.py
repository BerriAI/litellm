# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from datetime import datetime
from typing_extensions import Literal

from ..._models import BaseModel

__all__ = ["BetaModelInfo"]


class BetaModelInfo(BaseModel):
    id: str
    """Unique model identifier."""

    created_at: datetime
    """RFC 3339 datetime string representing the time at which the model was released.

    May be set to an epoch value if the release date is unknown.
    """

    display_name: str
    """A human-readable name for the model."""

    type: Literal["model"]
    """Object type.

    For Models, this is always `"model"`.
    """
