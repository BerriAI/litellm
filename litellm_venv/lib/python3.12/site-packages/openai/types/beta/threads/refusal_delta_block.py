# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import Optional
from typing_extensions import Literal

from ...._models import BaseModel

__all__ = ["RefusalDeltaBlock"]


class RefusalDeltaBlock(BaseModel):
    index: int
    """The index of the refusal part in the message."""

    type: Literal["refusal"]
    """Always `refusal`."""

    refusal: Optional[str] = None
