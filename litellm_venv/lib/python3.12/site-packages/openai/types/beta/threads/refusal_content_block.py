# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing_extensions import Literal

from ...._models import BaseModel

__all__ = ["RefusalContentBlock"]


class RefusalContentBlock(BaseModel):
    refusal: str

    type: Literal["refusal"]
    """Always `refusal`."""
