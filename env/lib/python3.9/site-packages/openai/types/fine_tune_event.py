# File generated from our OpenAPI spec by Stainless.

from typing_extensions import Literal

from .._models import BaseModel

__all__ = ["FineTuneEvent"]


class FineTuneEvent(BaseModel):
    created_at: int

    level: str

    message: str

    object: Literal["fine-tune-event"]
