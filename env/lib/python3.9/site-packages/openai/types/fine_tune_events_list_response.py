# File generated from our OpenAPI spec by Stainless.

from typing import List
from typing_extensions import Literal

from .._models import BaseModel
from .fine_tune_event import FineTuneEvent

__all__ = ["FineTuneEventsListResponse"]


class FineTuneEventsListResponse(BaseModel):
    data: List[FineTuneEvent]

    object: Literal["list"]
