# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import List

from ...._models import BaseModel
from .annotation import Annotation

__all__ = ["Text"]


class Text(BaseModel):
    annotations: List[Annotation]

    value: str
    """The data that makes up the text."""
