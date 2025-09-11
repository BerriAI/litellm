# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import List, Optional

from ...._models import BaseModel
from .annotation_delta import AnnotationDelta

__all__ = ["TextDelta"]


class TextDelta(BaseModel):
    annotations: Optional[List[AnnotationDelta]] = None

    value: Optional[str] = None
    """The data that makes up the text."""
