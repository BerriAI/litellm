# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import List, Union
from typing_extensions import Literal, TypeAlias

from ..._models import BaseModel
from .comparison_filter import ComparisonFilter

__all__ = ["CompoundFilter", "Filter"]

Filter: TypeAlias = Union[ComparisonFilter, object]


class CompoundFilter(BaseModel):
    filters: List[Filter]
    """Array of filters to combine.

    Items can be `ComparisonFilter` or `CompoundFilter`.
    """

    type: Literal["and", "or"]
    """Type of operation: `and` or `or`."""
