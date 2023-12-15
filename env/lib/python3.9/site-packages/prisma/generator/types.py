from typing import Mapping, Optional
from .._types import TypedDict


__all__ = (
    'PartialModel',
    'PartialModelField',
)


class PartialModelField(TypedDict):
    name: str
    is_list: bool
    optional: bool
    type: str
    documentation: Optional[str]
    is_relational: bool


class PartialModel(TypedDict):
    name: str
    from_model: str
    fields: Mapping[str, PartialModelField]
