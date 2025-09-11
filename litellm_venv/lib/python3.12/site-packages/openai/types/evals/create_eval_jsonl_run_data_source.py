# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import Dict, List, Union, Optional
from typing_extensions import Literal, Annotated, TypeAlias

from ..._utils import PropertyInfo
from ..._models import BaseModel

__all__ = [
    "CreateEvalJSONLRunDataSource",
    "Source",
    "SourceFileContent",
    "SourceFileContentContent",
    "SourceFileID",
]


class SourceFileContentContent(BaseModel):
    item: Dict[str, object]

    sample: Optional[Dict[str, object]] = None


class SourceFileContent(BaseModel):
    content: List[SourceFileContentContent]
    """The content of the jsonl file."""

    type: Literal["file_content"]
    """The type of jsonl source. Always `file_content`."""


class SourceFileID(BaseModel):
    id: str
    """The identifier of the file."""

    type: Literal["file_id"]
    """The type of jsonl source. Always `file_id`."""


Source: TypeAlias = Annotated[
    Union[SourceFileContent, SourceFileID], PropertyInfo(discriminator="type")
]


class CreateEvalJSONLRunDataSource(BaseModel):
    source: Source
    """Determines what populates the `item` namespace in the data source."""

    type: Literal["jsonl"]
    """The type of data source. Always `jsonl`."""
