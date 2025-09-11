# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing import Dict, Union, Iterable
from typing_extensions import Literal, Required, TypeAlias, TypedDict

__all__ = [
    "CreateEvalJSONLRunDataSourceParam",
    "Source",
    "SourceFileContent",
    "SourceFileContentContent",
    "SourceFileID",
]


class SourceFileContentContent(TypedDict, total=False):
    item: Required[Dict[str, object]]

    sample: Dict[str, object]


class SourceFileContent(TypedDict, total=False):
    content: Required[Iterable[SourceFileContentContent]]
    """The content of the jsonl file."""

    type: Required[Literal["file_content"]]
    """The type of jsonl source. Always `file_content`."""


class SourceFileID(TypedDict, total=False):
    id: Required[str]
    """The identifier of the file."""

    type: Required[Literal["file_id"]]
    """The type of jsonl source. Always `file_id`."""


Source: TypeAlias = Union[SourceFileContent, SourceFileID]


class CreateEvalJSONLRunDataSourceParam(TypedDict, total=False):
    source: Required[Source]
    """Determines what populates the `item` namespace in the data source."""

    type: Required[Literal["jsonl"]]
    """The type of data source. Always `jsonl`."""
