# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import List, Optional
from typing_extensions import Literal

from ....._models import BaseModel

__all__ = [
    "FileSearchToolCall",
    "FileSearch",
    "FileSearchRankingOptions",
    "FileSearchResult",
    "FileSearchResultContent",
]


class FileSearchRankingOptions(BaseModel):
    ranker: Literal["auto", "default_2024_08_21"]
    """The ranker to use for the file search.

    If not specified will use the `auto` ranker.
    """

    score_threshold: float
    """The score threshold for the file search.

    All values must be a floating point number between 0 and 1.
    """


class FileSearchResultContent(BaseModel):
    text: Optional[str] = None
    """The text content of the file."""

    type: Optional[Literal["text"]] = None
    """The type of the content."""


class FileSearchResult(BaseModel):
    file_id: str
    """The ID of the file that result was found in."""

    file_name: str
    """The name of the file that result was found in."""

    score: float
    """The score of the result.

    All values must be a floating point number between 0 and 1.
    """

    content: Optional[List[FileSearchResultContent]] = None
    """The content of the result that was found.

    The content is only included if requested via the include query parameter.
    """


class FileSearch(BaseModel):
    ranking_options: Optional[FileSearchRankingOptions] = None
    """The ranking options for the file search."""

    results: Optional[List[FileSearchResult]] = None
    """The results of the file search."""


class FileSearchToolCall(BaseModel):
    id: str
    """The ID of the tool call object."""

    file_search: FileSearch
    """For now, this is always going to be an empty object."""

    type: Literal["file_search"]
    """The type of tool call.

    This is always going to be `file_search` for this type of tool call.
    """
